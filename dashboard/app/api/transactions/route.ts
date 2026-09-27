import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

type GraphRow = {
  txId: string;
  timeStep: number;
  label: number;
  labeled: boolean;
  in_degree: number;
  out_degree: number;
  total_degree: number;
  class_name: string;
};

type NeighborNode = {
  txId: string;
  direction: "incoming" | "outgoing";
  label: string;
  labeled: boolean;
  timeStep: number | null;
  degree: number;
};

const ROOT = path.resolve(process.cwd(), "..");

/*
 * ============================================================
 * FULL TRANSACTION METADATA
 * ============================================================
 *
 * This file was generated from:
 *
 * data/raw/elliptic/elliptic_txs_features.csv
 * data/raw/elliptic/elliptic_txs_classes.csv
 *
 * It contains only:
 *
 * txId
 * time_step
 * class_name
 *
 * This avoids loading the ~658 MB feature matrix into memory.
 */
const TRANSACTION_METADATA_PATH = path.join(
  ROOT,
  "data",
  "processed",
  "elliptic_transaction_metadata.csv"
);

/*
 * Complete Elliptic edge list.
 */
const EDGE_LIST_PATH = path.join(
  ROOT,
  "data",
  "raw",
  "elliptic",
  "elliptic_txs_edgelist.csv"
);

/*
 * Test-set research data.
 *
 * These remain useful for:
 * - test-set neighbor statistics
 * - test transaction IDs
 * - model predictions
 */
const NEIGHBOR_STRUCTURE_PATH = path.join(
  ROOT,
  "results",
  "neighbor_structure_test.csv"
);

const TEST_TEMPORAL_PATH = path.join(
  ROOT,
  "data",
  "processed",
  "test_temporal.csv"
);

const PREDICTION_DIR = path.join(
  ROOT,
  "results",
  "predictions"
);

/*
 * Validation-selected thresholds from the
 * GraphShield research experiments.
 */
const THRESHOLDS: Record<string, number> = {
  "Logistic Regression": 0.93,
  "Random Forest": 0.57,
  XGBoost: 0.63,
  GCN: 0.53,
  GraphSAGE: 0.90,
  GAT: 0.88,
};

/*
 * Prediction files contain probabilities for
 * the labeled test transactions.
 */
const PREDICTION_FILES: Record<string, string> = {
  "Logistic Regression":
    "logistic_regression_test.csv",

  "Random Forest":
    "random_forest_test.csv",

  XGBoost:
    "xgboost_test.csv",

  GCN:
    "gcn_test.csv",

  GraphSAGE:
    "graphsage_test.csv",

  GAT:
    "gat_test.csv",
};

/*
 * ============================================================
 * CSV HELPERS
 * ============================================================
 */

function parseCsvLine(line: string): string[] {
  return line
    .split(",")
    .map((value) =>
      value.trim().replace(/^"|"$/g, "")
    );
}

function readCsv(filePath: string): string[][] {
  if (!fs.existsSync(filePath)) {
    throw new Error(
      `Required file not found: ${filePath}`
    );
  }

  const content = fs.readFileSync(
    filePath,
    "utf8"
  );

  return content
    .split(/\r?\n/)
    .filter(
      (line) => line.trim().length > 0
    )
    .map(parseCsvLine);
}

/*
 * ============================================================
 * FULL TRANSACTION GRAPH
 * ============================================================
 *
 * Reads the lightweight metadata file and then
 * calculates degree information from the complete
 * Elliptic edge list.
 */
function loadFullTransactionGraph(): Map<
  string,
  GraphRow
> {
  const metadataRows =
    readCsv(
      TRANSACTION_METADATA_PATH
    );

  const graphMap =
    new Map<string, GraphRow>();

  if (metadataRows.length === 0) {
    throw new Error(
      "Transaction metadata file is empty."
    );
  }

  const header =
    metadataRows[0];

  const txIdIndex =
    header.indexOf("txId");

  const timeStepIndex =
    header.indexOf("time_step");

  const classNameIndex =
    header.indexOf("class_name");

  if (
    txIdIndex === -1 ||
    timeStepIndex === -1 ||
    classNameIndex === -1
  ) {
    throw new Error(
      "Invalid transaction metadata file. Expected txId,time_step,class_name."
    );
  }

  /*
   * Build all 203,769 transaction nodes.
   */
  for (
    let i = 1;
    i < metadataRows.length;
    i++
  ) {
    const row =
      metadataRows[i];

    const txId =
      row[txIdIndex];

    if (!txId) {
      continue;
    }

    const timeStep =
      Number(
        row[timeStepIndex]
      );

    const className =
      row[classNameIndex] ||
      "unknown";

    /*
     * Internal class encoding:
     *
     * 1  = illicit
     * 0  = licit
     * -1 = unknown
     */
    let label = -1;

    if (className === "illicit") {
      label = 1;
    } else if (
      className === "licit"
    ) {
      label = 0;
    }

    graphMap.set(txId, {
      txId,

      timeStep:
        Number.isFinite(timeStep)
          ? timeStep
          : 0,

      label,

      labeled:
        label >= 0,

      in_degree: 0,

      out_degree: 0,

      total_degree: 0,

      class_name:
        className,
    });
  }

  /*
   * ==========================================================
   * BUILD COMPLETE GRAPH DEGREE INFORMATION
   * ==========================================================
   */

  const edgeRows =
    readCsv(EDGE_LIST_PATH);

  if (edgeRows.length > 0) {
    const header =
      edgeRows[0];

    let sourceIndex =
      header.indexOf("txId1");

    let targetIndex =
      header.indexOf("txId2");

    let start = 1;

    /*
     * Fallback for a headerless edge file.
     */
    if (
      sourceIndex === -1 ||
      targetIndex === -1
    ) {
      sourceIndex = 0;
      targetIndex = 1;
      start = 0;
    }

    for (
      let i = start;
      i < edgeRows.length;
      i++
    ) {
      const row =
        edgeRows[i];

      const source =
        row[sourceIndex];

      const target =
        row[targetIndex];

      if (!source || !target) {
        continue;
      }

      const sourceNode =
        graphMap.get(source);

      const targetNode =
        graphMap.get(target);

      /*
       * source -> target
       *
       * Therefore:
       * source gets outgoing degree
       * target gets incoming degree
       */
      if (sourceNode) {
        sourceNode.out_degree++;
        sourceNode.total_degree++;
      }

      if (targetNode) {
        targetNode.in_degree++;
        targetNode.total_degree++;
      }
    }
  }

  return graphMap;
}

/*
 * ============================================================
 * EDGE LIST
 * ============================================================
 */

function loadEdges(): Array<{
  source: string;
  target: string;
}> {
  const rows =
    readCsv(EDGE_LIST_PATH);

  if (rows.length === 0) {
    return [];
  }

  const header =
    rows[0];

  let sourceIndex =
    header.indexOf("txId1");

  let targetIndex =
    header.indexOf("txId2");

  let start = 1;

  /*
   * Fallback for headerless files.
   */
  if (
    sourceIndex === -1 ||
    targetIndex === -1
  ) {
    sourceIndex = 0;
    targetIndex = 1;
    start = 0;
  }

  const edges: Array<{
    source: string;
    target: string;
  }> = [];

  for (
    let i = start;
    i < rows.length;
    i++
  ) {
    const row =
      rows[i];

    const source =
      row[sourceIndex];

    const target =
      row[targetIndex];

    if (!source || !target) {
      continue;
    }

    edges.push({
      source,
      target,
    });
  }

  return edges;
}

/*
 * ============================================================
 * BUILD TRANSACTION NEIGHBORS
 * ============================================================
 */

function buildNeighbors(
  txId: string,
  graphMap: Map<string, GraphRow>,
  edges: Array<{
    source: string;
    target: string;
  }>
): NeighborNode[] {
  const neighbors: NeighborNode[] = [];

  for (const edge of edges) {
    /*
     * Incoming:
     *
     * edge.source -> txId
     */
    if (edge.target === txId) {
      const node =
        graphMap.get(edge.source);

      neighbors.push({
        txId: edge.source,

        direction: "incoming",

        label:
          node?.class_name ??
          "unknown",

        labeled:
          node?.labeled ??
          false,

        timeStep:
          node?.timeStep ??
          null,

        degree:
          node?.total_degree ??
          0,
      });
    }

    /*
     * Outgoing:
     *
     * txId -> edge.target
     */
    if (edge.source === txId) {
      const node =
        graphMap.get(edge.target);

      neighbors.push({
        txId: edge.target,

        direction: "outgoing",

        label:
          node?.class_name ??
          "unknown",

        labeled:
          node?.labeled ??
          false,

        timeStep:
          node?.timeStep ??
          null,

        degree:
          node?.total_degree ??
          0,
      });
    }
  }

  /*
   * Remove duplicate relationships.
   */
  const unique =
    new Map<
      string,
      NeighborNode
    >();

  for (const neighbor of neighbors) {
    const key =
      `${neighbor.direction}:${neighbor.txId}`;

    if (!unique.has(key)) {
      unique.set(
        key,
        neighbor
      );
    }
  }

  return Array.from(
    unique.values()
  );
}

/*
 * ============================================================
 * TEST TRANSACTION IDS
 * ============================================================
 */

function loadTestTransactionIds(): string[] {
  const rows =
    readCsv(
      TEST_TEMPORAL_PATH
    );

  if (rows.length === 0) {
    return [];
  }

  const header =
    rows[0];

  const txIndex =
    header.indexOf("txId");

  if (txIndex === -1) {
    throw new Error(
      "txId column not found in test_temporal.csv"
    );
  }

  return rows
    .slice(1)
    .map(
      (row) =>
        row[txIndex]
    )
    .filter(Boolean);
}

/*
 * ============================================================
 * MODEL PREDICTIONS
 * ============================================================
 */

function loadPredictions(
  model: string
): number[] {
  const fileName =
    PREDICTION_FILES[model];

  const filePath =
    path.join(
      PREDICTION_DIR,
      fileName
    );

  const rows =
    readCsv(filePath);

  const probabilities: number[] =
    [];

  for (
    let i = 1;
    i < rows.length;
    i++
  ) {
    if (!rows[i][1]) {
      continue;
    }

    const probability =
      Number(rows[i][1]);

    if (
      Number.isFinite(
        probability
      )
    ) {
      probabilities.push(
        probability
      );
    }
  }

  return probabilities;
}

/*
 * ============================================================
 * API
 * ============================================================
 */

export async function GET(
  request: NextRequest
) {
  try {
    /*
     * --------------------------------------------------------
     * Requested transaction
     * --------------------------------------------------------
     */
    const searchParams =
      request.nextUrl.searchParams;

    const txId =
      searchParams.get("txId");

    if (!txId) {
      return NextResponse.json(
        {
          error:
            "Transaction ID is required.",
        },
        { status: 400 }
      );
    }

    /*
     * --------------------------------------------------------
     * 1. Load the COMPLETE Elliptic transaction graph
     * --------------------------------------------------------
     */
    const graphMap =
      loadFullTransactionGraph();

    const graphRow =
      graphMap.get(txId);

    if (!graphRow) {
      return NextResponse.json(
        {
          error:
            `Transaction ${txId} was not found in the Elliptic graph.`,
        },
        { status: 404 }
      );
    }

    /*
     * --------------------------------------------------------
     * 2. Load complete edge list
     * --------------------------------------------------------
     */
    const edges =
      loadEdges();

    /*
     * --------------------------------------------------------
     * 3. Build complete neighborhood
     * --------------------------------------------------------
     */
    const allNeighbors =
      buildNeighbors(
        txId,
        graphMap,
        edges
      );

    /*
     * --------------------------------------------------------
     * 4. Sort neighbors
     *
     * Priority:
     * 1. illicit
     * 2. licit
     * 3. unknown
     * --------------------------------------------------------
     */
    const priority = (
      label: string
    ) => {
      if (
        label.toLowerCase() ===
        "illicit"
      ) {
        return 0;
      }

      if (
        label.toLowerCase() ===
        "licit"
      ) {
        return 1;
      }

      return 2;
    };

    allNeighbors.sort(
      (a, b) =>
        priority(a.label) -
        priority(b.label)
    );

    /*
     * Only display the first 20 in
     * the detailed graph panel.
     */
    const displayNeighbors =
      allNeighbors.slice(
        0,
        20
      );

    /*
     * --------------------------------------------------------
     * 5. Calculate complete-graph neighbor statistics
     * --------------------------------------------------------
     */
    const illicitNeighbors =
      allNeighbors.filter(
        (node) =>
          node.label ===
          "illicit"
      ).length;

    const licitNeighbors =
      allNeighbors.filter(
        (node) =>
          node.label ===
          "licit"
      ).length;

    const unknownNeighbors =
      allNeighbors.filter(
        (node) =>
          node.label ===
          "unknown"
      ).length;

    const labeledNeighborCount =
      illicitNeighbors +
      licitNeighbors;

    const calculatedIllicitRatio =
      labeledNeighborCount > 0
        ? illicitNeighbors /
          labeledNeighborCount
        : 0;

    /*
     * --------------------------------------------------------
     * 6. Load test-set neighbor statistics if available
     * --------------------------------------------------------
     *
     * These statistics are only available for
     * transactions belonging to the research
     * test population.
     */
    let testNeighborRow:
      string[] | undefined;

    let testNeighborHeader:
      string[] = [];

    if (
      fs.existsSync(
        NEIGHBOR_STRUCTURE_PATH
      )
    ) {
      const neighborRows =
        readCsv(
          NEIGHBOR_STRUCTURE_PATH
        );

      if (neighborRows.length > 0) {
        testNeighborHeader =
          neighborRows[0];

        const neighborTxIndex =
          testNeighborHeader.indexOf(
            "txId"
          );

        if (neighborTxIndex >= 0) {
          for (
            let i = 1;
            i < neighborRows.length;
            i++
          ) {
            if (
              neighborRows[i][
                neighborTxIndex
              ] === txId
            ) {
              testNeighborRow =
                neighborRows[i];

              break;
            }
          }
        }
      }
    }

    /*
     * Helper for test-set-only columns.
     */
    const getTestMetric = (
      column: string
    ): number | null => {
      if (
        !testNeighborRow
      ) {
        return null;
      }

      const index =
        testNeighborHeader.indexOf(
          column
        );

      if (index < 0) {
        return null;
      }

      const value =
        Number(
          testNeighborRow[index]
        );

      return Number.isFinite(value)
        ? value
        : null;
    };

    /*
     * --------------------------------------------------------
     * 7. Transaction information
     * --------------------------------------------------------
     */
    const transaction = {
      txId,

      label:
        graphRow.class_name,

      labelValue:
        String(graphRow.label),

      labeled:
        graphRow.labeled,

      timeStep:
        graphRow.timeStep,

      incoming:
        graphRow.in_degree,

      outgoing:
        graphRow.out_degree,

      totalDegree:
        graphRow.total_degree,

      /*
       * For test transactions:
       * use the research-computed neighbor statistics.
       *
       * For other graph transactions:
       * calculate directly from the complete graph.
       */
      illicitNeighborRatio:
        testNeighborRow
          ? getTestMetric(
              "illicit_neighbor_ratio"
            ) ??
            calculatedIllicitRatio
          : calculatedIllicitRatio,

      illicitNeighbors:
        testNeighborRow
          ? getTestMetric(
              "illicit_neighbors"
            ) ??
            illicitNeighbors
          : illicitNeighbors,

      licitNeighbors:
        testNeighborRow
          ? getTestMetric(
              "licit_neighbors"
            ) ??
            licitNeighbors
          : licitNeighbors,

      unknownNeighbors:
        testNeighborRow
          ? getTestMetric(
              "unknown_neighbors"
            ) ??
            unknownNeighbors
          : unknownNeighbors,

      totalNeighbors:
        testNeighborRow
          ? getTestMetric(
              "total_neighbors"
            ) ??
            allNeighbors.length
          : allNeighbors.length,
    };

    /*
     * --------------------------------------------------------
     * 8. Model predictions
     * --------------------------------------------------------
     *
     * The prediction files contain predictions
     * only for the 9,973 labeled test transactions.
     *
     * Therefore:
     *
     * Test transaction:
     * probability available
     *
     * Unknown/full-graph transaction:
     * probability = null
     * prediction = N/A
     */
    const testIds =
      loadTestTransactionIds();

    const traditionalIndex =
      testIds.indexOf(txId);

    /*
     * The GNN prediction files are aligned
     * to the same labeled test transaction
     * population.
     */
    const labeledGraphIds =
      testIds;

    const gnnIndex =
      labeledGraphIds.indexOf(txId);

    const models: Array<{
      model: string;
      type: string;
      probability:
        | number
        | null;
      threshold: number;
      prediction:
        | "ILLICIT"
        | "LICIT"
        | "N/A";
    }> = [];

    for (
      const model of Object.keys(
        PREDICTION_FILES
      )
    ) {
      const probabilities =
        loadPredictions(model);

      const isGnn =
        model === "GCN" ||
        model === "GraphSAGE" ||
        model === "GAT";

      const index =
        isGnn
          ? gnnIndex
          : traditionalIndex;

      const probability =
        index >= 0 &&
        index <
          probabilities.length
          ? probabilities[index]
          : null;

      const threshold =
        THRESHOLDS[model];

      let prediction:
        | "ILLICIT"
        | "LICIT"
        | "N/A" =
        "N/A";

      if (
        probability !== null
      ) {
        prediction =
          probability >=
          threshold
            ? "ILLICIT"
            : "LICIT";
      }

      models.push({
        model,

        type: isGnn
          ? "Graph Neural Network"
          : "Traditional ML",

        probability,

        threshold,

        prediction,
      });
    }

    /*
     * --------------------------------------------------------
     * 9. Consensus
     * --------------------------------------------------------
     */
    const illicitVotes =
      models.filter(
        (model) =>
          model.prediction ===
          "ILLICIT"
      ).length;

    const availableModels =
      models.filter(
        (model) =>
          model.probability !==
          null
      ).length;

    /*
     * --------------------------------------------------------
     * 10. Graph summary
     * --------------------------------------------------------
     */
    const graph = {
      center: {
        txId,

        label:
          graphRow.class_name,

        timeStep:
          graphRow.timeStep,

        degree:
          graphRow.total_degree,
      },

      neighbors:
        displayNeighbors,

      totalNeighbors:
        allNeighbors.length,

      displayedNeighbors:
        displayNeighbors.length,

      incomingNeighbors:
        displayNeighbors.filter(
          (node) =>
            node.direction ===
            "incoming"
        ).length,

      outgoingNeighbors:
        displayNeighbors.filter(
          (node) =>
            node.direction ===
            "outgoing"
        ).length,

      truncated:
        allNeighbors.length >
        displayNeighbors.length,
    };

    /*
     * --------------------------------------------------------
     * 11. Return response
     * --------------------------------------------------------
     */
    return NextResponse.json({
      transaction,

      models,

      consensus: {
        illicitVotes,

        totalModels:
          availableModels,
      },

      graph,

      source: {
        transactionMetadata:
          "data/processed/elliptic_transaction_metadata.csv",

        neighborStructure:
          "results/neighbor_structure_test.csv",

        edgeList:
          "data/raw/elliptic/elliptic_txs_edgelist.csv",

        predictions:
          "results/predictions/",
      },
    });
  } catch (error) {
    console.error(
      "Transaction API error:",
      error
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Internal server error.",
      },
      { status: 500 }
    );
  }
}