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

const GRAPH_STRUCTURE_PATH = path.join(
  ROOT,
  "results",
  "graph_structure_test.csv"
);

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

const EDGE_LIST_PATH = path.join(
  ROOT,
  "data",
  "raw",
  "elliptic",
  "elliptic_txs_edgelist.csv"
);

const PREDICTION_DIR = path.join(
  ROOT,
  "results",
  "predictions"
);

const THRESHOLDS: Record<string, number> = {
  "Logistic Regression": 0.93,
  "Random Forest": 0.57,
  XGBoost: 0.63,
  GCN: 0.53,
  GraphSAGE: 0.90,
  GAT: 0.88,
};

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

function parseCsvLine(line: string): string[] {
  return line.split(",").map((value) =>
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
    .filter((line) => line.trim().length > 0)
    .map(parseCsvLine);
}

function loadGraphStructure(): Map<
  string,
  GraphRow
> {
  const rows = readCsv(GRAPH_STRUCTURE_PATH);

  const map = new Map<string, GraphRow>();

  for (let i = 1; i < rows.length; i++) {
    const row = rows[i];

    if (!row[0]) continue;

    map.set(row[0], {
      txId: row[0],
      timeStep: Number(row[1]),
      label: Number(row[2]),
      labeled:
        row[3]?.toLowerCase() === "true",
      in_degree: Number(row[4]),
      out_degree: Number(row[5]),
      total_degree: Number(row[6]),
      class_name:
        row[7] || "unknown",
    });
  }

  return map;
}

function loadPredictions(
  model: string
): number[] {
  const fileName =
    PREDICTION_FILES[model];

  const filePath = path.join(
    PREDICTION_DIR,
    fileName
  );

  const rows = readCsv(filePath);

  const probabilities: number[] = [];

  for (let i = 1; i < rows.length; i++) {
    if (!rows[i][1]) continue;

    probabilities.push(
      Number(rows[i][1])
    );
  }

  return probabilities;
}

function loadTestTransactionIds(): string[] {
  const rows =
    readCsv(TEST_TEMPORAL_PATH);

  const header = rows[0];

  const txIndex =
    header.indexOf("txId");

  if (txIndex === -1) {
    throw new Error(
      "txId column not found in test_temporal.csv"
    );
  }

  return rows
    .slice(1)
    .map((row) => row[txIndex])
    .filter(Boolean);
}

function loadEdges(): Array<{
  source: string;
  target: string;
}> {
  const rows =
    readCsv(EDGE_LIST_PATH);

  if (rows.length === 0) {
    return [];
  }

  const header = rows[0];

  let sourceIndex =
    header.indexOf("txId1");

  let targetIndex =
    header.indexOf("txId2");

  /*
   * Fallback in case the edge file has no
   * header or uses different column names.
   */
  if (
    sourceIndex === -1 ||
    targetIndex === -1
  ) {
    sourceIndex = 0;
    targetIndex = 1;
  }

  const edges: Array<{
    source: string;
    target: string;
  }> = [];

  const start =
    header[0] === "txId1" ? 1 : 0;

  for (
    let i = start;
    i < rows.length;
    i++
  ) {
    const row = rows[i];

    if (!row[sourceIndex] || !row[targetIndex]) {
      continue;
    }

    edges.push({
      source: row[sourceIndex],
      target: row[targetIndex],
    });
  }

  return edges;
}

function buildNeighbors(
  txId: string,
  graphMap: Map<string, GraphRow>
): NeighborNode[] {
  const edges = loadEdges();

  const neighbors: NeighborNode[] = [];

  for (const edge of edges) {

    if (edge.target === txId) {
      const node =
        graphMap.get(edge.source);

      neighbors.push({
        txId: edge.source,
        direction: "incoming",
        label:
          node?.class_name || "unknown",
        labeled:
          node?.labeled ?? false,
        timeStep:
          node?.timeStep ?? null,
        degree:
          node?.total_degree ?? 0,
      });
    }

    if (edge.source === txId) {
      const node =
        graphMap.get(edge.target);

      neighbors.push({
        txId: edge.target,
        direction: "outgoing",
        label:
          node?.class_name || "unknown",
        labeled:
          node?.labeled ?? false,
        timeStep:
          node?.timeStep ?? null,
        degree:
          node?.total_degree ?? 0,
      });
    }
  }

  /*
   * Remove duplicate relationships.
   */
  const unique = new Map<
    string,
    NeighborNode
  >();

  for (const neighbor of neighbors) {
    const key =
      `${neighbor.direction}:${neighbor.txId}`;

    if (!unique.has(key)) {
      unique.set(key, neighbor);
    }
  }

  return Array.from(unique.values());
}

export async function GET(
  request: NextRequest
) {
  try {

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
     * Load graph structure.
     */
    const graphMap =
      loadGraphStructure();

    const graphRow =
      graphMap.get(txId);

    if (!graphRow) {
      return NextResponse.json(
        {
          error:
            `Transaction ${txId} was not found in the test graph.`,
        },
        { status: 404 }
      );
    }

    /*
     * Neighbor structure.
     */
    const neighborRows =
      readCsv(
        NEIGHBOR_STRUCTURE_PATH
      );

    const neighborHeader =
      neighborRows[0];

    const neighborMap =
      new Map<string, string[]>();

    const neighborTxIndex =
      neighborHeader.indexOf("txId");

    for (
      let i = 1;
      i < neighborRows.length;
      i++
    ) {
      const row =
        neighborRows[i];

      if (
        neighborTxIndex >= 0 &&
        row[neighborTxIndex]
      ) {
        neighborMap.set(
          row[neighborTxIndex],
          row
        );
      }
    }

    const neighborRow =
      neighborMap.get(txId);

    /*
     * Transaction information.
     */
    const transaction = {
      txId,

      label:
        graphRow.class_name ||
        "unknown",

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

      illicitNeighborRatio:
        neighborRow
          ? Number(
              neighborRow[
                neighborHeader.indexOf(
                  "illicit_neighbor_ratio"
                )
              ]
            )
          : 0,

      illicitNeighbors:
        neighborRow
          ? Number(
              neighborRow[
                neighborHeader.indexOf(
                  "illicit_neighbors"
                )
              ]
            )
          : 0,

      licitNeighbors:
        neighborRow
          ? Number(
              neighborRow[
                neighborHeader.indexOf(
                  "licit_neighbors"
                )
              ]
            )
          : 0,

      unknownNeighbors:
        neighborRow
          ? Number(
              neighborRow[
                neighborHeader.indexOf(
                  "unknown_neighbors"
                )
              ]
            )
          : 0,

      totalNeighbors:
        neighborRow
          ? Number(
              neighborRow[
                neighborHeader.indexOf(
                  "total_neighbors"
                )
              ]
            )
          : graphRow.total_degree,
    };

    /*
     * Get actual graph neighbors.
     */
    const allNeighbors =
      buildNeighbors(
        txId,
        graphMap
      );

    /*
     * Keep the graph visualization manageable.
     *
     * We prioritize:
     * 1. Illicit
     * 2. Licit
     * 3. Unknown
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

    const displayNeighbors =
      allNeighbors.slice(0, 20);

    /*
     * Load prediction IDs.
     */
    const testIds =
      loadTestTransactionIds();

    const traditionalIndex =
      testIds.indexOf(txId);

    /*
     * GNN predictions are aligned with
     * labeled graph nodes. The test graph
     * structure contains the same labeled
     * transaction population.
     */
    const labeledGraphIds =
      Array.from(graphMap.values())
        .filter((node) => node.labeled)
        .map((node) => node.txId);

    const gnnIndex =
      labeledGraphIds.indexOf(txId);

    const models: Array<{
      model: string;
      type: string;
      probability: number | null;
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
        index < probabilities.length
          ? probabilities[index]
          : null;

      const threshold =
        THRESHOLDS[model];

      let prediction:
        | "ILLICIT"
        | "LICIT"
        | "N/A" = "N/A";

      if (
        probability !== null
      ) {
        prediction =
          probability >= threshold
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

    const illicitVotes =
      models.filter(
        (model) =>
          model.prediction ===
          "ILLICIT"
      ).length;

    /*
     * Graph summary.
     */
    const graph = {
      center: {
        txId,
        label:
          graphRow.class_name ||
          "unknown",
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

    return NextResponse.json({
      transaction,

      models,

      consensus: {
        illicitVotes,
        totalModels:
          models.length,
      },

      graph,

      source: {
        graphStructure:
          "results/graph_structure_test.csv",

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