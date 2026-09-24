import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

type Label = "illicit" | "licit" | "unknown";

type NodeInfo = {
  id: string;
  label: Label;
  incoming: number;
  outgoing: number;
  neighbors: Set<string>;
};

let nodesCache: Map<string, NodeInfo> | null = null;

function getLabel(value: string): Label {
  if (value.trim() === "1") return "illicit";
  if (value.trim() === "2") return "licit";
  return "unknown";
}

function loadGraph() {
  if (nodesCache) {
    return nodesCache;
  }

  const root = path.resolve(
    process.cwd(),
    "../data/raw/elliptic"
  );

  const classesFile = path.join(
    root,
    "elliptic_txs_classes.csv"
  );

  const edgesFile = path.join(
    root,
    "elliptic_txs_edgelist.csv"
  );

  const nodes = new Map<string, NodeInfo>();

  const classes = fs.readFileSync(
    classesFile,
    "utf8"
  );

  const classLines = classes
    .split(/\r?\n/)
    .filter(Boolean);

  for (let i = 1; i < classLines.length; i++) {
    const [id, classValue] = classLines[i]
      .split(",");

    if (!id) continue;

    nodes.set(id.trim(), {
      id: id.trim(),
      label: getLabel(classValue),
      incoming: 0,
      outgoing: 0,
      neighbors: new Set(),
    });
  }

  const edges = fs.readFileSync(
    edgesFile,
    "utf8"
  );

  const edgeLines = edges
    .split(/\r?\n/)
    .filter(Boolean);

  for (let i = 1; i < edgeLines.length; i++) {
    const [source, target] = edgeLines[i]
      .split(",")
      .map((value) => value.trim());

    if (!source || !target) continue;

    if (!nodes.has(source)) {
      nodes.set(source, {
        id: source,
        label: "unknown",
        incoming: 0,
        outgoing: 0,
        neighbors: new Set(),
      });
    }

    if (!nodes.has(target)) {
      nodes.set(target, {
        id: target,
        label: "unknown",
        incoming: 0,
        outgoing: 0,
        neighbors: new Set(),
      });
    }

    nodes.get(source)!.outgoing++;
    nodes.get(target)!.incoming++;

    nodes.get(source)!.neighbors.add(target);
    nodes.get(target)!.neighbors.add(source);
  }

  nodesCache = nodes;

  return nodes;
}

export async function GET(request: NextRequest) {
  try {
    const nodes = loadGraph();

    const requestedId =
      request.nextUrl.searchParams.get("txId");

    const firstId = nodes.keys().next().value;

    const txId = requestedId || firstId;

    if (!txId) {
      return NextResponse.json(
        { error: "No transaction found." },
        { status: 404 }
      );
    }

    const node = nodes.get(txId);

    if (!node) {
      return NextResponse.json(
        {
          error: `Transaction ${txId} was not found.`,
        },
        { status: 404 }
      );
    }

    const neighbors = Array.from(node.neighbors)
      .slice(0, 30)
      .map((id) => {
        const neighbor = nodes.get(id)!;

        return {
          id,
          label: neighbor.label,
          incoming: neighbor.incoming,
          outgoing: neighbor.outgoing,
          degree:
            neighbor.incoming +
            neighbor.outgoing,
        };
      });

    const illicit =
      neighbors.filter(
        (n) => n.label === "illicit"
      ).length;

    const licit =
      neighbors.filter(
        (n) => n.label === "licit"
      ).length;

    const unknown =
      neighbors.filter(
        (n) => n.label === "unknown"
      ).length;

    const labeled = illicit + licit;

    return NextResponse.json({
      transaction: {
        id: node.id,
        label: node.label,
        incoming: node.incoming,
        outgoing: node.outgoing,
        degree: node.neighbors.size,
      },

      neighborhood: {
        total: neighbors.length,
        illicit,
        licit,
        unknown,
        illicitRatio:
          labeled > 0 ? illicit / labeled : 0,
      },

      neighbors,

      graph: {
        totalNodes: nodes.size,
        queriedNeighbors: neighbors.length,
      },
    });
  } catch (error) {
    console.error(error);

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Graph loading failed.",
      },
      { status: 500 }
    );
  }
}