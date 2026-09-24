import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const ROOT = path.resolve(process.cwd(), "..");

function readCSV(relativePath: string) {
  const filePath = path.join(ROOT, relativePath);

  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${relativePath}`);
  }

  const lines = fs
    .readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim());

  if (!lines.length) return [];

  const headers = lines[0]
    .split(",")
    .map((x) => x.trim().toLowerCase().replace(/[\s-]+/g, "_"));

  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: Record<string, string> = {};

    headers.forEach((header, index) => {
      row[header] = values[index]?.trim() ?? "";
    });

    return row;
  });
}

export async function GET() {
  try {
    return NextResponse.json({
      featureAblation: readCSV(
        "results/final/master_feature_ablation.csv"
      ),

      gnnAblation: readCSV(
        "results/final/master_gnn_feature_ablation.csv"
      ),

      robustness: readCSV(
        "results/final/master_robustness_results.csv"
      ),

      statistics: readCSV(
        "results/final/master_statistical_comparisons.csv"
      ),

      efficiency: readCSV(
        "results/final/master_computational_efficiency.csv"
      ),
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to load experiment results.",
      },
      { status: 500 }
    );
  }
}