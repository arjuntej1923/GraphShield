import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const ROOT = path.resolve(process.cwd(), "..");
const RESULTS = path.join(ROOT, "results");

type CsvRow = Record<string, string>;

function readCsv(filePath: string): CsvRow[] {
  const content = fs.readFileSync(filePath, "utf8").trim();

  if (!content) return [];

  const lines = content.split(/\r?\n/);

  if (lines.length < 2) return [];

  const headers = lines[0]
    .split(",")
    .map((h) => h.trim().replace(/^"|"$/g, ""));

  return lines.slice(1).map((line) => {
    const values = line.split(",");

    const row: CsvRow = {};

    headers.forEach((header, index) => {
      row[header] = values[index]?.trim().replace(/^"|"$/g, "") ?? "";
    });

    return row;
  });
}

function normalizeKey(key: string): string {
  return key
    .toLowerCase()
    .replace(/[\s_-]/g, "");
}

function getValue(
  row: CsvRow,
  possibleKeys: string[]
): string | undefined {
  const normalized = new Map<string, string>();

  Object.entries(row).forEach(([key, value]) => {
    normalized.set(normalizeKey(key), value);
  });

  for (const key of possibleKeys) {
    const value = normalized.get(normalizeKey(key));

    if (value !== undefined && value !== "") {
      return value;
    }
  }

  return undefined;
}

function getNumber(
  row: CsvRow,
  possibleKeys: string[],
  fallback = 0
): number {
  const value = getValue(row, possibleKeys);

  if (value === undefined) return fallback;

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : fallback;
}

function getRank(row: CsvRow): number | null {
  const value = getValue(row, [
    "rank",
    "Rank",
    "ranking",
    "feature_rank",
    "featureRank",
    "order",
    "position",
  ]);

  if (value === undefined) return null;

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : null;
}

function getFeature(row: CsvRow): string {
  return (
    getValue(row, [
      "feature",
      "Feature",
      "feature_name",
      "featureName",
      "name",
    ]) ?? ""
  );
}

function getImportance(row: CsvRow): number {
  const direct = getValue(row, [
    "importance",
    "Importance",
    "importance_score",
    "importanceScore",
    "feature_importance",
    "featureImportance",
    "mean_importance",
    "meanImportance",
    "score",
    "value",
    "gain",
    "weight",
  ]);

  if (direct !== undefined) {
    const parsed = Number(direct);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  /*
   * Fallback:
   * If the CSV uses an unexpected importance column name,
   * find the first numeric column that is not a rank/index.
   */
  for (const [key, value] of Object.entries(row)) {
    const normalizedKey = normalizeKey(key);

    if (
      normalizedKey.includes("rank") ||
      normalizedKey.includes("index")
    ) {
      continue;
    }

    const parsed = Number(value);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return 0;
}

function parseModelFeatureRows(rows: CsvRow[]) {
  return rows
    .map((row) => ({
      feature: getFeature(row),
      importance: getImportance(row),
      rank: getRank(row),
    }))
    .filter((row) => row.feature !== "");
}

function getOptionalNumber(
  row: CsvRow,
  possibleKeys: string[]
): number | null {
  const value = getValue(row, possibleKeys);

  if (value === undefined) return null;

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : null;
}

export async function GET() {
  try {
    const rfPath = path.join(
      RESULTS,
      "feature_importance",
      "random_forest_feature_importance.csv"
    );

    const xgbPath = path.join(
      RESULTS,
      "feature_importance",
      "xgboost_feature_importance.csv"
    );

    const combinedPath = path.join(
      RESULTS,
      "feature_importance",
      "combined_feature_importance.csv"
    );

    const overlapPath = path.join(
      RESULTS,
      "feature_importance",
      "top_k_feature_overlap.csv"
    );

    const top30Path = path.join(
      RESULTS,
      "feature_importance",
      "top_30_features.csv"
    );

    const ablationPath = path.join(
      RESULTS,
      "feature_ablation",
      "top_k_feature_ablation.csv"
    );

    const rfRaw = readCsv(rfPath);
    const xgbRaw = readCsv(xgbPath);
    const combinedRaw = readCsv(combinedPath);
    const overlapRaw = readCsv(overlapPath);
    const top30Raw = readCsv(top30Path);
    const ablationRaw = readCsv(ablationPath);

    const randomForest = parseModelFeatureRows(rfRaw);

    const xgboost = parseModelFeatureRows(xgbRaw);

    const combined = combinedRaw
      .map((row) => ({
        feature: getFeature(row),

        meanImportance: getOptionalNumber(row, [
          "mean_importance",
          "meanImportance",
          "importance",
        ]) ?? 0,

        rfImportance: getOptionalNumber(row, [
          "rf_importance",
          "random_forest_importance",
          "randomForestImportance",
        ]),

        xgbImportance: getOptionalNumber(row, [
          "xgb_importance",
          "xgboost_importance",
          "xgboostImportance",
        ]),

        rank: getRank(row),
      }))
      .filter((row) => row.feature !== "");

    const overlap = overlapRaw
      .map((row) => ({
        k: getOptionalNumber(row, [
          "k",
          "top_k",
          "features",
          "feature_count",
        ]),

        overlap: getOptionalNumber(row, [
          "overlap",
          "common_features",
          "overlap_count",
        ]),
      }))
      .filter((row) => row.k !== null);

    const top30 = top30Raw
      .map((row) => ({
        feature: getFeature(row),

        importance:
          getOptionalNumber(row, [
            "mean_importance",
            "meanImportance",
            "importance",
          ]) ?? 0,

        rank: getRank(row),
      }))
      .filter((row) => row.feature !== "");

    const ablation = ablationRaw
      .map((row) => ({
        model:
          getValue(row, [
            "model",
            "Model",
          ]) ?? "",

        featureCount: getOptionalNumber(row, [
          "feature_count",
          "features",
          "k",
          "top_k",
        ]),

        precision: getOptionalNumber(row, ["precision"]),

        recall: getOptionalNumber(row, ["recall"]),

        f1: getOptionalNumber(row, ["f1", "F1"]),

        rocAuc: getOptionalNumber(row, [
          "roc_auc",
          "rocAuc",
          "roc",
        ]),

        prAuc: getOptionalNumber(row, [
          "pr_auc",
          "prAuc",
          "pr",
        ]),
      }))
      .filter((row) => row.model !== "");

    console.log("Feature Intelligence loaded:", {
      randomForestRows: randomForest.length,
      xgboostRows: xgboost.length,
      combinedRows: combined.length,
      top30Rows: top30.length,
      ablationRows: ablation.length,

      rfFirst: randomForest[0],
      xgbFirst: xgboost[0],
      combinedFirst: combined[0],
    });

    return NextResponse.json({
      randomForest,
      xgboost,
      combined,
      overlap,
      top30,
      ablation,

      metadata: {
        rfXgbSpearman: 0.7362,

        top5Overlap: "1/5",
        top10Overlap: "4/10",
        top20Overlap: "10/20",
        top30Overlap: "16/30",
        top50Overlap: "32/50",
      },
    });
  } catch (error) {
    console.error(
      "Feature Intelligence API error:",
      error
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to load feature intelligence data.",
      },
      {
        status: 500,
      }
    );
  }
}