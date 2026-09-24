import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const ROOT = path.resolve(process.cwd(), "..");

function parseCSV(content: string) {
  const lines = content
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);

  if (lines.length === 0) {
    return [];
  }

  const headers = lines[0]
    .split(",")
    .map((header) =>
      header
        .trim()
        .replace(/^"|"$/g, "")
        .toLowerCase()
        .replace(/[\s-]+/g, "_")
    );

  return lines.slice(1).map((line) => {
    const values = line
      .split(",")
      .map((value) =>
        value
          .trim()
          .replace(/^"|"$/g, "")
      );

    const row: Record<string, string> = {};

    headers.forEach((header, index) => {
      row[header] =
        values[index] ?? "";
    });

    return row;
  });
}

function readCSV(relativePath: string) {
  const filePath = path.join(
    ROOT,
    relativePath
  );

  if (!fs.existsSync(filePath)) {
    throw new Error(
      `File not found: ${relativePath}`
    );
  }

  return parseCSV(
    fs.readFileSync(
      filePath,
      "utf8"
    )
  );
}

function numberValue(
  row: Record<string, string>,
  keys: string[]
): number {
  for (const key of keys) {
    const value = row[key];

    if (
      value !== undefined &&
      value !== null &&
      value !== ""
    ) {
      const parsed = Number(value);

      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }

  return 0;
}

function stringValue(
  row: Record<string, string>,
  keys: string[]
): string {
  for (const key of keys) {
    const value = row[key];

    if (
      value !== undefined &&
      value !== null &&
      value !== ""
    ) {
      return value;
    }
  }

  return "";
}

export async function GET() {
  try {

    const rawModels = readCSV(
      "results/final/master_model_results.csv"
    );

    const rawRobustness = readCSV(
      "results/final/master_robustness_results.csv"
    );

    const rawDataset = readCSV(
      "results/final/master_dataset_summary.csv"
    );

    const rawFeatureAblation = readCSV(
      "results/final/master_feature_ablation.csv"
    );

    const rawGnnAblation = readCSV(
      "results/final/master_gnn_feature_ablation.csv"
    );

    const rawStatistics = readCSV(
      "results/final/master_statistical_comparisons.csv"
    );

    const rawEfficiency = readCSV(
      "results/final/master_computational_efficiency.csv"
    );


    /*
     * Normalize the primary model results.
     *
     * The dashboard uses these exact fields,
     * while allowing several possible CSV
     * column-name variations.
     */

    const models = rawModels.map((row) => {

      const model =
        stringValue(row, [
          "model",
          "model_name",
          "name",
        ]);

      return {
        model,

        validation_pr_auc:
          numberValue(row, [
            "validation_pr_auc",
            "val_pr_auc",
            "validation_prauc",
            "val_prauc",
          ]),

        test_pr_auc:
          numberValue(row, [
            "test_pr_auc",
            "test_prauc",
            "pr_auc",
            "test_pr",
          ]),

        test_roc_auc:
          numberValue(row, [
            "test_roc_auc",
            "roc_auc",
            "test_roc",
            "test_rocauc",
          ]),

        test_precision:
          numberValue(row, [
            "test_precision",
            "precision",
            "test_prec",
          ]),

        test_recall:
          numberValue(row, [
            "test_recall",
            "recall",
            "test_rec",
          ]),

        test_f1:
          numberValue(row, [
            "test_f1",
            "f1",
            "f1_score",
          ]),

        selected_threshold:
          numberValue(row, [
            "selected_threshold",
            "threshold",
            "decision_threshold",
          ]),
      };
    });


    /*
     * Normalize robustness data.
     */

    const robustness =
      rawRobustness.map((row) => ({
        model:
          stringValue(row, [
            "model",
            "model_name",
            "name",
          ]),

        seed:
          stringValue(row, [
            "seed",
            "random_seed",
          ]),

        test_pr_auc:
          numberValue(row, [
            "test_pr_auc",
            "pr_auc",
            "test_prauc",
          ]),

        test_f1:
          numberValue(row, [
            "test_f1",
            "f1",
            "f1_score",
          ]),

        test_precision:
          numberValue(row, [
            "test_precision",
            "precision",
          ]),

        test_recall:
          numberValue(row, [
            "test_recall",
            "recall",
          ]),
      }));


    /*
     * Keep supporting research tables available
     * for the next dashboard sections.
     */

    const dataset =
      rawDataset;

    const featureAblation =
      rawFeatureAblation;

    const gnnAblation =
      rawGnnAblation;

    const statistics =
      rawStatistics;

    const efficiency =
      rawEfficiency;


    return NextResponse.json({
      models,
      robustness,
      dataset,
      featureAblation,
      gnnAblation,
      statistics,
      efficiency,
    });

  } catch (error) {

    console.error(
      "GraphShield analytics API error:",
      error
    );

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to load analytics data.",
      },
      {
        status: 500,
      }
    );
  }
}