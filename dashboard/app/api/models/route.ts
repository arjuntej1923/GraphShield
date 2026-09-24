import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

function parseCsv(text: string) {
  const lines = text
    .split(/\r?\n/)
    .filter((line) => line.trim());

  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0]
    .split(",")
    .map((header) => header.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",");

    const row: Record<string, string> = {};

    headers.forEach((header, index) => {
      row[header] = values[index]?.trim() ?? "";
    });

    return row;
  });
}

function getNumber(
  row: Record<string, string>,
  names: string[]
) {
  for (const name of names) {
    const key = Object.keys(row).find(
      (existingKey) =>
        existingKey.toLowerCase().trim() ===
        name.toLowerCase().trim()
    );

    if (key && row[key] !== "") {
      const value = Number(row[key]);

      if (!Number.isNaN(value)) {
        return value;
      }
    }
  }

  return null;
}

export async function GET() {
  try {
    const filePath = path.resolve(
      process.cwd(),
      "../results/final/master_model_results.csv"
    );

    if (!fs.existsSync(filePath)) {
      return NextResponse.json(
        {
          error: "Model results file not found.",
          path: filePath,
        },
        { status: 404 }
      );
    }

    const csv = fs.readFileSync(filePath, "utf8");

    const rows = parseCsv(csv);

    const models = rows.map((row) => ({
      model:
        row.model ||
        row.Model ||
        row.MODEL ||
        "Unknown",

      validationPrAuc: getNumber(row, [
        "validation_pr_auc",
        "val_pr_auc",
        "validation_pr",
        "val_pr",
      ]),

      testPrAuc: getNumber(row, [
        "test_pr_auc",
        "test_pr",
        "pr_auc",
      ]),

      testRocAuc: getNumber(row, [
        "test_roc_auc",
        "test_roc",
        "roc_auc",
      ]),

      precision: getNumber(row, [
        "test_precision",
        "precision",
      ]),

      recall: getNumber(row, [
        "test_recall",
        "recall",
      ]),

      f1: getNumber(row, [
        "test_f1",
        "f1",
        "f1_score",
      ]),

      threshold: getNumber(row, [
        "threshold",
        "selected_threshold",
      ]),
    }));

    return NextResponse.json({
      models,
      source: "results/final/master_model_results.csv",
    });
  } catch (error) {
    console.error(error);

    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to load model results.",
      },
      { status: 500 }
    );
  }
}