import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

type CsvRow = Record<string, string>;

function parseCsv(content: string): CsvRow[] {
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    return [];
  }

  const headers = lines[0].split(",").map((h) => h.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",");

    const row: CsvRow = {};

    headers.forEach((header, index) => {
      row[header] = values[index]?.trim() ?? "";
    });

    return row;
  });
}

function readCsv(filePath: string): CsvRow[] {
  if (!fs.existsSync(filePath)) {
    console.error("Missing CSV:", filePath);
    return [];
  }

  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function toNumber(value: string | undefined): number | undefined {
  if (value === undefined || value === "") {
    return undefined;
  }

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : undefined;
}

function toInteger(value: string | undefined): number | undefined {
  const parsed = toNumber(value);

  if (parsed === undefined) {
    return undefined;
  }

  return Math.round(parsed);
}

export async function GET() {
  try {
    const projectRoot = path.resolve(process.cwd(), "..");

    const errorDir = path.join(
      projectRoot,
      "results",
      "error_analysis"
    );

    const modelSummaryRows = readCsv(
      path.join(
        errorDir,
        "model_error_summary.csv"
      )
    );

    const errorCountRows = readCsv(
      path.join(
        errorDir,
        "error_counts.csv"
      )
    );

    const confidenceRows = readCsv(
      path.join(
        errorDir,
        "confidence_analysis.csv"
      )
    );

    const temporalRows = readCsv(
      path.join(
        errorDir,
        "errors_by_time_step.csv"
      )
    );

    const disagreementRows = readCsv(
      path.join(
        errorDir,
        "model_disagreement_distribution.csv"
      )
    );

    const agreementRows = readCsv(
      path.join(
        errorDir,
        "prediction_agreement_vs_truth.csv"
      )
    );

    /*
     * -------------------------------------------------------
     * MODEL SUMMARY
     * -------------------------------------------------------
     */

    const modelSummary = modelSummaryRows.map((row) => ({
      model: row.model,

      threshold: toNumber(row.threshold),

      precision: toNumber(row.precision),

      recall: toNumber(row.recall),

      f1: toNumber(row.f1),

      roc_auc: toNumber(row.roc_auc),

      pr_auc: toNumber(row.pr_auc),

      true_negatives:
        toInteger(row.true_negatives),

      false_positives:
        toInteger(row.false_positives),

      false_negatives:
        toInteger(row.false_negatives),

      true_positives:
        toInteger(row.true_positives),

      total_errors:
        toInteger(row.total_errors),
    }));


    /*
     * -------------------------------------------------------
     * ERROR COUNTS
     * -------------------------------------------------------
     */

    const errorCounts = errorCountRows.map((row) => ({
      model: row.model,

      true_negatives:
        toInteger(row.true_negatives),

      false_positives:
        toInteger(row.false_positives),

      false_negatives:
        toInteger(row.false_negatives),

      total_errors:
        toInteger(row.total_errors),

      true_positives:
        toInteger(row.true_positives),
    }));


    /*
     * -------------------------------------------------------
     * CONFIDENCE
     * -------------------------------------------------------
     */

    const confidence = confidenceRows.map((row) => ({
      model: row.model,

      correct_count:
        toInteger(row.correct_count),

      incorrect_count:
        toInteger(row.incorrect_count),

      mean_probability_correct:
        toNumber(
          row.mean_probability_correct
        ),

      mean_probability_incorrect:
        toNumber(
          row.mean_probability_incorrect
        ),

      mean_probability_true_illicit:
        toNumber(
          row.mean_probability_true_illicit
        ),

      mean_probability_true_licit:
        toNumber(
          row.mean_probability_true_licit
        ),

      mean_probability_false_positive:
        toNumber(
          row.mean_probability_false_positive
        ),

      mean_probability_false_negative:
        toNumber(
          row.mean_probability_false_negative
        ),
    }));


    /*
     * -------------------------------------------------------
     * TEMPORAL ANALYSIS
     * -------------------------------------------------------
     */

    const temporal = temporalRows.map((row) => ({
      model: row.model,

      time_step:
        toInteger(row.time_step) ?? 0,

      samples:
        toInteger(row.samples),

      illicit_transactions:
        toInteger(row.illicit_transactions),

      precision:
        toNumber(row.precision),

      recall:
        toNumber(row.recall),

      false_positives:
        toInteger(row.false_positives),

      false_negatives:
        toInteger(row.false_negatives),
    }));


    /*
     * -------------------------------------------------------
     * MODEL DISAGREEMENT
     * -------------------------------------------------------
     */

    const disagreement = disagreementRows.map(
      (row, index) => ({
        id: index,

        models_predicting_illicit:
          toInteger(
            row.models_predicting_illicit
          ) ?? 0,

        transactions:
          toInteger(row.transactions) ?? 0,
      })
    );


    /*
     * -------------------------------------------------------
     * AGREEMENT VS TRUTH
     * -------------------------------------------------------
     */

    const agreement = agreementRows.map(
      (row, index) => ({
        id: index,

        models_predicting_illicit:
          toInteger(
            row.models_predicting_illicit
          ) ?? 0,

        transactions:
          toInteger(row.transactions) ?? 0,

        actual_illicit_rate:
          toNumber(
            row.actual_illicit_rate
          ) ?? 0,
      })
    );


    /*
     * -------------------------------------------------------
     * METADATA
     * -------------------------------------------------------
     */

    const totalTestTransactions = 9973;

    const totalIllicitTransactions = 524;

    const testTimeSteps = Array.from(
      new Set(
        temporal
          .map((row) => row.time_step)
          .filter(
            (value) =>
              typeof value === "number"
          )
      )
    ).sort((a, b) => a - b);


    /*
     * -------------------------------------------------------
     * RESPONSE
     * -------------------------------------------------------
     */

    return NextResponse.json({
      metadata: {
        totalTestTransactions,
        totalIllicitTransactions,
        testTimeSteps,
        modelsEvaluated: modelSummary.length,
      },

      summary: {
        modelSummary,
        errorCounts,
      },

      confidence,

      temporal,

      disagreement,

      agreement,

      sources: {
        modelSummary:
          "results/error_analysis/model_error_summary.csv",

        errorCounts:
          "results/error_analysis/error_counts.csv",

        confidence:
          "results/error_analysis/confidence_analysis.csv",

        temporal:
          "results/error_analysis/errors_by_time_step.csv",

        disagreement:
          "results/error_analysis/model_disagreement_distribution.csv",

        agreement:
          "results/error_analysis/prediction_agreement_vs_truth.csv",
      },
    });

  } catch (error) {

    console.error(
      "Error Analysis API failure:",
      error
    );

    return NextResponse.json(
      {
        error:
          "Failed to load error analysis data.",
      },
      {
        status: 500,
      }
    );
  }
}