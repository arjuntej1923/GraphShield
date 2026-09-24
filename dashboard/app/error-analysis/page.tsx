"use client";

import { useEffect, useMemo, useState } from "react";

type ModelSummary = {
  model: string;
  threshold?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;
  pr_auc?: number;
  true_negatives?: number;
  false_positives?: number;
  false_negatives?: number;
  true_positives?: number;
  total_errors?: number;
};

type ErrorCount = {
  model: string;
  true_negatives?: number;
  false_positives?: number;
  false_negatives?: number;
  total_errors?: number;
  true_positives?: number;
};

type Confidence = {
  model: string;
  correct_count?: number;
  incorrect_count?: number;
  mean_probability_correct?: number;
  mean_probability_incorrect?: number;
  mean_probability_true_illicit?: number;
  mean_probability_true_licit?: number;
  mean_probability_false_positive?: number;
  mean_probability_false_negative?: number;
};

type TemporalRow = {
  model: string;
  time_step: number;
  samples?: number;
  illicit_transactions?: number;
  precision?: number;
  recall?: number;
  false_positives?: number;
  false_negatives?: number;
};

type DisagreementRow = {
  id: number;
  models_predicting_illicit: number;
  transactions: number;
};

type AgreementRow = {
  id: number;
  models_predicting_illicit: number;
  transactions: number;
  actual_illicit_rate: number;
};

type ErrorAnalysisData = {
  metadata: {
    totalTestTransactions: number;
    totalIllicitTransactions: number;
    testTimeSteps: number[];
    modelsEvaluated: number;
  };

  summary: {
    modelSummary: ModelSummary[];
    errorCounts: ErrorCount[];
  };

  confidence: Confidence[];
  temporal: TemporalRow[];
  disagreement: DisagreementRow[];
  agreement: AgreementRow[];
};

const MODEL_COLORS: Record<string, string> = {
  logistic_regression: "#60a5fa",
  random_forest: "#34d399",
  xgboost: "#fbbf24",
  gcn: "#a78bfa",
  graphsage: "#fb7185",
  gat: "#22d3ee",
};

const DISPLAY_NAMES: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  gcn: "GCN",
  graphsage: "GraphSAGE",
  gat: "GAT",
};

function modelName(model: string) {
  return DISPLAY_NAMES[model] || model;
}

function safeNumber(value: unknown): number | null {
  if (typeof value !== "number") {
    return null;
  }

  if (!Number.isFinite(value)) {
    return null;
  }

  return value;
}

function number(value: number | undefined | null) {
  const safe = safeNumber(value);

  if (safe === null) {
    return "—";
  }

  return safe.toLocaleString();
}

function decimal(
  value: number | undefined | null,
  digits = 3
) {
  const safe = safeNumber(value);

  if (safe === null) {
    return "—";
  }

  return safe.toFixed(digits);
}

function pct(
  value: number | undefined | null,
  digits = 1
) {
  const safe = safeNumber(value);

  if (safe === null) {
    return "—";
  }

  return `${(safe * 100).toFixed(digits)}%`;
}

export default function ErrorAnalysisPage() {
  const [data, setData] =
    useState<ErrorAnalysisData | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [selectedModel, setSelectedModel] =
    useState("all");

  useEffect(() => {
    fetch("/api/error-analysis")
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            "Failed to load error analysis"
          );
        }

        return res.json();
      })
      .then((json) => {
        console.log(
          "GraphShield Error Analysis:",
          json
        );

        setData(json);
        setLoading(false);
      })
      .catch((error) => {
        console.error(
          "Error Analysis API:",
          error
        );

        setLoading(false);
      });
  }, []);

  const selectedSummary = useMemo(() => {
    if (!data) {
      return [];
    }

    if (selectedModel === "all") {
      return data.summary.modelSummary;
    }

    return data.summary.modelSummary.filter(
      (item) =>
        item.model === selectedModel
    );
  }, [data, selectedModel]);

  const temporalRows = useMemo(() => {
    if (!data) {
      return [];
    }

    if (selectedModel === "all") {
      return data.temporal;
    }

    return data.temporal.filter(
      (item) =>
        item.model === selectedModel
    );
  }, [data, selectedModel]);

  if (loading) {
    return (
      <main className="research-page">
        <div className="page-loading">
          <div className="loading-dot" />
          Loading error analysis...
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="research-page">
        <div className="error-state">
          <h2>
            ERROR ANALYSIS UNAVAILABLE
          </h2>

          <p>
            Unable to load the research
            error-analysis dataset.
          </p>
        </div>
      </main>
    );
  }

  const bestPr =
    [...data.summary.modelSummary].sort(
      (a, b) =>
        (safeNumber(b.pr_auc) ?? -1) -
        (safeNumber(a.pr_auc) ?? -1)
    )[0];

  return (
    <main className="error-page">

      {/* =====================================================
          HERO
          ===================================================== */}

      <section className="error-hero">

        <div>

          <div className="eyebrow">
            GRAPHSHIELD · ERROR ANALYSIS
          </div>

          <h1>
            Model Error Intelligence
          </h1>

          <p>
            Transaction-level error patterns,
            confidence behavior, model
            disagreement, and temporal
            performance across the held-out
            test period.
          </p>

        </div>

        <div className="hero-status">
          <span className="status-dot" />
          TEST SET ANALYSIS
        </div>

      </section>


      {/* =====================================================
          KPI STRIP
          ===================================================== */}

      <section className="error-kpis">

        <div className="error-kpi">

          <span className="kpi-label">
            TEST TRANSACTIONS
          </span>

          <strong>
            {number(
              data.metadata
                .totalTestTransactions
            )}
          </strong>

          <small>
            Held-out transactions
          </small>

        </div>


        <div className="error-kpi">

          <span className="kpi-label">
            ILLICIT TRANSACTIONS
          </span>

          <strong>
            {number(
              data.metadata
                .totalIllicitTransactions
            )}
          </strong>

          <small>
            {pct(
              data.metadata
                .totalIllicitTransactions /
                data.metadata
                  .totalTestTransactions
            )}{" "}
            of test set
          </small>

        </div>


        <div className="error-kpi">

          <span className="kpi-label">
            MODELS ANALYZED
          </span>

          <strong>
            {number(
              data.metadata
                .modelsEvaluated
            )}
          </strong>

          <small>
            3 traditional · 3 GNN
          </small>

        </div>


        <div className="error-kpi">

          <span className="kpi-label">
            TOP TEST PR-AUC
          </span>

          <strong>
            {pct(
              bestPr?.pr_auc,
              2
            )}
          </strong>

          <small>
            {bestPr
              ? modelName(bestPr.model)
              : "—"}
          </small>

        </div>

      </section>


      {/* =====================================================
          MODEL ERROR PROFILE
          ===================================================== */}

      <section className="analysis-panel">

        <div className="panel-heading">

          <div>

            <span className="panel-index">
              01
            </span>

            <h2>
              MODEL ERROR PROFILE
            </h2>

          </div>


          <select
            value={selectedModel}
            onChange={(e) =>
              setSelectedModel(
                e.target.value
              )
            }
            className="model-select"
          >

            <option value="all">
              All models
            </option>

            {data.summary.modelSummary.map(
              (item) => (
                <option
                  key={item.model}
                  value={item.model}
                >
                  {modelName(item.model)}
                </option>
              )
            )}

          </select>

        </div>


        <div className="model-error-grid">

          {selectedSummary.map(
            (item) => {

              const color =
                MODEL_COLORS[
                  item.model
                ] || "#94a3b8";

              const totalErrors =
                safeNumber(
                  item.total_errors
                ) ?? 0;

              const truePositives =
                safeNumber(
                  item.true_positives
                ) ?? 0;

              const falsePositives =
                safeNumber(
                  item.false_positives
                ) ?? 0;

              const falseNegatives =
                safeNumber(
                  item.false_negatives
                ) ?? 0;

              const trueNegatives =
                safeNumber(
                  item.true_negatives
                ) ?? 0;

              return (

                <article
                  className="model-error-card"
                  key={item.model}
                >

                  <div className="model-card-top">

                    <div>

                      <span
                        className="model-marker"
                        style={{
                          background:
                            color,
                        }}
                      />

                      <span>
                        {modelName(
                          item.model
                        )}
                      </span>

                    </div>

                    <span className="threshold">
                      τ{" "}
                      {decimal(
                        item.threshold,
                        2
                      )}
                    </span>

                  </div>


                  <div className="metric-row">

                    <div>
                      <small>
                        PRECISION
                      </small>

                      <strong>
                        {pct(
                          item.precision
                        )}
                      </strong>
                    </div>


                    <div>
                      <small>
                        RECALL
                      </small>

                      <strong>
                        {pct(
                          item.recall
                        )}
                      </strong>
                    </div>


                    <div>
                      <small>
                        F1
                      </small>

                      <strong>
                        {pct(item.f1)}
                      </strong>
                    </div>


                    <div>
                      <small>
                        PR-AUC
                      </small>

                      <strong>
                        {pct(
                          item.pr_auc
                        )}
                      </strong>
                    </div>

                  </div>


                  <div className="error-bar-wrap">

                    <div className="error-bar-label">

                      <span>
                        ERRORS
                      </span>

                      <b>
                        {number(
                          totalErrors
                        )}
                      </b>

                    </div>


                    <div className="error-bar">

                      <div
                        className="error-bar-fill"
                        style={{
                          width: `${Math.min(
                            100,
                            (totalErrors /
                              1100) *
                              100
                          )}%`,
                          background:
                            color,
                        }}
                      />

                    </div>

                  </div>


                  <div className="confusion-grid">

                    <div>
                      <span>TP</span>

                      <b>
                        {number(
                          truePositives
                        )}
                      </b>
                    </div>


                    <div>
                      <span>FP</span>

                      <b>
                        {number(
                          falsePositives
                        )}
                      </b>
                    </div>


                    <div>
                      <span>FN</span>

                      <b>
                        {number(
                          falseNegatives
                        )}
                      </b>
                    </div>


                    <div>
                      <span>TN</span>

                      <b>
                        {number(
                          trueNegatives
                        )}
                      </b>
                    </div>

                  </div>

                </article>

              );
            }
          )}

        </div>

      </section>


      {/* =====================================================
          CONFIDENCE ANALYSIS
          ===================================================== */}

      <section className="analysis-panel">

        <div className="panel-heading">

          <div>

            <span className="panel-index">
              02
            </span>

            <h2>
              CONFIDENCE ANALYSIS
            </h2>

          </div>

        </div>


        <div className="confidence-table">

          <div className="confidence-head">

            <span>MODEL</span>

            <span>
              CORRECT
            </span>

            <span>
              INCORRECT
            </span>

            <span>
              MEAN CONFIDENCE
            </span>

            <span>
              FALSE POSITIVE
            </span>

            <span>
              FALSE NEGATIVE
            </span>

          </div>


          {data.confidence.map(
            (item) => {

              const correct =
                safeNumber(
                  item.correct_count
                ) ?? 0;

              const incorrect =
                safeNumber(
                  item.incorrect_count
                ) ?? 0;

              const total =
                correct + incorrect;

              return (

                <div
                  className="confidence-row"
                  key={item.model}
                >

                  <strong>
                    {modelName(
                      item.model
                    )}
                  </strong>


                  <span>

                    {number(correct)}

                    <small>
                      {total > 0
                        ? pct(
                            correct /
                              total
                          )
                        : "—"}
                    </small>

                  </span>


                  <span>

                    {number(incorrect)}

                    <small>
                      {total > 0
                        ? pct(
                            incorrect /
                              total
                          )
                        : "—"}
                    </small>

                  </span>


                  <span>

                    {decimal(
                      item.mean_probability_correct
                    )}

                    <small>
                      correct
                    </small>

                  </span>


                  <span>

                    {decimal(
                      item.mean_probability_false_positive
                    )}

                    <small>
                      FP confidence
                    </small>

                  </span>


                  <span>

                    {decimal(
                      item.mean_probability_false_negative
                    )}

                    <small>
                      FN confidence
                    </small>

                  </span>

                </div>

              );
            }
          )}

        </div>

      </section>


      {/* =====================================================
          TEMPORAL ERROR PATTERNS
          ===================================================== */}

      <section className="analysis-panel">

        <div className="panel-heading">

          <div>

            <span className="panel-index">
              03
            </span>

            <h2>
              TEMPORAL ERROR PATTERNS
            </h2>

          </div>


          <span className="panel-note">

            Test time steps{" "}

            {data.metadata.testTimeSteps[0]}

            {"–"}

            {data.metadata.testTimeSteps.at(-1)}

          </span>

        </div>


        <div className="temporal-table">

          <div className="temporal-head">

            <span>MODEL</span>
            <span>STEP</span>
            <span>SAMPLES</span>
            <span>ILLICIT</span>
            <span>PRECISION</span>
            <span>RECALL</span>
            <span>FP</span>
            <span>FN</span>

          </div>


          {temporalRows.map(
            (row, index) => (

              <div
                className="temporal-row"
                key={`${row.model}-${row.time_step}-${index}`}
              >

                <strong>
                  {modelName(
                    row.model
                  )}
                </strong>

                <span>
                  {number(
                    row.time_step
                  )}
                </span>

                <span>
                  {number(
                    row.samples
                  )}
                </span>

                <span>
                  {number(
                    row.illicit_transactions
                  )}
                </span>

                <span>
                  {pct(
                    row.precision
                  )}
                </span>

                <span>
                  {pct(
                    row.recall
                  )}
                </span>

                <span>
                  {number(
                    row.false_positives
                  )}
                </span>

                <span>
                  {number(
                    row.false_negatives
                  )}
                </span>

              </div>

            )
          )}

        </div>


        <div className="analysis-note">

          <span>
            NOTE
          </span>

          Per-time-step precision and
          recall should be interpreted
          with the number of illicit
          transactions in each period.
          Several later test periods
          contain only a small number
          of illicit examples.

        </div>

      </section>


      {/* =====================================================
          DISAGREEMENT + AGREEMENT
          ===================================================== */}

      <section className="analysis-two-column">


        {/* MODEL DISAGREEMENT */}

        <div className="analysis-panel">

          <div className="panel-heading">

            <div>

              <span className="panel-index">
                04
              </span>

              <h2>
                MODEL DISAGREEMENT
              </h2>

            </div>

          </div>


          <div className="disagreement-list">

            {data.disagreement.map(
              (row) => {

                const percentage =
                  data.metadata
                    .totalTestTransactions >
                  0
                    ? (
                        row.transactions /
                        data.metadata
                          .totalTestTransactions
                      ) * 100
                    : 0;

                return (

                  <div
                    className="disagreement-row"
                    key={`disagreement-${row.id}`}
                  >

                    <div className="disagreement-label">

                      <strong>
                        {
                          row.models_predicting_illicit
                        }
                        /6
                      </strong>

                      <span>
                        models predict illicit
                      </span>

                    </div>


                    <div className="disagreement-bar">

                      <div
                        style={{
                          width: `${Math.min(
                            100,
                            percentage * 4
                          )}%`,
                        }}
                      />

                    </div>


                    <b>
                      {number(
                        row.transactions
                      )}
                    </b>

                  </div>

                );
              }
            )}

          </div>

        </div>


        {/* AGREEMENT VS TRUTH */}

        <div className="analysis-panel">

          <div className="panel-heading">

            <div>

              <span className="panel-index">
                05
              </span>

              <h2>
                AGREEMENT VS TRUTH
              </h2>

            </div>

          </div>


          <div className="agreement-list">

            {data.agreement.map(
              (row) => (

                <div
                  className="agreement-row"
                  key={`agreement-${row.id}`}
                >

                  <div>

                    <strong>
                      {
                        row.models_predicting_illicit
                      }
                      /6
                    </strong>

                    <span>
                      illicit votes
                    </span>

                  </div>


                  <div>

                    <strong>
                      {number(
                        row.transactions
                      )}
                    </strong>

                    <span>
                      transactions
                    </span>

                  </div>


                  <div className="agreement-rate">

                    <strong>
                      {pct(
                        row.actual_illicit_rate
                      )}
                    </strong>

                    <span>
                      actual illicit
                    </span>

                  </div>

                </div>

              )
            )}

          </div>

        </div>

      </section>


      {/* =====================================================
          RESEARCH OBSERVATION
          ===================================================== */}

      <section className="research-observation">

        <div className="observation-tag">
          RESEARCH OBSERVATION
        </div>


        <h2>
          Error analysis exposes where
          model agreement, confidence,
          and temporal behavior diverge.
        </h2>


        <p>
          The test-set analysis separates
          aggregate performance from the
          underlying error structure.
          False positives, false negatives,
          confidence behavior, model
          disagreement, and temporal
          variation provide additional
          evidence for evaluating how the
          six approaches behave on unseen
          transactions.
        </p>


        <div className="observation-grid">

          <div>

            <span>01</span>

            <strong>
              ERROR COUNTS
            </strong>

            <p>
              Across the six model outputs,
              the recorded total error
              counts provide a direct view
              of classification mistakes.
            </p>

          </div>


          <div>

            <span>02</span>

            <strong>
              CONFIDENCE
            </strong>

            <p>
              Confidence statistics
              distinguish correct predictions
              from incorrect predictions
              rather than relying only on
              aggregate performance.
            </p>

          </div>


          <div>

            <span>03</span>

            <strong>
              TEMPORAL DRIFT
            </strong>

            <p>
              Performance varies across
              the held-out time steps and
              should be interpreted
              alongside the changing
              illicit class frequency.
            </p>

          </div>

        </div>

      </section>


      {/* =====================================================
          FOOTER
          ===================================================== */}

      <div className="error-footer">

        <span>
          GRAPHSHIELD · ERROR INTELLIGENCE
        </span>

        <span>
          TEST SET ·{" "}
          {number(
            data.metadata
              .totalTestTransactions
          )}{" "}
          TRANSACTIONS ·{" "}
          {data.metadata.testTimeSteps[0]}
          –
          {data.metadata.testTimeSteps.at(-1)}
        </span>

      </div>

    </main>
  );
}