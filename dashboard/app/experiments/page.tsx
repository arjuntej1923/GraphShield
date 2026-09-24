"use client";

import { useEffect, useMemo, useState } from "react";

type Row = Record<string, string | number>;

type ExperimentData = {
  featureAblation: Row[];
  gnnAblation: Row[];
  robustness: Row[];
  statistics: Row[];
  efficiency: Row[];
};

function num(row: Row, keys: string[]) {
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

function text(row: Row, keys: string[]) {
  for (const key of keys) {
    const value = row[key];

    if (
      value !== undefined &&
      value !== null &&
      String(value) !== ""
    ) {
      return String(value);
    }
  }

  return "";
}

function percent(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export default function ExperimentsPage() {
  const [data, setData] =
    useState<ExperimentData | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  useEffect(() => {
    fetch("/api/experiments")
      .then(async (response) => {
        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.error ||
              "Failed to load experiments."
          );
        }

        return result;
      })
      .then(setData)
      .catch((err) => {
        setError(
          err instanceof Error
            ? err.message
            : "Failed to load experiments."
        );
      })
      .finally(() => setLoading(false));
  }, []);

  const robustnessModels = useMemo(() => {
    if (!data) return [];

    return Array.from(
      new Set(
        data.robustness
          .map((row) =>
            text(row, [
              "model",
              "model_name",
            ])
          )
          .filter(Boolean)
      )
    );
  }, [data]);

  if (loading) {
    return (
      <main className="loading">
        Loading experiment laboratory...
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="loading error">
        {error || "No experiment data."}
      </main>
    );
  }

  const featureRows =
    data.featureAblation;

  const gnnRows =
    data.gnnAblation;

  const rfFeatureRows =
    featureRows.filter(
      (row) =>
        text(row, ["model"]) ===
        "Random Forest"
    );

  const xgbFeatureRows =
    featureRows.filter(
      (row) =>
        text(row, ["model"]) ===
        "XGBoost"
    );

  const gnnModels = [
    "GCN",
    "GraphSAGE",
    "GAT",
  ];

  const seeds = Array.from(
    new Set(
      data.robustness
        .map((row) =>
          text(row, [
            "seed",
            "random_seed",
          ])
        )
        .filter(Boolean)
    )
  );

  return (
    <>
      <style jsx global>{`

        * {
          box-sizing: border-box;
        }

        body {
          margin: 0;
          background: #050914;
          color: #e5e7eb;
          font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            sans-serif;
        }

        .page {
          min-height: 100vh;
          padding: 42px 48px 90px;
          background:
            radial-gradient(
              circle at 15% 0%,
              rgba(37,99,235,.12),
              transparent 30%
            ),
            radial-gradient(
              circle at 90% 20%,
              rgba(14,165,233,.07),
              transparent 28%
            ),
            #050914;
        }

        .container {
          max-width: 1500px;
          margin: auto;
        }

        .eyebrow {
          color: #60a5fa;
          font-size: 10px;
          font-weight: 800;
          letter-spacing: .16em;
        }

        .header {
          display: flex;
          justify-content: space-between;
          gap: 30px;
          margin-bottom: 35px;
        }

        h1 {
          margin: 9px 0 0;
          color: #f8fafc;
          font-size: 38px;
          line-height: 1.1;
          letter-spacing: -.025em;
        }

        .subtitle {
          max-width: 850px;
          margin-top: 13px;
          color: #94a3b8;
          font-size: 15px;
          line-height: 1.7;
        }

        .badge {
          height: fit-content;
          padding: 10px 14px;
          border: 1px solid #1e293b;
          border-radius: 8px;
          color: #94a3b8;
          background: #09101e;
          font-size: 9px;
          font-weight: 800;
          letter-spacing: .12em;
        }

        .section {
          margin-top: 38px;
        }

        .section-title {
          margin: 5px 0 0;
          color: #f8fafc;
          font-size: 21px;
        }

        .section-description {
          margin-top: 7px;
          color: #64748b;
          font-size: 12px;
        }

        .card {
          border: 1px solid #1e293b;
          border-radius: 12px;
          background: #090f1c;
        }

        .summary-grid {
          display: grid;
          grid-template-columns:
            repeat(4,minmax(0,1fr));
          gap: 14px;
        }

        .summary {
          padding: 20px;
        }

        .summary-label {
          color: #64748b;
          font-size: 9px;
          font-weight: 800;
          letter-spacing: .13em;
        }

        .summary-value {
          margin-top: 10px;
          color: #f8fafc;
          font-size: 29px;
          font-weight: 750;
        }

        .summary-note {
          margin-top: 5px;
          color: #64748b;
          font-size: 11px;
        }

        .table-wrap {
          overflow-x: auto;
        }

        table {
          width: 100%;
          border-collapse: collapse;
          min-width: 800px;
        }

        th {
          padding: 14px 17px;
          background: #070d18;
          border-bottom: 1px solid #1e293b;
          color: #64748b;
          text-align: left;
          font-size: 9px;
          letter-spacing: .09em;
        }

        td {
          padding: 14px 17px;
          border-bottom: 1px solid #111827;
          color: #cbd5e1;
          font-size: 11px;
        }

        tr:last-child td {
          border-bottom: 0;
        }

        .strong {
          color: #f8fafc;
          font-weight: 700;
        }

        .blue {
          color: #60a5fa;
          font-weight: 700;
        }

        .charts {
          display: grid;
          grid-template-columns:
            repeat(2,minmax(0,1fr));
          gap: 16px;
        }

        .chart {
          padding: 22px;
        }

        .chart-title {
          color: #f8fafc;
          font-size: 15px;
          font-weight: 700;
        }

        .chart-note {
          margin-top: 5px;
          color: #64748b;
          font-size: 11px;
        }

        .bars {
          display: flex;
          flex-direction: column;
          gap: 14px;
          margin-top: 22px;
        }

        .bar-line {
          display: grid;
          grid-template-columns:
            110px 1fr 60px;
          gap: 10px;
          align-items: center;
        }

        .bar-label {
          color: #cbd5e1;
          font-size: 10px;
        }

        .track {
          height: 8px;
          overflow: hidden;
          border-radius: 999px;
          background: #111827;
        }

        .fill {
          height: 100%;
          border-radius: inherit;
          background:
            linear-gradient(
              90deg,
              #2563eb,
              #38bdf8
            );
        }

        .bar-value {
          color: #f8fafc;
          font-size: 10px;
          font-weight: 700;
          text-align: right;
        }

        .seed-grid {
          display: grid;
          grid-template-columns:
            repeat(5,minmax(0,1fr));
          gap: 10px;
          margin-top: 18px;
        }

        .seed {
          padding: 14px;
          border: 1px solid #1e293b;
          border-radius: 8px;
          background: #070d18;
        }

        .seed-number {
          color: #60a5fa;
          font-size: 10px;
          font-weight: 800;
        }

        .seed-value {
          margin-top: 7px;
          color: #f8fafc;
          font-size: 17px;
          font-weight: 700;
        }

        .finding-grid {
          display: grid;
          grid-template-columns:
            repeat(3,minmax(0,1fr));
          gap: 14px;
        }

        .finding {
          min-height: 145px;
          padding: 20px;
        }

        .finding-number {
          color: #60a5fa;
          font-size: 25px;
          font-weight: 750;
        }

        .finding-title {
          margin-top: 9px;
          color: #f8fafc;
          font-size: 14px;
          font-weight: 700;
        }

        .finding-text {
          margin-top: 7px;
          color: #64748b;
          font-size: 11px;
          line-height: 1.6;
        }

        .efficiency-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
        }

        .efficiency-card {
          padding: 20px;
        }

        .efficiency-model {
          color: #f8fafc;
          font-size: 14px;
          font-weight: 700;
        }

        .efficiency-type {
          margin-top: 4px;
          color: #64748b;
          font-size: 9px;
          text-transform: uppercase;
          letter-spacing: .08em;
        }

        .efficiency-metrics {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 10px;
          margin-top: 18px;
        }

        .efficiency-metric {
          padding: 12px;
          border: 1px solid #1e293b;
          border-radius: 8px;
          background: #070d18;
        }

        .efficiency-label {
          color: #64748b;
          font-size: 8px;
          font-weight: 800;
          letter-spacing: .1em;
          text-transform: uppercase;
        }

        .efficiency-value {
          margin-top: 6px;
          color: #f8fafc;
          font-family: "SFMono-Regular", Consolas, monospace;
          font-size: 14px;
          font-weight: 700;
        }

        .efficiency-note {
          margin-top: 16px;
          padding: 15px 17px;
          border: 1px solid #1e293b;
          border-radius: 9px;
          background: #070d18;
          color: #64748b;
          font-size: 11px;
          line-height: 1.6;
        }

        .efficiency-note strong {
          color: #cbd5e1;
        }

        .loading {
          min-height: 100vh;
          display: flex;
          justify-content: center;
          align-items: center;
          background: #050914;
          color: #94a3b8;
        }

        .error {
          color: #fca5a5;
        }

        @media(max-width:1000px) {
          .page {
            padding: 30px 22px 60px;
          }

          .summary-grid {
            grid-template-columns:
              repeat(2,1fr);
          }

          .charts {
            grid-template-columns: 1fr;
          }

          .finding-grid {
            grid-template-columns: 1fr;
          }
        }

        @media(max-width:650px) {
          .page {
            padding: 22px 14px 50px;
          }

          .header {
            flex-direction: column;
          }

          h1 {
            font-size: 30px;
          }

          .summary-grid {
            grid-template-columns: 1fr;
          }

          .seed-grid {
            grid-template-columns:
              repeat(2,1fr);
          }
        }

      `}</style>

      <main className="page">
        <div className="container">

          <header className="header">
            <div>
              <div className="eyebrow">
                GRAPHSHIELD · EXPERIMENT LABORATORY
              </div>

              <h1>
                Ablation & Robustness
              </h1>

              <p className="subtitle">
                Deeper experimental analysis covering
                feature selection, graph architecture
                sensitivity, random-seed robustness,
                statistical comparisons, and
                computational efficiency.
              </p>
            </div>

            <div className="badge">
              RESEARCH VALIDATION
            </div>
          </header>


          {/* OVERVIEW */}

          <section className="summary-grid">

            <div className="card summary">
              <div className="summary-label">
                FEATURE SETTINGS
              </div>

              <div className="summary-value">
                5
              </div>

              <div className="summary-note">
                10 · 20 · 30 · 50 · 165
              </div>
            </div>

            <div className="card summary">
              <div className="summary-label">
                GNN ARCHITECTURES
              </div>

              <div className="summary-value">
                3
              </div>

              <div className="summary-note">
                GCN · GraphSAGE · GAT
              </div>
            </div>

            <div className="card summary">
              <div className="summary-label">
                RANDOM SEEDS
              </div>

              <div className="summary-value">
                {seeds.length || 5}
              </div>

              <div className="summary-note">
                Robustness experiment
              </div>
            </div>

            <div className="card summary">
              <div className="summary-label">
                STATISTICAL TEST
              </div>

              <div className="summary-value">
                Wilcoxon
              </div>

              <div className="summary-note">
                Paired PR-AUC comparisons
              </div>
            </div>

          </section>


          {/* FEATURE ABLATION */}

          <section className="section">

            <div className="eyebrow">
              FEATURE ABLATION
            </div>

            <h2 className="section-title">
              Traditional ML feature sensitivity
            </h2>

            <p className="section-description">
              Test performance as the number of selected
              transaction features changes.
            </p>


            <div className="charts">

              <div className="card chart">

                <div className="chart-title">
                  Random Forest
                </div>

                <div className="chart-note">
                  Test PR-AUC
                </div>

                <div className="bars">

                  {rfFeatureRows.map(
                    (row, index) => {

                      const value =
                        num(row, [
                          "test_pr_auc",
                          "pr_auc",
                          "test_prauc",
                        ]);

                      const features =
                        num(row, [
                          "n_features",
                          "features",
                          "feature_count",
                          "top_k",
                        ]);

                      return (
                        <div
                          className="bar-line"
                          key={index}
                        >
                          <div className="bar-label">
                            {features || "All"} features
                          </div>

                          <div className="track">
                            <div
                              className="fill"
                              style={{
                                width:
                                  `${value * 100}%`,
                              }}
                            />
                          </div>

                          <div className="bar-value">
                            {percent(value)}
                          </div>
                        </div>
                      );
                    }
                  )}

                </div>
              </div>


              <div className="card chart">

                <div className="chart-title">
                  XGBoost
                </div>

                <div className="chart-note">
                  Test PR-AUC
                </div>

                <div className="bars">

                  {xgbFeatureRows.map(
                    (row, index) => {

                      const value =
                        num(row, [
                          "test_pr_auc",
                          "pr_auc",
                          "test_prauc",
                        ]);

                      const features =
                        num(row, [
                          "n_features",
                          "features",
                          "feature_count",
                          "top_k",
                        ]);

                      return (
                        <div
                          className="bar-line"
                          key={index}
                        >
                          <div className="bar-label">
                            {features || "All"} features
                          </div>

                          <div className="track">
                            <div
                              className="fill"
                              style={{
                                width:
                                  `${value * 100}%`,
                              }}
                            />
                          </div>

                          <div className="bar-value">
                            {percent(value)}
                          </div>
                        </div>
                      );
                    }
                  )}

                </div>
              </div>

            </div>

          </section>


          {/* GNN ABLATION */}

          <section className="section">

            <div className="eyebrow">
              GNN FEATURE ABLATION
            </div>

            <h2 className="section-title">
              Graph architecture sensitivity
            </h2>

            <p className="section-description">
              Comparison of GCN, GraphSAGE, and GAT
              under different feature subsets.
            </p>

            <div className="card table-wrap">

              <table>

                <thead>
                  <tr>
                    <th>MODEL</th>
                    <th>FEATURES</th>
                    <th>VAL PR-AUC</th>
                    <th>TEST PR-AUC</th>
                    <th>TEST F1</th>
                    <th>ROC-AUC</th>
                  </tr>
                </thead>

                <tbody>

                  {gnnRows
                    .filter((row) =>
                      gnnModels.includes(
                        text(row, ["model"])
                      )
                    )
                    .map((row, index) => {

                      const featureCount =
                        num(row, [
                          "n_features",
                          "features",
                          "feature_count",
                          "top_k",
                        ]);

                      const valPr =
                        num(row, [
                          "validation_pr_auc",
                          "val_pr_auc",
                          "val_prauc",
                        ]);

                      const testPr =
                        num(row, [
                          "test_pr_auc",
                          "pr_auc",
                          "test_prauc",
                        ]);

                      const f1 =
                        num(row, [
                          "test_f1",
                          "f1",
                          "f1_score",
                        ]);

                      const roc =
                        num(row, [
                          "test_roc_auc",
                          "roc_auc",
                        ]);

                      return (
                        <tr key={index}>
                          <td className="strong">
                            {text(row, [
                              "model",
                            ])}
                          </td>

                          <td>
                            {featureCount || "All"}
                          </td>

                          <td>
                            {percent(valPr)}
                          </td>

                          <td className="blue">
                            {percent(testPr)}
                          </td>

                          <td>
                            {percent(f1)}
                          </td>

                          <td>
                            {percent(roc)}
                          </td>
                        </tr>
                      );
                    })}

                </tbody>

              </table>

            </div>

          </section>


          {/* ROBUSTNESS */}

          <section className="section">

            <div className="eyebrow">
              ROBUSTNESS ANALYSIS
            </div>

            <h2 className="section-title">
              Five-seed model stability
            </h2>

            <p className="section-description">
              Test PR-AUC across independent random
              seeds.
            </p>


            <div className="charts">

              {robustnessModels.map(
                (model) => {

                  const rows =
                    data.robustness.filter(
                      (row) =>
                        text(row, [
                          "model",
                          "model_name",
                        ]) === model
                    );

                  return (
                    <div
                      className="card chart"
                      key={model}
                    >

                      <div className="chart-title">
                        {model}
                      </div>

                      <div className="chart-note">
                        Test PR-AUC by seed
                      </div>

                      <div className="seed-grid">

                        {rows.map(
                          (row, index) => {

                            const seed =
                              text(row, [
                                "seed",
                                "random_seed",
                              ]);

                            const pr =
                              num(row, [
                                "test_pr_auc",
                                "pr_auc",
                              ]);

                            return (
                              <div
                                className="seed"
                                key={index}
                              >
                                <div className="seed-number">
                                  SEED {seed}
                                </div>

                                <div className="seed-value">
                                  {percent(pr)}
                                </div>
                              </div>
                            );
                          }
                        )}

                      </div>

                    </div>
                  );
                }
              )}

            </div>

          </section>


          {/* STATISTICS */}

          <section className="section">

            <div className="eyebrow">
              STATISTICAL ANALYSIS
            </div>

            <h2 className="section-title">
              Paired PR-AUC comparisons
            </h2>

            <p className="section-description">
              Wilcoxon signed-rank comparisons across
              the five paired seeds.
            </p>


            <div className="card table-wrap">

              <table>

                <thead>
                  <tr>
                    <th>COMPARISON</th>
                    <th>MEAN DIFFERENCE</th>
                    <th>W</th>
                    <th>P-VALUE</th>
                  </tr>
                </thead>

                <tbody>

                  {data.statistics.map(
                    (row, index) => (
                      <tr key={index}>

                        <td className="strong">
                          {text(row, [
                            "comparison",
                            "pair",
                            "model_comparison",
                          ])}
                        </td>

                        <td>
                          {num(row, [
                            "mean_difference",
                            "mean_diff",
                            "difference",
                          ]).toFixed(4)}
                        </td>

                        <td>
                          {text(row, [
                            "w",
                            "statistic",
                            "wilcoxon_statistic",
                          ])}
                        </td>

                        <td>
                          {num(row, [
                            "p_value",
                            "p",
                            "pvalue",
                          ]).toFixed(4)}
                        </td>

                      </tr>
                    )
                  )}

                </tbody>

              </table>

            </div>

          </section>


          {/* COMPUTATIONAL EFFICIENCY */}
          <section className="section">
            <div className="eyebrow">
              COMPUTATIONAL EFFICIENCY
            </div>

            <h2 className="section-title">
              Training and inference cost
            </h2>

            <p className="section-description">
              Recorded computational measurements for the primary model
              architectures under the GraphShield benchmark configuration.
            </p>

            <div className="efficiency-grid">
              {data.efficiency.map((row, index) => {
                const model = text(row, [
                  "model",
                  "model_name",
                ]) || "Unknown model";

                const training = num(row, [
                  "training_seconds",
                  "training_time",
                  "train_seconds",
                  "train_time",
                  "training_time_seconds",
                ]);

                const inference = num(row, [
                  "inference_seconds",
                  "inference_time",
                  "infer_seconds",
                  "inference_time_seconds",
                ]);

                const parameters = num(row, [
                  "parameters",
                  "parameter_count",
                  "params",
                ]);

                const isXGB = model.toLowerCase().includes("xgboost");

                return (
                  <div
                    className="card efficiency-card"
                    key={`${model}-${index}`}
                  >
                    <div className="efficiency-model">
                      {model}
                    </div>

                    <div className="efficiency-type">
                      {["GCN", "GraphSAGE", "GAT"].includes(model)
                        ? "Graph neural network"
                        : "Traditional ML"}
                    </div>

                    <div className="efficiency-metrics">
                      <div className="efficiency-metric">
                        <div className="efficiency-label">
                          Training
                        </div>

                        <div className="efficiency-value">
                          {isXGB && training === 0
                            ? "N/A"
                            : `${training.toFixed(3)} s`}
                        </div>
                      </div>

                      <div className="efficiency-metric">
                        <div className="efficiency-label">
                          Inference
                        </div>

                        <div className="efficiency-value">
                          {isXGB && inference === 0
                            ? "N/A"
                            : `${inference.toFixed(4)} s`}
                        </div>
                      </div>
                    </div>

                    {parameters > 0 && (
                      <div className="efficiency-note">
                        <strong>Parameters:</strong>{" "}
                        {parameters.toLocaleString()}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="efficiency-note">
              <strong>Benchmark note:</strong>{" "}
              XGBoost is shown as N/A where the native segmentation fault
              prevented a reliable efficiency measurement for this workload.
              The displayed measurements should be interpreted as benchmark
              observations under the recorded experimental configuration.
            </div>
          </section>

          {/* RESEARCH NOTES */}

          <section className="section">

            <div className="eyebrow">
              EXPERIMENTAL INTERPRETATION
            </div>

            <h2 className="section-title">
              What these experiments test
            </h2>


            <div className="finding-grid">

              <div className="card finding">

                <div className="finding-number">
                  01
                </div>

                <div className="finding-title">
                  Feature dependence
                </div>

                <div className="finding-text">
                  Feature ablation examines whether model
                  performance is concentrated in a small
                  subset of transaction features or depends
                  on a broader feature representation.
                </div>

              </div>


              <div className="card finding">

                <div className="finding-number">
                  02
                </div>

                <div className="finding-title">
                  Graph sensitivity
                </div>

                <div className="finding-text">
                  GNN ablation compares how GCN, GraphSAGE,
                  and GAT behave as the transaction feature
                  representation changes.
                </div>

              </div>


              <div className="card finding">

                <div className="finding-number">
                  03
                </div>

                <div className="finding-title">
                  Reproducibility
                </div>

                <div className="finding-text">
                  Five random seeds provide a direct view
                  of variation in model performance instead
                  of relying on a single experimental run.
                </div>

              </div>

            </div>

          </section>

        </div>
      </main>
    </>
  );
}