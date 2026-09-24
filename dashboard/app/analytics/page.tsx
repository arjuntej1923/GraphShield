"use client";

import { useEffect, useState } from "react";

type ModelResult = {
  model: string;
  validation_pr_auc: string;
  test_pr_auc: string;
  test_roc_auc: string;
  test_precision: string;
  test_recall: string;
  test_f1: string;
  selected_threshold: string;
};

type RobustnessRow = {
  model: string;
  seed: string;
  test_pr_auc: string;
  test_f1: string;
  test_precision: string;
  test_recall: string;
};

type AnalyticsData = {
  models: ModelResult[];
  robustness: RobustnessRow[];
  dataset: Record<string, string>[];
  featureAblation: Record<string, string>[];
  gnnAblation: Record<string, string>[];
  statistics: Record<string, string>[];
  efficiency: Record<string, string>[];
};

export default function ResearchAnalytics() {

  const [data, setData] =
    useState<AnalyticsData | null>(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  useEffect(() => {

    async function loadAnalytics() {

      try {

        const response =
          await fetch(
            "/api/analytics"
          );

        const result =
          await response.json();

        if (!response.ok) {
          throw new Error(
            result.error ||
              "Failed to load analytics."
          );
        }

        setData(result);

      } catch (err) {

        setError(
          err instanceof Error
            ? err.message
            : "Failed to load analytics."
        );

      } finally {

        setLoading(false);

      }
    }

    loadAnalytics();

  }, []);

  if (loading) {

    return (
      <main className="analytics-page">
        <div className="loading">
          Loading GraphShield research results...
        </div>
      </main>
    );

  }

  if (error || !data) {

    return (
      <main className="analytics-page">
        <div className="error">
          {error || "No analytics data available."}
        </div>
      </main>
    );

  }

  const models =
    data.models;

  const maxPRAUC =
    Math.max(
      ...models.map(
        (m) =>
          Number(m.test_pr_auc)
      )
    );

  const maxROCAUC =
    Math.max(
      ...models.map(
        (m) =>
          Number(m.test_roc_auc)
      )
    );

  const maxF1 =
    Math.max(
      ...models.map(
        (m) =>
          Number(m.test_f1)
      )
    );

  const bestPR =
    models.reduce(
      (best, current) =>
        Number(current.test_pr_auc) >
        Number(best.test_pr_auc)
          ? current
          : best
    );

  const traditional =
    models.filter(
      (m) =>
        ![
          "GCN",
          "GraphSAGE",
          "GAT",
        ].includes(m.model)
    );

  const gnn =
    models.filter(
      (m) =>
        [
          "GCN",
          "GraphSAGE",
          "GAT",
        ].includes(m.model)
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
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
        }

        .analytics-page {
          min-height: 100vh;
          padding:
            42px
            48px
            90px;

          background:
            radial-gradient(
              circle at 15% 0%,
              rgba(37,99,235,.12),
              transparent 30%
            ),
            radial-gradient(
              circle at 90% 15%,
              rgba(14,165,233,.08),
              transparent 28%
            ),
            #050914;
        }

        .analytics-container {
          max-width: 1500px;
          margin: 0 auto;
        }

        .eyebrow {
          color: #60a5fa;
          font-size: 10px;
          font-weight: 800;
          letter-spacing: .16em;
          text-transform: uppercase;
        }

        .header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          gap: 30px;
          margin-bottom: 35px;
        }

        .title {
          margin: 9px 0 0;
          color: #f8fafc;
          font-size: 38px;
          line-height: 1.1;
          font-weight: 750;
          letter-spacing: -.025em;
        }

        .subtitle {
          max-width: 800px;
          margin-top: 13px;
          color: #94a3b8;
          font-size: 15px;
          line-height: 1.7;
        }

        .research-badge {
          padding: 10px 14px;
          border-radius: 8px;
          border: 1px solid #1e293b;
          background: #09101e;
          color: #94a3b8;
          font-size: 9px;
          font-weight: 800;
          letter-spacing: .12em;
          white-space: nowrap;
        }

        .research-question {
          padding: 24px;
          margin-bottom: 35px;
          border-radius: 12px;
          border: 1px solid #1e293b;
          background:
            linear-gradient(
              120deg,
              rgba(37,99,235,.1),
              rgba(9,15,28,.95)
            );
        }

        .question-text {
          margin-top: 10px;
          color: #f8fafc;
          font-size: 21px;
          font-weight: 650;
          line-height: 1.5;
        }

        .stats-grid {
          display: grid;
          grid-template-columns:
            repeat(4,minmax(0,1fr));
          gap: 14px;
          margin-bottom: 38px;
        }

        .stat-card {
          padding: 20px;
          border: 1px solid #1e293b;
          border-radius: 11px;
          background: #090f1c;
        }

        .stat-label {
          color: #64748b;
          font-size: 9px;
          font-weight: 800;
          letter-spacing: .13em;
        }

        .stat-value {
          margin-top: 12px;
          color: #f8fafc;
          font-size: 30px;
          font-weight: 750;
        }

        .stat-note {
          margin-top: 5px;
          color: #64748b;
          font-size: 11px;
        }

        .section {
          margin-top: 38px;
        }

        .section-header {
          margin-bottom: 17px;
        }

        .section-title {
          margin: 5px 0 0;
          color: #f8fafc;
          font-size: 21px;
          font-weight: 700;
        }

        .section-description {
          margin-top: 7px;
          color: #64748b;
          font-size: 12px;
        }

        .model-table {
          overflow: hidden;
          border: 1px solid #1e293b;
          border-radius: 12px;
          background: #090f1c;
        }

        .table-header,
        .table-row {
          display: grid;
          grid-template-columns:
            1.7fr
            repeat(6,1fr);
          gap: 10px;
          align-items: center;
          padding: 15px 18px;
        }

        .table-header {
          color: #64748b;
          background: #070d18;
          border-bottom: 1px solid #1e293b;
          font-size: 9px;
          font-weight: 800;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .table-row {
          color: #cbd5e1;
          font-size: 12px;
          border-bottom: 1px solid #111827;
        }

        .table-row:last-child {
          border-bottom: 0;
        }

        .model-name {
          color: #f8fafc;
          font-weight: 700;
        }

        .model-type {
          margin-top: 4px;
          color: #64748b;
          font-size: 9px;
        }

        .metric-number {
          font-family:
            "SFMono-Regular",
            Consolas,
            monospace;
        }

        .highlight {
          color: #60a5fa;
          font-weight: 750;
        }

        .charts-grid {
          display: grid;
          grid-template-columns:
            repeat(2,minmax(0,1fr));
          gap: 16px;
        }

        .chart-card {
          padding: 22px;
          border: 1px solid #1e293b;
          border-radius: 12px;
          background: #090f1c;
        }

        .chart-title {
          color: #f8fafc;
          font-size: 15px;
          font-weight: 700;
        }

        .chart-subtitle {
          margin-top: 5px;
          color: #64748b;
          font-size: 11px;
        }

        .bars {
          display: flex;
          flex-direction: column;
          gap: 15px;
          margin-top: 23px;
        }

        .bar-row {
          display: grid;
          grid-template-columns:
            105px 1fr 55px;
          gap: 10px;
          align-items: center;
        }

        .bar-label {
          color: #cbd5e1;
          font-size: 10px;
          font-weight: 600;
        }

        .bar-track {
          height: 9px;
          overflow: hidden;
          border-radius: 999px;
          background: #111827;
        }

        .bar-fill {
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
          color: #e2e8f0;
          font-size: 10px;
          font-weight: 700;
          text-align: right;
        }

        .comparison-grid {
          display: grid;
          grid-template-columns:
            repeat(2,minmax(0,1fr));
          gap: 16px;
        }

        .comparison-card {
          padding: 22px;
          border: 1px solid #1e293b;
          border-radius: 12px;
          background: #090f1c;
        }

        .comparison-title {
          color: #f8fafc;
          font-size: 15px;
          font-weight: 700;
        }

        .comparison-subtitle {
          margin-top: 5px;
          color: #64748b;
          font-size: 11px;
        }

        .model-chip {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 12px 0;
          border-bottom: 1px solid #111827;
        }

        .model-chip:last-child {
          border-bottom: 0;
        }

        .chip-name {
          color: #cbd5e1;
          font-size: 12px;
        }

        .chip-value {
          color: #f8fafc;
          font-family:
            "SFMono-Regular",
            Consolas,
            monospace;
          font-size: 12px;
          font-weight: 700;
        }

        .finding-grid {
          display: grid;
          grid-template-columns:
            repeat(3,minmax(0,1fr));
          gap: 14px;
        }

        .finding {
          min-height: 160px;
          padding: 21px;
          border: 1px solid #1e293b;
          border-radius: 11px;
          background: #090f1c;
        }

        .finding-number {
          color: #60a5fa;
          font-size: 25px;
          font-weight: 750;
        }

        .finding-title {
          margin-top: 10px;
          color: #f8fafc;
          font-size: 14px;
          font-weight: 700;
        }

        .finding-text {
          margin-top: 7px;
          color: #64748b;
          font-size: 11px;
          line-height: 1.65;
        }

        .loading,
        .error {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #94a3b8;
        }

        .error {
          color: #fca5a5;
        }

        @media(max-width:1000px) {

          .analytics-page {
            padding: 30px 22px 60px;
          }

          .stats-grid {
            grid-template-columns:
              repeat(2,1fr);
          }

          .charts-grid,
          .comparison-grid {
            grid-template-columns: 1fr;
          }

          .finding-grid {
            grid-template-columns: 1fr;
          }

        }

        @media(max-width:650px) {

          .analytics-page {
            padding: 22px 14px 50px;
          }

          .header {
            flex-direction: column;
          }

          .title {
            font-size: 30px;
          }

          .stats-grid {
            grid-template-columns: 1fr;
          }

          .table-header,
          .table-row {
            min-width: 800px;
          }

          .model-table {
            overflow-x: auto;
          }

        }

      `}</style>


      <main className="analytics-page">

        <div className="analytics-container">

          {/* HEADER */}

          <header className="header">

            <div>

              <div className="eyebrow">
                GRAPHSHEILD · RESEARCH ANALYTICS
              </div>

              <h1 className="title">
                Research Analytics
              </h1>

              <p className="subtitle">
                Experimental results from the
                GraphShield fraud detection study,
                including traditional machine learning,
                graph neural networks, ablation studies,
                and robustness experiments.
              </p>

            </div>

            <div className="research-badge">
              EXPERIMENTAL RESULTS
            </div>

          </header>


          {/* RESEARCH QUESTION */}

          <section className="research-question">

            <div className="eyebrow">
              RESEARCH QUESTION
            </div>

            <div className="question-text">
              Does relational graph information improve
              fraud detection compared with treating
              transactions independently?
            </div>

          </section>


          {/* OVERVIEW */}

          <div className="stats-grid">

            <div className="stat-card">

              <div className="stat-label">
                BEST TEST PR-AUC
              </div>

              <div className="stat-value">
                {(
                  Number(
                    bestPR.test_pr_auc
                  ) * 100
                ).toFixed(2)}
                %
              </div>

              <div className="stat-note">
                {bestPR.model}
              </div>

            </div>


            <div className="stat-card">

              <div className="stat-label">
                MODELS EVALUATED
              </div>

              <div className="stat-value">
                {models.length}
              </div>

              <div className="stat-note">
                Traditional ML + GNN
              </div>

            </div>


            <div className="stat-card">

              <div className="stat-label">
                GNN ARCHITECTURES
              </div>

              <div className="stat-value">
                3
              </div>

              <div className="stat-note">
                GCN · GraphSAGE · GAT
              </div>

            </div>


            <div className="stat-card">

              <div className="stat-label">
                ROBUSTNESS SEEDS
              </div>

              <div className="stat-value">
                5
              </div>

              <div className="stat-note">
                Independent random seeds
              </div>

            </div>

          </div>


          {/* MODEL TABLE */}

          <section className="section">

            <div className="section-header">

              <div className="eyebrow">
                PRIMARY EXPERIMENT
              </div>

              <h2 className="section-title">
                Model performance
              </h2>

              <div className="section-description">
                Test-set metrics using validation-selected
                decision thresholds.
              </div>

            </div>


            <div className="model-table">

              <div className="table-header">

                <div>Model</div>
                <div>Val PR</div>
                <div>Test PR</div>
                <div>ROC-AUC</div>
                <div>Precision</div>
                <div>Recall</div>
                <div>F1</div>

              </div>


              {models.map((model) => (

                <div
                  className="table-row"
                  key={model.model}
                >

                  <div>

                    <div className="model-name">
                      {model.model}
                    </div>

                    <div className="model-type">
                      {[
                        "GCN",
                        "GraphSAGE",
                        "GAT",
                      ].includes(model.model)
                        ? "Graph Neural Network"
                        : "Traditional ML"}
                    </div>

                  </div>

                  <div className="metric-number">
                    {(
                      Number(
                        model.validation_pr_auc
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                  <div className="metric-number highlight">
                    {(
                      Number(
                        model.test_pr_auc
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                  <div className="metric-number">
                    {(
                      Number(
                        model.test_roc_auc
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                  <div className="metric-number">
                    {(
                      Number(
                        model.test_precision
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                  <div className="metric-number">
                    {(
                      Number(
                        model.test_recall
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                  <div className="metric-number">
                    {(
                      Number(
                        model.test_f1
                      ) * 100
                    ).toFixed(2)}%
                  </div>

                </div>

              ))}

            </div>

          </section>


          {/* PERFORMANCE BARS */}

          <section className="section">

            <div className="section-header">

              <div className="eyebrow">
                PERFORMANCE COMPARISON
              </div>

              <h2 className="section-title">
                Model metrics
              </h2>

            </div>


            <div className="charts-grid">

              {/* PR AUC */}

              <div className="chart-card">

                <div className="chart-title">
                  Test PR-AUC
                </div>

                <div className="chart-subtitle">
                  Precision-recall area under the curve
                </div>

                <div className="bars">

                  {models.map((model) => {

                    const value =
                      Number(
                        model.test_pr_auc
                      );

                    return (

                      <div
                        className="bar-row"
                        key={`pr-${model.model}`}
                      >

                        <div className="bar-label">
                          {model.model}
                        </div>

                        <div className="bar-track">

                          <div
                            className="bar-fill"
                            style={{
                              width:
                                `${(
                                  value /
                                  maxPRAUC
                                ) * 100}%`,
                            }}
                          />

                        </div>

                        <div className="bar-value">
                          {(value * 100).toFixed(1)}%
                        </div>

                      </div>

                    );

                  })}

                </div>

              </div>


              {/* ROC */}

              <div className="chart-card">

                <div className="chart-title">
                  Test ROC-AUC
                </div>

                <div className="chart-subtitle">
                  Receiver operating characteristic
                  area under the curve
                </div>

                <div className="bars">

                  {models.map((model) => {

                    const value =
                      Number(
                        model.test_roc_auc
                      );

                    return (

                      <div
                        className="bar-row"
                        key={`roc-${model.model}`}
                      >

                        <div className="bar-label">
                          {model.model}
                        </div>

                        <div className="bar-track">

                          <div
                            className="bar-fill"
                            style={{
                              width:
                                `${(
                                  value /
                                  maxROCAUC
                                ) * 100}%`,
                            }}
                          />

                        </div>

                        <div className="bar-value">
                          {(value * 100).toFixed(1)}%
                        </div>

                      </div>

                    );

                  })}

                </div>

              </div>


              {/* F1 */}

              <div className="chart-card">

                <div className="chart-title">
                  Test F1
                </div>

                <div className="chart-subtitle">
                  Balance between precision and recall
                </div>

                <div className="bars">

                  {models.map((model) => {

                    const value =
                      Number(
                        model.test_f1
                      );

                    return (

                      <div
                        className="bar-row"
                        key={`f1-${model.model}`}
                      >

                        <div className="bar-label">
                          {model.model}
                        </div>

                        <div className="bar-track">

                          <div
                            className="bar-fill"
                            style={{
                              width:
                                `${(
                                  value /
                                  maxF1
                                ) * 100}%`,
                            }}
                          />

                        </div>

                        <div className="bar-value">
                          {(value * 100).toFixed(1)}%
                        </div>

                      </div>

                    );

                  })}

                </div>

              </div>


              {/* PRECISION / RECALL */}

              <div className="chart-card">

                <div className="chart-title">
                  Precision and Recall
                </div>

                <div className="chart-subtitle">
                  Test-set operating characteristics
                </div>

                <div className="bars">

                  {models.map((model) => {

                    const precision =
                      Number(
                        model.test_precision
                      );

                    const recall =
                      Number(
                        model.test_recall
                      );

                    return (

                      <div
                        key={`prr-${model.model}`}
                      >

                        <div
                          style={{
                            display: "flex",
                            justifyContent:
                              "space-between",
                            marginBottom: "7px",
                          }}
                        >

                          <span className="bar-label">
                            {model.model}
                          </span>

                          <span
                            className="bar-value"
                          >
                            P{" "}
                            {(
                              precision * 100
                            ).toFixed(0)}
                            % · R{" "}
                            {(
                              recall * 100
                            ).toFixed(0)}
                            %
                          </span>

                        </div>

                        <div
                          className="bar-track"
                          style={{
                            marginBottom: "7px",
                          }}
                        >

                          <div
                            className="bar-fill"
                            style={{
                              width:
                                `${precision * 100}%`,
                            }}
                          />

                        </div>

                      </div>

                    );

                  })}

                </div>

              </div>

            </div>

          </section>


          {/* MODEL GROUPS */}

          <section className="section">

            <div className="section-header">

              <div className="eyebrow">
                ARCHITECTURE GROUPS
              </div>

              <h2 className="section-title">
                Traditional ML and GNN results
              </h2>

            </div>


            <div className="comparison-grid">

              <div className="comparison-card">

                <div className="comparison-title">
                  Traditional Machine Learning
                </div>

                <div className="comparison-subtitle">
                  Transaction-level feature models
                </div>

                {traditional.map((model) => (

                  <div
                    className="model-chip"
                    key={model.model}
                  >

                    <span className="chip-name">
                      {model.model}
                    </span>

                    <span className="chip-value">
                      PR-AUC{" "}
                      {(
                        Number(
                          model.test_pr_auc
                        ) * 100
                      ).toFixed(2)}
                      %
                    </span>

                  </div>

                ))}

              </div>


              <div className="comparison-card">

                <div className="comparison-title">
                  Graph Neural Networks
                </div>

                <div className="comparison-subtitle">
                  Relational graph-based models
                </div>

                {gnn.map((model) => (

                  <div
                    className="model-chip"
                    key={model.model}
                  >

                    <span className="chip-name">
                      {model.model}
                    </span>

                    <span className="chip-value">
                      PR-AUC{" "}
                      {(
                        Number(
                          model.test_pr_auc
                        ) * 100
                      ).toFixed(2)}
                      %
                    </span>

                  </div>

                ))}

              </div>

            </div>

          </section>


          {/* RESEARCH FINDINGS */}

          <section className="section">

            <div className="section-header">

              <div className="eyebrow">
                RESEARCH OBSERVATIONS
              </div>

              <h2 className="section-title">
                Experimental findings
              </h2>

            </div>


            <div className="finding-grid">

              <div className="finding">

                <div className="finding-number">
                  {(
                    Number(
                      bestPR.test_pr_auc
                    ) * 100
                  ).toFixed(1)}%
                </div>

                <div className="finding-title">
                  Highest test PR-AUC
                </div>

                <div className="finding-text">
                  {bestPR.model} produced the highest
                  test PR-AUC among the six primary
                  models in the recorded experiment.
                </div>

              </div>


              <div className="finding">

                <div className="finding-number">
                  5
                </div>

                <div className="finding-title">
                  Robustness seeds
                </div>

                <div className="finding-text">
                  The robustness experiment evaluates
                  model behavior across five independent
                  random seeds.
                </div>

              </div>


              <div className="finding">

                <div className="finding-number">
                  3
                </div>

                <div className="finding-title">
                  GNN architectures
                </div>

                <div className="finding-text">
                  GCN, GraphSAGE, and GAT were evaluated
                  as graph-based alternatives to
                  transaction-level models.
                </div>

              </div>

            </div>

          </section>


        </div>

      </main>
    </>
  );
}