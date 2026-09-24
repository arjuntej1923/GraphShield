"use client";

import { FormEvent, useEffect, useState } from "react";

type Transaction = {
  txId: string;
  label: string;
  labelValue: string;
  timeStep: number;
  labeled: boolean;
  incoming: number;
  outgoing: number;
  totalDegree: number;
  illicitNeighborRatio: number;
  illicitNeighbors: number;
  licitNeighbors: number;
  unknownNeighbors: number;
  totalNeighbors: number;
};

type ModelResult = {
  model: string;
  type: string;
  probability: number | null;
  threshold: number;
  prediction: "ILLICIT" | "LICIT" | "N/A";
};

type Consensus = {
  illicitVotes: number;
  totalModels: number;
};

type ApiResponse = {
  transaction: Transaction;
  models: ModelResult[];
  consensus: Consensus;
};

const MODEL_ORDER = [
  "Logistic Regression",
  "Random Forest",
  "XGBoost",
  "GCN",
  "GraphSAGE",
  "GAT",
];

function formatProbability(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return "N/A";
  }

  if (value < 0.0001) {
    return `${(value * 100).toFixed(4)}%`;
  }

  return `${(value * 100).toFixed(2)}%`;
}

function clampProbability(value: number | null) {
  if (value === null || Number.isNaN(value)) {
    return 0;
  }

  return Math.max(0, Math.min(1, value));
}

function labelClass(label: string) {
  const normalized = label.toLowerCase();

  if (normalized === "illicit") {
    return "tx-badge tx-badge-danger";
  }

  if (normalized === "licit") {
    return "tx-badge tx-badge-success";
  }

  return "tx-badge";
}

function predictionClass(prediction: string) {
  if (prediction === "ILLICIT") {
    return "tx-prediction tx-prediction-danger";
  }

  if (prediction === "LICIT") {
    return "tx-prediction tx-prediction-success";
  }

  return "tx-prediction";
}

export default function TransactionAnalysis() {
  const [txId, setTxId] = useState("12781680");
  const [transaction, setTransaction] =
    useState<Transaction | null>(null);

  const [models, setModels] = useState<ModelResult[]>([]);
  const [consensus, setConsensus] =
    useState<Consensus | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyzeTransaction(id: string) {
    const cleanId = id.trim();

    if (!cleanId) {
      setError("Please enter a transaction ID.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `/api/transactions?txId=${encodeURIComponent(cleanId)}`,
        {
          cache: "no-store",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error ||
            "Unable to analyze this transaction."
        );
      }

      setTransaction(data.transaction);
      setModels(data.models || []);
      setConsensus(data.consensus || null);
    } catch (err) {
      setTransaction(null);
      setModels([]);
      setConsensus(null);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to analyze this transaction."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    analyzeTransaction("12781680");
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    analyzeTransaction(txId);
  }

  const illicitRatio =
    transaction?.illicitNeighborRatio ?? 0;

  return (
    <>
      <style jsx global>{`
        .transaction-page {
          min-height: 100vh;
          background:
            radial-gradient(
              circle at 20% 0%,
              rgba(37, 99, 235, 0.08),
              transparent 32%
            ),
            radial-gradient(
              circle at 85% 15%,
              rgba(14, 165, 233, 0.05),
              transparent 28%
            ),
            #050914;

          color: #e2e8f0;

          padding: 46px 24px 80px;
        }

        .transaction-container {
          max-width: 1420px;
          margin: 0 auto;
        }

        .transaction-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;

          gap: 30px;

          margin-bottom: 34px;
        }

        .transaction-eyebrow {
          color: #38bdf8;

          font-size: 10px;
          font-weight: 800;

          letter-spacing: 0.22em;
          text-transform: uppercase;

          margin-bottom: 12px;
        }

        .transaction-title {
          margin: 0;

          color: #f8fafc;

          font-size: clamp(34px, 4vw, 54px);
          line-height: 1.05;

          letter-spacing: -0.04em;
          font-weight: 800;
        }

        .transaction-subtitle {
          max-width: 780px;

          margin-top: 16px;

          color: #8190a8;

          font-size: 14px;
          line-height: 1.7;
        }

        .live-indicator {
          display: flex;
          align-items: center;
          gap: 8px;

          padding: 10px 14px;

          border: 1px solid #1e293b;
          border-radius: 9px;

          color: #94a3b8;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.12em;
          white-space: nowrap;
        }

        .live-dot {
          width: 7px;
          height: 7px;

          border-radius: 50%;

          background: #22c55e;

          box-shadow:
            0 0 12px
            rgba(34, 197, 94, 0.7);
        }

        .search-card {
          padding: 22px;

          border: 1px solid #1e293b;
          border-radius: 14px;

          background:
            linear-gradient(
              145deg,
              rgba(15, 23, 42, 0.92),
              rgba(7, 13, 27, 0.92)
            );

          margin-bottom: 24px;
        }

        .search-label {
          display: block;

          margin-bottom: 10px;

          color: #71809a;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.18em;
          text-transform: uppercase;
        }

        .search-row {
          display: flex;
          gap: 12px;
        }

        .search-input {
          flex: 1;

          min-width: 0;

          height: 50px;

          padding: 0 16px;

          border: 1px solid #26354d;
          border-radius: 9px;

          background: #070d1b;

          color: #f8fafc;

          font-size: 14px;

          outline: none;

          transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease;
        }

        .search-input:focus {
          border-color: #3b82f6;

          box-shadow:
            0 0 0 3px
            rgba(59, 130, 246, 0.12);
        }

        .search-button {
          height: 50px;

          padding: 0 24px;

          border: 0;
          border-radius: 9px;

          background:
            linear-gradient(
              135deg,
              #2563eb,
              #0ea5e9
            );

          color: white;

          font-size: 12px;
          font-weight: 800;

          cursor: pointer;

          transition:
            transform 0.15s ease,
            opacity 0.15s ease;
        }

        .search-button:hover {
          transform: translateY(-1px);
        }

        .search-button:disabled {
          opacity: 0.55;
          cursor: wait;
          transform: none;
        }

        .error-box {
          margin-top: 14px;

          padding: 13px 15px;

          border:
            1px solid
            rgba(239, 68, 68, 0.3);

          border-radius: 8px;

          background:
            rgba(239, 68, 68, 0.07);

          color: #fca5a5;

          font-size: 12px;
        }

        .overview-grid {
          display: grid;

          grid-template-columns:
            minmax(0, 1fr)
            300px;

          gap: 18px;

          margin-bottom: 30px;
        }

        .panel {
          border:
            1px solid
            #1e293b;

          border-radius: 14px;

          background:
            rgba(7, 13, 27, 0.86);
        }

        .transaction-summary {
          padding: 26px;
        }

        .panel-label {
          color: #64748b;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.18em;
          text-transform: uppercase;
        }

        .transaction-id {
          margin-top: 10px;

          color: #f8fafc;

          font-size: 31px;
          font-weight: 800;

          letter-spacing: -0.03em;
        }

        .badges {
          display: flex;

          flex-wrap: wrap;

          gap: 8px;

          margin-top: 16px;
        }

        .tx-badge {
          display: inline-flex;
          align-items: center;

          min-height: 28px;

          padding: 0 10px;

          border:
            1px solid
            #26354d;

          border-radius: 7px;

          color: #94a3b8;

          background: #0a1221;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .tx-badge-success {
          border-color:
            rgba(34, 197, 94, 0.3);

          background:
            rgba(34, 197, 94, 0.08);

          color: #4ade80;
        }

        .tx-badge-danger {
          border-color:
            rgba(239, 68, 68, 0.35);

          background:
            rgba(239, 68, 68, 0.08);

          color: #f87171;
        }

        .consensus-card {
          padding: 26px;

          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        .consensus-number {
          margin-top: 10px;

          color: #f8fafc;

          font-size: 45px;
          line-height: 1;

          font-weight: 850;

          letter-spacing: -0.05em;
        }

        .consensus-number span {
          color: #64748b;

          font-size: 19px;
          font-weight: 700;
        }

        .consensus-label {
          margin-top: 10px;

          color: #64748b;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.15em;
          text-transform: uppercase;
        }

        .section {
          margin-top: 34px;
        }

        .section-heading {
          margin-bottom: 15px;
        }

        .section-eyebrow {
          color: #38bdf8;

          font-size: 9px;
          font-weight: 800;

          letter-spacing: 0.2em;
          text-transform: uppercase;
        }

        .section-title {
          margin: 7px 0 0;

          color: #f1f5f9;

          font-size: 22px;
          font-weight: 750;

          letter-spacing: -0.025em;
        }

        .section-description {
          margin-top: 5px;

          color: #64748b;

          font-size: 12px;
        }

        .metrics-grid {
          display: grid;

          grid-template-columns:
            repeat(4, minmax(0, 1fr));

          gap: 14px;
        }

        .metric-card {
          min-height: 130px;

          padding: 20px;

          border:
            1px solid
            #1e293b;

          border-radius: 12px;

          background:
            linear-gradient(
              145deg,
              rgba(9, 17, 32, 0.95),
              rgba(5, 10, 20, 0.95)
            );
        }

        .metric-label {
          color: #64748b;

          font-size: 8px;
          font-weight: 800;

          letter-spacing: 0.18em;
          text-transform: uppercase;
        }

        .metric-value {
          margin-top: 18px;

          color: #f1f5f9;

          font-size: 32px;
          font-weight: 800;

          letter-spacing: -0.04em;
        }

        .metric-caption {
          margin-top: 6px;

          color: #64748b;

          font-size: 10px;
        }

        .neighbor-grid {
          display: grid;

          grid-template-columns:
            repeat(3, minmax(0, 1fr));

          gap: 14px;
        }

        .neighbor-card {
          padding: 20px;

          border:
            1px solid
            #1e293b;

          border-radius: 12px;

          background: #080f1e;
        }

        .neighbor-card.illicit {
          border-color:
            rgba(239, 68, 68, 0.2);
        }

        .neighbor-card.licit {
          border-color:
            rgba(34, 197, 94, 0.18);
        }

        .neighbor-card.unknown {
          border-color:
            rgba(148, 163, 184, 0.15);
        }

        .neighbor-count {
          margin-top: 13px;

          color: #f8fafc;

          font-size: 28px;
          font-weight: 800;
        }

        .neighbor-ratio {
          margin-top: 4px;

          color: #64748b;

          font-size: 11px;
        }

        .models-grid {
          display: grid;

          grid-template-columns:
            repeat(2, minmax(0, 1fr));

          gap: 14px;
        }

        .model-card {
          padding: 22px;

          border:
            1px solid
            #1e293b;

          border-radius: 13px;

          background:
            linear-gradient(
              145deg,
              rgba(9, 17, 32, 0.96),
              rgba(5, 10, 20, 0.96)
            );
        }

        .model-top {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;

          gap: 15px;
        }

        .model-type {
          color: #64748b;

          font-size: 8px;
          font-weight: 800;

          letter-spacing: 0.17em;
          text-transform: uppercase;
        }

        .model-name {
          margin-top: 7px;

          color: #f1f5f9;

          font-size: 17px;
          font-weight: 750;
        }

        .tx-prediction {
          padding: 7px 10px;

          border:
            1px solid
            #26354d;

          border-radius: 7px;

          color: #94a3b8;

          background: #0a1221;

          font-size: 8px;
          font-weight: 900;

          letter-spacing: 0.1em;
        }

        .tx-prediction-success {
          border-color:
            rgba(34, 197, 94, 0.28);

          background:
            rgba(34, 197, 94, 0.07);

          color: #4ade80;
        }

        .tx-prediction-danger {
          border-color:
            rgba(239, 68, 68, 0.3);

          background:
            rgba(239, 68, 68, 0.08);

          color: #f87171;
        }

        .probability-row {
          display: flex;

          justify-content: space-between;
          align-items: center;

          margin-top: 25px;
          margin-bottom: 8px;
        }

        .probability-label {
          color: #64748b;

          font-size: 10px;
        }

        .probability-value {
          color: #cbd5e1;

          font-size: 11px;
          font-weight: 750;
        }

        .probability-track {
          height: 6px;

          overflow: hidden;

          border-radius: 999px;

          background: #111c2e;
        }

        .probability-fill {
          height: 100%;

          border-radius: inherit;

          background:
            linear-gradient(
              90deg,
              #2563eb,
              #38bdf8
            );

          transition:
            width 0.5s ease;
        }

        .threshold-row {
          display: flex;

          justify-content: space-between;

          margin-top: 10px;

          color: #475569;

          font-size: 9px;
        }

        .assessment {
          display: flex;

          justify-content: space-between;
          align-items: center;

          gap: 30px;

          padding: 28px;

          border:
            1px solid
            #1e293b;

          border-radius: 14px;

          background:
            linear-gradient(
              120deg,
              rgba(15, 23, 42, 0.9),
              rgba(7, 13, 27, 0.95)
            );
        }

        .assessment-title {
          margin-top: 7px;

          color: #f8fafc;

          font-size: 21px;
          font-weight: 750;
        }

        .assessment-text {
          max-width: 780px;

          margin-top: 9px;

          color: #64748b;

          font-size: 12px;
          line-height: 1.7;
        }

        .assessment-number {
          flex-shrink: 0;

          color: #f8fafc;

          font-size: 46px;
          font-weight: 850;

          letter-spacing: -0.05em;
        }

        .assessment-number span {
          color: #64748b;

          font-size: 18px;
        }

        .loading-state {
          min-height: 320px;

          display: flex;
          align-items: center;
          justify-content: center;

          color: #64748b;

          font-size: 13px;
        }

        @media (max-width: 950px) {
          .overview-grid {
            grid-template-columns: 1fr;
          }

          .metrics-grid {
            grid-template-columns:
              repeat(2, minmax(0, 1fr));
          }

          .models-grid {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 650px) {
          .transaction-page {
            padding: 30px 14px 60px;
          }

          .transaction-header {
            flex-direction: column;
          }

          .search-row {
            flex-direction: column;
          }

          .metrics-grid,
          .neighbor-grid {
            grid-template-columns: 1fr;
          }

          .assessment {
            flex-direction: column;
            align-items: flex-start;
          }
        }
      `}</style>

      <main className="transaction-page">
        <div className="transaction-container">

          {/* HEADER */}
          <header className="transaction-header">

            <div>
              <div className="transaction-eyebrow">
                GraphShield · Transaction Intelligence
              </div>

              <h1 className="transaction-title">
                Transaction Analysis
              </h1>

              <p className="transaction-subtitle">
                Investigate individual transactions using
                graph structure, neighborhood relationships,
                and six fraud detection models.
              </p>
            </div>

            <div className="live-indicator">
              <span className="live-dot" />
              LIVE ANALYSIS
            </div>

          </header>


          {/* SEARCH */}
          <section className="search-card">

            <form
              onSubmit={handleSubmit}
            >

              <label className="search-label">
                Transaction ID
              </label>

              <div className="search-row">

                <input
                  className="search-input"
                  value={txId}
                  onChange={(event) =>
                    setTxId(event.target.value)
                  }
                  placeholder="Enter transaction ID..."
                  spellCheck={false}
                  autoComplete="off"
                />

                <button
                  type="submit"
                  className="search-button"
                  disabled={loading}
                >
                  {loading
                    ? "Analyzing..."
                    : "Analyze →"}
                </button>

              </div>

            </form>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

          </section>


          {loading && !transaction ? (
            <div className="panel loading-state">
              Loading transaction intelligence...
            </div>
          ) : transaction ? (

            <>
              {/* TRANSACTION OVERVIEW */}
              <section className="overview-grid">

                <div className="panel transaction-summary">

                  <div className="panel-label">
                    Transaction
                  </div>

                  <div className="transaction-id">
                    {transaction.txId}
                  </div>

                  <div className="badges">

                    <span
                      className={labelClass(
                        transaction.label
                      )}
                    >
                      {transaction.label}
                    </span>

                    <span className="tx-badge">
                      TIME STEP {transaction.timeStep}
                    </span>

                    <span className="tx-badge">
                      {transaction.labeled
                        ? "LABELED"
                        : "UNLABELED"}
                    </span>

                  </div>

                </div>


                <div className="panel consensus-card">

                  <div className="panel-label">
                    Model Consensus
                  </div>

                  <div className="consensus-number">

                    {consensus?.illicitVotes ?? 0}

                    <span>
                      /{consensus?.totalModels ?? 6}
                    </span>

                  </div>

                  <div className="consensus-label">
                    Models flagged illicit
                  </div>

                </div>

              </section>


              {/* GRAPH STRUCTURE */}
              <section className="section">

                <div className="section-heading">

                  <div className="section-eyebrow">
                    Graph Structure
                  </div>

                  <h2 className="section-title">
                    Transaction topology
                  </h2>

                  <p className="section-description">
                    Local connectivity around the selected
                    transaction.
                  </p>

                </div>


                <div className="metrics-grid">

                  <div className="metric-card">

                    <div className="metric-label">
                      Incoming
                    </div>

                    <div className="metric-value">
                      {transaction.incoming}
                    </div>

                    <div className="metric-caption">
                      Incoming connections
                    </div>

                  </div>


                  <div className="metric-card">

                    <div className="metric-label">
                      Outgoing
                    </div>

                    <div className="metric-value">
                      {transaction.outgoing}
                    </div>

                    <div className="metric-caption">
                      Outgoing connections
                    </div>

                  </div>


                  <div className="metric-card">

                    <div className="metric-label">
                      Total Degree
                    </div>

                    <div className="metric-value">
                      {transaction.totalDegree}
                    </div>

                    <div className="metric-caption">
                      Total graph connectivity
                    </div>

                  </div>


                  <div className="metric-card">

                    <div className="metric-label">
                      Total Neighbors
                    </div>

                    <div className="metric-value">
                      {transaction.totalNeighbors}
                    </div>

                    <div className="metric-caption">
                      Connected transactions
                    </div>

                  </div>

                </div>

              </section>


              {/* NEIGHBOR ANALYSIS */}
              <section className="section">

                <div className="section-heading">

                  <div className="section-eyebrow">
                    Neighbor Analysis
                  </div>

                  <h2 className="section-title">
                    Local graph neighborhood
                  </h2>

                  <p className="section-description">
                    Relationship-based signals around the
                    selected transaction.
                  </p>

                </div>


                <div className="neighbor-grid">

                  <div className="neighbor-card illicit">

                    <div className="metric-label">
                      Illicit Neighbors
                    </div>

                    <div className="neighbor-count">
                      {transaction.illicitNeighbors}
                    </div>

                    <div className="neighbor-ratio">
                      {formatProbability(
                        transaction.illicitNeighborRatio
                      )}
                    </div>

                  </div>


                  <div className="neighbor-card licit">

                    <div className="metric-label">
                      Licit Neighbors
                    </div>

                    <div className="neighbor-count">
                      {transaction.licitNeighbors}
                    </div>

                    <div className="neighbor-ratio">
                      {transaction.totalNeighbors > 0
                        ? formatProbability(
                            transaction.licitNeighbors /
                              transaction.totalNeighbors
                          )
                        : "0.00%"}
                    </div>

                  </div>


                  <div className="neighbor-card unknown">

                    <div className="metric-label">
                      Unknown Neighbors
                    </div>

                    <div className="neighbor-count">
                      {transaction.unknownNeighbors}
                    </div>

                    <div className="neighbor-ratio">
                      {transaction.totalNeighbors > 0
                        ? formatProbability(
                            transaction.unknownNeighbors /
                              transaction.totalNeighbors
                          )
                        : "0.00%"}
                    </div>

                  </div>

                </div>

              </section>


              {/* MODEL ANALYSIS */}
              <section className="section">

                <div className="section-heading">

                  <div className="section-eyebrow">
                    Model Intelligence
                  </div>

                  <h2 className="section-title">
                    Six-model fraud assessment
                  </h2>

                  <p className="section-description">
                    Independent model probabilities and
                    validation-selected decision thresholds.
                  </p>

                </div>


                <div className="models-grid">

                  {MODEL_ORDER.map((modelName) => {

                    const model =
                      models.find(
                        (item) =>
                          item.model === modelName
                      );

                    if (!model) {
                      return (
                        <div
                          key={modelName}
                          className="model-card"
                        >
                          <div className="model-name">
                            {modelName}
                          </div>

                          <div className="probability-row">
                            <span className="probability-label">
                              Status
                            </span>

                            <span className="probability-value">
                              N/A
                            </span>
                          </div>
                        </div>
                      );
                    }

                    const probability =
                      clampProbability(
                        model.probability
                      );

                    return (
                      <div
                        key={model.model}
                        className="model-card"
                      >

                        <div className="model-top">

                          <div>

                            <div className="model-type">
                              {model.type}
                            </div>

                            <div className="model-name">
                              {model.model}
                            </div>

                          </div>

                          <div
                            className={predictionClass(
                              model.prediction
                            )}
                          >
                            {model.prediction}
                          </div>

                        </div>


                        <div className="probability-row">

                          <span className="probability-label">
                            Illicit probability
                          </span>

                          <span className="probability-value">
                            {formatProbability(
                              model.probability
                            )}
                          </span>

                        </div>


                        <div className="probability-track">

                          <div
                            className="probability-fill"
                            style={{
                              width: `${probability * 100}%`,
                            }}
                          />

                        </div>


                        <div className="threshold-row">

                          <span>
                            Decision threshold
                          </span>

                          <span>
                            {model.threshold}
                          </span>

                        </div>

                      </div>
                    );
                  })}

                </div>

              </section>


              {/* ASSESSMENT */}
              <section className="section">

                <div className="assessment">

                  <div>

                    <div className="section-eyebrow">
                      Investigation Summary
                    </div>

                    <div className="assessment-title">
                      GraphShield assessment
                    </div>

                    <p className="assessment-text">
                      This transaction is evaluated using
                      transaction-level features together
                      with graph connectivity and neighborhood
                      information. The six model predictions
                      are displayed independently so their
                      behavior can be compared.
                    </p>

                  </div>


                  <div>

                    <div className="assessment-number">

                      {consensus?.illicitVotes ?? 0}

                      <span>
                        /{consensus?.totalModels ?? 6}
                      </span>

                    </div>

                    <div className="consensus-label">
                      Illicit model votes
                    </div>

                  </div>

                </div>

              </section>

            </>

          ) : null}

        </div>
      </main>
    </>
  );
}