"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  BarChart3,
  BrainCircuit,
  Database,
  FlaskConical,
  Gauge,
  Target,
  TrendingUp,
} from "lucide-react";

type ModelResult = {
  model: string;
  validationPrAuc: number | null;
  testPrAuc: number | null;
  testRocAuc: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  threshold: number | null;
};

function formatScore(value: number | null) {
  if (value === null || value === undefined) {
    return "—";
  }

  return value.toFixed(4);
}

function formatPercent(value: number | null) {
  if (value === null || value === undefined) {
    return "—";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function modelShortName(model: string) {
  if (model.toLowerCase().includes("logistic")) {
    return "Logistic Regression";
  }

  if (model.toLowerCase().includes("random")) {
    return "Random Forest";
  }

  if (model.toLowerCase().includes("xgb")) {
    return "XGBoost";
  }

  if (model.toLowerCase().includes("graphsage")) {
    return "GraphSAGE";
  }

  if (model.toLowerCase().includes("gcn")) {
    return "GCN";
  }

  if (model.toLowerCase().includes("gat")) {
    return "GAT";
  }

  return model;
}

export default function ModelLaboratory() {
  const [models, setModels] = useState<ModelResult[]>([]);
  const [selected, setSelected] = useState<ModelResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadModels() {
      try {
        const response = await fetch("/api/models");

        const result = await response.json();

        if (!response.ok) {
          throw new Error(
            result.error || "Unable to load model results."
          );
        }

        setModels(result.models);

        if (result.models.length > 0) {
          setSelected(result.models[0]);
        }
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to load model results."
        );
      } finally {
        setLoading(false);
      }
    }

    loadModels();
  }, []);

  const bestModel = useMemo(() => {
    if (!models.length) return null;

    return [...models].sort(
      (a, b) =>
        (b.testPrAuc ?? 0) -
        (a.testPrAuc ?? 0)
    )[0];
  }, [models]);

  return (
    <main className="models-page">
      <header className="models-topbar">
        <Link href="/" className="models-back">
          <ArrowLeft size={15} />
          Command Center
        </Link>

        <div className="models-title">
          <div className="models-title-icon">
            <FlaskConical size={18} />
          </div>

          <div>
            <div className="models-eyebrow">
              EXPERIMENTAL RESEARCH
            </div>

            <h1>Model Laboratory</h1>
          </div>
        </div>

        <div className="models-dataset">
          <Database size={13} />
          Elliptic Bitcoin
        </div>
      </header>

      <section className="models-content">
        <div className="models-intro">
          <div className="models-section-label">
            MODEL BENCHMARK
          </div>

          <h2>
            Traditional ML vs Graph Neural Networks.
          </h2>

          <p>
            Compare the experimental performance of six
            classification approaches using the temporally
            separated Elliptic Bitcoin evaluation protocol.
          </p>
        </div>

        {loading && (
          <div className="models-loading">
            Loading experimental results...
          </div>
        )}

        {error && (
          <div className="models-error">
            {error}
          </div>
        )}

        {!loading && !error && models.length > 0 && (
          <>
            <section className="model-highlight-grid">
              <div className="model-highlight">
                <div className="highlight-icon blue">
                  <TrendingUp size={18} />
                </div>

                <span>TOP TEST PR-AUC</span>

                <strong>
                  {formatScore(
                    bestModel?.testPrAuc ?? null
                  )}
                </strong>

                <small>
                  {bestModel
                    ? modelShortName(bestModel.model)
                    : "—"}
                </small>
              </div>

              <div className="model-highlight">
                <div className="highlight-icon purple">
                  <Target size={18} />
                </div>

                <span>MODELS EVALUATED</span>

                <strong>{models.length}</strong>

                <small>
                  3 traditional + 3 graph models
                </small>
              </div>

              <div className="model-highlight">
                <div className="highlight-icon cyan">
                  <Database size={18} />
                </div>

                <span>EVALUATION</span>

                <strong>49</strong>

                <small>
                  temporal time steps
                </small>
              </div>

              <div className="model-highlight">
                <div className="highlight-icon red">
                  <Gauge size={18} />
                </div>

                <span>PRIMARY METRIC</span>

                <strong>PR-AUC</strong>

                <small>
                  class imbalance aware
                </small>
              </div>
            </section>

            <section className="model-comparison">
              <div className="comparison-header">
                <div>
                  <span>PERFORMANCE MATRIX</span>
                  <h3>Test-set comparison</h3>
                </div>

                <div className="research-badge">
                  Validation-selected thresholds
                </div>
              </div>

              <div className="model-table">
                <div className="model-table-head">
                  <span>MODEL</span>
                  <span>TYPE</span>
                  <span>PR-AUC</span>
                  <span>ROC-AUC</span>
                  <span>PRECISION</span>
                  <span>RECALL</span>
                  <span>F1</span>
                </div>

                {models.map((model) => {
                  const isSelected =
                    selected?.model === model.model;

                  const isBest =
                    bestModel?.model === model.model;

                  const isGraph =
                    ["gcn", "gat", "graphsage"].some(
                      (name) =>
                        model.model
                          .toLowerCase()
                          .includes(name)
                    );

                  return (
                    <button
                      key={model.model}
                      className={`model-row ${
                        isSelected ? "selected" : ""
                      }`}
                      onClick={() => setSelected(model)}
                    >
                      <span className="model-name">
                        <span
                          className={`model-dot ${
                            isGraph
                              ? "graph-dot"
                              : "traditional-dot"
                          }`}
                        />

                        {modelShortName(model.model)}

                        {isBest && (
                          <em>TOP PR-AUC</em>
                        )}
                      </span>

                      <span className="model-type">
                        {isGraph
                          ? "GRAPH NEURAL NETWORK"
                          : "TRADITIONAL ML"}
                      </span>

                      <strong>
                        {formatScore(model.testPrAuc)}
                      </strong>

                      <span>
                        {formatScore(model.testRocAuc)}
                      </span>

                      <span>
                        {formatPercent(model.precision)}
                      </span>

                      <span>
                        {formatPercent(model.recall)}
                      </span>

                      <span>
                        {formatPercent(model.f1)}
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>

            {selected && (
              <section className="model-detail">
                <div className="detail-heading">
                  <div>
                    <div className="models-section-label">
                      SELECTED EXPERIMENT
                    </div>

                    <h3>
                      {modelShortName(selected.model)}
                    </h3>

                    <p>
                      Detailed validation and test
                      performance.
                    </p>
                  </div>

                  <div className="detail-model-icon">
                    {selected.model
                      .toLowerCase()
                      .includes("graph")
                      ? (
                        <BrainCircuit size={25} />
                      ) : (
                        <BarChart3 size={25} />
                      )}
                  </div>
                </div>

                <div className="detail-grid">
                  <div className="metric-card">
                    <span>VALIDATION PR-AUC</span>

                    <strong>
                      {formatScore(
                        selected.validationPrAuc
                      )}
                    </strong>
                  </div>

                  <div className="metric-card primary">
                    <span>TEST PR-AUC</span>

                    <strong>
                      {formatScore(
                        selected.testPrAuc
                      )}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>TEST ROC-AUC</span>

                    <strong>
                      {formatScore(
                        selected.testRocAuc
                      )}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>PRECISION</span>

                    <strong>
                      {formatPercent(
                        selected.precision
                      )}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>RECALL</span>

                    <strong>
                      {formatPercent(
                        selected.recall
                      )}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>F1 SCORE</span>

                    <strong>
                      {formatPercent(
                        selected.f1
                      )}
                    </strong>
                  </div>

                  <div className="metric-card">
                    <span>SELECTED THRESHOLD</span>

                    <strong>
                      {formatScore(
                        selected.threshold
                      )}
                    </strong>
                  </div>
                </div>

                <div className="metric-bars">
                  <div className="bar-label">
                    <span>PR-AUC</span>
                    <strong>
                      {formatScore(
                        selected.testPrAuc
                      )}
                    </strong>
                  </div>

                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${
                          (selected.testPrAuc ?? 0) * 100
                        }%`,
                      }}
                    />
                  </div>

                  <div className="bar-label">
                    <span>ROC-AUC</span>
                    <strong>
                      {formatScore(
                        selected.testRocAuc
                      )}
                    </strong>
                  </div>

                  <div className="bar-track">
                    <div
                      className="bar-fill roc"
                      style={{
                        width: `${
                          (selected.testRocAuc ?? 0) * 100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              </section>
            )}

            <section className="research-note">
              <div className="research-note-icon">
                <BrainCircuit size={18} />
              </div>

              <div>
                <strong>
                  Why PR-AUC is emphasized
                </strong>

                <p>
                  Illicit transactions are a minority of
                  labeled transactions in the Elliptic
                  dataset. Precision-recall performance
                  therefore provides a useful view of
                  minority-class detection alongside ROC-AUC.
                </p>
              </div>
            </section>
          </>
        )}
      </section>
    </main>
  );
}