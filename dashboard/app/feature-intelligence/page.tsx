"use client";

import { useEffect, useMemo, useState } from "react";

type Feature = {
  feature: string;
  importance: number;
  rank: number | null;
};

type CombinedFeature = {
  feature: string;
  meanImportance: number;
  rfImportance: number | null;
  xgbImportance: number | null;
  rank: number | null;
};

type Ablation = {
  model: string;
  featureCount: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  rocAuc: number | null;
  prAuc: number | null;
};

type FeatureData = {
  randomForest: Feature[];
  xgboost: Feature[];
  combined: CombinedFeature[];
  overlap: {
    k: number | null;
    overlap: number | null;
  }[];
  top30: Feature[];
  ablation: Ablation[];
  metadata: {
    rfXgbSpearman: number;
    top5Overlap: string;
    top10Overlap: string;
    top20Overlap: string;
    top30Overlap: string;
    top50Overlap: string;
  };
};

function pct(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(2)}%`;
}

function num(value: number | null | undefined) {
  if (value === null || value === undefined) return "—";
  return value.toFixed(4);
}

function importanceWidth(value: number, max: number) {
  if (!max) return 0;
  return Math.max(3, (value / max) * 100);
}

export default function FeatureIntelligencePage() {
  const [data, setData] = useState<FeatureData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/feature-intelligence")
      .then(async (response) => {
        const json = await response.json();

        if (!response.ok) {
          throw new Error(json.error || "Failed to load data");
        }

        return json;
      })
      .then(setData)
      .catch((err) => {
        setError(err.message);
      });
  }, []);

  const topRF = useMemo(
    () => data?.randomForest.slice(0, 10) ?? [],
    [data]
  );

  const topXGB = useMemo(
    () => data?.xgboost.slice(0, 10) ?? [],
    [data]
  );

  const topCombined = useMemo(
    () => data?.combined.slice(0, 15) ?? [],
    [data]
  );

  const rfMax = Math.max(
    ...(topRF.map((x) => x.importance)),
    0
  );

  const xgbMax = Math.max(
    ...(topXGB.map((x) => x.importance)),
    0
  );

  const combinedMax = Math.max(
    ...(topCombined.map((x) => x.meanImportance)),
    0
  );

  if (error) {
    return (
      <main className="fi-page">
        <div className="fi-error">
          <div className="fi-error-title">
            Feature Intelligence unavailable
          </div>
          <div className="fi-error-message">{error}</div>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="fi-page">
        <div className="fi-loading">
          Loading feature intelligence...
        </div>
      </main>
    );
  }

  return (
    <main className="fi-page">
      <div className="fi-header">
        <div>
          <div className="fi-eyebrow">
            GRAPHSHIELD · FEATURE INTELLIGENCE
          </div>

          <h1>Which transaction features drive detection?</h1>

          <p>
            Feature importance and ablation analysis from the
            GraphShield fraud detection experiments.
          </p>
        </div>

        <div className="fi-header-badge">
          RESEARCH ANALYSIS
        </div>
      </div>

      {/* SUMMARY */}
      <section className="fi-summary-grid">
        <div className="fi-card">
          <span>FEATURES EVALUATED</span>
          <strong>165</strong>
          <small>Transaction features</small>
        </div>

        <div className="fi-card">
          <span>RF / XGB CORRELATION</span>
          <strong>
            {data.metadata.rfXgbSpearman.toFixed(3)}
          </strong>
          <small>Spearman rank correlation</small>
        </div>

        <div className="fi-card">
          <span>TOP-10 OVERLAP</span>
          <strong>{data.metadata.top10Overlap}</strong>
          <small>Shared important features</small>
        </div>

        <div className="fi-card">
          <span>TOP-30 OVERLAP</span>
          <strong>{data.metadata.top30Overlap}</strong>
          <small>Shared important features</small>
        </div>
      </section>

      {/* RF + XGB */}
      <section className="fi-section">
        <div className="fi-section-heading">
          <div>
            <div className="fi-eyebrow">MODEL INTERPRETATION</div>
            <h2>Feature importance</h2>
            <p>
              The strongest transaction features identified by
              Random Forest and XGBoost.
            </p>
          </div>
        </div>

        <div className="fi-two-column">
          <div className="fi-panel">
            <div className="fi-panel-title">
              <div>
                <strong>Random Forest</strong>
                <span>Top 10 features</span>
              </div>
            </div>

            <div className="fi-feature-list">
              {topRF.map((item, index) => (
                <div className="fi-feature-row" key={item.feature}>
                  <div className="fi-feature-rank">
                    {index + 1}
                  </div>

                  <div className="fi-feature-main">
                    <div className="fi-feature-label">
                      <span>{item.feature}</span>
                      <strong>
                        {item.importance.toFixed(4)}
                      </strong>
                    </div>

                    <div className="fi-bar-track">
                      <div
                        className="fi-bar rf"
                        style={{
                          width: `${importanceWidth(
                            item.importance,
                            rfMax
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="fi-panel">
            <div className="fi-panel-title">
              <div>
                <strong>XGBoost</strong>
                <span>Top 10 features</span>
              </div>
            </div>

            <div className="fi-feature-list">
              {topXGB.map((item, index) => (
                <div className="fi-feature-row" key={item.feature}>
                  <div className="fi-feature-rank">
                    {index + 1}
                  </div>

                  <div className="fi-feature-main">
                    <div className="fi-feature-label">
                      <span>{item.feature}</span>
                      <strong>
                        {item.importance.toFixed(4)}
                      </strong>
                    </div>

                    <div className="fi-bar-track">
                      <div
                        className="fi-bar xgb"
                        style={{
                          width: `${importanceWidth(
                            item.importance,
                            xgbMax
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* COMBINED */}
      <section className="fi-section">
        <div className="fi-section-heading">
          <div>
            <div className="fi-eyebrow">CROSS-MODEL ANALYSIS</div>
            <h2>Combined feature ranking</h2>
            <p>
              Mean importance across Random Forest and XGBoost.
            </p>
          </div>
        </div>

        <div className="fi-panel">
          <div className="fi-feature-list">
            {topCombined.map((item, index) => (
              <div className="fi-feature-row" key={item.feature}>
                <div className="fi-feature-rank">
                  {index + 1}
                </div>

                <div className="fi-feature-main">
                  <div className="fi-feature-label">
                    <span>{item.feature}</span>
                    <strong>
                      {item.meanImportance.toFixed(4)}
                    </strong>
                  </div>

                  <div className="fi-bar-track">
                    <div
                      className="fi-bar combined"
                      style={{
                        width: `${importanceWidth(
                          item.meanImportance,
                          combinedMax
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* OVERLAP */}
      <section className="fi-section">
        <div className="fi-section-heading">
          <div>
            <div className="fi-eyebrow">MODEL AGREEMENT</div>
            <h2>Feature ranking overlap</h2>
            <p>
              Shared features between the Random Forest and
              XGBoost rankings.
            </p>
          </div>
        </div>

        <div className="fi-overlap-grid">
          {[
            ["Top 5", data.metadata.top5Overlap],
            ["Top 10", data.metadata.top10Overlap],
            ["Top 20", data.metadata.top20Overlap],
            ["Top 30", data.metadata.top30Overlap],
            ["Top 50", data.metadata.top50Overlap],
          ].map(([label, value]) => (
            <div className="fi-overlap-card" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
              <small>shared features</small>
            </div>
          ))}
        </div>
      </section>

      {/* ABLATION */}
      <section className="fi-section">
        <div className="fi-section-heading">
          <div>
            <div className="fi-eyebrow">FEATURE ABLATION</div>
            <h2>How many features are necessary?</h2>
            <p>
              Test-set PR-AUC as the number of selected transaction
              features changes.
            </p>
          </div>
        </div>

        <div className="fi-ablation-grid">
          {["Random Forest", "XGBoost"].map((model) => {
            const rows = data.ablation
              .filter((x) => x.model === model)
              .sort(
                (a, b) =>
                  (a.featureCount ?? 0) -
                  (b.featureCount ?? 0)
              );

            const maxPR = Math.max(
              ...(rows.map((x) => x.prAuc ?? 0)),
              0
            );

            return (
              <div className="fi-panel" key={model}>
                <div className="fi-panel-title">
                  <div>
                    <strong>{model}</strong>
                    <span>Test PR-AUC</span>
                  </div>
                </div>

                <div className="fi-ablation-list">
                  {rows.map((row) => (
                    <div
                      className="fi-ablation-row"
                      key={`${model}-${row.featureCount}`}
                    >
                      <span>
                        {row.featureCount} features
                      </span>

                      <div className="fi-bar-track">
                        <div
                          className="fi-bar ablation"
                          style={{
                            width: `${importanceWidth(
                              row.prAuc ?? 0,
                              maxPR
                            )}%`,
                          }}
                        />
                      </div>

                      <strong>{pct(row.prAuc)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* RESEARCH NOTE */}
      <section className="fi-note">
        <div className="fi-eyebrow">RESEARCH NOTE</div>

        <h2>
          Full feature representation remains important.
        </h2>

        <p>
          The feature ablation experiments evaluate whether
          restricting the transaction representation changes
          predictive performance. The results show how PR-AUC
          changes across 10, 20, 30, 50 and 165 selected
          features. Feature importance rankings also show that
          the two tree-based models share a substantial subset
          of important features, while their exact rankings are
          not identical.
        </p>
      </section>
    </main>
  );
}