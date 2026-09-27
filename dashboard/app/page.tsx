"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  ChevronRight,
  CircleDot,
  Database,
  GitBranch,
  Network,
  Shield,
  Target,
  TrendingUp,
  Zap,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const modelData = [
  {
    name: "XGBoost",
    short: "XGB",
    type: "Traditional ML",
    pr: 65.85,
    roc: 89.16,
    precision: 92.33,
    recall: 57.44,
    f1: 70.82,
  },
  {
    name: "Random Forest",
    short: "RF",
    type: "Traditional ML",
    pr: 65.06,
    roc: 89.11,
    precision: 93.75,
    recall: 57.25,
    f1: 71.09,
  },
  {
    name: "GraphSAGE",
    short: "SAGE",
    type: "Graph Neural Network",
    pr: 29.71,
    roc: 84.0,
    precision: 33.53,
    recall: 42.94,
    f1: 37.66,
  },
  {
    name: "GAT",
    short: "GAT",
    type: "Graph Neural Network",
    pr: 22.65,
    roc: 81.38,
    precision: 24.69,
    recall: 30.34,
    f1: 27.23,
  },
  {
    name: "Logistic Regression",
    short: "LR",
    type: "Traditional ML",
    pr: 20.26,
    roc: 85.03,
    precision: 24.50,
    recall: 48.85,
    f1: 32.63,
  },
  {
    name: "GCN",
    short: "GCN",
    type: "Graph Neural Network",
    pr: 18.11,
    roc: 74.48,
    precision: 19.29,
    recall: 23.85,
    f1: 21.33,
  },
];

const prChartData = [...modelData]
  .reverse()
  .map((model) => ({
    model: model.short,
    value: model.pr,
  }));

const temporalData = [
  { time: "41", illicit: 116, samples: 1132 },
  { time: "42", illicit: 239, samples: 2154 },
  { time: "43", illicit: 24, samples: 1370 },
  { time: "44", illicit: 24, samples: 1591 },
  { time: "45", illicit: 5, samples: 1221 },
  { time: "46", illicit: 2, samples: 712 },
  { time: "47", illicit: 22, samples: 846 },
  { time: "48", illicit: 36, samples: 471 },
  { time: "49", illicit: 56, samples: 476 },
];

const featureData = [
  { feature: "feature_5", value: 7.44 },
  { feature: "feature_53", value: 6.64 },
  { feature: "feature_90", value: 6.61 },
  { feature: "feature_55", value: 4.01 },
  { feature: "feature_46", value: 3.42 },
  { feature: "feature_163", value: 2.82 },
  { feature: "feature_47", value: 2.72 },
  { feature: "feature_40", value: 2.67 },
];

const robustnessData = [
  { model: "RF", value: 65.15, error: 0.14 },
  { model: "SAGE", value: 40.47, error: 4.77 },
  { model: "GCN", value: 30.17, error: 3.61 },
  { model: "GAT", value: 24.32, error: 5.06 },
];

function MetricCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <div>
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        <div className="metric-detail">{detail}</div>
      </div>
    </div>
  );
}

function SectionHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-header">
      <div>
        <div className="section-eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      <ChevronRight size={18} />
    </div>
  );
}

export default function Home() {
  return (
    <main className="dashboard-shell">
      
      
      {/* MAIN CONTENT */}
      <section className="main-content">
        {/* TOP BAR */}
        <header className="topbar">
          <div>
            <div className="breadcrumb">RESEARCH / OVERVIEW</div>
            <h1>Fraud Detection Intelligence</h1>
          </div>

          <div className="topbar-right">
            <div className="live-indicator">
              <span className="status-dot" />
              EXPERIMENTS READY
            </div>

            <div className="research-badge">
              <GitBranch size={14} />
              Elliptic Bitcoin
            </div>
          </div>
        </header>

        {/* HERO */}
        <div className="hero">
          <div className="hero-copy">
            <div className="hero-tag">
              <CircleDot size={12} />
              GRAPH MACHINE LEARNING RESEARCH
            </div>

            <h2>
              Detecting illicit transactions
              <br />
              through <span>transaction relationships.</span>
            </h2>

            <p>
              GraphShield evaluates traditional machine learning against
              graph neural networks for illicit Bitcoin transaction detection,
              using temporally separated evaluation and graph-aware analysis.
            </p>

            <div className="hero-meta">
              <span>
                <Database size={14} />
                203,769 nodes
              </span>
              <span>
                <Network size={14} />
                234,355 edges
              </span>
              <span>
                <Activity size={14} />
                49 time steps
              </span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="network-orbit orbit-one" />
            <div className="network-orbit orbit-two" />
            <div className="network-core">
              <Shield size={34} />
            </div>

            <span className="node node-a" />
            <span className="node node-b" />
            <span className="node node-c" />
            <span className="node node-d" />
            <span className="node node-e" />
            <span className="node node-f" />
          </div>
        </div>

        {/* KPI CARDS */}
        <div className="metrics-grid">
          <MetricCard
            icon={<Database size={19} />}
            label="TRANSACTIONS"
            value="203,769"
            detail="Graph nodes"
          />

          <MetricCard
            icon={<Network size={19} />}
            label="RELATIONSHIPS"
            value="234,355"
            detail="Directed transaction edges"
          />

          <MetricCard
            icon={<Target size={19} />}
            label="LABELED"
            value="46,564"
            detail="42,019 licit · 4,545 illicit"
          />

          <MetricCard
            icon={<AlertTriangle size={19} />}
            label="ILLICIT RATE"
            value="9.76%"
            detail="Among labeled transactions"
          />
        </div>

        {/* MODEL PERFORMANCE */}
        <div className="section-block">
          <SectionHeader
            eyebrow="MODEL BENCHMARK"
            title="Model Performance"
            description="Test-set performance using validation-selected thresholds."
          />

          <div className="model-grid">
            {modelData.map((model, index) => (
              <div
                className={`model-card ${
                  index === 0 ? "model-card-highlight" : ""
                }`}
                key={model.name}
              >
                <div className="model-top">
                  <div>
                    <span className="model-type">{model.type}</span>
                    <h3>{model.name}</h3>
                  </div>

                  {index === 0 && (
                    <span className="research-chip">TOP PR-AUC</span>
                  )}
                </div>

                <div className="model-pr">
                  <span>{model.pr.toFixed(2)}%</span>
                  <small>PR-AUC</small>
                </div>

                <div className="model-metrics">
                  <div>
                    <span>ROC-AUC</span>
                    <strong>{model.roc.toFixed(2)}%</strong>
                  </div>
                  <div>
                    <span>Precision</span>
                    <strong>{model.precision.toFixed(2)}%</strong>
                  </div>
                  <div>
                    <span>Recall</span>
                    <strong>{model.recall.toFixed(2)}%</strong>
                  </div>
                  <div>
                    <span>F1</span>
                    <strong>{model.f1.toFixed(2)}%</strong>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* CHART ROW */}
        <div className="chart-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <span>PRIMARY METRIC</span>
                <h3>Test PR-AUC Comparison</h3>
              </div>
              <span className="panel-value">Higher is better</span>
            </div>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={prChartData}
                  layout="vertical"
                  margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    stroke="rgba(148,163,184,0.12)"
                  />
                  <XAxis
                    type="number"
                    domain={[0, 70]}
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="model"
                    type="category"
                    width={55}
                    tick={{ fill: "#94a3b8", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "10px",
                      color: "#fff",
                    }}
                    formatter={(value) => [`${value}%`, "PR-AUC"]}
                  />
                  <Bar dataKey="value" radius={[0, 5, 5, 0]}>
                    {prChartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.model === "XGB" ? "#38bdf8" : "#334155"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <span>TEST DISTRIBUTION</span>
                <h3>Illicit Transactions by Time</h3>
              </div>
              <span className="panel-value">Steps 41–49</span>
            </div>

            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={temporalData}
                  margin={{ top: 10, right: 10, left: -10, bottom: 10 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(148,163,184,0.12)"
                  />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "10px",
                      color: "#fff",
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="illicit"
                    stroke="#f87171"
                    strokeWidth={2.5}
                    dot={{ r: 3, fill: "#f87171" }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* FEATURE + ROBUSTNESS */}
        <div className="chart-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <span>FEATURE INTELLIGENCE</span>
                <h3>Top Combined Features</h3>
              </div>
              <span className="panel-value">RF + XGBoost</span>
            </div>

            <div className="feature-list">
              {featureData.map((feature, index) => (
                <div className="feature-row" key={feature.feature}>
                  <div className="feature-rank">{index + 1}</div>
                  <div className="feature-name">{feature.feature}</div>
                  <div className="feature-bar">
                    <span style={{ width: `${feature.value * 10}%` }} />
                  </div>
                  <div className="feature-score">
                    {feature.value.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <span>ROBUSTNESS</span>
                <h3>5-Seed PR-AUC</h3>
              </div>
              <span className="panel-value">Mean ± Std</span>
            </div>

            <div className="robustness-list">
              {robustnessData.map((item) => (
                <div className="robustness-row" key={item.model}>
                  <div className="robustness-label">
                    <strong>{item.model}</strong>
                    <span>{item.value.toFixed(2)}%</span>
                  </div>

                  <div className="robustness-track">
                    <span style={{ width: `${item.value}%` }} />
                  </div>

                  <div className="robustness-error">
                    ± {item.error.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>

            <div className="research-note">
              <Zap size={15} />
              Five independent random seeds were evaluated for robustness.
            </div>
          </div>
        </div>

        {/* RESEARCH FINDINGS */}
        <div className="section-block findings-block">
          <SectionHeader
            eyebrow="RESEARCH SUMMARY"
            title="Key Findings"
            description="Evidence generated from the GraphShield experimental pipeline."
          />

          <div className="findings-grid">
            <div className="finding">
              <div className="finding-icon blue">
                <BarChart3 size={18} />
              </div>
              <div>
                <h3>Strong traditional baselines</h3>
                <p>
                  Random Forest and XGBoost achieved test PR-AUC values of
                  65.06% and 65.85%, respectively.
                </p>
              </div>
            </div>

            <div className="finding">
              <div className="finding-icon purple">
                <Network size={18} />
              </div>
              <div>
                <h3>Graph structure contains signal</h3>
                <p>
                  Neighbor-label analysis shows a measurable relationship
                  between transaction connectivity and illicit activity.
                </p>
              </div>
            </div>

            <div className="finding">
              <div className="finding-icon amber">
                <BrainCircuit size={18} />
              </div>
              <div>
                <h3>GNN performance varies by architecture</h3>
                <p>
                  GraphSAGE achieved higher test PR-AUC than GCN and GAT in
                  the primary six-model comparison.
                </p>
              </div>
            </div>

            <div className="finding">
              <div className="finding-icon red">
                <AlertTriangle size={18} />
              </div>
              <div>
                <h3>Temporal distribution shift matters</h3>
                <p>
                  Illicit prevalence changes substantially across the held-out
                  test time steps, motivating temporal evaluation.
                </p>
              </div>
            </div>
          </div>
        </div>

        <footer className="dashboard-footer">
          <div>
            <Shield size={15} />
            GraphShield Research Platform
          </div>
          <span>
            Graph Neural Networks for Illicit Bitcoin Transaction Detection
          </span>
        </footer>
      </section>
    </main>
  );
}