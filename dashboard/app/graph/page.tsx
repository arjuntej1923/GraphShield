"use client";

import {
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import Link from "next/link";
import {
  ArrowLeft,
  ArrowRight,
  CircleAlert,
  CircleCheck,
  CircleHelp,
  Database,
  GitBranch,
  Network,
  Search,
  Shield,
  Target,
  Activity,
} from "lucide-react";

type Neighbor = {
  id: string;
  label: "illicit" | "licit" | "unknown";
  incoming: number;
  outgoing: number;
  degree: number;
};

type GraphResponse = {
  transaction: {
    id: string;
    label: "illicit" | "licit" | "unknown";
    incoming: number;
    outgoing: number;
    degree: number;
  };

  neighborhood: {
    total: number;
    illicit: number;
    licit: number;
    unknown: number;
    illicitRatio: number;
  };

  neighbors: Neighbor[];

  graph: {
    totalNodes: number;
    queriedNeighbors: number;
  };
};

function labelColor(label: string) {
  if (label === "illicit") return "red";
  if (label === "licit") return "green";
  return "gray";
}

function LabelIcon({ label }: { label: string }) {
  if (label === "illicit") {
    return <CircleAlert size={15} />;
  }

  if (label === "licit") {
    return <CircleCheck size={15} />;
  }

  return <CircleHelp size={15} />;
}

export default function GraphExplorer() {
  const [transactionId, setTransactionId] = useState("");
  const [data, setData] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Keeps track of the latest request.
  // If an older request finishes after a newer one,
  // its result will be ignored.
  const requestIdRef = useRef(0);

  async function loadTransaction(id?: string) {
    const requestId = ++requestIdRef.current;

    setLoading(true);
    setError("");

    try {
      const query = id
        ? `?txId=${encodeURIComponent(id)}`
        : "";

      const response = await fetch(`/api/graph${query}`);

      const result = await response.json();

      if (!response.ok) {
        throw new Error(
          result.error || "Unable to load transaction."
        );
      }

      // Only allow the latest request to update the UI.
      if (requestId === requestIdRef.current) {
        setData(result);
      }
    } catch (err) {
      // Ignore errors from stale requests.
      if (requestId === requestIdRef.current) {
        setError(
          err instanceof Error
            ? err.message
            : "Unable to load transaction."
        );

        setData(null);
      }
    } finally {
      // Only the latest request controls the loading state.
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    loadTransaction();
  }, []);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedId = transactionId.trim();

    if (!trimmedId) {
      loadTransaction();
      return;
    }

    loadTransaction(trimmedId);
  }

  return (
    <main className="graph-page">
      <header className="graph-topbar">
        <Link href="/" className="back-link">
          <ArrowLeft size={15} />
          Command Center
        </Link>

        <div className="graph-title">
          <div className="graph-title-icon">
            <Network size={18} />
          </div>

          <div>
            <div className="graph-eyebrow">
              GRAPH MACHINE LEARNING
            </div>

            <h1>Graph Explorer</h1>
          </div>
        </div>

        <div className="dataset-badge">
          <Database size={13} />
          Elliptic Bitcoin
        </div>
      </header>

      <section className="graph-content">
        <div className="graph-intro">
          <div>
            <div className="graph-section-label">
              TRANSACTION NETWORK
            </div>

            <h2>
              Investigate transaction relationships.
            </h2>

            <p>
              Explore the neighborhood structure of real
              transactions from the Elliptic Bitcoin graph.
            </p>
          </div>
        </div>

        <form
          className="transaction-search"
          onSubmit={handleSubmit}
        >
          <div className="search-icon">
            <Search size={18} />
          </div>

          <input
            value={transactionId}
            onChange={(event) =>
              setTransactionId(event.target.value)
            }
            placeholder="Enter a transaction ID..."
          />

          <button
            type="submit"
            disabled={loading}
          >
            {loading ? "Loading..." : "Investigate"}
            <ArrowRight size={15} />
          </button>
        </form>

        {error && (
          <div className="graph-error">
            <CircleAlert size={17} />
            <span>{error}</span>
          </div>
        )}

        {data && (
          <>
            <section className="transaction-header">
              <div>
                <div className="transaction-label">
                  TRANSACTION
                </div>

                <div className="transaction-id">
                  {data.transaction.id}
                </div>

                <div
                  className={`classification ${labelColor(
                    data.transaction.label
                  )}`}
                >
                  <LabelIcon
                    label={data.transaction.label}
                  />

                  {data.transaction.label.toUpperCase()}
                </div>
              </div>

              <div className="transaction-status">
                <div className="status-caption">
                  GRAPH CONTEXT
                </div>

                <strong>
                  {data.neighborhood.total}
                </strong>

                <span>visible neighbors</span>
              </div>
            </section>

            <section className="stat-grid">
              <div className="graph-stat">
                <div className="stat-icon blue">
                  <GitBranch size={17} />
                </div>

                <span>INCOMING</span>

                <strong>
                  {data.transaction.incoming}
                </strong>
              </div>

              <div className="graph-stat">
                <div className="stat-icon purple">
                  <GitBranch size={17} />
                </div>

                <span>OUTGOING</span>

                <strong>
                  {data.transaction.outgoing}
                </strong>
              </div>

              <div className="graph-stat">
                <div className="stat-icon cyan">
                  <Network size={17} />
                </div>

                <span>TOTAL DEGREE</span>

                <strong>
                  {data.transaction.degree}
                </strong>
              </div>

              <div className="graph-stat">
                <div className="stat-icon red">
                  <Target size={17} />
                </div>

                <span>ILLICIT RATIO</span>

                <strong>
                  {(
                    data.neighborhood.illicitRatio * 100
                  ).toFixed(1)}
                  %
                </strong>
              </div>
            </section>

            <section className="explorer-grid">
              <div className="network-panel">
                <div className="panel-header">
                  <div>
                    <span>
                      RELATIONAL STRUCTURE
                    </span>

                    <h3>
                      Transaction Neighborhood
                    </h3>
                  </div>
                </div>

                <div className="network-canvas">
                  <div className="central-node">
                    <Shield size={25} />
                    <span>TX</span>
                  </div>

                  {data.neighbors
                    .slice(0, 12)
                    .map((neighbor, index) => {
                      const angle =
                        (index /
                          Math.max(
                            1,
                            Math.min(
                              data.neighbors.length,
                              12
                            )
                          )) *
                          Math.PI *
                          2 -
                        Math.PI / 2;

                      const x =
                        50 + Math.cos(angle) * 34;

                      const y =
                        50 + Math.sin(angle) * 34;

                      return (
                        <div
                          key={neighbor.id}
                          className={`neighbor-node ${labelColor(
                            neighbor.label
                          )}`}
                          style={{
                            left: `${x}%`,
                            top: `${y}%`,
                          }}
                          title={neighbor.id}
                        >
                          <span />
                        </div>
                      );
                    })}

                  <div className="network-caption">
                    <Activity size={13} />

                    {data.neighbors.length} neighboring
                    transactions loaded
                  </div>
                </div>
              </div>

              <div className="neighbors-panel">
                <div className="panel-header">
                  <div>
                    <span>
                      CONNECTED TRANSACTIONS
                    </span>

                    <h3>Neighbor Analysis</h3>
                  </div>

                  <span className="count-badge">
                    {data.neighbors.length}
                  </span>
                </div>

                <div className="neighbor-summary">
                  <div>
                    <span>ILLICIT</span>

                    <strong className="red-text">
                      {data.neighborhood.illicit}
                    </strong>
                  </div>

                  <div>
                    <span>LICIT</span>

                    <strong className="green-text">
                      {data.neighborhood.licit}
                    </strong>
                  </div>

                  <div>
                    <span>UNKNOWN</span>

                    <strong>
                      {data.neighborhood.unknown}
                    </strong>
                  </div>
                </div>

                <div className="neighbor-table">
                  <div className="neighbor-table-head">
                    <span>TRANSACTION</span>
                    <span>LABEL</span>
                    <span>DEGREE</span>
                  </div>

                  {data.neighbors.map((neighbor) => (
                    <button
                      key={neighbor.id}
                      type="button"
                      className="neighbor-row"
                      onClick={() =>
                        loadTransaction(
                          neighbor.id
                        )
                      }
                    >
                      <span className="neighbor-id">
                        {neighbor.id}
                      </span>

                      <span
                        className={`neighbor-label ${labelColor(
                          neighbor.label
                        )}`}
                      >
                        <LabelIcon
                          label={neighbor.label}
                        />

                        {neighbor.label}
                      </span>

                      <span className="neighbor-degree">
                        {neighbor.degree}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="research-context">
              <div className="context-icon">
                <Activity size={18} />
              </div>

              <div>
                <strong>Research context</strong>

                <p>
                  GraphShield evaluates whether transaction
                  relationships provide useful information
                  beyond transaction-level features.
                </p>
              </div>
            </section>
          </>
        )}
      </section>
    </main>
  );
}