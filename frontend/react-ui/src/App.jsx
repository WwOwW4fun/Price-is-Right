import { startTransition, useEffect, useState } from "react";

const emptyDashboard = {
  running: false,
  error: null,
  mode: "Autonomous scan",
  lastRunAt: null,
  summaryCards: [
    { label: "System Status", value: "Offline", note: "API not connected" },
    { label: "Deals Found", value: "0", note: "Saved opportunities in memory" },
    { label: "Best Discount", value: "0.0%", note: "No deal yet" },
    { label: "Last Run", value: "Never", note: "Background pipeline" },
  ],
  dealColumns: [
    { title: "Recent Opportunities", items: [] },
    { title: "Best Discounts", items: [] },
    { title: "Alerted Deals", items: [] },
  ],
  agentTimeline: [],
  alerts: [],
};

function StatusPill({ running }) {
  return (
    <span className={`status-pill ${running ? "running" : "stopped"}`}>
      {running ? "Running" : "Idle"}
    </span>
  );
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value ?? 0);
}

function formatPercent(value) {
  return `${Number(value ?? 0).toFixed(1)}%`;
}

function App() {
  const [dashboard, setDashboard] = useState(emptyDashboard);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunningRequest, setIsRunningRequest] = useState(false);
  const [connectionError, setConnectionError] = useState("");

  async function fetchDashboard() {
    try {
      const response = await fetch("/api/status");
      if (!response.ok) {
        throw new Error(`Status request failed with ${response.status}`);
      }

      const data = await response.json();
      startTransition(() => {
        setDashboard(data);
      });
      setConnectionError("");
    } catch (error) {
      setConnectionError(error.message || "Unable to reach the backend API.");
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    let active = true;

    async function load() {
      if (!active) {
        return;
      }
      await fetchDashboard();
    }

    load();
    const intervalId = window.setInterval(load, 5000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  async function handleRunAgents() {
    setIsRunningRequest(true);
    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(`Run request failed with ${response.status}`);
      }

      const payload = await response.json();
      startTransition(() => {
        setDashboard(payload.state);
      });
      setConnectionError("");
    } catch (error) {
      setConnectionError(error.message || "Unable to start the agents.");
    } finally {
      setIsRunningRequest(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Multi-Agent Deal Operations</p>
          <h1>Deal Agent Command Center</h1>
          <p className="hero-copy">
            Monitor your planner, scanner, pricing ensemble, and messaging agent from one live dashboard backed by the Python runtime.
          </p>
          {connectionError ? (
            <div className="connection-banner error">
              <strong>Backend unavailable.</strong>
              <span>{connectionError}</span>
            </div>
          ) : (
            <div className="connection-banner">
              <strong>Backend connected.</strong>
              <span>Polling live system state every 5 seconds.</span>
            </div>
          )}
        </div>
        <div className="control-panel">
          <div className="control-head">
            <span>Agent Runtime</span>
            <StatusPill running={dashboard.running} />
          </div>
          <div className="control-actions">
            <button
              className="primary-action"
              disabled={dashboard.running || isRunningRequest || Boolean(connectionError)}
              onClick={handleRunAgents}
            >
              {dashboard.running ? "Agents Running" : isRunningRequest ? "Starting..." : "Run Agents"}
            </button>
            <button className="secondary-action" onClick={fetchDashboard}>
              Refresh Data
            </button>
          </div>
          <div className="control-meta">
            <div>
              <span className="meta-label">Last run</span>
              <strong>{dashboard.lastRunAt || "Never"}</strong>
            </div>
            <div>
              <span className="meta-label">Current mode</span>
              <strong>{dashboard.mode}</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="summary-grid">
        {dashboard.summaryCards.map((card) => (
          <article className="summary-card" key={card.label}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.note}</p>
          </article>
        ))}
      </section>

      <section className="workspace">
        <div className="main-column">
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="panel-kicker">Deals Board</p>
                <h2>Track every opportunity</h2>
              </div>
              <div className="filters">
                <button className="filter active">
                  {isLoading ? "Loading..." : "Live memory"}
                </button>
                <button className="filter">
                  {dashboard.running ? "Run in progress" : "Ready"}
                </button>
              </div>
            </div>
            <div className="board">
              {dashboard.dealColumns.map((column) => (
                <div className="board-column" key={column.title}>
                  <div className="column-head">
                    <h3>{column.title}</h3>
                    <span>{column.items.length}</span>
                  </div>
                  {column.items.length ? (
                    column.items.map((deal) => (
                      <article className="deal-card" key={`${column.title}-${deal.url}`}>
                        <div className="deal-topline">
                          <span className="source-tag">{deal.source}</span>
                          <strong>{formatPercent(deal.discountPercent)}</strong>
                        </div>
                        <h4>{deal.product}</h4>
                        <dl className="deal-metrics">
                          <div>
                            <dt>Price</dt>
                            <dd>{formatCurrency(deal.price)}</dd>
                          </div>
                          <div>
                            <dt>Estimate</dt>
                            <dd>{formatCurrency(deal.estimate)}</dd>
                          </div>
                        </dl>
                      </article>
                    ))
                  ) : (
                    <div className="empty-state">
                      <p>No saved opportunities yet.</p>
                      <span>Run the agent system to populate the board.</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        </div>

        <aside className="side-column">
          <section className="panel">
            <div className="panel-header stacked">
              <div>
                <p className="panel-kicker">Live Activity</p>
                <h2>Agent timeline</h2>
              </div>
            </div>
            <div className="timeline">
              {dashboard.agentTimeline.length ? (
                dashboard.agentTimeline.map((event, index) => (
                  <article className={`timeline-item ${event.level}`} key={`${event.time}-${event.agent}-${index}`}>
                    <span className="timeline-time">{event.time}</span>
                    <div>
                      <h3>{event.agent}</h3>
                      <p>{event.message}</p>
                    </div>
                  </article>
                ))
              ) : (
                <div className="empty-state">
                  <p>No agent logs yet.</p>
                  <span>Timeline events will appear here during a run.</span>
                </div>
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-header stacked">
              <div>
                <p className="panel-kicker">Notifications</p>
                <h2>Recent alerts</h2>
              </div>
            </div>
            <div className="alerts">
              {dashboard.alerts.length ? (
                dashboard.alerts.map((alert) => (
                  <article className="alert-card" key={`${alert.title}-${alert.body}`}>
                    <h3>{alert.title}</h3>
                    <p>{alert.body}</p>
                  </article>
                ))
              ) : (
                <div className="empty-state">
                  <p>No alerts have been generated yet.</p>
                  <span>Qualified deals will show up here after the next run.</span>
                </div>
              )}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

export default App;
