import { useState } from "react";
import {
  Shield,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Send,
  Clock,
  Zap,
  ShieldAlert,
} from "lucide-react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const analyzePrompt = async () => {
    if (!prompt.trim()) {
      setError("Please enter a prompt to analyze.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
        }),
      });

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();

      setResult(data);

      const historyItem = {
        id: Date.now(),
        prompt: data.prompt,
        risk_score: data.risk_score,
        action: data.action,
      };

      setHistory((previous) => [historyItem, ...previous].slice(0, 6));
    } catch (err) {
      setError(
        "Unable to connect to SentinelAI API. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const getActionClass = (action) => {
    if (action === "BLOCK") return "block";
    if (action === "SUSPICIOUS") return "suspicious";
    return "allow";
  };

  const getActionIcon = (action) => {
    if (action === "BLOCK") return <XCircle size={22} />;
    if (action === "SUSPICIOUS") return <AlertTriangle size={22} />;
    return <CheckCircle size={22} />;
  };

  const getRiskLevel = (score) => {
    if (score >= 80) return "CRITICAL";
    if (score >= 50) return "MEDIUM";
    return "LOW";
  };

  return (
    <div className="app">

      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={25} />
          </div>

          <div>
            <h1>SentinelAI</h1>
            <span>AI Security Engine</span>
          </div>
        </div>

        <nav className="navigation">
          <div className="nav-item active">
            <Activity size={19} />
            Dashboard
          </div>

          <div className="nav-item">
            <ShieldAlert size={19} />
            Threat Analysis
          </div>

          <div className="nav-item">
            <Clock size={19} />
            Activity
          </div>
        </nav>

        <div className="sidebar-bottom">
          <div className="engine-status">
            <span className="status-dot"></span>

            <div>
              <strong>Detection Engine</strong>
              <small>Online</small>
            </div>
          </div>

          <div className="version">
            SentinelAI v1.0
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main className="main">

        {/* HEADER */}
        <header className="topbar">
          <div>
            <h2>Security Dashboard</h2>
            <p>AI-powered prompt injection detection</p>
          </div>

          <div className="api-status">
            <span className="status-dot"></span>
            API ONLINE
          </div>
        </header>

        {/* STAT CARDS */}
        <section className="stats">

          <div className="stat-card">
            <div className="stat-icon">
              <Zap size={21} />
            </div>

            <div>
              <span>Detection Model</span>
              <strong>MiniLM + LR</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Shield size={21} />
            </div>

            <div>
              <span>Detection Mode</span>
              <strong>Semantic Analysis</strong>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon">
              <Activity size={21} />
            </div>

            <div>
              <span>Prompts Analyzed</span>
              <strong>{history.length}</strong>
            </div>
          </div>

        </section>

        {/* ANALYZER */}
        <section className="analyzer-card">

          <div className="section-heading">
            <div>
              <h3>Prompt Security Analyzer</h3>
              <p>
                Analyze a prompt for potential prompt injection attacks.
              </p>
            </div>

            <div className="model-badge">
              <span></span>
              ML ENGINE ACTIVE
            </div>
          </div>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter a prompt to analyze..."
          />

          <div className="analyzer-footer">
            <span>
              {prompt.length} characters
            </span>

            <button
              onClick={analyzePrompt}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  ANALYZING...
                </>
              ) : (
                <>
                  <Send size={17} />
                  ANALYZE PROMPT
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="error-message">
              <AlertTriangle size={18} />
              {error}
            </div>
          )}

        </section>

        {/* RESULT */}
        {result && (
          <section className="results">

            <div className="result-header">
              <div>
                <h3>Analysis Result</h3>
                <p>Security classification generated by SentinelAI</p>
              </div>

              <div
                className={`action-badge ${getActionClass(result.action)}`}
              >
                {getActionIcon(result.action)}
                {result.action}
              </div>
            </div>

            <div className="result-grid">

              {/* RISK SCORE */}
              <div className="result-card risk-card">

                <span className="result-label">
                  RISK SCORE
                </span>

                <div className="risk-score">
                  {result.risk_score.toFixed(1)}
                  <small>/100</small>
                </div>

                <div className="risk-bar">
                  <div
                    style={{
                      width: `${result.risk_score}%`,
                    }}
                  ></div>
                </div>

                <span className="risk-level">
                  {getRiskLevel(result.risk_score)} RISK
                </span>

              </div>

              {/* PROBABILITY */}
              <div className="result-card">

                <span className="result-label">
                  INJECTION PROBABILITY
                </span>

                <div className="probability">
                  {(result.injection_probability * 100).toFixed(1)}
                  <small>%</small>
                </div>

                <p>
                  Estimated probability that the prompt
                  contains injection behavior.
                </p>

              </div>

              {/* ACTION */}
              <div className="result-card action-card">

                <span className="result-label">
                  RECOMMENDED ACTION
                </span>

                <div
                  className={`large-action ${getActionClass(
                    result.action
                  )}`}
                >
                  {getActionIcon(result.action)}
                  {result.action}
                </div>

                <p>
                  {result.action === "BLOCK"
                    ? "Prompt should be blocked."
                    : result.action === "SUSPICIOUS"
                    ? "Prompt requires additional review."
                    : "Prompt appears safe to process."}
                </p>

              </div>

            </div>

            <div className="analyzed-prompt">
              <span>ANALYZED PROMPT</span>

              <p>{result.prompt}</p>
            </div>

          </section>
        )}

        {/* HISTORY */}
        <section className="history-card">

          <div className="section-heading">
            <div>
              <h3>Recent Analysis</h3>
              <p>Latest prompts analyzed by the engine.</p>
            </div>
          </div>

          {history.length === 0 ? (

            <div className="empty-state">
              <Shield size={30} />
              <p>No prompts analyzed yet.</p>
              <span>
                Submit a prompt above to begin analysis.
              </span>
            </div>

          ) : (

            <div className="history-table">

              <div className="history-header">
                <span>Prompt</span>
                <span>Risk</span>
                <span>Action</span>
              </div>

              {history.map((item) => (

                <div
                  className="history-row"
                  key={item.id}
                >

                  <span className="history-prompt">
                    {item.prompt}
                  </span>

                  <span>
                    {item.risk_score.toFixed(1)}
                  </span>

                  <span>
                    <span
                      className={`table-action ${getActionClass(
                        item.action
                      )}`}
                    >
                      {item.action}
                    </span>
                  </span>

                </div>

              ))}

            </div>

          )}

        </section>

        <footer>
          SentinelAI • AI-Powered Prompt Injection Detection
        </footer>

      </main>
    </div>
  );
}

export default App;