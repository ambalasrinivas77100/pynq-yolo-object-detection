
"use client";

import { useEffect, useMemo, useState } from "react";

type Detection = {
  class: string;
  confidence: number;
  box: number[];
};

type Result = {
  success: boolean;
  total_objects: number;
  counts: Record<string, number>;
  detections: Detection[];
  annotated_image: string;
};

type HistoryItem = {
  name: string;
  total: number;
  time: string;
};

const API = process.env.NEXT_PUBLIC_API_URL;

const navigation = [
  { name: "Overview", icon: "◫" },
  { name: "Object Detection", icon: "⌗" },
  { name: "Analytics", icon: "▥" },
  { name: "Detection History", icon: "◷" },
  { name: "Model & Hardware", icon: "▦" },
];

export default function Home() {
  const [active, setActive] = useState("Overview");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("Checking");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let activeRequest = true;

    async function checkApi() {
      try {
        const response = await fetch(`${API}/health`);
        const data = await response.json();

        if (activeRequest) {
          setApiStatus(response.ok && data.status === "healthy" ? "Online" : "Unavailable");
        }
      } catch {
        if (activeRequest) setApiStatus("Unavailable");
      }
    }

    if (API) checkApi();
    else setApiStatus("Not configured");

    return () => {
      activeRequest = false;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const totalObjects = history.reduce((sum, item) => sum + item.total, 0);

  const classTotals = useMemo(() => {
    return Object.entries(result?.counts || {}).sort((a, b) => b[1] - a[1]);
  }, [result]);

  function selectFile(selected: File | null) {
    if (!selected) return;

    if (!selected.type.startsWith("image/")) {
      setError("Please select a valid image.");
      return;
    }

    if (selected.size > 10 * 1024 * 1024) {
      setError("Image size must be below 10 MB.");
      return;
    }

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
    setActive("Object Detection");
  }

  async function runDetection() {
    if (!file || !API) {
      setError("Select an image and configure the API URL.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/detect`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Detection request failed.");
      }

      setResult(data);

      setHistory((old) => [
        {
          name: file.name,
          total: data.total_objects,
          time: new Date().toLocaleTimeString(),
        },
        ...old,
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to detect objects.");
    } finally {
      setLoading(false);
    }
  }

  const card = "stat-card";

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">V</div>
          <div>
            <strong>VISION AI</strong>
            <span>INTELLIGENCE PLATFORM</span>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav className="navigation">
          {navigation.map((item) => (
            <button
              key={item.name}
              onClick={() => setActive(item.name)}
              className={`nav-item ${active === item.name ? "active" : ""}`}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
              {item.name === "Object Detection" && (
                <span className="nav-dot" />
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="system-card">
            <div className="system-heading">
              <span className="online-dot" />
              System status
            </div>
            <strong>{apiStatus}</strong>
            <span>YOLO API connection</span>
          </div>

          <div className="profile">
            <div className="avatar">AI</div>
            <div>
              <strong>Vision Workspace</strong>
              <span>YOLO11 Project</span>
            </div>
          </div>
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <strong>{active}</strong>
          </div>
          <div className="top-actions">
            <span className={`status-pill ${apiStatus === "Online" ? "connected" : ""}`}>
              <span className="online-dot" /> API {apiStatus}
            </span>
            <div className="top-avatar">AI</div>
          </div>
        </header>

        <div className="content">
          <div className="welcome-row">
            <div>
              <div className="eyebrow">COMPUTER VISION PLATFORM</div>
              <h1>{active === "Overview" ? "Dashboard Overview" : active}</h1>
              <p>Monitor, analyze and understand your visual data with YOLO11.</p>
            </div>
            <div className="model-pill">
              <span className="online-dot" />
              YOLO11n Model
            </div>
          </div>

          <section className="stats-grid">
            <div className={card}>
              <div className="stat-top">
                <span>Total objects detected</span>
                <div className="stat-icon blue">⌗</div>
              </div>
              <strong>{totalObjects.toLocaleString()}</strong>
              <small>Across this session</small>
            </div>

            <div className={card}>
              <div className="stat-top">
                <span>Images processed</span>
                <div className="stat-icon purple">▧</div>
              </div>
              <strong>{history.length}</strong>
              <small>Current browser session</small>
            </div>

            <div className={card}>
              <div className="stat-top">
                <span>Unique classes</span>
                <div className="stat-icon cyan">◈</div>
              </div>
              <strong>{Object.keys(result?.counts || {}).length}</strong>
              <small>In latest detection</small>
            </div>

            <div className={card}>
              <div className="stat-top">
                <span>API connection</span>
                <div className="stat-icon green">⌁</div>
              </div>
              <strong className="status-number">{apiStatus}</strong>
              <small>Backend health check</small>
            </div>
          </section>

          <div className="section-heading">
            <div>
              <h2>Detection workspace</h2>
              <p>Upload an image and run AI-powered object detection.</p>
            </div>
            <span className="mode-label">IMAGE INFERENCE</span>
          </div>

          <section className="workspace-grid">
            <div className="panel upload-panel">
              <div className="panel-heading">
                <div>
                  <h3>Image input</h3>
                  <p>Select an image for analysis</p>
                </div>
                <span className="step-label">STEP 01</span>
              </div>

              <label
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  selectFile(e.dataTransfer.files[0] || null);
                }}
              >
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(e) => selectFile(e.target.files?.[0] || null)}
                />

                {preview ? (
                  <img src={preview} alt="Selected upload preview" className="preview-image" />
                ) : (
                  <>
                    <div className="upload-icon">↑</div>
                    <strong>Drop your image here</strong>
                    <span>or click to browse files</span>
                    <small>PNG, JPG or WEBP · Maximum 10 MB</small>
                  </>
                )}
              </label>

              {file && (
                <div className="file-info">
                  <div className="file-symbol">IMG</div>
                  <div className="file-name">
                    <strong>{file.name}</strong>
                    <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                  <button
                    className="remove-file"
                    onClick={() => {
                      setFile(null);
                      setPreview("");
                      setResult(null);
                    }}
                  >
                    Remove
                  </button>
                </div>
              )}

              <button
                className="primary-button"
                disabled={!file || loading || apiStatus !== "Online"}
                onClick={runDetection}
              >
                {loading ? "Processing image..." : "⌗  Run YOLO Detection"}
              </button>

              {error && <div className="error-message">{error}</div>}
              <div className="upload-footnote">
                <span>✦</span> Powered by Ultralytics YOLO11
              </div>
            </div>

            <div className="panel result-panel">
              <div className="panel-heading">
                <div>
                  <h3>Detection output</h3>
                  <p>AI-annotated image and bounding boxes</p>
                </div>
                <span className="step-label">STEP 02</span>
              </div>

              {result ? (
                <>
                  <div className="result-image-wrap">
                    <img
                      src={`data:image/jpeg;base64,${result.annotated_image}`}
                      alt="YOLO annotated detection"
                      className="result-image"
                    />
                  </div>
                  <div className="result-summary">
                    <div>
                      <strong>{result.total_objects}</strong>
                      <span>Objects found</span>
                    </div>
                    <div>
                      <strong>{classTotals.length}</strong>
                      <span>Classes detected</span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty-result">
                  <div className="target-icon">◎</div>
                  <strong>Waiting for detection</strong>
                  <span>Your annotated image will appear here.</span>
                </div>
              )}
            </div>
          </section>

          <div className="lower-grid">
            <section className="panel analytics-panel">
              <div className="panel-heading">
                <div>
                  <h3>Object distribution</h3>
                  <p>Class-wise count from latest detection</p>
                </div>
                <span className="step-label">ANALYTICS</span>
              </div>

              {classTotals.length ? (
                <div className="distribution-list">
                  {classTotals.map(([name, count], index) => {
                    const max = Math.max(...classTotals.map((item) => item[1]));
                    return (
                      <div className="distribution-item" key={name}>
                        <div className="distribution-label">
                          <span>{name}</span>
                          <strong>{count}</strong>
                        </div>
                        <div className="bar-track">
                          <div
                            className={`bar-fill color-${index % 5}`}
                            style={{ width: `${(count / max) * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-small">
                  Object analytics will appear after your first detection.
                </div>
              )}
            </section>

            <section className="panel history-panel">
              <div className="panel-heading">
                <div>
                  <h3>Recent activity</h3>
                  <p>Images processed in this session</p>
                </div>
                <span className="step-label">HISTORY</span>
              </div>

              {history.length ? (
                <div className="history-list">
                  {history.slice(0, 5).map((item, index) => (
                    <div className="history-item" key={`${item.name}-${index}`}>
                      <div className="history-file">▧</div>
                      <div className="history-name">
                        <strong>{item.name}</strong>
                        <span>{item.time}</span>
                      </div>
                      <span className="history-count">{item.total} objects</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-small">
                  Your recent detection activity will appear here.
                </div>
              )}
            </section>
          </div>

          <section className="panel model-panel">
            <div>
              <div className="model-label">ACTIVE DETECTION MODEL</div>
              <h3>YOLO11n Object Detection</h3>
              <p>Ultralytics inference served through your FastAPI backend on Render.</p>
            </div>
            <div className="model-details">
              <div><span>Model</span><strong>YOLO11n</strong></div>
              <div><span>Inference API</span><strong>{apiStatus}</strong></div>
              <div><span>Platform</span><strong>Cloud API</strong></div>
            </div>
          </section>

          <footer className="footer">
            <span>VISION AI · PYNQ OBJECT DETECTION PROJECT</span>
            <span>YOLO11 · COMPUTER VISION</span>
          </footer>
        </div>
      </section>
    </main>
  );
}
