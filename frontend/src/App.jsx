import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Box,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  Cpu,
  Download,
  Eye,
  History,
  Image as ImageIcon,
  Info,
  Layers,
  RefreshCw,
  ScanSearch,
  Settings as SettingsIcon,
  Sparkles,
  Trash2,
  Upload,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const DEFAULT_API_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: Layers },
  { id: "detection", label: "Object Detection", icon: ScanSearch },
  { id: "history", label: "Detection History", icon: History },
  { id: "analytics", label: "Analytics", icon: BarChart3 },
  { id: "model", label: "Model Information", icon: Cpu },
  { id: "about", label: "About Project", icon: Info },
  { id: "settings", label: "Settings", icon: SettingsIcon },
];

function App() {
  const fileInputRef = useRef(null);

  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState("detection");

  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_URL);
  const [draftApiUrl, setDraftApiUrl] = useState(DEFAULT_API_URL);
  const [settingsSavedMsg, setSettingsSavedMsg] = useState("");

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState("annotated");

  const [apiStatus, setApiStatus] = useState("checking");
  const [modelName, setModelName] = useState("yolo11n.pt");

  const [isDetecting, setIsDetecting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const checkHealth = useCallback(async (targetUrl) => {
    setApiStatus("checking");
    try {
      const res = await axios.get(`${targetUrl}/health`, { timeout: 15000 });
      setApiStatus("online");
      if (res.data?.model) setModelName(res.data.model);
    } catch {
      setApiStatus("offline");
    }
  }, []);

  useEffect(() => {
    checkHealth(apiBaseUrl);
  }, [apiBaseUrl, checkHealth]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleFileSelect(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setErrorMsg("Please select a valid image file (JPG, PNG, WEBP).");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg("File exceeds the 10 MB upload limit.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setErrorMsg("");
    setResult(null);
    setViewMode("annotated");
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function handleClearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl("");
    setResult(null);
    setErrorMsg("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  async function handleDetect() {
    if (!selectedFile) {
      setErrorMsg("Please upload an image before running detection.");
      return;
    }
    setIsDetecting(true);
    setErrorMsg("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await axios.post(`${apiBaseUrl}/detect`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });

      const data = res.data;
      setResult(data);
      setViewMode("annotated");
      setApiStatus("online");

      const entry = {
        id: Date.now(),
        fileName: selectedFile.name,
        time: new Date().toLocaleTimeString(),
        totalObjects: data.total_objects ?? 0,
        counts: data.counts || {},
        detections: data.detections || [],
        annotatedImage: data.annotated_image || "",
      };
      setHistory((prev) => [entry, ...prev]);
    } catch (err) {
      const detail =
        err.response?.data?.detail ||
        err.message ||
        `Unable to reach FastAPI backend at ${apiBaseUrl}`;
      setErrorMsg(detail);
      // Keep status unchanged on /detect timeout or transient network error
    } finally {
      setIsDetecting(false);
    }
  }

  function handleSaveSettings(e) {
    e.preventDefault();
    const cleaned = (draftApiUrl || DEFAULT_API_URL).replace(/\/+$/, "");
    setApiBaseUrl(cleaned);
    setDraftApiUrl(cleaned);
    setSettingsSavedMsg(`Updated backend endpoint to ${cleaned}`);
  }

  const totalObjectsDetected = history.reduce(
    (acc, item) => acc + (item.totalObjects || 0),
    0
  );
  const imagesProcessed = history.length;
  const uniqueClassesCount = result?.counts
    ? Object.keys(result.counts).length
    : 0;

  const chartData = result?.counts
    ? Object.entries(result.counts).map(([name, count]) => ({
        name,
        count,
      }))
    : [];

  const annotatedSrc = result?.annotated_image
    ? `data:image/jpeg;base64,${result.annotated_image}`
    : "";

  const currentPageLabel =
    NAV_ITEMS.find((n) => n.id === activeTab)?.label || "Object Detection";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${collapsed ? "collapsed" : ""}`}>
        <div>
          <div className="brand-row">
            <div className="brand-left">
              <div className="brand-logo">
                <Sparkles size={18} />
              </div>
              {!collapsed && (
                <div>
                  <div className="brand-name">VISION AI LAB</div>
                  <div className="brand-tag">YOLOv11 · PYNQ</div>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => setCollapsed((c) => !c)}
              className="collapse-btn"
              title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {collapsed ? <ChevronRight size={15} /> : <ChevronLeft size={15} />}
            </button>
          </div>

          {!collapsed && <div className="nav-label">Navigation</div>}
          <nav className="nav-menu">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  className={`nav-item ${isActive ? "active" : ""}`}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon size={16} />
                  {!collapsed && <span>{item.label}</span>}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="sidebar-status">
          <div className="status-line">
            <span className={`status-dot ${apiStatus}`} />
            {!collapsed && <span>{apiStatus}</span>}
          </div>
          {!collapsed && (
            <div className="endpoint-sub" title={apiBaseUrl}>
              {apiBaseUrl}
            </div>
          )}
        </div>
      </aside>

      <div className="workspace-column">
        <header className="topnav">
          <div className="topnav-title-group">
            <span className="lab-pill">AI Vision Suite</span>
            <h1 className="topnav-heading">{currentPageLabel}</h1>
          </div>

          <div className="topnav-badges">
            <span className="badge-chip">
              <Cpu size={14} color="#22d3ee" />
              {modelName}
            </span>
            <span
              className={`badge-chip ${apiStatus === "online" ? "online" : "offline"}`}
            >
              <Activity size={14} />
              {apiStatus === "online" ? "API Connected" : "API Offline"}
            </span>
            <button
              type="button"
              onClick={() => checkHealth(apiBaseUrl)}
              className="btn-secondary"
              style={{ padding: "6px 10px" }}
              title="Refresh backend health"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </header>

        <main className="main-container">
          <section className="metrics-grid">
            <div className="metric-card">
              <div className="metric-top">
                <span>Total Objects Detected</span>
                <Box size={15} color="#22d3ee" />
              </div>
              <div className="metric-num">{totalObjectsDetected}</div>
              <div className="metric-sub">Cumulative session detections</div>
            </div>

            <div className="metric-card">
              <div className="metric-top">
                <span>Images Processed</span>
                <ImageIcon size={15} color="#3b82f6" />
              </div>
              <div className="metric-num">{imagesProcessed}</div>
              <div className="metric-sub">Verified /detect inferences</div>
            </div>

            <div className="metric-card">
              <div className="metric-top">
                <span>Detected Classes</span>
                <Layers size={15} color="#8b5cf6" />
              </div>
              <div className="metric-num">{uniqueClassesCount}</div>
              <div className="metric-sub">Distinct categories in latest run</div>
            </div>

            <div className="metric-card">
              <div className="metric-top">
                <span>Inference Server</span>
                <Activity size={15} color="#10b981" />
              </div>
              <div className="metric-num" style={{ textTransform: "capitalize" }}>
                {apiStatus}
              </div>
              <div className="metric-sub">{apiBaseUrl}</div>
            </div>
          </section>

          {activeTab === "history" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2 className="panel-title">Session Detection History</h2>
                  <p className="panel-desc">
                    Records generated from real backend /detect responses
                  </p>
                </div>
              </div>
              {history.length === 0 ? (
                <div className="placeholder-box">
                  <div style={{ fontWeight: 600 }}>No detection runs yet</div>
                  <p className="panel-desc">
                    Run Object Detection on an image to populate session history.
                  </p>
                </div>
              ) : (
                history.map((item) => (
                  <div key={item.id} className="history-row">
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 13.5 }}>
                        {item.fileName}
                      </div>
                      <div
                        className="panel-desc"
                        style={{ display: "flex", alignItems: "center", gap: 5 }}
                      >
                        <Clock size={12} /> {item.time}
                      </div>
                    </div>
                    <div style={{ fontWeight: 700, color: "#22d3ee", fontSize: 13 }}>
                      {item.totalObjects} object(s)
                    </div>
                  </div>
                ))
              )}
            </section>
          )}

          {activeTab === "analytics" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2 className="panel-title">Class Distribution Analytics</h2>
                  <p className="panel-desc">
                    Object counts from the latest detection response
                  </p>
                </div>
              </div>
              {chartData.length === 0 ? (
                <div className="placeholder-box">
                  <div style={{ fontWeight: 600 }}>No class analytics available</div>
                  <p className="panel-desc">
                    Detect objects in an image first to view class distribution.
                  </p>
                </div>
              ) : (
                <div style={{ width: "100%", height: 300 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                      <YAxis allowDecimals={false} stroke="#94a3b8" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0b1324",
                          border: "1px solid rgba(34,211,238,0.35)",
                          borderRadius: 10,
                          color: "#f8fafc",
                        }}
                      />
                      <Bar dataKey="count" fill="#22d3ee" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </section>
          )}

          {activeTab === "model" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2 className="panel-title">Model & Backend Details</h2>
                  <p className="panel-desc">
                    Live status queried directly from FastAPI GET /health
                  </p>
                </div>
              </div>
              <p className="panel-desc" style={{ marginBottom: 8 }}>
                Active Model File: <strong>{modelName}</strong>
              </p>
              <p className="panel-desc" style={{ marginBottom: 8 }}>
                API Base URL: <strong>{apiBaseUrl}</strong>
              </p>
              <p className="panel-desc">
                Detection Confidence Threshold: <strong>0.25</strong> (configured in backend/main.py)
              </p>
            </section>
          )}

          {activeTab === "about" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2 className="panel-title">About Capstone Project</h2>
                  <p className="panel-desc">
                    AI-Based Object Detection System Using YOLOv11 and PYNQ
                  </p>
                </div>
              </div>
              <p className="panel-desc" style={{ lineHeight: 1.6 }}>
                This application pairs a React + Vite frontend with a Python FastAPI and
                Ultralytics YOLOv11 inference backend to process uploaded images, return
                annotated outputs, class counts, confidence scores, and bounding boxes.
              </p>
            </section>
          )}

          {activeTab === "settings" && (
            <section className="panel">
              <div className="panel-head">
                <div>
                  <h2 className="panel-title">API Endpoint Settings</h2>
                  <p className="panel-desc">
                    Configure the FastAPI backend URL used for /health and /detect
                  </p>
                </div>
              </div>
              <form onSubmit={handleSaveSettings}>
                <label htmlFor="api-url-input" className="panel-desc">
                  Backend Base URL
                </label>
                <input
                  id="api-url-input"
                  type="text"
                  value={draftApiUrl}
                  onChange={(e) => setDraftApiUrl(e.target.value)}
                  className="input-field"
                  placeholder="http://127.0.0.1:8000"
                />
                <div className="action-row">
                  <button type="submit" className="btn-primary">
                    Save & Reconnect
                  </button>
                </div>
                {settingsSavedMsg && (
                  <p className="panel-desc" style={{ color: "#34d399", marginTop: 10 }}>
                    {settingsSavedMsg}
                  </p>
                )}
              </form>
            </section>
          )}

          {(activeTab === "detection" || activeTab === "overview") && (
            <section className="two-col">
              <div className="panel">
                <div className="panel-head">
                  <div>
                    <h2 className="panel-title">Image Input</h2>
                    <p className="panel-desc">
                      Drag & drop or choose an image up to 10 MB
                    </p>
                  </div>
                  <span className="tag-step">STEP 01</span>
                </div>

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    handleFileSelect(e.dataTransfer.files?.[0]);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  className={`upload-zone ${isDragging ? "drag-active" : ""}`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={(e) => handleFileSelect(e.target.files?.[0])}
                  />

                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt="Uploaded preview"
                      className="media-frame"
                    />
                  ) : (
                    <>
                      <Upload size={32} color="#22d3ee" style={{ marginBottom: 10 }} />
                      <div style={{ fontWeight: 600, fontSize: 14 }}>
                        Drop your image here
                      </div>
                      <p className="panel-desc">
                        or click to browse (JPG, PNG, WEBP · max 10 MB)
                      </p>
                    </>
                  )}
                </div>

                {selectedFile && (
                  <div className="file-bar">
                    <div className="file-meta">
                      <ImageIcon size={15} color="#22d3ee" />
                      <span className="file-text">{selectedFile.name}</span>
                      <span style={{ color: "#64748b", flexShrink: 0 }}>
                        ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={handleClearImage}
                      className="btn-link-danger"
                    >
                      <Trash2 size={13} /> Remove
                    </button>
                  </div>
                )}

                {errorMsg && (
                  <div className="alert-err">
                    <AlertCircle size={16} color="#fb7185" style={{ flexShrink: 0 }} />
                    <span>{errorMsg}</span>
                  </div>
                )}

                <div className="action-row">
                  <button
                    type="button"
                    onClick={handleDetect}
                    disabled={!selectedFile || isDetecting}
                    className="btn-primary"
                  >
                    <ScanSearch size={16} />
                    {isDetecting ? "Running YOLOv11..." : "Run YOLO Detection"}
                  </button>
                </div>
              </div>

              <div className="panel">
                <div className="panel-head">
                  <div>
                    <h2 className="panel-title">Detection Output</h2>
                    <p className="panel-desc">
                      Annotated bounding boxes, class summary & confidence
                    </p>
                  </div>
                  <span className="tag-step">STEP 02</span>
                </div>

                {!result ? (
                  <div className="placeholder-box">
                    <CheckCircle2
                      size={30}
                      color="#475569"
                      style={{ marginBottom: 10 }}
                    />
                    <div style={{ fontWeight: 600, fontSize: 14 }}>
                      {isDetecting
                        ? "Analyzing image with YOLOv11..."
                        : "Waiting for detection"}
                    </div>
                    <p className="panel-desc">
                      Annotated results will appear here after detection finishes.
                    </p>
                  </div>
                ) : (
                  <div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: 8,
                        marginBottom: 10,
                        flexWrap: "wrap",
                      }}
                    >
                      <div style={{ display: "flex", gap: 6 }}>
                        <button
                          type="button"
                          onClick={() => setViewMode("annotated")}
                          className="btn-secondary"
                          style={{
                            padding: "6px 10px",
                            borderColor:
                              viewMode === "annotated" ? "#22d3ee" : undefined,
                          }}
                        >
                          Annotated
                        </button>
                        {previewUrl && (
                          <button
                            type="button"
                            onClick={() => setViewMode("original")}
                            className="btn-secondary"
                            style={{
                              padding: "6px 10px",
                              borderColor:
                                viewMode === "original" ? "#22d3ee" : undefined,
                            }}
                          >
                            <Eye size={13} /> Original
                          </button>
                        )}
                      </div>

                      {annotatedSrc && (
                        <a
                          href={annotatedSrc}
                          download={`detected-${selectedFile?.name || "result.jpg"}`}
                          className="btn-secondary"
                          style={{ padding: "6px 10px" }}
                        >
                          <Download size={13} /> Download
                        </a>
                      )}
                    </div>

                    <img
                      src={viewMode === "original" && previewUrl ? previewUrl : annotatedSrc}
                      alt={
                        viewMode === "original"
                          ? "Original uploaded image"
                          : "YOLOv11 annotated image"
                      }
                      className="media-frame"
                    />

                    <div className="class-badges">
                      <span className="class-pill total">
                        Total: {result.total_objects ?? 0}
                      </span>
                      {result.counts &&
                        Object.entries(result.counts).map(([cls, count]) => (
                          <span key={cls} className="class-pill">
                            {cls}: {count}
                          </span>
                        ))}
                    </div>

                    {Array.isArray(result.detections) &&
                      result.detections.length > 0 && (
                        <div className="det-scroll">
                          {result.detections.map((det, idx) => {
                            const pct = Math.min(
                              100,
                              Math.max(0, (det.confidence || 0) * 100)
                            );
                            return (
                              <div key={idx} className="det-card">
                                <div className="det-top">
                                  <span style={{ fontWeight: 700 }}>{det.class}</span>
                                  <span className="conf-val">{pct.toFixed(1)}%</span>
                                  <span className="coords-txt">
                                    [{det.box?.join(", ")}]
                                  </span>
                                </div>
                                <div className="conf-bar-track">
                                  <div
                                    className="conf-bar-fill"
                                    style={{ width: `${pct}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                  </div>
                )}
              </div>
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;