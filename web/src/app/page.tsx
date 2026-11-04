"use client";

import { useState } from "react";

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

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function detectObjects() {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const api = process.env.NEXT_PUBLIC_API_URL;

      if (!api) throw new Error("Backend API URL is not configured.");

      const response = await fetch(`${api}/detect`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Detection failed.");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 1050, margin: "40px auto", padding: 24, fontFamily: "Arial, sans-serif" }}>
      <header>
        <p style={{ color: "#2563eb", fontWeight: 700 }}>PYNQ-Z2 · YOLO11</p>
        <h1 style={{ fontSize: 36, marginBottom: 8 }}>AI Object Detection</h1>
        <p style={{ color: "#64748b" }}>Upload an image and detect objects using YOLO11.</p>
      </header>

      <section style={{ border: "1px solid #ddd", borderRadius: 14, padding: 24, marginTop: 28 }}>
        <h2>Upload image</h2>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            setResult(null);
            setError("");
          }}
        />
        {file && <p>Selected: {file.name}</p>}
        <button
          onClick={detectObjects}
          disabled={!file || loading}
          style={{ background: "#2563eb", color: "white", border: 0, padding: "12px 22px", borderRadius: 8, marginTop: 14, cursor: "pointer" }}
        >
          {loading ? "Detecting..." : "Run object detection"}
        </button>
        {error && <p style={{ color: "red" }}>{error}</p>}
      </section>

      {result && (
        <section style={{ marginTop: 30 }}>
          <h2>Detection results</h2>
          <h3>Total objects: {result.total_objects}</h3>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
            {Object.entries(result.counts).map(([name, count]) => (
              <div key={name} style={{ border: "1px solid #ddd", borderRadius: 10, padding: 16 }}>
                <strong>{name}</strong>
                <p style={{ fontSize: 24, margin: "8px 0 0" }}>{count}</p>
              </div>
            ))}
          </div>

          <h3>Annotated image</h3>
          <img
            src={`data:image/jpeg;base64,${result.annotated_image}`}
            alt="YOLO detection result"
            style={{ width: "100%", maxWidth: 800, borderRadius: 12 }}
          />

          <h3>Detected objects</h3>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th align="left">Object</th>
                  <th align="left">Confidence</th>
                  <th align="left">Bounding box</th>
                </tr>
              </thead>
              <tbody>
                {result.detections.map((item, index) => (
                  <tr key={index}>
                    <td style={{ padding: 10, borderTop: "1px solid #ddd" }}>{item.class}</td>
                    <td style={{ padding: 10, borderTop: "1px solid #ddd" }}>{(item.confidence * 100).toFixed(1)}%</td>
                    <td style={{ padding: 10, borderTop: "1px solid #ddd" }}>{item.box.join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
