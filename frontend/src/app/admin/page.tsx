"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import {
  CheckCircle,
  XCircle,
  Edit3,
  Download,
  UploadCloud,
  ChevronLeft,
  ChevronRight,
  Layers,
  Trash2,
  CheckSquare,
} from "lucide-react";
import styles from "./page.module.css";

const BACKEND =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "https://plant-disease-detection-32l7.onrender.com";

interface BoundingBox {
  class_name: string;
  confidence?: number;
  x_center: number;
  y_center: number;
  width: number;
  height: number;
  crop_label?: string;
}

interface VerificationFrame {
  id: number;
  raw_input_id: number | null;
  frame_index: number;
  storage_path: string;
  image_url: string;
  parent_crop_predicted: string | null;
  parent_confidence: number;
  model_predictions: BoundingBox[];
  status: "pending" | "approved" | "rejected" | "corrected";
  human_crop_label: string | null;
  human_annotations: BoundingBox[];
  verified_by: string;
  ready_for_retraining: boolean;
  created_at: string | null;
  verified_at: string | null;
}

interface Metrics {
  total_count: number;
  pending_count: number;
  approved_count: number;
  rejected_count: number;
  corrected_count: number;
  low_confidence_count: number;
}

const CROP_OPTIONS = [
  "Apple",
  "Banana",
  "BitterGourd",
  "Brinjal",
  "Cashew",
  "Cassava",
  "Cauliflower",
  "Cherry",
  "Coconut",
  "Coffee",
  "Coriander",
  "Corn",
  "Grape",
  "Groundnut",
  "Guava",
  "Jackfruit",
  "Lemon",
  "Mango",
  "Neem",
  "Papaya",
  "Peach",
  "PepperBell",
  "Potato",
  "Pumpkin",
  "Rice",
  "Rose",
  "Sesame",
  "Soybean",
  "Strawberry",
  "SugarCane",
  "Sunflower",
  "Tobacco",
  "Tomato",
  "Wheat",
];

const DISEASE_OPTIONS = [
  "Healthy",
  "Apple_Black_rot",
  "Apple_scab",
  "Cedar_apple_rust",
  "Corn_Common_rust",
  "Corn_Northern_Leaf_Blight",
  "Grape_Black_rot",
  "Grape_Leaf_blight",
  "Potato_Early_blight",
  "Potato_Late_blight",
  "Tomato_Bacterial_spot",
  "Tomato_Early_blight",
  "Tomato_Late_blight",
  "Tomato_Leaf_Mold",
  "Tomato_Septoria_leaf_spot",
  "Tomato_Spider_mites",
  "Tomato_Target_Spot",
  "Tomato_Yellow_Leaf_Curl_Virus",
  "Tomato_mosaic_virus",
  "Wheat_brown_rust",
  "Wheat_yellow_rust",
];

export default function AdminDashboardPage() {
  const [frames, setFrames] = useState<VerificationFrame[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    total_count: 0,
    pending_count: 0,
    approved_count: 0,
    rejected_count: 0,
    corrected_count: 0,
    low_confidence_count: 0,
  });

  // Filter States
  const [statusFilter, setStatusFilter] = useState<string>("pending");
  const [cropFilter, setCropFilter] = useState<string>("");
  const [lowConfOnly, setLowConfOnly] = useState<boolean>(false);
  const [page, setPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);

  // Selected Active Frame for Verification
  const [selectedFrame, setSelectedFrame] = useState<VerificationFrame | null>(null);
  const [humanCropLabel, setHumanCropLabel] = useState<string>("");
  const [humanBoxes, setHumanBoxes] = useState<BoundingBox[]>([]);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState<number | null>(null);

  // Annotation Form Controls
  const [newBoxLabel, setNewBoxLabel] = useState<string>("Healthy");

  // Ingestion Modal State
  const [showIngestModal, setShowIngestModal] = useState<boolean>(false);
  const [ingestFile, setIngestFile] = useState<File | null>(null);
  const [ingestSourceType, setIngestSourceType] = useState<string>("single_image");
  const [isIngesting, setIsIngesting] = useState<boolean>(false);

  // Canvas Refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null);
  const [drawCurrent, setDrawCurrent] = useState<{ x: number; y: number } | null>(null);

  // Fetch Verification Frames from shared backend
  const fetchFrames = useCallback(async () => {
    try {
      let minConf = 0.0;
      let maxConf = 1.0;
      if (lowConfOnly) {
        minConf = 0.0;
        maxConf = 0.80;
      }

      const params = new URLSearchParams({
        page: page.toString(),
        limit: "10",
        status: statusFilter,
        min_confidence: minConf.toString(),
        max_confidence: maxConf.toString(),
      });
      if (cropFilter) params.append("crop_type", cropFilter);

      const res = await fetch(`${BACKEND}/api/admin/verification/frames?${params.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setFrames(data.frames || []);
      setMetrics(data.metrics || metrics);
      setTotalPages(data.pagination?.total_pages || 1);

      if (data.frames && data.frames.length > 0 && !selectedFrame) {
        const first = data.frames[0];
        setSelectedFrame(first);
        setHumanCropLabel(first.human_crop_label || first.parent_crop_predicted || "Plant");
        setHumanBoxes(first.human_annotations.length > 0 ? first.human_annotations : first.model_predictions);
      }
    } catch (err) {
      console.error("[Admin] Error fetching verification frames:", err);
    }
  }, [page, statusFilter, cropFilter, lowConfOnly]);

  useEffect(() => {
    fetchFrames();
  }, [fetchFrames]);

  const handleSelectFrame = (frame: VerificationFrame) => {
    setSelectedFrame(frame);
    setHumanCropLabel(frame.human_crop_label || frame.parent_crop_predicted || "Plant");
    const initialBoxes = frame.human_annotations && frame.human_annotations.length > 0
      ? frame.human_annotations
      : frame.model_predictions;
    setHumanBoxes(initialBoxes);
    setSelectedBoxIndex(null);
  };

  // Render HTML5 Canvas
  useEffect(() => {
    if (!selectedFrame || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = "anonymous";
    
    const imgUrl = selectedFrame.image_url.startsWith("http")
      ? selectedFrame.image_url
      : `${BACKEND}${selectedFrame.image_url}`;

    img.src = imgUrl;
    img.onload = () => {
      const maxW = 720;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;

      const cw = canvas.width;
      const ch = canvas.height;

      ctx.drawImage(img, 0, 0, cw, ch);

      // Model Predictions (Dashed lines / Orange)
      selectedFrame.model_predictions.forEach((box) => {
        const bx = (box.x_center - box.width / 2) * cw;
        const by = (box.y_center - box.height / 2) * ch;
        const bw = box.width * cw;
        const bh = box.height * ch;

        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "#F59E0B";
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = "#F59E0B";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText(`[Model] ${box.class_name} (${((box.confidence || 0.9) * 100).toFixed(0)}%)`, bx + 4, Math.max(14, by - 4));
      });

      // Human Annotations (Solid Purple/Cyan lines)
      ctx.setLineDash([]);
      humanBoxes.forEach((box, idx) => {
        const bx = (box.x_center - box.width / 2) * cw;
        const by = (box.y_center - box.height / 2) * ch;
        const bw = box.width * cw;
        const bh = box.height * ch;

        const isSel = selectedBoxIndex === idx;
        ctx.strokeStyle = isSel ? "#38BDF8" : "#8B5CF6";
        ctx.lineWidth = isSel ? 3 : 2;
        ctx.strokeRect(bx, by, bw, bh);

        ctx.fillStyle = isSel ? "#0284C7" : "#7C3AED";
        ctx.fillRect(bx, Math.max(0, by - 20), ctx.measureText(box.class_name).width + 12, 18);

        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 10px sans-serif";
        ctx.fillText(box.class_name, bx + 6, Math.max(12, by - 6));
      });

      // Active Mouse Drag Box
      if (isDrawing && drawStart && drawCurrent) {
        const dx = Math.min(drawStart.x, drawCurrent.x);
        const dy = Math.min(drawStart.y, drawCurrent.y);
        const dw = Math.abs(drawCurrent.x - drawStart.x);
        const dh = Math.abs(drawCurrent.y - drawStart.y);

        ctx.setLineDash([]);
        ctx.strokeStyle = "#38BDF8";
        ctx.lineWidth = 2;
        ctx.strokeRect(dx, dy, dw, dh);
      }
    };
  }, [selectedFrame, humanBoxes, selectedBoxIndex, isDrawing, drawStart, drawCurrent]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setIsDrawing(true);
    setDrawStart({ x, y });
    setDrawCurrent({ x, y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawCurrent({ x, y });
  };

  const handleMouseUp = () => {
    if (!isDrawing || !drawStart || !drawCurrent || !canvasRef.current) {
      setIsDrawing(false);
      return;
    }

    const cw = canvasRef.current.width;
    const ch = canvasRef.current.height;

    const x1 = Math.min(drawStart.x, drawCurrent.x);
    const y1 = Math.min(drawStart.y, drawCurrent.y);
    const w = Math.abs(drawCurrent.x - drawStart.x);
    const h = Math.abs(drawCurrent.y - drawStart.y);

    if (w > 10 && h > 10) {
      const x_center = (x1 + w / 2) / cw;
      const y_center = (y1 + h / 2) / ch;
      const norm_w = w / cw;
      const norm_h = h / ch;

      const newBox: BoundingBox = {
        class_name: newBoxLabel,
        confidence: 1.0,
        x_center: Math.max(0, Math.min(1, x_center)),
        y_center: Math.max(0, Math.min(1, y_center)),
        width: Math.max(0.01, Math.min(1, norm_w)),
        height: Math.max(0.01, Math.min(1, norm_h)),
        crop_label: humanCropLabel,
      };

      setHumanBoxes((prev) => [...prev, newBox]);
      setSelectedBoxIndex(humanBoxes.length);
    }

    setIsDrawing(false);
    setDrawStart(null);
    setDrawCurrent(null);
  };

  const handleDecision = async (statusDecision: "approved" | "rejected" | "corrected") => {
    if (!selectedFrame) return;

    try {
      const body = {
        frame_id: selectedFrame.id,
        status: statusDecision,
        human_crop_label: humanCropLabel,
        human_annotations: humanBoxes,
        verified_by: "Human_Annotator_1",
      };

      const res = await fetch(`${BACKEND}/api/admin/verification/annotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      fetchFrames();
      const currIdx = frames.findIndex((f) => f.id === selectedFrame.id);
      if (currIdx >= 0 && currIdx < frames.length - 1) {
        handleSelectFrame(frames[currIdx + 1]);
      }
    } catch (err) {
      console.error("[Admin] Error updating annotation decision:", err);
    }
  };

  const handleExport = (format: "json" | "yolo") => {
    window.open(`${BACKEND}/api/admin/verification/export?format=${format}`, "_blank");
  };

  const handleIngest = async () => {
    if (!ingestFile) return;
    setIsIngesting(true);
    try {
      const formData = new FormData();
      formData.append("file", ingestFile);
      formData.append("source_type", ingestSourceType);

      const res = await fetch(`${BACKEND}/api/admin/verification/ingest`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setShowIngestModal(false);
      setIngestFile(null);
      fetchFrames();
    } catch (err) {
      console.error("[Admin] Ingestion failed:", err);
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <div className={styles.page}>
      {/* ── HEADER NAVBAR ── */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.logoWrap}>
            <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
              <polygon points="16,2 30,26 2,26" stroke="#38BDF8" strokeWidth="2.2" fill="rgba(56,189,248,0.15)" strokeLinejoin="round" />
              <circle cx="16" cy="18" r="3.5" fill="#38BDF8" />
            </svg>
          </div>
          <div>
            <div className={styles.title}>PROJECT JATAYU</div>
            <div className={styles.subtitle}>ADMINISTRATOR CONTROL & HITL VERIFICATION HUB</div>
          </div>
        </div>

        <nav className={styles.navLinks}>
          <Link href="/" className={styles.navBtn}>
            <Layers size={14} /> MISSION CONTROL
          </Link>
          <Link href="/admin" className={`${styles.navBtn} ${styles.navBtnActive}`}>
            <CheckSquare size={14} /> ADMIN DASHBOARD
          </Link>
        </nav>

        <div className={styles.headerActions}>
          <button className={styles.actionBtn} onClick={() => setShowIngestModal(true)}>
            <UploadCloud size={14} /> INGEST MEDIA
          </button>
          <button className={`${styles.actionBtn} ${styles.exportBtn}`} onClick={() => handleExport("yolo")}>
            <Download size={14} /> EXPORT YOLO DATASET
          </button>
        </div>
      </header>

      {/* ── KPI METRICS BANNER ── */}
      <div className={styles.kpiBanner}>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel}>SAMPLED FRAMES</span>
          <span className={styles.kpiValue}>{metrics.total_count}</span>
          <span className={styles.kpiSub}>Total Ingested Telemetry</span>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel} style={{ color: "#3B82F6" }}>
            PENDING REVIEW
          </span>
          <span className={styles.kpiValue} style={{ color: "#3B82F6" }}>
            {metrics.pending_count}
          </span>
          <span className={styles.kpiSub}>Awaiting Verifier Decision</span>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel} style={{ color: "#F59E0B" }}>
            LOW CONFIDENCE (&lt;80%)
          </span>
          <span className={styles.kpiValue} style={{ color: "#F59E0B" }}>
            {metrics.low_confidence_count}
          </span>
          <span className={styles.kpiSub}>Priority HITL Samples</span>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel} style={{ color: "#10B981" }}>
            APPROVED SAMPLES
          </span>
          <span className={styles.kpiValue} style={{ color: "#10B981" }}>
            {metrics.approved_count}
          </span>
          <span className={styles.kpiSub}>Ground Truth Verified</span>
        </div>
        <div className={styles.kpiCard}>
          <span className={styles.kpiLabel} style={{ color: "#8B5CF6" }}>
            RETRAINING READY
          </span>
          <span className={styles.kpiValue} style={{ color: "#8B5CF6" }}>
            {metrics.approved_count + metrics.corrected_count}
          </span>
          <span className={styles.kpiSub}>Approved + Corrected Dataset</span>
        </div>
      </div>

      {/* ── MAIN WORKSPACE GRID ── */}
      <div className={styles.mainLayout}>
        {/* LEFT SIDEBAR: FILTERS & THUMBNAIL LIST */}
        <aside className={styles.sidebar}>
          <div className={styles.filterHeader}>
            <div className={styles.statusTabs}>
              {["all", "pending", "approved", "corrected", "rejected"].map((st) => (
                <button
                  key={st}
                  className={`${styles.tabBtn} ${statusFilter === st ? styles.tabActive : ""}`}
                  onClick={() => {
                    setStatusFilter(st);
                    setPage(1);
                  }}
                >
                  {st.toUpperCase()}
                </button>
              ))}
            </div>

            <div className={styles.filterControls}>
              <select className={styles.selectInput} value={cropFilter} onChange={(e) => setCropFilter(e.target.value)}>
                <option value="">All Crops</option>
                {CROP_OPTIONS.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>

              <button
                className={`${styles.toggleBtn} ${lowConfOnly ? styles.toggleBtnActive : ""}`}
                onClick={() => setLowConfOnly(!lowConfOnly)}
              >
                &lt;80% CONFIDENCE
              </button>
            </div>
          </div>

          <div className={styles.thumbList}>
            {frames.length === 0 ? (
              <div style={{ textAlign: "center", padding: "32px 16px", color: "var(--text-muted)", fontSize: "12px" }}>
                No frames match current filter.
              </div>
            ) : (
              frames.map((frame) => {
                const isSel = selectedFrame?.id === frame.id;
                const isLowConf = frame.parent_confidence < 0.8;
                const imgUrl = frame.image_url.startsWith("http") ? frame.image_url : `${BACKEND}${frame.image_url}`;

                return (
                  <div
                    key={frame.id}
                    className={`${styles.thumbCard} ${isSel ? styles.thumbSelected : ""}`}
                    onClick={() => handleSelectFrame(frame)}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imgUrl} alt={`Frame ${frame.id}`} className={styles.thumbImg} />
                    <div className={styles.thumbMeta}>
                      <div className={styles.thumbTitle}>
                        Frame #{frame.id} · {frame.parent_crop_predicted || "Plant"}
                      </div>
                      <div className={styles.thumbBadges}>
                        <span className={`${styles.badge} ${styles[`badge${frame.status.charAt(0).toUpperCase() + frame.status.slice(1)}`]}`}>
                          {frame.status}
                        </span>
                        <span className={isLowConf ? styles.confBadgeLow : styles.confBadgeHigh}>
                          {(frame.parent_confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </aside>

        {/* RIGHT WORKSPACE: INTERACTIVE CANVAS & VERIFIER PANEL */}
        <main className={styles.workspace}>
          {selectedFrame ? (
            <>
              <div className={styles.workHeader}>
                <div className={styles.workTitle}>
                  <span>FRAME #{selectedFrame.id} VERIFICATION WORKSPACE</span>
                  <span className={styles.badge} style={{ background: "var(--bg-panel-alt)", color: "var(--accent-primary)" }}>
                    Predicted Crop: {selectedFrame.parent_crop_predicted || "Plant"} ({(selectedFrame.parent_confidence * 100).toFixed(1)}%)
                  </span>
                </div>

                <div className={styles.navFrameBtns}>
                  <button
                    className={styles.iconBtn}
                    onClick={() => {
                      const idx = frames.findIndex((f) => f.id === selectedFrame.id);
                      if (idx > 0) handleSelectFrame(frames[idx - 1]);
                    }}
                  >
                    <ChevronLeft size={14} /> PREV
                  </button>
                  <button
                    className={styles.iconBtn}
                    onClick={() => {
                      const idx = frames.findIndex((f) => f.id === selectedFrame.id);
                      if (idx >= 0 && idx < frames.length - 1) handleSelectFrame(frames[idx + 1]);
                    }}
                  >
                    NEXT <ChevronRight size={14} />
                  </button>
                </div>
              </div>

              {/* INTERACTIVE CANVAS */}
              <div className={styles.canvasContainer}>
                <div className={styles.canvasWrapper}>
                  <canvas
                    ref={canvasRef}
                    className={styles.canvasEl}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                  />
                </div>
              </div>

              {/* DECISION & ANNOTATION FORM PANEL */}
              <div className={styles.decisionPanel}>
                <div className={styles.annoForm}>
                  <label style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-secondary)" }}>Verified Crop:</label>
                  <select
                    className={styles.selectInput}
                    value={humanCropLabel}
                    onChange={(e) => setHumanCropLabel(e.target.value)}
                  >
                    {CROP_OPTIONS.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>

                  <label style={{ fontSize: "11px", fontWeight: 700, color: "var(--text-secondary)", marginLeft: "8px" }}>Draw Bounding Box Strain:</label>
                  <select
                    className={styles.selectInput}
                    value={newBoxLabel}
                    onChange={(e) => setNewBoxLabel(e.target.value)}
                  >
                    {DISEASE_OPTIONS.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>

                  {humanBoxes.length > 0 && (
                    <button
                      className={styles.iconBtn}
                      style={{ color: "#EF4444", borderColor: "rgba(239, 68, 68, 0.4)" }}
                      onClick={() => {
                        setHumanBoxes([]);
                        setSelectedBoxIndex(null);
                      }}
                    >
                      <Trash2 size={13} /> Clear Boxes
                    </button>
                  )}
                </div>

                <div className={styles.decisionBtns}>
                  <button className={styles.approveBtn} onClick={() => handleDecision("approved")}>
                    <CheckCircle size={15} /> APPROVE (GROUND TRUTH)
                  </button>
                  <button className={styles.correctBtn} onClick={() => handleDecision("corrected")}>
                    <Edit3 size={15} /> SUBMIT CORRECTION
                  </button>
                  <button className={styles.rejectBtn} onClick={() => handleDecision("rejected")}>
                    <XCircle size={15} /> REJECT
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "13px" }}>
              Select a sampled frame from the left sidebar to begin verification.
            </div>
          )}
        </main>
      </div>

      {/* ── INGEST MEDIA MODAL ── */}
      {showIngestModal && (
        <div className={styles.modalBackdrop} onClick={() => setShowIngestModal(false)}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "14px", fontWeight: 800, color: "var(--text-primary)" }}>INGEST MEDIA FOR HITL SAMPLING</span>
              <button style={{ background: "none", border: "none", color: "var(--text-muted)", cursor: "pointer" }} onClick={() => setShowIngestModal(false)}>
                ✕
              </button>
            </div>

            <div style={{ marginTop: "16px", display: "flex", gap: "12px" }}>
              <select className={styles.selectInput} style={{ flex: 1 }} value={ingestSourceType} onChange={(e) => setIngestSourceType(e.target.value)}>
                <option value="single_image">Single Image</option>
                <option value="video">UAV Flight Video (.mp4)</option>
              </select>
            </div>

            <div className={styles.dropzone} onClick={() => document.getElementById("fileInput")?.click()}>
              <UploadCloud size={32} style={{ color: "var(--accent-primary)" }} />
              <div style={{ fontSize: "12px", fontWeight: 700, color: "var(--text-primary)", marginTop: "8px" }}>
                {ingestFile ? ingestFile.name : "Click to select or drop image / video file"}
              </div>
              <input
                id="fileInput"
                type="file"
                style={{ display: "none" }}
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) setIngestFile(e.target.files[0]);
                }}
              />
            </div>

            <div style={{ marginTop: "20px", display: "flex", justifyContent: "flex-end", gap: "12px" }}>
              <button className={styles.iconBtn} onClick={() => setShowIngestModal(false)}>
                Cancel
              </button>
              <button className={styles.actionBtn} onClick={handleIngest} disabled={!ingestFile || isIngesting}>
                {isIngesting ? "INGESTING MEDIA..." : "START SAMPLING"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
