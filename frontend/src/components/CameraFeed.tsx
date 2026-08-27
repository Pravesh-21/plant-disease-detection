"use client";
import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  MapPin, Crosshair, Maximize2, Camera,
  Upload, Link2, RefreshCw, Play, Pause, Zap,
  Video, VideoOff, ShieldAlert, Radio, Activity, Eye
} from "lucide-react";
import { InputMode } from "./InputModeSelector";
import { RawDetection } from "@/hooks/useDetections";
import styles from "./CameraFeed.module.css";

const BACKEND = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "https://plant-disease-detection-sf8o.onrender.com";

interface Props {
  lat: number;
  lon: number;
  mode: InputMode;
  modelReady: boolean;
  /** Called with raw detection results from the backend inference endpoint */
  onDetections: (raws: RawDetection[]) => void;
  scanInterval?: number;
  /** Called when the live camera active state changes (live mode only) */
  onCameraActive?: (active: boolean) => void;
}

// Helper: post image blob to backend inference endpoint
async function runInferenceOnBlob(
  blob: Blob,
  endpoint: "infer/image" | "infer/video-frame",
  filename = "frame.jpg"
): Promise<RawDetection[]> {
  const fd = new FormData();
  fd.append("file", blob, filename);
  const res = await fetch(`${BACKEND}/api/inference/${endpoint}`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return (data.detections ?? []) as RawDetection[];
}

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE MODE
// ─────────────────────────────────────────────────────────────────────────────
function ImageMode({
  modelReady,
  onDetections,
}: {
  modelReady: boolean;
  onDetections: (raws: RawDetection[]) => void;
}) {
  const [preview,  setPreview]  = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [running,  setRunning]  = useState(false);
  const [runningPhase, setRunningPhase] = useState<"phase1" | "phase2" | null>(null);
  const [result, setResult] = useState<{
    count: number;
    cls: string[];
    crop?: string;
    childModel?: string;
  } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    setPreview(URL.createObjectURL(file));
    setResult(null);
    if (!modelReady) return;
    setRunning(true);
    setRunningPhase("phase1");
    try {
      // Simulate quick phase sequence for clear UI demonstration
      setTimeout(() => setRunningPhase("phase2"), 350);
      const dets = await runInferenceOnBlob(file, "infer/image", file.name);
      onDetections(dets);
      const firstDet = dets[0];
      setResult({
        count: dets.length,
        cls: [...new Set(dets.map(d => d.detected_class))],
        crop: firstDet?.plant_class,
        childModel: firstDet?.model_name,
      });
    } catch {
      setResult({ count: -1, cls: [] });
    } finally {
      setRunning(false);
      setRunningPhase(null);
    }
  };

  return (
    <div
      className={`${styles.dropZone} ${dragging ? styles.dragOver : ""}`}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        e.preventDefault(); setDragging(false);
        const f = e.dataTransfer.files[0]; if (f) handleFile(f);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef} type="file" accept="image/*" hidden
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
      />

      {preview ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={preview} alt="preview" className={styles.preview} />
      ) : (
        <div className={styles.dropPlaceholder}>
          <Upload size={32} color="#38BDF8" strokeWidth={1.4} />
          <span className={styles.dropTitle}>DROP SURVEY PAYLOAD IMAGE HERE</span>
          <span className={styles.dropSub}>or click to browse · PNG, JPG, WEBP</span>
        </div>
      )}

      {running && (
        <div className={styles.analysisOverlay}>
          <div className={styles.analysisSpin} />
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
            <span style={{ fontWeight: 800, color: "#38BDF8", letterSpacing: "0.04em" }}>
              {runningPhase === "phase1"
                ? "PHASE 1: PARENT MODEL IDENTIFYING CROP SPECIES…"
                : "PHASE 2: AWAKENING SPECIALIST CHILD MODEL…"}
            </span>
            <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
              ParentModel.pt ➔ Dynamic Crop Router ➔ Specialist Child Model
            </span>
          </div>
        </div>
      )}

      {result !== null && (
        <div
          className={styles.resultChip}
          style={{ background: result.count < 0 ? "rgba(239,68,68,0.15)" : "#151C2A", maxWidth: "90%", textAlign: "center" }}
          onClick={e => e.stopPropagation()}
        >
          {result.count < 0
            ? "⚠ Could not reach backend"
            : result.count === 0
              ? "ℹ No leaf detected (<50% confidence) or non-crop frame"
              : result.crop
                ? `✓ Phase 1: Identified Crop '${result.crop}' ➔ Phase 2: Awoke '${result.childModel}' ➔ ${result.count} Detection(s): ${result.cls.join(", ")}`
                : `✓ ${result.count} detection(s): ${result.cls.join(", ")}`
          }
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// VIDEO MODE
// ─────────────────────────────────────────────────────────────────────────────
function VideoMode({
  modelReady,
  onDetections,
  scanInterval = 1,
}: {
  modelReady: boolean;
  onDetections: (raws: RawDetection[]) => void;
  scanInterval?: number;
}) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const timerRef   = useRef<ReturnType<typeof setInterval> | null>(null);
  const [src,       setSrc]       = useState<string | null>(null);
  const [scanning,  setScanning]  = useState(false);
  const [dragging,  setDragging]  = useState(false);
  const [scanCount, setScanCount] = useState(0);
  const [scanTotal, setScanTotal] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const lastCapturedTimeRef = useRef<number>(-1);

  const [videoDets,  setVideoDets]  = useState<RawDetection[]>([]);

  const handleFile = (file: File) => {
    setSrc(URL.createObjectURL(file));
    setScanCount(0);
    setScanTotal(0);
    setVideoDets([]);
    lastCapturedTimeRef.current = -1;
    if (modelReady) {
      setScanning(true);
    }
  };

  const captureAndInfer = useCallback(async () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.ended) return;

    if (video.paused && video.currentTime === lastCapturedTimeRef.current) return;

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 360;
    canvas.getContext("2d")?.drawImage(video, 0, 0);

    lastCapturedTimeRef.current = video.currentTime;

    canvas.toBlob(async blob => {
      if (!blob) return;
      try {
        const dets = await runInferenceOnBlob(blob, "infer/video-frame");
        onDetections(dets);
        setVideoDets(dets);
        setScanCount(c => c + 1);
        setScanTotal(t => t + dets.length);
      } catch { /* silent */ }
    }, "image/jpeg", 0.85);
  }, [onDetections]);

  useEffect(() => {
    if (scanning && src && modelReady) {
      if (videoRef.current?.paused) {
        videoRef.current.play().catch(() => {});
      }
      captureAndInfer();
      timerRef.current = setInterval(captureAndInfer, scanInterval * 1000);
    } else {
      if (videoRef.current && !videoRef.current.paused) {
        videoRef.current.pause();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [scanning, src, modelReady, scanInterval, captureAndInfer]);

  useEffect(() => {
    if (modelReady && src) {
      setScanning(true);
    }
  }, [modelReady, src]);

  const toggleScan = () => setScanning(s => !s);
  const clearVideo = () => { setSrc(null); setScanning(false); };

  return (
    <div
      className={`${styles.videoMode} ${!src ? styles.dropZone : ""} ${dragging ? styles.dragOver : ""}`}
      onDragOver={e => {
        if (src) return;
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => {
        if (src) return;
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files[0];
        if (f) handleFile(f);
      }}
      onClick={() => {
        if (!src) inputRef.current?.click();
      }}
    >
      <input ref={inputRef} type="file" accept="video/*" hidden
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
      <canvas ref={canvasRef} hidden />

      {!src ? (
        <div className={styles.dropPlaceholder}>
          <Upload size={32} color="#38BDF8" strokeWidth={1.4} />
          <span className={styles.dropTitle}>DROP VIDEO SURVEY FILE HERE</span>
          <span className={styles.dropSub}>or click to browse · MP4, MOV, AVI, WEBM</span>
        </div>
      ) : (
        <div className={styles.videoPlayerContainer} onClick={e => e.stopPropagation()}>
          <video ref={videoRef} src={src} controls className={styles.videoPlayer} />
          
          <div className={styles.videoControls}>
            <button
              className={`${styles.scanBtn} ${scanning ? styles.scanActive : ""}`}
              onClick={toggleScan}
              disabled={!modelReady}
            >
              {scanning ? <Pause size={11} /> : <Play size={11} />}
              {scanning ? "STOP AI SCAN" : "START AI SCAN"}
            </button>
            {scanCount > 0 && (
              <span className={styles.scanCount}>
                {scanCount} FRAMES · {scanTotal} DETECTIONS
              </span>
            )}
            <button className={styles.clearBtn} onClick={clearVideo}>
              <RefreshCw size={10} /> CLEAR
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LIVE UAV MODE (Professional UAV HUD & Live Detection Overlays)
// ─────────────────────────────────────────────────────────────────────────────
function LiveUAVMode({
  lat, lon, modelReady, onDetections, onCameraActive, scanInterval = 1,
}: {
  lat: number; lon: number;
  modelReady: boolean;
  onDetections: (raws: RawDetection[]) => void;
  onCameraActive?: (active: boolean) => void;
  scanInterval?: number;
}) {
  const videoRef   = useRef<HTMLVideoElement>(null);
  const canvasRef  = useRef<HTMLCanvasElement>(null);
  const wrapRef    = useRef<HTMLDivElement>(null);
  const captureRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [camStatus,   setCamStatus]   = useState<"requesting"|"active"|"unavailable"|"closed">("requesting");
  const [camErrorMsg, setCamErrorMsg] = useState<string | null>(null);
  const [camOn,       setCamOn]       = useState(true);
  const [retryCount,  setRetryCount]  = useState(0);
  const [rtspUrl,     setRtspUrl]     = useState("");
  const [showRtsp,    setShowRtsp]    = useState(false);
  const [autoCapture, setAutoCapture] = useState(true);
  const [lastResult,  setLastResult]  = useState<string | null>(null);
  const isInferringRef = useRef(false);
  const [currentDets, setCurrentDets] = useState<RawDetection[]>([]);

  const cameraIsLive = camStatus === "active";

  useEffect(() => {
    onCameraActive?.(cameraIsLive);
  }, [cameraIsLive, onCameraActive]);

  const startStream = useCallback(async () => {
    if (!camOn) {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
        videoRef.current.srcObject = null;
      }
      setCamStatus("closed");
      setAutoCapture(false);
      setLastResult(null);
      setCurrentDets([]);
      return;
    }

    setCamStatus("requesting");
    setCamErrorMsg(null);

    let stream: MediaStream | null = null;
    let lastError: any = null;

    if (typeof navigator !== "undefined" && navigator.mediaDevices?.getUserMedia) {
      const constraintsList: MediaStreamConstraints[] = [
        { video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" } },
        { video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: true },
      ];

      for (const constraints of constraintsList) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (stream) break;
        } catch (err: any) {
          lastError = err;
        }
      }
    } else {
      lastError = new Error("MediaDevices API not supported or insecure context");
    }
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(console.error);
      };
      videoRef.current.play().catch(console.error);
      setCamStatus("active");
      return () => stream?.getTracks().forEach(t => t.stop());
    } else if (stream) {
      setCamStatus("active");
      return () => stream?.getTracks().forEach(t => t.stop());
    } else {
      setCamStatus("unavailable");
      setCamErrorMsg(lastError?.message || "Camera hardware unavailable or permission denied");
      return;
    }
  }, [camOn]);

  useEffect(() => {
    let active = true;
    let cleanupFn: (() => void) | undefined;

    startStream().then(c => {
      if (!active) {
        c?.();
      } else {
        cleanupFn = c;
      }
    });

    return () => {
      active = false;
      cleanupFn?.();
    };
  }, [camOn, retryCount, startStream]);

  const handleRetry = () => {
    setCamOn(true);
    setRetryCount(c => c + 1);
  };

  const captureFrame = useCallback(async () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!camOn || camStatus !== "active" || !canvas || !modelReady || !video) return;
    if (isInferringRef.current) return;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 360;
    const scale = Math.min(1.0, 512 / Math.max(w, h, 1));
    canvas.width  = Math.max(64, Math.round(w * scale));
    canvas.height = Math.max(64, Math.round(h * scale));
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async blob => {
      if (!blob) return;
      isInferringRef.current = true;
      try {
        const dets = await runInferenceOnBlob(blob, "infer/video-frame");
        onDetections(dets);
        setCurrentDets(dets);
        setLastResult(
          dets.length > 0
            ? `${dets.length} detection(s): ${[...new Set(dets.map(d => d.detected_class))].join(", ")}`
            : "✓ Frame scanned — no disease detected"
        );
      } catch { 
        setLastResult("Backend inference endpoint offline"); 
      } finally {
        isInferringRef.current = false;
      }
    }, "image/jpeg", 0.80);
  }, [camOn, camStatus, modelReady, onDetections]);

  useEffect(() => {
    if (autoCapture && cameraIsLive && modelReady) {
      captureFrame();
      captureRef.current = setInterval(captureFrame, scanInterval * 1000);
    } else {
      if (captureRef.current) {
        clearInterval(captureRef.current);
        captureRef.current = null;
      }
    }

    return () => {
      if (captureRef.current) {
        clearInterval(captureRef.current);
        captureRef.current = null;
      }
    };
  }, [autoCapture, cameraIsLive, modelReady, scanInterval, captureFrame]);

  // Continuous scanning active whenever camera is live and model is ready
  useEffect(() => {
    if (cameraIsLive && modelReady) {
      setAutoCapture(true);
    } else {
      setAutoCapture(false);
    }
  }, [cameraIsLive, modelReady]);

  const toggleAutoCapture = () => setAutoCapture(v => !v);
  const toggleFullscreen  = () => {
    if (!document.fullscreenElement && wrapRef.current) wrapRef.current.requestFullscreen();
    else document.exitFullscreen();
  };

  return (
    <div className={styles.wrap} ref={wrapRef}>
      <canvas ref={canvasRef} hidden />

      {/* Full-Viewport Webcam Video Screen Ripple Overlay */}
      <div className={styles.fullViewportRippleWrap}>
        <div className={styles.fullViewportRippleRing} />
        <div className={styles.fullViewportRippleRing} />
        <div className={styles.fullViewportRippleRing} />
        <div className={styles.fullViewportRippleRing} />
      </div>

      {/* Persistent Video Element to always capture media stream */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={styles.video}
        style={{ display: cameraIsLive ? "block" : "none" }}
      />

      {/* Placeholder shown during requesting / standby / unavailable */}
      {!cameraIsLive && (
        <div className={styles.placeholder}>
          <div className={styles.noSignalText}>
            {camStatus === "requesting" && (
              <><div className={styles.spinner} /><span>ESTABLISHING HIGH-DEFINITION UAV LINK…</span></>
            )}
            {camStatus === "closed" && (
              <>
                <VideoOff size={36} color="#38BDF8" />
                <span style={{ color: "#F8FAFC", letterSpacing: "0.04em", fontSize: "15px" }}>CAMERA STREAM STANDBY</span>
                <span className={styles.noSigSub}>UAV optical sensor array is shut down · AI detection is paused</span>
                <button className={styles.startCamBtn} onClick={handleRetry}>
                  <Video size={14} /> INITIALIZE LIVE UAV STREAM
                </button>
              </>
            )}
            {camStatus === "unavailable" && (
              <>
                <VideoOff size={36} color="#EF4444" />
                <span style={{ color: "#F8FAFC", letterSpacing: "0.04em", fontSize: "15px" }}>CAMERA HARDWARE UNREACHABLE</span>
                <span className={styles.noSigSub}>{camErrorMsg || "No camera permission or video device was detected"}</span>
                <button className={styles.startCamBtn} onClick={handleRetry}>
                  <RefreshCw size={14} /> RETRY STREAM CONNECTION
                </button>
              </>
            )}
          </div>
        </div>
      )}


      {/* RTSP Config Panel */}
      {showRtsp && (
        <div className={styles.rtspPanel}>
          <Link2 size={12} color="#38BDF8" />
          <input
            className={styles.rtspInput}
            placeholder="rtsp://192.168.1.120:554/live"
            value={rtspUrl}
            onChange={e => setRtspUrl(e.target.value)}
          />
          <button className={styles.rtspConnect}>CONNECT RTSP</button>
        </div>
      )}

      {/* Last detection result badge */}
      {lastResult && cameraIsLive && (
        <div className={styles.liveResultBadge}>
          <Zap size={11} color="#38BDF8" />
          {lastResult}
        </div>
      )}

      {/* Scan line */}
      <div className={styles.scanLine} />

      {/* Corner brackets */}
      <div className={`${styles.corner} ${styles.tl}`} />
      <div className={`${styles.corner} ${styles.tr}`} />
      <div className={`${styles.corner} ${styles.bl}`} />
      <div className={`${styles.corner} ${styles.br}`} />

      {/* Center crosshair */}
      {cameraIsLive && (
        <div className={styles.crosshair}>
          <Crosshair size={32} color="rgba(56,189,248,0.4)" strokeWidth={0.8} />
        </div>
      )}

      {/* Top HUD */}
      <div className={styles.topHud}>
        <div className={styles.hudChip}>
          <span className={styles.hudLabel}>STREAM</span>
          <span className={styles.recDot} style={{ background: cameraIsLive ? "#10B981" : "#64748B" }} />
          <span className={styles.hudValue} style={{ color: cameraIsLive ? "#10B981" : "#EF4444" }}>
            {cameraIsLive ? "1080p @ 60 FPS" : "STANDBY"}
          </span>
        </div>
        <div className={styles.hudChip}>
          <span className={styles.hudLabel}>CAM</span>
          <span className={styles.hudValue} style={{ color: cameraIsLive ? "#38BDF8" : "#EF4444" }}>
            {cameraIsLive ? "LIVE STREAM" : "OFFLINE"}
          </span>
        </div>
        <div className={styles.hudChip}>
          <span className={styles.hudLabel}>AI INFERENCE</span>
          <span className={styles.hudValue} style={{ color: autoCapture && cameraIsLive ? "#10B981" : "rgba(255,255,255,0.45)" }}>
            {autoCapture && cameraIsLive ? "CONTINUOUS SCANNING" : cameraIsLive ? "READY" : "PAUSED"}
          </span>
        </div>
        <div className={styles.spacer} />

        {/* AI scan toggle */}
        {modelReady && cameraIsLive && (
          <button
            className={styles.fsBtn}
            onClick={toggleAutoCapture}
            title={autoCapture ? "Stop AI frame capture" : "Start AI frame capture"}
            style={{ borderColor: autoCapture ? "#38BDF8" : undefined, background: autoCapture ? "rgba(56,189,248,0.12)" : undefined }}
          >
            <Zap size={13} color={autoCapture ? "#38BDF8" : "rgba(255,255,255,0.45)"} />
          </button>
        )}

        {/* Snapshot trigger */}
        {modelReady && cameraIsLive && (
          <button className={styles.fsBtn} onClick={captureFrame} title="Take snapshot & run AI classification">
            <Camera size={13} color="#38BDF8" />
          </button>
        )}

        <button className={styles.fsBtn} onClick={() => setShowRtsp(v => !v)} title="Connect Drone RTSP Stream">
          <Link2 size={13} color="rgba(255,255,255,0.45)" />
        </button>

        {/* Camera Power Toggle */}
        <button
          className={styles.fsBtn}
          onClick={() => setCamOn(v => !v)}
          title={camOn ? "Close Camera Stream" : "Open Camera Stream"}
          style={{
            borderColor: !camOn ? "rgba(239,68,68,0.4)" : undefined,
            background:  !camOn ? "rgba(239,68,68,0.12)" : undefined
          }}
        >
          {camOn && cameraIsLive ? (
            <Video size={13} color="#38BDF8" />
          ) : (
            <VideoOff size={13} color="#EF4444" />
          )}
        </button>

        <button className={styles.fsBtn} onClick={toggleFullscreen} title="Fullscreen Viewport">
          <Maximize2 size={13} color="rgba(255,255,255,0.45)" />
        </button>
      </div>

      {/* Bottom HUD Telemetry */}
      <div className={styles.bottomHud}>
        <div className={styles.hudTelRow}>
          <div className={`${styles.telBlock} ${styles.gpsTel}`}>
            <MapPin size={11} color="#38BDF8" />
            <span className={styles.hudLabel}>TARGET GPS</span>
            <span className={styles.gpsVal}>{lat.toFixed(4)}°N &nbsp;{lon.toFixed(4)}°E</span>
          </div>
          <div className={styles.telBlock}>
            <span className={styles.hudLabel}>SIGNAL LINK: <strong style={{ color: "#10B981" }}>98% (5.8 GHz)</strong></span>
          </div>
          <div className={styles.telBlock}>
            <span className={styles.hudLabel}>GIMBAL: <strong style={{ color: "#F8FAFC" }}>PITCH -30° | ROLL 0°</strong></span>
          </div>
        </div>
      </div>

      <div className={styles.gridOverlay} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
export default function CameraFeed({ lat, lon, mode, modelReady, onDetections, onCameraActive, scanInterval }: Props) {
  return (
    <div className={styles.container}>
      {mode === "image" && (
        <ImageMode modelReady={modelReady} onDetections={onDetections} />
      )}
      {mode === "video" && (
        <VideoMode modelReady={modelReady} onDetections={onDetections} scanInterval={scanInterval} />
      )}
      {mode === "live" && (
        <LiveUAVMode
          lat={lat} lon={lon}
          modelReady={modelReady}
          onDetections={onDetections}
          onCameraActive={onCameraActive}
          scanInterval={scanInterval}
        />
      )}

      {/* 2x2 Grid divider overlay */}
      <div className={styles.viewportGridOverlay}>
        <div className={styles.gridLineH} />
        <div className={styles.gridLineV} />
        <span className={`${styles.gridLabel} ${styles.glTl}`}>QUAD A1</span>
        <span className={`${styles.gridLabel} ${styles.glTr}`}>QUAD A2</span>
        <span className={`${styles.gridLabel} ${styles.glBl}`}>QUAD C1</span>
        <span className={`${styles.gridLabel} ${styles.glBr}`}>QUAD C2</span>
      </div>
    </div>
  );
}
