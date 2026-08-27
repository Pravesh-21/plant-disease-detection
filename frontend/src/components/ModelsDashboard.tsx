import React, { useState, useMemo } from "react";
import {
  Cpu,
  Zap,
  HardDrive,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  Box,
  Layers,
  ArrowRight,
  RotateCcw,
  Activity,
  Sparkles,
  Search,
  Filter,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useModelRegistry, ChildModelInfo } from "@/hooks/useModelRegistry";
import { useGroqStatus } from "@/hooks/useGroqStatus";
import styles from "./ModelsDashboard.module.css";

const BACKEND =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "https://plant-disease-detection-sf8o.onrender.com";

// ── Child Model Card Component ────────────────────────────────────────────────
function ChildModelCard({
  child,
  onAwaken,
  awakening,
}: {
  child: ChildModelInfo;
  onAwaken: (crop: string) => Promise<void>;
  awakening: boolean;
}) {
  const loaded = child.is_loaded;
  const missing = !child.has_weights;
  const [showClasses, setShowClasses] = useState(false);

  return (
    <div
      className={`${styles.childCard} ${loaded ? styles.childCardLoaded : ""} ${
        missing ? styles.childCardMissing : ""
      }`}
    >
      <div className={styles.childCardTop}>
        <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
          <span className={styles.childName}>{child.display_name}</span>
          <span className={styles.childFilename}>
            {child.folder || `${child.display_name}_best_int8.onnx`}
          </span>
        </div>

        <span
          className={`${styles.childStatusChip} ${
            loaded ? styles.chipLoaded : missing ? styles.chipMissing : styles.chipStandby
          }`}
        >
          <span
            className={styles.statusDot}
            style={{
              background: loaded ? "#10B981" : missing ? "#EF4444" : "#FBBF24",
              boxShadow: loaded ? "0 0 8px #10B981" : "none",
            }}
          />
          {loaded ? "AWOKEN (IN MEMORY)" : missing ? "NO WEIGHTS" : "STANDBY"}
        </span>
      </div>

      <div className={styles.childMeta}>
        {loaded && child.task && (
          <span className={styles.metaChip} style={{ color: "#38BDF8" }}>
            TASK: {child.task.toUpperCase()}
          </span>
        )}
        {loaded && child.class_count != null && (
          <span className={styles.metaChip} style={{ color: "#34D399" }}>
            {child.class_count} CLASSES
          </span>
        )}
        {!loaded && child.has_weights && (
          <span className={styles.metaChip} style={{ color: "#FBBF24" }}>
            ON-DEMAND ONNX
          </span>
        )}
        {missing && (
          <span className={styles.metaChip} style={{ color: "#FCA5A5" }}>
            WEIGHTS MISSING
          </span>
        )}
      </div>

      {loaded && child.class_names && child.class_names.length > 0 && (
        <div style={{ marginTop: "4px" }}>
          <button
            className={styles.toggleClassBtn}
            onClick={() => setShowClasses(!showClasses)}
          >
            {showClasses ? (
              <>
                <ChevronUp size={12} /> HIDE DISEASE CLASSES
              </>
            ) : (
              <>
                <ChevronDown size={12} /> SHOW {child.class_names.length} TRAINED CLASSES
              </>
            )}
          </button>

          {showClasses && (
            <div className={styles.classBadgesList}>
              {child.class_names.map((c, i) => (
                <span key={i} className={styles.classPill}>
                  {c}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      <div className={styles.childCardActions}>
        {!loaded && child.has_weights ? (
          <button
            className={styles.awakenBtn}
            onClick={() => onAwaken(child.display_name)}
            disabled={awakening}
            title="Load this specialist model into memory on-demand"
          >
            <Zap size={12} />
            {awakening ? "AWAKENING MODEL..." : "AWAKEN MODEL (ON-DEMAND)"}
          </button>
        ) : loaded ? (
          <div className={styles.awokenBadge}>
            <CheckCircle2 size={13} color="#10B981" />
            <span>AWOKEN &amp; ACTIVE IN INFERENCE PIPELINE</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ── Main Models Dashboard ──────────────────────────────────────────────────────
export default function ModelsDashboard() {
  const registry = useModelRegistry();
  const groqStatus = useGroqStatus();
  const { parent, children, total_available, total_loaded, loading, error, refresh } = registry;
  
  const [awakeningCrop, setAwakeningCrop] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);
  
  // Search & Filter States
  const [searchQuery, setSearchQuery] = useState("");
  const [statusTab, setStatusTab] = useState<"all" | "loaded" | "standby">("all");

  const handleAwaken = async (cropName: string) => {
    setAwakeningCrop(cropName);
    try {
      const res = await fetch(`${BACKEND}/api/inference/awaken/${encodeURIComponent(cropName)}`, {
        method: "POST",
      });
      if (res.ok) refresh();
    } catch {
      /* silent */
    } finally {
      setAwakeningCrop(null);
    }
  };

  const handleUnloadAll = async () => {
    setResetting(true);
    try {
      const res = await fetch(`${BACKEND}/api/inference/unload-children`, { method: "POST" });
      if (res.ok) refresh();
    } catch {
      /* silent */
    } finally {
      setResetting(false);
    }
  };

  // Filtered children models list
  const filteredChildren = useMemo(() => {
    return children.filter((child) => {
      const matchesSearch =
        child.display_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (child.folder && child.folder.toLowerCase().includes(searchQuery.toLowerCase()));

      if (!matchesSearch) return false;

      if (statusTab === "loaded") return child.is_loaded;
      if (statusTab === "standby") return !child.is_loaded && child.has_weights;
      return true;
    });
  }, [children, searchQuery, statusTab]);

  if (loading) {
    return (
      <div className={styles.modelsLayout}>
        <div className={styles.modelsLoading}>
          <div className={styles.loadingPulse} />
          <span>LOADING MODEL REGISTRY…</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.modelsLayout}>
        <div className={styles.modelsError}>
          <AlertTriangle size={24} />
          <span>COULD NOT REACH BACKEND MODEL REGISTRY</span>
          <button className={styles.refreshBtn} onClick={refresh}>
            <RefreshCw size={11} /> RETRY
          </button>
        </div>
      </div>
    );
  }

  const isGpu = parent.device?.includes("cuda") || parent.device?.includes("mps");

  return (
    <div className={styles.modelsLayout}>

      {/* ── Architecture Pipeline Banner ── */}
      <div className={styles.archBanner}>
        <div className={styles.archTitleRow}>
          <Activity size={16} color="#38BDF8" />
          <span className={styles.archTitle}>TWO-PHASE DYNAMIC MEMORY PIPELINE</span>
        </div>
        <div className={styles.archFlow}>
          <div className={styles.flowNode}>
            <span className={styles.nodeStep}>PHASE 1</span>
            <span className={styles.nodeTitle}>Parent Model</span>
            <span className={styles.nodeSub}>Classifies crop species</span>
          </div>
          <ArrowRight size={16} className={styles.flowArrow} />
          <div className={styles.flowNode}>
            <span className={styles.nodeStep}>DYNAMIC ROUTER</span>
            <span className={styles.nodeTitle}>Crop Trigger</span>
            <span className={styles.nodeSub}>Wakes matching model</span>
          </div>
          <ArrowRight size={16} className={styles.flowArrow} />
          <div className={styles.flowNode}>
            <span className={styles.nodeStep}>PHASE 2</span>
            <span className={styles.nodeTitle}>Child Specialist</span>
            <span className={styles.nodeSub}>Awoken into memory</span>
          </div>
          <ArrowRight size={16} className={styles.flowArrow} />
          <div className={styles.flowNode}>
            <span className={styles.nodeStep}>OUTPUT</span>
            <span className={styles.nodeTitle}>Disease Bounding Boxes</span>
            <span className={styles.nodeSub}>Exact localization &amp; remedy</span>
          </div>
        </div>
      </div>

      {/* ── Parent Model Card ── */}
      <div className={styles.parentCard}>
        <div className={styles.parentHeader}>
          <div className={styles.parentTitle}>
            <Cpu size={18} color="#38BDF8" />
            <div>
              <div className={styles.parentTitleText}>PARENT MODEL (PRIMARY CLASSIFIER)</div>
              <div className={styles.parentSubtext}>
                Continuously evaluates drone imagery · Identifies crop species to awaken specialist child models
              </div>
            </div>
          </div>
          <span
            className={`${styles.statusBadge} ${
              parent.ready
                ? parent.mock_mode ? styles.statusMock : styles.statusOnline
                : styles.statusOffline
            }`}
          >
            <span
              className={styles.statusDot}
              style={{
                background: parent.ready
                  ? parent.mock_mode ? "#FBBF24" : "#10B981"
                  : "#EF4444",
                boxShadow: parent.ready && !parent.mock_mode ? "0 0 8px #10B981" : "none",
              }}
            />
            {parent.ready
              ? parent.mock_mode ? "MOCK MODE" : "OPERATIONAL"
              : "OFFLINE"}
          </span>
        </div>

        <div className={styles.parentFields}>
          <div className={styles.fieldCard}>
            <span className={styles.fieldLabel}>Ensemble</span>
            <span className={styles.fieldValue}>{parent.model_name}</span>
          </div>
          <div className={styles.fieldCard}>
            <span className={styles.fieldLabel}>Task</span>
            <span className={styles.fieldValue} style={{ color: "#38BDF8" }}>
              {parent.model_task?.toUpperCase() || "CLASSIFY"}
            </span>
          </div>
          <div className={styles.fieldCard}>
            <span className={styles.fieldLabel}>Device</span>
            <span className={styles.fieldValue} style={{ color: isGpu ? "#10B981" : "#F59E0B" }}>
              {parent.device?.toUpperCase() || "CPU"}
            </span>
          </div>
          <div className={styles.fieldCard}>
            <span className={styles.fieldLabel}>Loaded Children</span>
            <span className={styles.fieldValue} style={{ color: "#34D399" }}>
              {total_loaded} / {total_available}
            </span>
          </div>
        </div>
      </div>

      {/* ── SEARCH & FILTER CONTROLS BAR ── */}
      <div className={styles.controlsBar}>
        <div className={styles.searchWrap}>
          <Search size={14} color="#64748B" />
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Search 35+ child models by crop or filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button className={styles.clearSearchBtn} onClick={() => setSearchQuery("")}>
              ✕
            </button>
          )}
        </div>

        <div className={styles.filterTabs}>
          <button
            className={`${styles.filterTab} ${statusTab === "all" ? styles.filterTabActive : ""}`}
            onClick={() => setStatusTab("all")}
          >
            ALL MODELS ({children.length})
          </button>
          <button
            className={`${styles.filterTab} ${statusTab === "loaded" ? styles.filterTabActive : ""}`}
            onClick={() => setStatusTab("loaded")}
          >
            AWOKEN IN MEMORY ({total_loaded})
          </button>
          <button
            className={`${styles.filterTab} ${statusTab === "standby" ? styles.filterTabActive : ""}`}
            onClick={() => setStatusTab("standby")}
          >
            STANDBY ({total_available - total_loaded})
          </button>
        </div>

        <div className={styles.actionGroup}>
          {total_loaded > 0 && (
            <button
              className={styles.resetBtn}
              onClick={handleUnloadAll}
              disabled={resetting}
              title="Reset all child models back to Standby"
            >
              <RotateCcw size={11} /> {resetting ? "RESETTING..." : "RESET ALL TO STANDBY"}
            </button>
          )}
          <button className={styles.refreshBtn} onClick={refresh}>
            <RefreshCw size={11} /> REFRESH
          </button>
        </div>
      </div>

      {/* ── Child Models Grid ── */}
      <div className={styles.childrenGrid}>
        {filteredChildren.map((child) => (
          <ChildModelCard
            key={child.folder}
            child={child}
            onAwaken={handleAwaken}
            awakening={awakeningCrop === child.display_name}
          />
        ))}
        {filteredChildren.length === 0 && (
          <div className={styles.modelsLoading}>
            <Box size={28} color="var(--text-muted)" />
            <span>No specialist models match current search or filter query.</span>
          </div>
        )}
      </div>

    </div>
  );
}
