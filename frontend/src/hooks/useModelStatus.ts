"use client";
import { useState, useEffect, useRef } from "react";

export type ModelStatus = {
  ready: boolean;
  mock_mode: boolean;
  model_name: string;
  loaded_at: string | null;
  torch_available: boolean;
  loading: boolean;   // true while the first fetch hasn't resolved yet
  error: boolean;     // true if backend is unreachable
  device?: string;
  model_task?: string;
};

const POLL_INTERVAL_MS = 2000;
const BACKEND_BASE = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "https://plant-disease-detection-sf8o.onrender.com";

const DEFAULT: ModelStatus = {
  ready: false,
  mock_mode: false,
  model_name: "ParentModel.pt",
  loaded_at: null,
  torch_available: false,
  loading: true,
  error: false,
};

export function useModelStatus(): ModelStatus {
  const [status, setStatus] = useState<ModelStatus>(DEFAULT);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = async () => {
    try {
      const res = await fetch(`${BACKEND_BASE}/api/inference/model-status`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setStatus({ ...data, loading: false, error: false });

      // Once ready, stop polling
      if (data.ready && timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    } catch {
      // Backend not reachable — keep polling but surface error state
      setStatus(prev => ({ ...prev, loading: false, error: true }));
    }
  };

  useEffect(() => {
    // Immediate first poll
    poll();
    timerRef.current = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return status;
}
