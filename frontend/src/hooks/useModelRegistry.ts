"use client";
import { useState, useEffect, useCallback } from "react";

const BACKEND = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || "https://plant-disease-detection-sf8o.onrender.com";

export interface ChildModelInfo {
  folder: string;
  display_name: string;
  has_weights: boolean;
  weights_path: string | null;
  is_loaded: boolean;
  task: string | null;
  class_count: number | null;
  class_names: string[] | null;
}

export interface ParentModelInfo {
  ready: boolean;
  mock_mode: boolean;
  model_name: string;
  model_task: string;
  device: string;
  loaded_at: string | null;
  torch_available: boolean;
  loaded_child_models: string[];
  parent_models?: { name: string; classes: number }[];
}

export interface ModelRegistryData {
  parent: ParentModelInfo;
  children: ChildModelInfo[];
  child_models_dir: string;
  total_available: number;
  total_loaded: number;
  loading: boolean;
  error: boolean;
}

const DEFAULT: ModelRegistryData = {
  parent: {
    ready: false,
    mock_mode: false,
    model_name: "ParentModel.pt",
    model_task: "unknown",
    device: "cpu",
    loaded_at: null,
    torch_available: false,
    loaded_child_models: [],
  },
  children: [],
  child_models_dir: "",
  total_available: 0,
  total_loaded: 0,
  loading: true,
  error: false,
};

export function useModelRegistry(): ModelRegistryData & { refresh: () => void } {
  const [data, setData] = useState<ModelRegistryData>(DEFAULT);

  const fetchRegistry = useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND}/api/inference/model-registry`, {
        cache: "no-store",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData({ ...json, loading: false, error: false });
    } catch {
      setData(prev => ({ ...prev, loading: false, error: true }));
    }
  }, []);

  useEffect(() => {
    fetchRegistry();
    // Poll every 8 seconds to pick up newly loaded child models
    const t = setInterval(fetchRegistry, 8000);
    return () => clearInterval(t);
  }, [fetchRegistry]);

  return { ...data, refresh: fetchRegistry };
}
