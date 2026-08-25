"""
onnx_engine.py — Self-contained ONNX inference engine for the child microservice.
Handles model loading from HF Hub, preprocessing, detection + classification inference.
No PyTorch required.
"""
import os
import ast
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

logger = logging.getLogger("onnx_engine")

HF_CHILD_REPO = os.getenv("HF_CHILD_REPO", "BeRam-Plant-Disease/Child-Models")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Maps normalized crop name → HF child model filename
_CHILD_HF_FILES: Dict[str, str] = {
    "apple": "Apple_best_int8.onnx",
    "banana": "Banana_best_int8.onnx",
    "bittergourd": "Bitter_Gourd_best_int8.onnx",
    "brinjal": "Brinjal_best_int8.onnx",
    "cashew": "Cashew_best_int8.onnx",
    "cassava": "Cassava_best_int8.onnx",
    "cauliflower": "Cauliflower_best_int8.onnx",
    "cherry": "Cherry_best_int8.onnx",
    "coconut": "Coconut_best_int8.onnx",
    "coffee": "Coffee_best_int8.onnx",
    "coriander": "Coriander_best_int8.onnx",
    "corn": "Corn_best_int8.onnx",
    "grape": "Grape_best_int8.onnx",
    "groundnut": "Groundnut_best_int8.onnx",
    "guava": "Guava_best_int8.onnx",
    "jackfruit": "Jackfruit_best_int8.onnx",
    "juniper": "Juniper_best_int8.onnx",
    "lemon": "Lemon_best_int8.onnx",
    "mango": "Mango_best_int8.onnx",
    "neem": "Neem_best_int8.onnx",
    "papaya": "Papaya_best_int8.onnx",
    "peach": "Peach_best_int8.onnx",
    "pepperbell": "Pepper_Bell_best_int8.onnx",
    "potato": "Potato_best_int8.onnx",
    "pumkin": "Pumkin_best_int8.onnx",
    "rice": "Rice_best_int8.onnx",
    "rose": "Rose_best_int8.onnx",
    "sesame": "Sesame_best_int8.onnx",
    "soyabean": "SoyaBean_best_int8.onnx",
    "strawberry": "Strawberry_best_int8.onnx",
    "sugarcane": "SugarCane_best_int8.onnx",
    "sunflower": "Sunflower_best_int8.onnx",
    "tobacco": "Tobacco_best_int8.onnx",
    "tomato": "Tomato_best_int8.onnx",
    "wheat": "Wheat_best_int8.onnx",
}

CROP_ALIASES: Dict[str, str] = {
    "eggplant": "brinjal", "aubergine": "brinjal",
    "soybean": "soyabean", "soya": "soyabean", "soy": "soyabean",
    "pumpkin": "pumkin",
    "peanut": "groundnut",
    "pepper": "pepperbell", "bellpepper": "pepperbell", "capsicum": "pepperbell",
    "sugarcane": "sugarcane", "sugar_cane": "sugarcane",
    "bittergourd": "bittergourd", "bitter_gourd": "bittergourd",
}


def _normalize_crop(name: str) -> str:
    raw = name.lower().replace("_", "").replace(" ", "").replace("-", "")
    return CROP_ALIASES.get(raw, raw)


def _resolve_hf_filename(crop_name: str) -> Optional[str]:
    norm = _normalize_crop(crop_name)
    return _CHILD_HF_FILES.get(norm)


def _download_model(hf_filename: str) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_CHILD_REPO,
            filename=hf_filename,
            token=HF_TOKEN or None,
        )
        logger.info(f"[HF Hub] Downloaded {hf_filename} -> {path}")
        return path
    except Exception as exc:
        logger.error(f"[HF Hub] Failed to download {hf_filename}: {exc}")
        return None


def _load_onnx_metadata(model_path: str) -> Dict[str, Any]:
    try:
        import onnx
        model = onnx.load(model_path)
        meta = {p.key: p.value for p in model.metadata_props}
        names_str = meta.get("names", "{}")
        try:
            names = ast.literal_eval(names_str)
        except Exception:
            names = {}
        imgsz_str = meta.get("imgsz", "[640, 640]")
        try:
            imgsz = ast.literal_eval(imgsz_str)
        except Exception:
            imgsz = [640, 640]
        task = meta.get("task", "detect")
        return {"names": names, "task": task, "imgsz": imgsz if isinstance(imgsz, list) else [imgsz, imgsz]}
    except Exception as exc:
        logger.warning(f"[ONNX Metadata] Failed to read metadata: {exc}")
        return {"names": {}, "task": "detect", "imgsz": [640, 640]}


def _preprocess(image: Image.Image, imgsz: List[int]) -> np.ndarray:
    h, w = (imgsz[0], imgsz[1]) if len(imgsz) > 1 else (imgsz[0], imgsz[0])
    img = image.convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)           # HWC -> CHW
    return np.expand_dims(arr, 0)          # NCHW


def _numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _nms_postprocess(raw_output: np.ndarray, num_classes: int, conf_threshold: float = 0.05) -> List[Dict]:
    output = raw_output[0] if raw_output.ndim == 4 else raw_output
    if output.ndim == 3:
        output = output[0]
    # output shape: [4+num_classes, 8400]
    output = output.T  # [8400, 4+num_classes]
    boxes_xywh = output[:, :4]
    class_scores = output[:, 4:4 + num_classes]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    mask = confidences >= conf_threshold
    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    if len(confidences) == 0:
        return []
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = _numpy_nms(boxes_xyxy, confidences, 0.45)
    results = []
    for idx in keep:
        results.append({
            "cls_idx": int(class_ids[idx]),
            "confidence": float(confidences[idx]),
            "x_center": float(boxes_xywh[idx, 0]),
            "y_center": float(boxes_xywh[idx, 1]),
            "width": float(boxes_xywh[idx, 2]),
            "height": float(boxes_xywh[idx, 3]),
        })
    return results


class PureONNX:
    """
    Loads a single child ONNX model from HF Hub and runs disease detection inference.
    One instance per request (stateless) to keep memory usage minimal on Render.
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.enable_cpu_mem_arena = False
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        self.metadata = _load_onnx_metadata(model_path)
        self.names: Dict[int, str] = self.metadata["names"]
        self.imgsz: List[int] = self.metadata["imgsz"]
        self.task: str = self.metadata["task"]
        logger.info(f"[PureONNX] Loaded {os.path.basename(model_path)} — task={self.task}, classes={len(self.names)}")

    def predict(self, image: Image.Image, crop_name: str = "Crop", conf_threshold: float = 0.05) -> List[Dict]:
        """Run inference on a PIL Image. Returns list of detection dicts."""
        input_tensor = _preprocess(image, self.imgsz)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        raw_output = self.session.run([output_name], {input_name: input_tensor})

        img_h, img_w = self.imgsz[0], self.imgsz[1] if len(self.imgsz) > 1 else self.imgsz[0]

        if self.task == "classify":
            # Classification output: [1, num_classes]
            probs = raw_output[0][0]
            if abs(float(np.sum(probs)) - 1.0) > 0.1:
                probs = np.exp(probs) / np.sum(np.exp(probs))
            top5_idx = np.argsort(probs)[::-1][:5]
            return [{"detected_class": self.names.get(int(i), f"class_{i}"), "confidence_score": round(float(probs[i]), 4)} for i in top5_idx]
        else:
            # Detection output: [1, 4+num_classes, 8400]
            raw_dets = _nms_postprocess(raw_output[0], num_classes=len(self.names), conf_threshold=conf_threshold)
            results = []
            for det in raw_dets:
                raw_class = self.names.get(det["cls_idx"], f"disease_{det['cls_idx']}")
                if "healthy" in raw_class.lower() and not raw_class.lower().startswith(crop_name.lower()):
                    display_class = f"{crop_name}_Healthy"
                elif not raw_class.lower().startswith(crop_name.lower()):
                    display_class = f"{crop_name}_{raw_class}"
                else:
                    display_class = raw_class
                results.append({
                    "detected_class": display_class,
                    "confidence_score": round(det["confidence"], 4),
                    "x_center": round(det["x_center"] / img_w, 4),
                    "y_center": round(det["y_center"] / img_h, 4),
                    "width": round(det["width"] / img_w, 4),
                    "height": round(det["height"] / img_h, 4),
                    "plant_class": crop_name,
                    "child_status": "AWOKEN (IN MEMORY)",
                })
            return results


def load_child_model(crop_name: str) -> Optional["PureONNX"]:
    """Resolve crop_name -> HF filename -> download -> load PureONNX session."""
    hf_filename = _resolve_hf_filename(crop_name)
    if not hf_filename:
        logger.warning(f"[ChildLoader] No HF filename mapping for crop '{crop_name}'")
        return None
    model_path = _download_model(hf_filename)
    if not model_path:
        return None
    return PureONNX(model_path)
