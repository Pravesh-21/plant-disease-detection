import os
import json
import uuid
import time
import logging
import cv2
import numpy as np
from PIL import Image
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.verification import RawInput, VerificationFrame
from app.services.inference import ModelRegistry

logger = logging.getLogger("app.services.frame_sampler")

# Base directory for storing frame images
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "uploads", "verification_frames"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Strict Parent Crop -> Child Specialist ONNX Model File Mapping
CROP_TO_CHILD_MODEL: Dict[str, str] = {
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
    "pumpkin": "Pumkin_best_int8.onnx",
    "pumkin": "Pumkin_best_int8.onnx",
    "rice": "Rice_best_int8.onnx",
    "rose": "Rose_best_int8.onnx",
    "sesame": "Sesame_best_int8.onnx",
    "soybean": "SoyaBean_best_int8.onnx",
    "soyabean": "SoyaBean_best_int8.onnx",
    "strawberry": "Strawberry_best_int8.onnx",
    "sugarcane": "SugarCane_best_int8.onnx",
    "sunflower": "Sunflower_best_int8.onnx",
    "tobacco": "Tobacco_best_int8.onnx",
    "tomato": "Tomato_best_int8.onnx",
    "wheat": "Wheat_best_int8.onnx",
}

def resolve_target_model(crop_name: Optional[str]) -> str:
    """Maps predicted or ground-truth crop species to the exact child ONNX model filename."""
    if not crop_name:
        return "ParentEnsemble.onnx"
    norm = crop_name.lower().replace("_", "").replace(" ", "").replace("-", "")
    return CROP_TO_CHILD_MODEL.get(norm, f"{crop_name}_best_int8.onnx")


class FrameSamplerService:
    """
    Frame Ingestion & Sampling Pipeline Service for Project Jatayu.
    Enforces strict Parent-to-Child routing (Parent predicts crop -> routes only to matched child ONNX).
    """

    _last_stream_sample_time: float = 0.0

    @classmethod
    async def process_single_image(
        cls,
        db: AsyncSession,
        image_bytes: bytes,
        original_filename: str
    ) -> Tuple[RawInput, VerificationFrame]:
        """Processes a single uploaded image, runs model inference, and stores DB entries."""
        ext = os.path.splitext(original_filename)[1] or ".jpg"
        unique_name = f"frame_{uuid.uuid4().hex[:10]}{ext}"
        storage_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(storage_path, "wb") as f:
            f.write(image_bytes)

        raw_input = RawInput(
            source_type="single_image",
            original_filename=original_filename,
            storage_path=storage_path,
            status="processed"
        )
        db.add(raw_input)
        await db.commit()
        await db.refresh(raw_input)

        # Run Two-Phase Model Inference
        predictions, crop_name, conf = await cls._run_inference_on_file(storage_path)
        target_model = resolve_target_model(crop_name)

        rel_url = f"/uploads/verification_frames/{unique_name}"
        frame = VerificationFrame(
            raw_input_id=raw_input.id,
            frame_index=0,
            storage_path=storage_path,
            image_url=rel_url,
            parent_crop_predicted=crop_name,
            target_model_name=target_model,
            parent_confidence=conf,
            model_predictions=json.dumps(predictions),
            status="pending",
            verification_status="pending",
            ready_for_retraining=False
        )
        db.add(frame)
        await db.commit()
        await db.refresh(frame)

        logger.info(f"[FrameSampler] Processed single image ID={frame.id}, crop='{crop_name}', target='{target_model}', conf={conf:.2f}")
        return raw_input, frame

    @classmethod
    async def process_video_upload(
        cls,
        db: AsyncSession,
        video_bytes: bytes,
        original_filename: str,
        sample_fps: float = 1.0
    ) -> Tuple[RawInput, List[VerificationFrame]]:
        """
        Processes a video file using OpenCV.
        Samples 1 frame per second, runs parent crop classification -> matches child ONNX model,
        and logs batch database rows.
        """
        temp_video_path = os.path.join(UPLOAD_DIR, f"temp_vid_{uuid.uuid4().hex[:8]}.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_bytes)

        raw_input = RawInput(
            source_type="video",
            original_filename=original_filename,
            storage_path=temp_video_path,
            status="processing"
        )
        db.add(raw_input)
        await db.commit()
        await db.refresh(raw_input)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            logger.error(f"[FrameSampler] Failed to open video file: {temp_video_path}")
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            raw_input.status = "failed"
            await db.commit()
            return raw_input, []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / sample_fps))
        
        frames_created: List[VerificationFrame] = []
        frame_count = 0
        saved_count = 0

        try:
            while True:
                ret, frame_mat = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    saved_count += 1
                    frame_filename = f"video_{raw_input.id}_f{saved_count:04d}_{uuid.uuid4().hex[:6]}.jpg"
                    frame_storage_path = os.path.join(UPLOAD_DIR, frame_filename)

                    cv2.imwrite(frame_storage_path, frame_mat)

                    predictions, crop_name, conf = await cls._run_inference_on_file(frame_storage_path)
                    target_model = resolve_target_model(crop_name)

                    rel_url = f"/uploads/verification_frames/{frame_filename}"
                    vf = VerificationFrame(
                        raw_input_id=raw_input.id,
                        frame_index=frame_count,
                        storage_path=frame_storage_path,
                        image_url=rel_url,
                        parent_crop_predicted=crop_name,
                        target_model_name=target_model,
                        parent_confidence=conf,
                        model_predictions=json.dumps(predictions),
                        status="pending",
                        verification_status="pending",
                        ready_for_retraining=False
                    )
                    db.add(vf)
                    frames_created.append(vf)

                frame_count += 1

            raw_input.status = "processed"
            await db.commit()
            logger.info(f"[FrameSampler] Sampled {len(frames_created)} frame(s) from video '{original_filename}'.")
        finally:
            cap.release()

        return raw_input, frames_created

    @classmethod
    async def process_live_stream_frame(
        cls,
        db: AsyncSession,
        image_bytes: bytes,
        force_sample: bool = False
    ) -> Optional[VerificationFrame]:
        """
        Processes an incoming UAV live stream frame.
        Samples keyframe every 3 seconds OR whenever confidence < 0.80.
        """
        now = time.time()
        time_elapsed = now - cls._last_stream_sample_time

        temp_path = os.path.join(UPLOAD_DIR, f"stream_check_{uuid.uuid4().hex[:6]}.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        try:
            predictions, crop_name, conf = await cls._run_inference_on_file(temp_path)
            should_sample = force_sample or (time_elapsed >= 3.0) or (conf < 0.80)

            if not should_sample:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return None

            cls._last_stream_sample_time = now

            permanent_name = f"stream_{uuid.uuid4().hex[:10]}.jpg"
            permanent_path = os.path.join(UPLOAD_DIR, permanent_name)
            os.rename(temp_path, permanent_path)

            target_model = resolve_target_model(crop_name)
            rel_url = f"/uploads/verification_frames/{permanent_name}"
            vf = VerificationFrame(
                raw_input_id=None,
                frame_index=0,
                storage_path=permanent_path,
                image_url=rel_url,
                parent_crop_predicted=crop_name,
                target_model_name=target_model,
                parent_confidence=conf,
                model_predictions=json.dumps(predictions),
                status="pending",
                verification_status="pending",
                ready_for_retraining=False
            )
            db.add(vf)
            await db.commit()
            await db.refresh(vf)

            logger.info(f"[FrameSampler] Stream keyframe sampled ID={vf.id}, crop='{crop_name}', target='{target_model}', conf={conf:.2f}")
            return vf
        except Exception as exc:
            logger.error(f"[FrameSampler] Error processing stream frame: {exc}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    @classmethod
    async def _run_inference_on_file(cls, file_path: str) -> Tuple[List[Dict[str, Any]], str, float]:
        """Executes Parent classification -> matches child ONNX model for two-phase detection."""
        try:
            registry = ModelRegistry.get()
            detections = await registry.async_run_inference(file_path, skip_vlm=True)
            
            top_det = detections[0] if detections else {}
            crop_name = top_det.get("plant_class") or top_det.get("parent_crop") or "Plant"
            conf = float(top_det.get("parent_confidence") or top_det.get("confidence_score") or 0.85)

            formatted_preds = []
            for d in detections:
                formatted_preds.append({
                    "class_name": d.get("detected_class", "Healthy"),
                    "confidence": float(d.get("confidence_score", 0.90)),
                    "x_center": float(d.get("x_center", 0.5)),
                    "y_center": float(d.get("y_center", 0.5)),
                    "width": float(d.get("width", 0.4)),
                    "height": float(d.get("height", 0.4)),
                    "parent_crop": d.get("plant_class", crop_name),
                })

            return formatted_preds, crop_name, conf
        except Exception as exc:
            logger.warning(f"[FrameSampler] Model inference error: {exc}")
            return [], "Plant", 0.75
