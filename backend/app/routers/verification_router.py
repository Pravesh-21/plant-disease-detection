import os
import json
import io
import zipfile
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_

from app.core.database import get_db
from app.models.verification import RawInput, VerificationFrame
from app.services.frame_sampler import FrameSamplerService, resolve_target_model

logger = logging.getLogger("app.routers.verification_router")

router = APIRouter(prefix="/admin/verification", tags=["HITL Verification"])

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class AnnotationItem(BaseModel):
    class_name: str
    confidence: Optional[float] = 1.0
    x_center: float
    y_center: float
    width: float
    height: float
    crop_label: Optional[str] = None

class AnnotateRequest(BaseModel):
    frame_id: int
    status: str = Field(..., description="'approved', 'rejected', or 'corrected'")
    human_crop_label: Optional[str] = None
    human_annotations: Optional[List[AnnotationItem]] = None
    verified_by: Optional[str] = "Human_Annotator_1"

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/frames")
async def list_verification_frames(
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
    status: str = Query("all", description="'pending', 'approved', 'rejected', 'corrected', 'all'"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0),
    crop_type: Optional[str] = Query(None),
    target_model_name: Optional[str] = Query(None, description="Filter by exact child model, e.g., Apple_best_int8.onnx"),
    source_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves paginated verification frames from Neon DB with filtering
    by status, target child model, confidence range, crop type, and source media type.
    """
    if db is None:
        return {
            "frames": [],
            "pagination": {"page": page, "limit": limit, "total_frames": 0, "total_pages": 0},
            "metrics": {"pending_count": 0, "approved_count": 0, "rejected_count": 0, "corrected_count": 0, "low_confidence_count": 0, "total_count": 0}
        }

    query = select(VerificationFrame).outerjoin(RawInput)

    filters = []
    if status and status.lower() != "all":
        filters.append(
            or_(
                VerificationFrame.status == status.lower(),
                VerificationFrame.verification_status == status.lower()
            )
        )
    
    if min_confidence > 0.0 or max_confidence < 1.0:
        filters.append(
            and_(
                VerificationFrame.parent_confidence >= min_confidence,
                VerificationFrame.parent_confidence <= max_confidence
            )
        )

    if crop_type and crop_type.strip():
        filters.append(
            or_(
                VerificationFrame.parent_crop_predicted.ilike(f"%{crop_type}%"),
                VerificationFrame.human_crop_label.ilike(f"%{crop_type}%")
            )
        )

    if target_model_name and target_model_name.strip():
        filters.append(VerificationFrame.target_model_name.ilike(f"%{target_model_name.strip()}%"))

    if source_type and source_type.strip() and source_type.lower() != "all":
        filters.append(RawInput.source_type == source_type.lower())

    if filters:
        query = query.where(and_(*filters))

    # Calculate total matching count
    count_query = select(func.count(VerificationFrame.id)).outerjoin(RawInput)
    if filters:
        count_query = count_query.where(and_(*filters))
    
    count_res = await db.execute(count_query)
    total_matching = count_res.scalar() or 0

    # Apply pagination & order by pending first, then lowest confidence, then newest
    offset = (page - 1) * limit
    paginated_query = query.order_by(
        VerificationFrame.status.asc(),
        VerificationFrame.parent_confidence.asc(),
        VerificationFrame.id.desc()
    ).offset(offset).limit(limit)

    res = await db.execute(paginated_query)
    rows = res.scalars().all()

    # Calculate overall metrics
    metrics_query = select(
        func.count(VerificationFrame.id).label("total"),
        func.count(func.nullif(VerificationFrame.status != "pending", True)).label("pending"),
        func.count(func.nullif(VerificationFrame.status != "approved", True)).label("approved"),
        func.count(func.nullif(VerificationFrame.status != "rejected", True)).label("rejected"),
        func.count(func.nullif(VerificationFrame.status != "corrected", True)).label("corrected"),
        func.count(func.nullif(VerificationFrame.parent_confidence >= 0.80, True)).label("low_conf"),
    )
    m_res = await db.execute(metrics_query)
    m_row = m_res.one()

    serialized_frames = []
    for f in rows:
        preds = []
        try:
            preds = json.loads(f.model_predictions) if f.model_predictions else []
        except Exception:
            preds = []

        h_annos = []
        try:
            h_annos = json.loads(f.human_annotations) if f.human_annotations else []
        except Exception:
            h_annos = []

        # Ensure target_model_name is populated
        eff_crop = f.human_crop_label or f.parent_crop_predicted
        target_model = f.target_model_name or resolve_target_model(eff_crop)

        serialized_frames.append({
            "id": f.id,
            "raw_input_id": f.raw_input_id,
            "frame_index": f.frame_index,
            "storage_path": f.storage_path,
            "image_url": f.image_url or f"/api/admin/verification/frame-image/{f.id}",
            "parent_crop_predicted": f.parent_crop_predicted,
            "target_model_name": target_model,
            "parent_confidence": f.parent_confidence,
            "model_predictions": preds,
            "status": f.status,
            "verification_status": f.verification_status or f.status,
            "human_crop_label": f.human_crop_label,
            "human_annotations": h_annos,
            "verified_by": f.verified_by,
            "ready_for_retraining": f.ready_for_retraining,
            "created_at": f.created_at.isoformat() if f.created_at else None,
            "verified_at": f.verified_at.isoformat() if f.verified_at else None,
        })

    total_pages = max(1, (total_matching + limit - 1) // limit)

    return {
        "frames": serialized_frames,
        "pagination": {
            "page": page,
            "limit": limit,
            "total_frames": total_matching,
            "total_pages": total_pages,
        },
        "metrics": {
            "total_count": m_row.total or 0,
            "pending_count": m_row.pending or 0,
            "approved_count": m_row.approved or 0,
            "rejected_count": m_row.rejected or 0,
            "corrected_count": m_row.corrected or 0,
            "low_confidence_count": m_row.low_conf or 0,
        }
    }


@router.post("/annotate")
async def annotate_verification_frame(
    payload: AnnotateRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Accepts human verification updates (approve, reject, or modify bounding boxes).
    Strict Rule: If human_crop_label is updated, automatically re-assigns target_model_name 
    to the matched specialist child ONNX model (e.g. 'Apple' -> 'Apple_best_int8.onnx').
    Sets ready_for_retraining = True when status is 'approved' or 'corrected'.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection unavailable.")

    stmt = select(VerificationFrame).where(VerificationFrame.id == payload.frame_id)
    res = await db.execute(stmt)
    frame = res.scalar_one_or_none()

    if not frame:
        raise HTTPException(status_code=404, detail=f"VerificationFrame #{payload.frame_id} not found.")

    new_status = payload.status.lower().strip()
    if new_status not in ["approved", "rejected", "corrected"]:
        raise HTTPException(status_code=400, detail="Status must be 'approved', 'rejected', or 'corrected'.")

    frame.status = new_status
    frame.verification_status = new_status
    frame.verified_by = payload.verified_by or "Human_Annotator_1"
    frame.verified_at = datetime.now(timezone.utc)

    # Re-assign target_model_name strictly based on updated human_crop_label
    if payload.human_crop_label and payload.human_crop_label.strip():
        new_crop = payload.human_crop_label.strip()
        frame.human_crop_label = new_crop
        frame.target_model_name = resolve_target_model(new_crop)
    elif frame.parent_crop_predicted:
        frame.target_model_name = resolve_target_model(frame.parent_crop_predicted)

    if payload.human_annotations is not None:
        annos_json = [a.model_dump() for a in payload.human_annotations]
        frame.human_annotations = json.dumps(annos_json)

    # Set ready_for_retraining flag
    if new_status in ["approved", "corrected"]:
        frame.ready_for_retraining = True
    else:
        frame.ready_for_retraining = False

    await db.commit()
    await db.refresh(frame)

    logger.info(f"[HITL] Frame #{frame.id} updated: status='{new_status}', target_model='{frame.target_model_name}', ready_for_retraining={frame.ready_for_retraining}")

    h_annos = []
    try:
        h_annos = json.loads(frame.human_annotations) if frame.human_annotations else []
    except Exception:
        h_annos = []

    return {
        "status": "success",
        "message": f"Frame #{frame.id} updated to '{new_status}' with target model '{frame.target_model_name}'.",
        "frame": {
            "id": frame.id,
            "status": frame.status,
            "verification_status": frame.verification_status,
            "human_crop_label": frame.human_crop_label,
            "target_model_name": frame.target_model_name,
            "human_annotations": h_annos,
            "ready_for_retraining": frame.ready_for_retraining,
            "verified_at": frame.verified_at.isoformat() if frame.verified_at else None,
        }
    }


@router.post("/ingest")
async def ingest_media_file(
    file: UploadFile = File(...),
    source_type: str = Form("single_image"),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingests media files (single image, video, or stream frame) into sampled verification frames.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection unavailable.")

    contents = await file.read()
    st = source_type.lower().strip()

    if "video" in file.content_type or st == "video":
        raw_input, frames = await FrameSamplerService.process_video_upload(
            db=db,
            video_bytes=contents,
            original_filename=file.filename or "video.mp4"
        )
        return {
            "status": "success",
            "source_type": "video",
            "raw_input_id": raw_input.id,
            "frames_created": len(frames)
        }
    else:
        raw_input, frame = await FrameSamplerService.process_single_image(
            db=db,
            image_bytes=contents,
            original_filename=file.filename or "image.jpg"
        )
        return {
            "status": "success",
            "source_type": "single_image",
            "raw_input_id": raw_input.id,
            "frames_created": 1,
            "frame_id": frame.id
        }


@router.get("/export")
async def export_retraining_dataset(
    format: str = Query("yolo", description="'json' or 'yolo'"),
    target_model_name: Optional[str] = Query(None, description="Filter dataset export by exact child ONNX model, e.g. Apple_best_int8.onnx"),
    db: AsyncSession = Depends(get_db)
):
    """
    Exports all verified frames marked ready_for_retraining = True 
    filtered by target child model pool formatted as a downloadable YOLO dataset zip or JSON.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection unavailable.")

    query = select(VerificationFrame).where(VerificationFrame.ready_for_retraining == True)
    if target_model_name and target_model_name.strip():
        query = query.where(VerificationFrame.target_model_name.ilike(f"%{target_model_name.strip()}%"))

    res = await db.execute(query)
    frames = res.scalars().all()

    if not frames:
        raise HTTPException(status_code=404, detail="No verified frames ready for retraining found matching the specified model target.")

    model_tag = target_model_name.replace(".onnx", "").replace(".pt", "") if target_model_name else "all_models"

    if format.lower() == "json":
        dataset_records = []
        for f in frames:
            annos = []
            try:
                annos = json.loads(f.human_annotations) if f.human_annotations else json.loads(f.model_predictions or "[]")
            except Exception:
                annos = []

            dataset_records.append({
                "frame_id": f.id,
                "crop_label": f.human_crop_label or f.parent_crop_predicted or "Plant",
                "target_model_name": f.target_model_name or resolve_target_model(f.human_crop_label or f.parent_crop_predicted),
                "status": f.status,
                "storage_path": f.storage_path,
                "image_url": f.image_url,
                "annotations": annos,
                "verified_by": f.verified_by,
                "verified_at": f.verified_at.isoformat() if f.verified_at else None,
            })

        payload = {
            "dataset_name": f"Project_Jatayu_FineTuning_Dataset_{model_tag}",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "target_model_name": target_model_name or "All Child Models",
            "total_samples": len(dataset_records),
            "samples": dataset_records
        }

        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        return Response(
            content=json_bytes,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=jatayu_retraining_{model_tag}_{len(frames)}samples.json"}
        )

    # YOLO Zip Format Export
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        class_mapping: Dict[str, int] = {}
        class_counter = 0

        for f in frames:
            if os.path.exists(f.storage_path):
                zf.write(f.storage_path, arcname=f"images/frame_{f.id}.jpg")

            annos = []
            try:
                annos = json.loads(f.human_annotations) if f.human_annotations else json.loads(f.model_predictions or "[]")
            except Exception:
                annos = []

            txt_lines = []
            for a in annos:
                cls_name = a.get("class_name", "Healthy")
                if cls_name not in class_mapping:
                    class_mapping[cls_name] = class_counter
                    class_counter += 1
                cls_idx = class_mapping[cls_name]
                xc = a.get("x_center", 0.5)
                yc = a.get("y_center", 0.5)
                w = a.get("width", 0.4)
                h = a.get("height", 0.4)
                txt_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            zf.writestr(f"labels/frame_{f.id}.txt", "\n".join(txt_lines))

        classes_yaml = "\n".join([f"  {idx}: '{name}'" for name, idx in class_mapping.items()])
        yaml_content = f"# Project Jatayu Fine-Tuning Dataset for {target_model_name or 'All Models'}\npath: ./dataset\ntrain: images\nval: images\nnames:\n{classes_yaml}\n"
        zf.writestr("dataset.yaml", yaml_content)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=jatayu_yolo_{model_tag}_{len(frames)}samples.zip"}
    )


@router.get("/frame-image/{frame_id}")
async def get_frame_image(frame_id: int, db: AsyncSession = Depends(get_db)):
    """Serves the frame image file for rendering in the HITL verification canvas."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection unavailable.")

    stmt = select(VerificationFrame).where(VerificationFrame.id == frame_id)
    res = await db.execute(stmt)
    frame = res.scalar_one_or_none()

    if not frame or not os.path.exists(frame.storage_path):
        raise HTTPException(status_code=404, detail="Frame image file not found.")

    return FileResponse(frame.storage_path, media_type="image/jpeg")
