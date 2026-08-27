from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class RawInput(Base):
    __tablename__ = "raw_inputs"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50), nullable=False)  # 'single_image', 'video', 'live_stream'
    original_filename = Column(String(255), nullable=True)
    storage_path = Column(String(500), nullable=True)
    status = Column(String(50), default="processed", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    frames = relationship("VerificationFrame", back_populates="raw_input", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<RawInput id={self.id} type={self.source_type} file={self.original_filename}>"


class VerificationFrame(Base):
    __tablename__ = "verification_frames"

    id = Column(Integer, primary_key=True, index=True)
    raw_input_id = Column(
        Integer,
        ForeignKey("raw_inputs.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )
    frame_index = Column(Integer, default=0, nullable=False)
    storage_path = Column(String(500), nullable=False)
    image_url = Column(String(500), nullable=True)
    
    parent_crop_predicted = Column(String(100), nullable=True)
    target_model_name = Column(String(150), nullable=True, index=True)  # Exact child ONNX filename, e.g. 'Apple_best_int8.onnx'
    parent_confidence = Column(Float, default=0.0, nullable=False)
    model_predictions = Column(Text, nullable=True)  # JSON string of bounding boxes
    
    status = Column(String(50), default="pending", nullable=False, index=True)  # 'pending', 'approved', 'rejected', 'corrected'
    verification_status = Column(String(50), default="pending", nullable=False, index=True)  # Explicit verification_status field
    human_crop_label = Column(String(100), nullable=True)
    human_annotations = Column(Text, nullable=True)  # JSON string of human bounding boxes
    verified_by = Column(String(100), default="Human_Annotator_1", nullable=True)
    ready_for_retraining = Column(Boolean, default=False, nullable=False, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    verified_at = Column(DateTime(timezone=True), nullable=True)

    raw_input = relationship("RawInput", back_populates="frames")

    def __repr__(self) -> str:
        return f"<VerificationFrame id={self.id} status={self.status} crop={self.parent_crop_predicted} target={self.target_model_name} conf={self.parent_confidence}>"
