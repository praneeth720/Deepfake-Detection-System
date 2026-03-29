from typing import Dict
from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    duration: float
    sample_rate: int
    channels: int
    format: str


class AudioAnalysisResult(BaseModel):
    filename: str
    is_deepfake: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    deepfake_probability: float = Field(..., ge=0.0, le=1.0)
    authentic_probability: float = Field(..., ge=0.0, le=1.0)
    features: Dict[str, float]
    metadata: AudioMetadata
    message: str