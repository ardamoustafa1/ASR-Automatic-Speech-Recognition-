from typing import Optional

from pydantic import BaseModel, Field


class SegmentOut(BaseModel):
    id: str
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class KeywordHitOut(BaseModel):
    id: str
    keyword: str
    matched_text: str
    match_type: str
    confidence: float
    timestamp_sec: float
    speaker: Optional[str]
    context_before: str
    rule_id: Optional[str]
    topic_id: Optional[str]
    severity: str = "info"


class ConversationOut(BaseModel):
    id: str
    sector: str
    duration_sec: float
    full_transcript: str
    asr_confidence: float
    quality_gate_passed: bool
    created_at: str
    hit_count: int = 0
    topics: list[dict] = []


class ConversationDetail(ConversationOut):
    segments: list[SegmentOut]
    hits: list[KeywordHitOut]


class AnalyzeTextRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=50000)
    sector: str = Field("omni", max_length=50)


class AnalyzeSegmentsRequest(BaseModel):
    segments: list[dict] = Field(..., min_length=1, max_length=10000)
    full_transcript: str = Field("", max_length=500000)
    sector: str = Field("omni", max_length=50)
    uploaded_name: Optional[str] = Field(None, max_length=255)
    audio_path: Optional[str] = Field(None, max_length=1000)
    asr_confidence: float = Field(0.0, ge=0.0, le=1.0)
    quality_gate_passed: bool = True
