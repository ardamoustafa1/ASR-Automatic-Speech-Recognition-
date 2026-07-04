# API routes for uploading audio recordings and retrieving analysis results.
from __future__ import annotations

"""API route: conversations — list, detail, and analysis endpoints."""

import shutil

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db, limiter
from asr_pro.api.routes.auth import get_current_user
from asr_pro.api.schemas.conversations import (
    AnalyzeSegmentsRequest,
    AnalyzeTextRequest,
    ConversationDetail,
    ConversationOut,
    KeywordHitOut,
    SegmentOut,
)
from asr_pro.config import TEMP_AUDIO_DIR
from asr_pro.core.keyword_engine import SegmentInput, hits_to_dict
from asr_pro.db.models import Conversation, KeywordHit, Topic, TranscriptSegmentRow, new_uuid
from asr_pro.db.session import SessionLocal
from asr_pro.services.asr_service import ASRService
from asr_pro.services.conversation_service import (
    analyze_without_save,
    save_conversation_with_analysis,
)

router = APIRouter(
    prefix="/conversations", tags=["conversations"], dependencies=[Depends(get_current_user)]
)


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    limit: int = Query(50, le=200),
    sector: str | None = None,
    db: Session = Depends(get_db),
):
    """List conversations with aggregated hit counts — uses a single subquery (no N+1)."""
    q = db.query(Conversation).order_by(Conversation.created_at.desc())
    if sector:
        q = q.filter(Conversation.sector == sector)
    convs = q.limit(limit).all()

    if not convs:
        return []

    # ── Single query: aggregate hit counts for all returned conversations ──────
    conv_ids = [c.id for c in convs]
    hit_count_map: dict[str, int] = dict(
        db.query(KeywordHit.conversation_id, func.count(KeywordHit.id))
        .filter(KeywordHit.conversation_id.in_(conv_ids))
        .group_by(KeywordHit.conversation_id)
        .all()
    )

    return [
        ConversationOut(
            id=c.id,
            sector=c.sector,
            duration_sec=c.duration_sec,
            full_transcript=c.full_transcript[:500],
            asr_confidence=c.asr_confidence,
            quality_gate_passed=c.quality_gate_passed,
            created_at=c.created_at.isoformat() if c.created_at else "",
            hit_count=hit_count_map.get(c.id, 0),
            metadata_json=c.metadata_json,
        )
        for c in convs
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(404, "Görüşme bulunamadı")

    segments = (
        db.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == conversation_id)
        .order_by(TranscriptSegmentRow.start)
        .all()
    )
    hits = db.query(KeywordHit).filter(KeywordHit.conversation_id == conversation_id).all()

    # ── Batch-load topics in one query ────────────────────────────────────────
    topic_ids = list({h.topic_id for h in hits if h.topic_id})
    topic_map: dict[str, Topic] = {}
    if topic_ids:
        topics_found = db.query(Topic).filter(Topic.id.in_(topic_ids)).all()
        topic_map = {t.id: t for t in topics_found}

    topic_slugs = {t.slug: {"slug": t.slug, "label_tr": t.label_tr} for t in topic_map.values()}
    topics = list(topic_slugs.values())

    return ConversationDetail(
        id=conv.id,
        sector=conv.sector,
        duration_sec=conv.duration_sec,
        full_transcript=conv.full_transcript,
        asr_confidence=conv.asr_confidence,
        quality_gate_passed=conv.quality_gate_passed,
        created_at=conv.created_at.isoformat() if conv.created_at else "",
        hit_count=len(hits),
        topics=topics,
        metadata_json=conv.metadata_json,
        segments=[
            SegmentOut(id=s.id, start=s.start, end=s.end, text=s.text, speaker=s.speaker)
            for s in segments
        ],
        hits=[
            KeywordHitOut(
                id=h.id,
                keyword=h.keyword,
                matched_text=h.matched_text,
                match_type=h.match_type,
                confidence=h.confidence,
                timestamp_sec=h.timestamp_sec,
                speaker=h.speaker,
                context_before=h.context_before,
                rule_id=h.rule_id,
                topic_id=h.topic_id,
                severity="info",
            )
            for h in hits
        ],
    )


def _process_analysis_background(payload_data: dict, segments_data: list) -> None:
    """Background task: save conversation with full NLP analysis."""
    db = SessionLocal()
    try:
        save_conversation_with_analysis(
            db,
            segments_data=segments_data,
            full_transcript=payload_data.get("full_transcript")
            or " ".join(s.text for s in segments_data),
            sector=payload_data.get("sector"),
            audio_path=payload_data.get("audio_path"),
            uploaded_name=payload_data.get("uploaded_name"),
            asr_confidence=payload_data.get("asr_confidence", 0.0),
            quality_gate_passed=payload_data.get("quality_gate_passed", True),
        )
    finally:
        db.close()


def _process_audio_upload_background(file_path: str, filename: str, sector: str) -> None:
    """Background task: transcribe audio file and run full NLP + Diarization analysis."""
    from loguru import logger

    db = SessionLocal()
    try:
        logger.info(f"Starting background ASR transcription for uploaded file: {filename}")
        asr = ASRService.get_instance()
        transcribe_res = asr.transcribe(file_path)
        if isinstance(transcribe_res, tuple) and len(transcribe_res) >= 2:
            raw_segments, duration = transcribe_res[0], transcribe_res[1]
            confidence = 0.85
            full_text = " ".join(getattr(s, "text", "") for s in raw_segments)
        elif isinstance(transcribe_res, dict):
            raw_segments = transcribe_res.get("segments", [])
            confidence = float(transcribe_res.get("confidence", 0.85))
            full_text = transcribe_res.get("text", "") or " ".join(
                s.get("text", "") for s in raw_segments if isinstance(s, dict)
            )
        else:
            raw_segments = []
            confidence = 0.0
            full_text = ""

        segments = [
            SegmentInput(
                start=float(
                    getattr(s, "start", s.get("start", 0)) if not isinstance(s, dict) else s.get("start", 0)
                ),
                end=float(
                    getattr(s, "end", s.get("end", 0)) if not isinstance(s, dict) else s.get("end", 0)
                ),
                text=str(
                    getattr(s, "text", s.get("text", "")) if not isinstance(s, dict) else s.get("text", "")
                ),
                speaker=getattr(s, "speaker", s.get("speaker")) if not isinstance(s, dict) else s.get("speaker"),
                segment_index=i,
            )
            for i, s in enumerate(raw_segments)
        ]
        save_conversation_with_analysis(
            db,
            segments_data=segments,
            full_transcript=full_text,
            sector=sector,
            audio_path=file_path,
            uploaded_name=filename,
            asr_confidence=confidence,
            quality_gate_passed=True,
        )
        logger.info(f"Successfully processed uploaded conversation for {filename}")
    except Exception as exc:
        logger.error(f"Error processing uploaded audio {filename}: {exc}")
    finally:
        db.close()


@router.post("/upload", status_code=202)
@limiter.limit("20/minute")
def upload_audio_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sector: str = Query("omni"),
):
    """Upload an audio file (.wav, .mp3) for automated transcription, diarization, and NLP analysis."""
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{new_uuid()}_{file.filename or 'audio.wav'}"
    dest_path = str(TEMP_AUDIO_DIR / safe_name)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        _process_audio_upload_background, dest_path, file.filename or "audio.wav", sector
    )
    return {
        "message": "Ses dosyası yüklendi ve transkripsiyon/diarization işlemi arka plana alındı.",
        "status": "processing",
        "filename": file.filename,
    }


@router.post("/analyze", status_code=202)
@limiter.limit("60/minute")
def analyze_conversation(
    request: Request, payload: AnalyzeSegmentsRequest, background_tasks: BackgroundTasks
):
    """Queue a conversation for background NLP analysis (non-blocking)."""
    segments = [
        SegmentInput(
            start=float(s.get("start", 0)),
            end=float(s.get("end", 0)),
            text=str(s.get("text", "")),
            speaker=s.get("speaker"),
            segment_index=i,
        )
        for i, s in enumerate(payload.segments)
    ]
    background_tasks.add_task(_process_analysis_background, payload.model_dump(), segments)
    return {"message": "Analiz işlemi arka plana alındı.", "status": "processing"}


@router.post("/analyze-text")
@limiter.limit("120/minute")
def analyze_text(request: Request, payload: AnalyzeTextRequest, db: Session = Depends(get_db)):
    """Analyze a raw text string for keyword hits."""
    segment = SegmentInput(start=0, end=0, text=payload.text)
    hits = analyze_without_save(db, [segment], sector=payload.sector)
    return {"hits": hits_to_dict(hits), "hit_count": len(hits)}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation entirely (GDPR Right to be Forgotten)."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conv)
    db.commit()
    return {"status": "success", "id": conversation_id}


@router.get("/{conversation_id}/export")
def export_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Export conversation data (GDPR Right to Data Portability)."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    segments = (
        db.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == conversation_id)
        .order_by(TranscriptSegmentRow.start)
        .all()
    )

    return {
        "conversation": {
            "id": conv.id,
            "full_transcript": conv.full_transcript,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "sector": conv.sector,
            "duration_sec": conv.duration_sec,
        },
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker} for s in segments
        ],
    }
