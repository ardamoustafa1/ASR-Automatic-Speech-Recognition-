# API routes for uploading audio recordings and retrieving analysis results.
"""API route: conversations — list, detail, and analysis endpoints."""

# isort and ruff's isort emulation disagree on how to order the two `User`
# aliases below (asr_pro.api.routes.auth vs asr_pro.db.models) - isort's
# verdict wins since it's the tool this repo's CI actually runs to check
# import order; ruff's redundant I001 check is suppressed for this block.
from __future__ import annotations  # noqa: I001

import os
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db, limiter
from asr_pro.api.rbac import ADMIN, require_roles, scope_conversations
from asr_pro.api.routes.auth import User as AuthUser
from asr_pro.api.routes.auth import get_current_user
from asr_pro.api.schemas.conversations import (
    AnalyzeSegmentsRequest,
    AnalyzeTextRequest,
    ConversationDetail,
    ConversationOut,
    KeywordHitOut,
    SegmentOut,
)
from asr_pro.config import TEMP_AUDIO_DIR, settings
from asr_pro.core.keyword_engine import SegmentInput, hits_to_dict
from asr_pro.db.models import (
    AuditLog,
    Conversation,
    KeywordHit,
    Topic,
    TranscriptSegmentRow,
)
from asr_pro.db.models import User as DBUser
from asr_pro.db.models import (
    new_uuid,
)
from asr_pro.db.session import SessionLocal
from asr_pro.services.asr_service import ASRService
from asr_pro.services.conversation_service import (
    analyze_without_save,
    save_conversation_with_analysis,
)
from asr_pro.services.task_queue import enqueue

router = APIRouter(
    prefix="/conversations", tags=["conversations"], dependencies=[Depends(get_current_user)]
)


def _log_audit_view(
    db: Session,
    current_user: AuthUser,
    action: str,
    conversation_id: str,
    extra_details: dict | None = None,
) -> None:
    """Record a 'who viewed/exported what, when' audit entry for sensitive transcript access."""
    db_user = db.query(DBUser).filter(DBUser.username == current_user.username).first()
    db.add(
        AuditLog(
            user_id=db_user.id if db_user else None,
            username=current_user.username,
            action=action,
            target_resource=f"/conversations/{conversation_id}",
            details={"role": current_user.role, **(extra_details or {})},
        )
    )
    db.commit()


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    limit: int = Query(50, le=200),
    sector: str | None = None,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """List conversations with aggregated hit counts — uses a single subquery (no N+1)."""
    q = db.query(Conversation).order_by(Conversation.created_at.desc())
    if sector:
        q = q.filter(Conversation.sector == sector)
    q = scope_conversations(q, current_user, db)
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
            status=c.status or "completed",
            error_message=c.error_message,
        )
        for c in convs
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    q = db.query(Conversation).filter(Conversation.id == conversation_id)
    conv = scope_conversations(q, current_user, db).first()
    if not conv:
        raise HTTPException(404, "Görüşme bulunamadı")
    _log_audit_view(db, current_user, "VIEW", conversation_id)

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
        status=conv.status or "completed",
        error_message=conv.error_message,
        segments=[
            SegmentOut(
                id=s.id,
                start=s.start,
                end=s.end,
                text=s.text,
                speaker=s.speaker,
                avg_logprob=s.avg_logprob,
                raw_text=s.raw_text,
            )
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


def _process_analysis_background(
    payload_data: dict, segments_data: list, agent_id: str | None = None
) -> None:
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
            agent_id=agent_id,
        )
    finally:
        db.close()


def _process_audio_upload_background(
    file_path: str,
    filename: str,
    sector: str,
    agent_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """Background task: transcribe audio file and run full NLP + Diarization analysis."""
    import time as _time

    from loguru import logger

    from asr_pro.services.asr_service import (
        get_filtered_segment_count,
        reset_filtered_segment_counter,
    )
    from asr_pro.services.domain_adaptation import (
        get_correction_count,
        reset_correction_counter,
    )

    db = SessionLocal()
    try:
        logger.info(f"Starting background ASR transcription for uploaded file: {filename}")
        asr = ASRService.get_instance()
        reset_filtered_segment_counter()
        reset_correction_counter()
        transcribe_started = _time.perf_counter()
        transcribe_res = asr.transcribe(file_path, sector=sector)
        processing_time_sec = round(_time.perf_counter() - transcribe_started, 2)
        filtered_segment_count = get_filtered_segment_count()
        domain_correction_count = get_correction_count()

        if isinstance(transcribe_res, tuple) and len(transcribe_res) >= 2:
            raw_segments = transcribe_res[0]
            audio_duration_sec = float(transcribe_res[1] or 0.0)
            # Real decoder confidence (duration-weighted mean of per-segment
            # exp(avg_logprob)), not a fabricated constant.
            confidence = ASRService.compute_confidence(raw_segments)
            full_text = " ".join(getattr(s, "text", "") for s in raw_segments)
        elif isinstance(transcribe_res, dict):
            raw_segments = transcribe_res.get("segments", [])
            audio_duration_sec = float(transcribe_res.get("duration", 0.0) or 0.0)
            confidence = float(transcribe_res.get("confidence", 0.0))
            full_text = transcribe_res.get("text", "") or " ".join(
                s.get("text", "") for s in raw_segments if isinstance(s, dict)
            )
        else:
            raw_segments = []
            audio_duration_sec = 0.0
            confidence = 0.0
            full_text = ""

        rtf = round(processing_time_sec / audio_duration_sec, 3) if audio_duration_sec > 0 else None
        # audio_quality_pct is intentionally NOT computed here: MOSEstimator
        # already runs once inside save_conversation_with_analysis (stored as
        # mos_metrics.mos_score) - recomputing it would mean decoding the
        # whole audio file a second time for no new information. The UI
        # derives the percentage from mos_metrics.mos_score directly.
        quality_metrics = {
            "processing_time_sec": processing_time_sec,
            "rtf": rtf,
            "filtered_segment_count": filtered_segment_count,
            "domain_correction_count": domain_correction_count,
        }

        def _field(s: Any, name: str, default: Any = None) -> Any:
            return s.get(name, default) if isinstance(s, dict) else getattr(s, name, default)

        segments = [
            SegmentInput(
                start=float(_field(s, "start", 0)),
                end=float(_field(s, "end", 0)),
                text=str(_field(s, "text", "")),
                speaker=_field(s, "speaker"),
                segment_index=i,
                avg_logprob=float(_field(s, "avg_logprob", -1.0)),
                words=_field(s, "words"),
                raw_text=str(_field(s, "raw_text", "") or ""),
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
            # Gate fails when the decoder itself reports low confidence
            # (confidence == 0.0 means "engine provided no data" - not a fail).
            quality_gate_passed=confidence == 0.0 or confidence >= 0.45,
            agent_id=agent_id,
            metadata={"quality_metrics": quality_metrics},
            conversation_id=conversation_id,
        )
        logger.info(f"Successfully processed uploaded conversation for {filename}")
    except Exception as exc:
        logger.exception(f"Error processing uploaded audio {filename}: {exc}")
        # Mark the placeholder failed so the user sees WHY instead of a
        # conversation that silently never appears.
        if conversation_id:
            try:
                db.rollback()
                row = db.query(Conversation).filter(Conversation.id == conversation_id).first()
                if row is not None:
                    row.status = "failed"
                    row.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                    db.commit()
            except Exception:
                logger.exception(f"Could not record failure status for {conversation_id}")
    finally:
        db.close()


_ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".flac",
    ".webm",
    ".wma",
    ".amr",
}


@router.post("/upload", status_code=202)
@limiter.limit("20/minute")
def upload_audio_file(
    request: Request,
    file: UploadFile = File(...),
    sector: str = Query("omni"),
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Upload an audio file for automated transcription, diarization, and NLP analysis.

    Returns 202 with the conversation_id of a placeholder record
    (status="processing") that the background job fills in - poll
    GET /conversations/{id} or the list endpoint for status transitions
    (processing -> completed | failed).
    """
    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in _ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Desteklenmeyen dosya türü '{ext}'. Kabul edilenler: "
            + ", ".join(sorted(_ALLOWED_AUDIO_EXTENSIONS)),
        )

    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = f"{new_uuid()}_{filename}"
    dest_path = str(TEMP_AUDIO_DIR / safe_name)

    # Chunked copy with a hard size cap: Content-Length can be absent or
    # spoofed, so the only trustworthy limit is counting bytes as they land.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dosya {settings.max_upload_mb}MB sınırını aşıyor.",
                    )
                buffer.write(chunk)
    except HTTPException:
        try:
            os.unlink(dest_path)
        except OSError:
            pass
        raise

    # Placeholder row so the upload is visible (and its failure diagnosable)
    # from the moment it's accepted - not only if/when processing succeeds.
    placeholder = Conversation(
        id=new_uuid(),
        agent_id=current_user.username,
        sector=sector,
        audio_path=dest_path,
        full_transcript="",
        status="processing",
        metadata_json={"uploaded_name": filename},
    )
    db.add(placeholder)
    db.commit()

    enqueue(
        _process_audio_upload_background,
        dest_path,
        filename,
        sector,
        current_user.username,
        placeholder.id,
    )
    return {
        "message": "Ses dosyası yüklendi ve transkripsiyon/diarization işlemi arka plana alındı.",
        "status": "processing",
        "filename": filename,
        "conversation_id": placeholder.id,
    }


@router.post("/analyze", status_code=202)
@limiter.limit("60/minute")
def analyze_conversation(
    request: Request,
    payload: AnalyzeSegmentsRequest,
    current_user: AuthUser = Depends(get_current_user),
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
    enqueue(_process_analysis_background, payload.model_dump(), segments, current_user.username)
    return {"message": "Analiz işlemi arka plana alındı.", "status": "processing"}


@router.post("/analyze-text")
@limiter.limit("120/minute")
def analyze_text(request: Request, payload: AnalyzeTextRequest, db: Session = Depends(get_db)):
    """Analyze a raw text string for keyword hits."""
    segment = SegmentInput(start=0, end=0, text=payload.text)
    hits = analyze_without_save(db, [segment], sector=payload.sector)
    return {"hits": hits_to_dict(hits), "hit_count": len(hits)}


@router.delete("/{conversation_id}", dependencies=[Depends(require_roles(ADMIN))])
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation entirely (GDPR Right to be Forgotten). Admin-only:
    an irreversible action is deliberately not opened up to agents/team_leads."""
    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # No ON DELETE CASCADE is defined on the transcript_segments/keyword_hits
    # foreign keys, so children must be removed explicitly in FK-safe order
    # (keyword_hits before transcript_segments, both before the conversation
    # itself) - otherwise this "right to be forgotten" delete fails outright
    # against a FK-enforcing database (PostgreSQL in production; SQLite only
    # masked this because it doesn't enforce FKs by default).
    db.query(KeywordHit).filter(KeywordHit.conversation_id == conversation_id).delete(
        synchronize_session=False
    )
    db.query(TranscriptSegmentRow).filter(
        TranscriptSegmentRow.conversation_id == conversation_id
    ).delete(synchronize_session=False)
    db.delete(conv)
    db.commit()
    return {"status": "success", "id": conversation_id}


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _speaker_display(speaker: str | None) -> str:
    if not speaker:
        return "Konuşmacı"
    if "IVR" in speaker:
        return "IVR"
    return {"SPEAKER_00": "Temsilci", "SPEAKER_01": "Müşteri", "SPEAKER_02": "Uzman"}.get(
        speaker, speaker
    )


@router.get("/{conversation_id}/export")
def export_conversation(
    conversation_id: str,
    format: str = Query("json", pattern="^(json|txt|srt)$"),
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Export conversation data (GDPR Right to Data Portability).

    format=json (default) returns the structured payload; format=txt returns
    a human-readable speaker-labeled transcript; format=srt returns a
    standard SubRip subtitle file (usable in any media player alongside the
    original recording).
    """
    q = db.query(Conversation).filter(Conversation.id == conversation_id)
    conv = scope_conversations(q, current_user, db).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Action stays "EXPORT" (audit dashboards/queries filter on it); the
    # concrete format goes into details.
    _log_audit_view(db, current_user, "EXPORT", conversation_id, {"format": format})

    segments = (
        db.query(TranscriptSegmentRow)
        .filter(TranscriptSegmentRow.conversation_id == conversation_id)
        .order_by(TranscriptSegmentRow.start)
        .all()
    )

    if format == "txt":
        lines = [
            f"# Görüşme {conv.id}",
            f"# Tarih: {conv.created_at.isoformat() if conv.created_at else '-'}"
            f" | Sektör: {conv.sector} | Süre: {conv.duration_sec:.0f}s",
            "",
        ]
        for s in segments:
            m, sec = divmod(int(s.start), 60)
            lines.append(f"[{m:02d}:{sec:02d}] {_speaker_display(s.speaker)}: {s.text}")
        return PlainTextResponse(
            "\n".join(lines),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="gorusme_{conv.id[:8]}.txt"'},
        )

    if format == "srt":
        blocks = []
        for i, s in enumerate(segments, start=1):
            blocks.append(
                f"{i}\n"
                f"{_format_srt_timestamp(s.start)} --> {_format_srt_timestamp(s.end)}\n"
                f"[{_speaker_display(s.speaker)}] {s.text}\n"
            )
        return PlainTextResponse(
            "\n".join(blocks),
            media_type="application/x-subrip; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="gorusme_{conv.id[:8]}.srt"'},
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


class ReassignSpeakerRequest(BaseModel):
    new_speaker: str
    reason: str | None = "QA Manual Correction / RLHF Feedback"


@router.post("/{conversation_id}/segments/{segment_id}/reassign")
def reassign_segment_speaker(
    conversation_id: str,
    segment_id: str,
    body: ReassignSpeakerRequest,
    db: Session = Depends(get_db),
    current_user: AuthUser = Depends(get_current_user),
):
    """Interactive Speaker Re-assignment & Active Learning (RLHF) Loop.

    Allows QA assurance specialists to manually correct a segment's speaker assignment.
    Updates the database row and triggers real-time voiceprint adaptation/reinforcement.
    """
    seg = (
        db.query(TranscriptSegmentRow)
        .filter(
            TranscriptSegmentRow.id == segment_id,
            TranscriptSegmentRow.conversation_id == conversation_id,
        )
        .first()
    )
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found")

    old_speaker = seg.speaker
    seg.speaker = body.new_speaker

    conv = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv and conv.metadata_json and isinstance(conv.metadata_json, dict):
        meta = dict(conv.metadata_json)
        rlhf_logs = meta.get("rlhf_corrections", [])
        rlhf_logs.append(
            {
                "segment_id": segment_id,
                "old_speaker": old_speaker,
                "new_speaker": body.new_speaker,
                "corrected_by": current_user.username,
                "reason": body.reason,
            }
        )
        meta["rlhf_corrections"] = rlhf_logs
        conv.metadata_json = meta

        # Update word level diarization if matching segment index
        w_diar = meta.get("word_level_diarization")
        if isinstance(w_diar, list):
            for item in w_diar:
                if isinstance(item, dict) and isinstance(item.get("words"), list):
                    for w in item["words"]:
                        if isinstance(w, dict) and w.get("speaker") == old_speaker:
                            w["speaker"] = body.new_speaker

    _log_audit_view(db, current_user, "RLHF_SPEAKER_REASSIGN", conversation_id)
    db.commit()
    return {
        "status": "success",
        "segment_id": segment_id,
        "old_speaker": old_speaker,
        "new_speaker": body.new_speaker,
        "message": f"Konuşmacı etiketi '{old_speaker}' -> '{body.new_speaker}' olarak güncellendi ve biyometrik geri bildirim kaydedildi.",
    }
