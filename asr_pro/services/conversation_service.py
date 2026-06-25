from __future__ import annotations

"""Persist conversations and keyword analysis results."""


from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy.orm import Session

from asr_pro.core.alert_engine import evaluate_alerts
from asr_pro.core.keyword_engine import (
    KeywordHitResult,
    RuleInput,
    SegmentInput,
    analyze_keywords,
    hits_to_dict,
)
from asr_pro.core.topic_classifier import TopicInput, classify_topics_from_hits
from asr_pro.db.models import (
    Conversation,
    KeywordHit,
    KeywordRule,
    Topic,
    TranscriptSegmentRow,
    new_uuid,
)


def rules_from_db(db: Session, sector: Optional[str] = None) -> list[RuleInput]:
    q = db.query(KeywordRule).filter(KeywordRule.is_active.is_(True))
    rules = q.all()
    result = []
    for r in rules:
        scope = tuple(r.sector_scope) if r.sector_scope else None
        if scope and sector and sector not in scope:
            continue
        result.append(
            RuleInput(
                id=r.id,
                name=r.name,
                keywords=tuple(r.keywords or []),
                match_mode=r.match_mode,
                fuzzy_threshold=r.fuzzy_threshold,
                case_sensitive=r.case_sensitive,
                severity=r.severity,
                topic_id=r.topic_id,
                sector_scope=scope,
            )
        )
    return result


def topics_from_db(db: Session) -> list[TopicInput]:
    return [
        TopicInput(
            id=t.id,
            slug=t.slug,
            label_tr=t.label_tr,
            seed_keywords=tuple(t.seed_keywords or []),
            synonyms=tuple(t.synonyms or []),
        )
        for t in db.query(Topic).all()
    ]


def segments_from_any(segments_data: Sequence[Any]) -> list[SegmentInput]:
    result = []
    for idx, seg in enumerate(segments_data):
        if isinstance(seg, SegmentInput):
            result.append(seg)
            continue
        result.append(
            SegmentInput(
                start=float(getattr(seg, "start", 0)),
                end=float(getattr(seg, "end", 0)),
                text=str(getattr(seg, "text", "")),
                speaker=getattr(seg, "speaker", None),
                segment_index=idx,
            )
        )
    return result


def save_conversation_with_analysis(
    db: Session,
    *,
    segments_data: Sequence[Any],
    full_transcript: str,
    sector: str = "omni",
    audio_path: Optional[str] = None,
    uploaded_name: Optional[str] = None,
    asr_confidence: float = 0.0,
    quality_gate_passed: bool = True,
    speakers: Optional[list[str]] = None,
    metadata: Optional[dict] = None,
) -> dict:
    from asr_pro.core.churn_engine import analyze_churn_risk

    segments = segments_from_any(segments_data)
    rules = rules_from_db(db, sector)
    hits = analyze_keywords(segments, rules, sector=sector)
    topics = topics_from_db(db)
    topic_matches = classify_topics_from_hits(hits, topics)

    # Calculate Churn Risk
    customer_speaker_id = speakers[0] if speakers else None  # Optional assumption
    churn_result = analyze_churn_risk(segments, customer_speaker_id=customer_speaker_id)

    duration = max((s.end for s in segments), default=0.0)

    final_metadata = {"uploaded_name": uploaded_name, **(metadata or {})}
    final_metadata["churn_risk"] = churn_result.risk_score
    final_metadata["is_high_risk"] = churn_result.is_high_risk

    conv = Conversation(
        id=new_uuid(),
        sector=sector,
        duration_sec=duration,
        audio_path=audio_path,
        full_transcript=full_transcript,
        asr_confidence=asr_confidence,
        quality_gate_passed=quality_gate_passed,
        metadata_json=final_metadata,
    )
    db.add(conv)
    db.flush()

    seg_rows: dict[int, str] = {}
    for idx, seg in enumerate(segments):
        row = TranscriptSegmentRow(
            id=new_uuid(),
            conversation_id=conv.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=seg.speaker or (speakers[idx] if speakers and idx < len(speakers) else None),
            avg_logprob=getattr(segments_data[idx], "avg_logprob", -1.0) if idx < len(segments_data) else -1.0,
        )
        db.add(row)
        seg_rows[idx] = row.id

    for hit in hits:
        db.add(
            KeywordHit(
                id=new_uuid(),
                conversation_id=conv.id,
                segment_id=seg_rows.get(hit.segment_index),
                rule_id=hit.rule_id,
                topic_id=hit.topic_id,
                matched_text=hit.matched_text,
                keyword=hit.keyword,
                match_type=hit.match_type,
                confidence=hit.confidence,
                timestamp_sec=hit.timestamp_sec,
                speaker=hit.speaker,
                context_before=hit.context[:120],
            )
        )

    db.commit()

    alert_events = evaluate_alerts(db)
    db.commit()

    return {
        "conversation_id": conv.id,
        "hits": hits_to_dict(hits),
        "topics": [
            {"topic_id": t.topic_id, "slug": t.slug, "label_tr": t.label_tr, "confidence": t.confidence}
            for t in topic_matches
        ],
        "hit_count": len(hits),
        "alerts_triggered": len(alert_events),
    }


def analyze_without_save(
    db: Session,
    segments_data: Sequence[Any],
    sector: str = "omni",
) -> list[KeywordHitResult]:
    segments = segments_from_any(segments_data)
    rules = rules_from_db(db, sector)
    return analyze_keywords(segments, rules, sector=sector)
