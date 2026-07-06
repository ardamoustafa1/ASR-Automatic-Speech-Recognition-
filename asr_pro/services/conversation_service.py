# Business logic orchestration layer coordinating transcription, diarization, and NLP engines.
from __future__ import annotations

"""Persist conversations and keyword analysis results."""


from collections.abc import Sequence
from typing import Any

import numpy as np
from loguru import logger
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
from asr_pro.observability.metrics import churn_risk_score, nlp_engine_duration_seconds, time_block


def rules_from_db(db: Session, sector: str | None = None) -> list[RuleInput]:
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
    audio_path: str | None = None,
    uploaded_name: str | None = None,
    asr_confidence: float = 0.0,
    quality_gate_passed: bool = True,
    speakers: list[str] | None = None,
    metadata: dict | None = None,
    agent_id: str | None = None,
) -> dict:
    from asr_pro.core.churn_engine import analyze_churn_risk
    from asr_pro.core.compliance_engine import analyze_compliance_risk
    from asr_pro.core.empathy_engine import analyze_soft_skills
    from asr_pro.services.diarization_service import DiarizationService

    # 1. Perform Speaker Diarization & Role Assignment (Agent vs Customer)
    diarizer = DiarizationService.get_instance()
    segments, agent_speaker_id, customer_speaker_id = diarizer.assign_speakers_to_segments(
        segments_data, audio_path=audio_path
    )
    if speakers and not agent_speaker_id:
        agent_speaker_id = speakers[0] if len(speakers) > 0 else "SPEAKER_00"
        customer_speaker_id = speakers[1] if len(speakers) > 1 else "SPEAKER_01"

    # Diarization can degenerate to a single detected speaker (very short calls,
    # overlapping stereo channels, etc.). When that happens, analyze_soft_skills
    # falls back to treating the WHOLE transcript as the agent, and
    # analyze_churn_risk falls back to analyzing the WHOLE transcript unfiltered
    # (customer_speaker_id=None disables its speaker filter). Both fallbacks
    # exist so the engines still produce a result with no diarization data at
    # all - but here we DO have a diarization attempt, it just failed to
    # separate two speakers, so silently trusting the output would risk
    # attributing customer speech to the agent (or vice versa). Surface that
    # explicitly instead of letting it look identical to a confident result.
    speaker_separation_reliable = bool(agent_speaker_id) and bool(customer_speaker_id) and (
        agent_speaker_id != customer_speaker_id
    )
    if not speaker_separation_reliable:
        logger.warning(
            "Diarization could not reliably separate agent/customer speakers "
            f"(agent={agent_speaker_id!r}, customer={customer_speaker_id!r}). "
            "Churn/empathy scores for this conversation may mix both parties' speech."
        )

    # 2. Analyze Keyword Rules & Topics
    with time_block(nlp_engine_duration_seconds, engine="keyword"):
        rules = rules_from_db(db, sector)
        hits = analyze_keywords(segments, rules, sector=sector)
    with time_block(nlp_engine_duration_seconds, engine="topic"):
        topics = topics_from_db(db)
        topic_matches = classify_topics_from_hits(hits, topics)

    # 3. Analyze Agent Soft Skills & Empathy (using identified agent_speaker_id)
    with time_block(nlp_engine_duration_seconds, engine="empathy"):
        empathy_result = analyze_soft_skills(segments, agent_speaker_id=agent_speaker_id)

    # 4. Calculate Customer Churn Risk (using identified customer_speaker_id)
    with time_block(nlp_engine_duration_seconds, engine="churn"):
        churn_result = analyze_churn_risk(segments, customer_speaker_id=customer_speaker_id)
    churn_risk_score.observe(churn_result.risk_score)

    # 5. Regulatory Compliance Monitoring (KVKK/BTK/SPK red-flag detection),
    # agent-only since these are obligations on the agent/company, not the customer.
    with time_block(nlp_engine_duration_seconds, engine="compliance"):
        compliance_violations = analyze_compliance_risk(
            segments, domain_key=sector, agent_speaker_id=agent_speaker_id
        )

    duration = max((s.end for s in segments), default=0.0)

    # Never report a confident-looking score off an unreliable speaker split -
    # downgrade churn confidence and flag the empathy summary regardless of
    # what the engines themselves computed from a mixed/unfiltered transcript.
    churn_confidence = churn_result.confidence if speaker_separation_reliable else "Düşük"
    empathy_summary = empathy_result.analysis_summary
    if not speaker_separation_reliable:
        empathy_summary = (
            "⚠️ Konuşmacı ayrımı güvenilir değil, bu skor müşteri ve temsilci "
            "konuşmalarının karışımı olabilir. " + empathy_summary
        )

    final_metadata = {"uploaded_name": uploaded_name, **(metadata or {})}
    final_metadata["churn_risk"] = churn_result.risk_score
    final_metadata["is_high_risk"] = churn_result.is_high_risk
    final_metadata["churn_risk_breakdown"] = churn_result.risk_breakdown
    final_metadata["churn_confidence"] = churn_confidence
    final_metadata["agent_retention_score"] = getattr(churn_result, "agent_retention_score", 100.0)
    final_metadata["was_deescalated"] = getattr(churn_result, "was_deescalated", False)
    final_metadata["average_filler_ratio"] = getattr(churn_result, "average_filler_ratio", 0.0)
    final_metadata["detected_prices"] = list(getattr(churn_result, "detected_prices", ()))
    final_metadata["churn_trajectory"] = list(getattr(churn_result, "trajectory", ()))
    final_metadata["empathy_score"] = empathy_result.score
    final_metadata["empathy_summary"] = empathy_summary
    final_metadata["agent_speaker_id"] = agent_speaker_id
    final_metadata["customer_speaker_id"] = customer_speaker_id
    final_metadata["speaker_separation_reliable"] = speaker_separation_reliable
    final_metadata["compliance_violations"] = [
        {
            "severity": v.severity,
            "category": v.category,
            "reason": v.reason,
            "segment_text": v.segment_text,
            "timestamp_start": v.timestamp_start,
            "timestamp_end": v.timestamp_end,
        }
        for v in compliance_violations
    ]

    # 6. Compute LLM Discourse Guard (FCR, CES, Adherence) & ITU-T P.863 MOS Quality
    from asr_pro.services.llm_discourse_guard import LLMDiscourseGuard
    from asr_pro.services.mos_estimator import MOSEstimator
    discourse_metrics = LLMDiscourseGuard.analyze_call_metrics(segments)
    mos_metrics = MOSEstimator.estimate_mos(audio_path) if audio_path else MOSEstimator.estimate_mos(np.array([]))
    crosstalk_events = diarizer.extract_crosstalk_events(segments)
    pitch_profiles = diarizer.extract_speaker_pitch_profiles(segments, audio_path)
    all_spks = sorted(list({s.speaker for s in segments if s.speaker}))
    supervisor_speaker_id = [s for s in all_spks if s not in (agent_speaker_id, customer_speaker_id)][0] if len(all_spks) > 2 else None
    word_level_diarization = [
        {"segment_index": idx, "words": getattr(seg, "words", None)}
        for idx, seg in enumerate(segments)
        if getattr(seg, "words", None)
    ]
    final_metadata.update({
        "discourse_metrics": discourse_metrics,
        "mos_metrics": mos_metrics,
        "crosstalk_events": crosstalk_events,
        "pitch_profiles": pitch_profiles,
        "supervisor_speaker_id": supervisor_speaker_id,
        "word_level_diarization": word_level_diarization,
    })

    conv = Conversation(
        id=new_uuid(),
        agent_id=agent_id,
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
            speaker=seg.speaker
            or (speakers[idx] if speakers and idx < len(speakers) else "SPEAKER_00"),
            avg_logprob=getattr(segments_data[idx], "avg_logprob", -1.0)
            if idx < len(segments_data)
            else -1.0,
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
            {
                "topic_id": t.topic_id,
                "slug": t.slug,
                "label_tr": t.label_tr,
                "confidence": t.confidence,
            }
            for t in topic_matches
        ],
        "hit_count": len(hits),
        "alerts_triggered": len(alert_events),
        "diarization": {
            "agent_speaker_id": agent_speaker_id,
            "customer_speaker_id": customer_speaker_id,
            "speaker_separation_reliable": speaker_separation_reliable,
        },
        "empathy": {
            "score": empathy_result.score,
            "summary": empathy_summary,
        },
        "churn": {
            "risk_score": churn_result.risk_score,
            "is_high_risk": churn_result.is_high_risk,
            "risk_breakdown": churn_result.risk_breakdown,
            "confidence": churn_confidence,
        },
        "compliance": final_metadata["compliance_violations"],
    }


def analyze_without_save(
    db: Session,
    segments_data: Sequence[Any],
    sector: str = "omni",
) -> list[KeywordHitResult]:
    from asr_pro.services.diarization_service import DiarizationService

    segments, _, _ = DiarizationService.get_instance().assign_speakers_to_segments(segments_data)
    rules = rules_from_db(db, sector)
    return analyze_keywords(segments, rules, sector=sector)
