from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(32), default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    customer_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sector: Mapped[str] = mapped_column(String(64), default="omni")
    duration_sec: Mapped[float] = mapped_column(Float, default=0.0)
    audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    full_transcript: Mapped[str] = mapped_column(Text, default="")
    asr_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    quality_gate_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TranscriptSegmentRow(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), index=True
    )
    start: Mapped[float] = mapped_column(Float, default=0.0)
    end: Mapped[float] = mapped_column(Float, default=0.0)
    text: Mapped[str] = mapped_column(Text, default="")
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    avg_logprob: Mapped[float] = mapped_column(Float, default=-1.0)


class KeywordRule(Base):
    __tablename__ = "keyword_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(128))
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    match_mode: Mapped[str] = mapped_column(String(32), default="exact")
    fuzzy_threshold: Mapped[float] = mapped_column(Float, default=0.85)
    case_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    sector_scope: Mapped[list | None] = mapped_column(JSON, nullable=True)
    severity: Mapped[str] = mapped_column(String(32), default="info")
    topic_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("topics.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label_tr: Mapped[str] = mapped_column(String(128))
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topics.id"), nullable=True
    )
    seed_keywords: Mapped[list] = mapped_column(JSON, default=list)
    synonyms: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class KeywordHit(Base):
    __tablename__ = "keyword_hits"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id"), index=True
    )
    segment_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("transcript_segments.id"), nullable=True
    )
    rule_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("keyword_rules.id"), nullable=True
    )
    topic_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("topics.id"), nullable=True)
    matched_text: Mapped[str] = mapped_column(String(256), default="")
    keyword: Mapped[str] = mapped_column(String(128), default="")
    match_type: Mapped[str] = mapped_column(String(32), default="exact")
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    timestamp_sec: Mapped[float] = mapped_column(Float, default=0.0)
    speaker: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_before: Mapped[str] = mapped_column(String(256), default="")
    context_after: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    rule_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("keyword_rules.id"), nullable=True
    )
    topic_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("topics.id"), nullable=True)
    keyword: Mapped[str] = mapped_column(String(128), default="")
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    conversation_count: Mapped[int] = mapped_column(Integer, default=0)
    window: Mapped[str] = mapped_column(String(16), default="7d")
    pct_change_vs_prev: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(128))
    target_type: Mapped[str] = mapped_column(String(32), default="keyword")
    target_id: Mapped[str] = mapped_column(String(36))
    condition: Mapped[dict] = mapped_column(JSON, default=dict)
    channels: Mapped[list] = mapped_column(JSON, default=list)
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=1440)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    alert_rule_id: Mapped[str] = mapped_column(String(36), ForeignKey("alert_rules.id"), index=True)
    title: Mapped[str] = mapped_column(String(256))
    summary: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(32), default="info")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class TrendCallLog(Base):
    """Replaces the sqlite3 call_logs table in trend_engine.py for PostgreSQL support."""

    __tablename__ = "trend_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(128), index=True)
    call_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_resource: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
