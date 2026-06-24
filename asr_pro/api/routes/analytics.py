from __future__ import annotations
from typing import Optional
"""API route: analytics — trends, dashboard summary, and top keywords."""

from fastapi import APIRouter, Depends, Query
from fastapi_cache.decorator import cache
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db
from asr_pro.api.schemas.analytics import DailyPointOut, DashboardOut, TrendOut
from asr_pro.core.trend_engine import compute_trend, dashboard_summary, top_keywords

from asr_pro.api.routes.auth import get_current_user

router = APIRouter(prefix="/analytics", tags=["analytics"], dependencies=[Depends(get_current_user)])


@router.get("/dashboard", response_model=DashboardOut)
def get_dashboard(db: Session = Depends(get_db)):
    """Return a high-level KPI summary for the dashboard."""
    return dashboard_summary(db)


@router.get("/trends", response_model=TrendOut)
@cache(expire=60)
def get_trends(
    keyword: Optional[str] = None,
    topic_id: Optional[str] = None,
    rule_id: Optional[str] = None,
    window: str = Query("7d", pattern="^(24h|7d|30d)$"),
    sector: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Return percentage-change trend data for a keyword or topic."""
    trend = compute_trend(
        db,
        topic_id=topic_id,
        rule_id=rule_id,
        window=window,
    )
    return {
        "keyword": trend.keyword,
        "topic_id": topic_id,
        "rule_id": rule_id,
        "window": window,
        "current_count": trend.current_count,
        "previous_count": trend.previous_count,
        "current_conversations": trend.current_count,
        "previous_conversations": trend.previous_count,
        "pct_change": trend.pct_change,
        "daily_series": [],
        "anomaly": False,
        "summary": "Trend hesaplandı."
    }


@router.get("/top-keywords")
def get_top_keywords(
    window: str = Query("7d", pattern="^(24h|7d|30d)$"),
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
):
    """Return the most frequently matched keywords in the given time window."""
    return top_keywords(db, window=window, limit=limit)
