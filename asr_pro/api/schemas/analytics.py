# Pydantic schemas for dashboard KPI metrics and performance trend data.
from typing import Optional

from pydantic import BaseModel


class TrendQuery(BaseModel):
    keyword: Optional[str] = None
    topic_id: Optional[str] = None
    rule_id: Optional[str] = None
    window: str = "7d"
    sector: Optional[str] = None


class DailyPointOut(BaseModel):
    date: str
    hit_count: int
    conversation_count: int


class TrendOut(BaseModel):
    keyword: str
    topic_id: Optional[str]
    rule_id: Optional[str]
    window: str
    current_count: int
    previous_count: int
    current_conversations: int
    previous_conversations: int
    pct_change: Optional[float]
    daily_series: list[DailyPointOut]
    anomaly: bool


class DashboardOut(BaseModel):
    hits_today: int
    conversations_total: int
    active_alerts: int
    top_rising_keyword: Optional[dict]
