from pydantic import BaseModel
from typing import Literal


class AlertRuleCreate(BaseModel):
    name: str
    target_type: Literal["keyword", "topic"] = "keyword"
    target_id: str
    condition: dict
    channels: list[str] = ["in_app"]
    cooldown_minutes: int = 1440
    is_active: bool = True


class AlertRuleOut(BaseModel):
    id: str
    name: str
    target_type: str
    target_id: str
    condition: dict
    channels: list
    cooldown_minutes: int
    is_active: bool

    model_config = {"from_attributes": True}


class AlertEventOut(BaseModel):
    id: str
    alert_rule_id: str
    title: str
    summary: str
    severity: str
    payload: dict
    acknowledged: bool
    created_at: str

    model_config = {"from_attributes": True}
