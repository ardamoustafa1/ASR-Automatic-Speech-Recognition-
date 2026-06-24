from pydantic import BaseModel, Field
from typing import Optional, Literal


class KeywordRuleCreate(BaseModel):
    name: str
    keywords: list[str]
    match_mode: Literal["exact", "fuzzy", "regex", "semantic"] = "semantic"
    fuzzy_threshold: float = 0.85
    case_sensitive: bool = False
    sector_scope: Optional[list[str]] = None
    severity: Literal["info", "warning", "critical"] = "info"
    topic_id: Optional[str] = None
    is_active: bool = True


class KeywordRuleUpdate(BaseModel):
    name: Optional[str] = None
    keywords: Optional[list[str]] = None
    match_mode: Optional[Literal["exact", "fuzzy", "regex", "semantic"]] = None
    fuzzy_threshold: Optional[float] = None
    case_sensitive: Optional[bool] = None
    sector_scope: Optional[list[str]] = None
    severity: Optional[Literal["info", "warning", "critical"]] = None
    topic_id: Optional[str] = None
    is_active: Optional[bool] = None


class KeywordRuleOut(BaseModel):
    id: str
    name: str
    keywords: list[str]
    match_mode: str
    fuzzy_threshold: float
    case_sensitive: bool
    sector_scope: Optional[list[str]]
    severity: str
    topic_id: Optional[str]
    is_active: bool

    model_config = {"from_attributes": True}


class KeywordTestRequest(BaseModel):
    text: str
    rule: Optional[KeywordRuleCreate] = None
    rule_id: Optional[str] = None


class KeywordTestResponse(BaseModel):
    hits: list[dict]


class TopicOut(BaseModel):
    id: str
    slug: str
    label_tr: str
    seed_keywords: list[str]
    synonyms: list[str]
    parent_id: Optional[str] = None

    model_config = {"from_attributes": True}
