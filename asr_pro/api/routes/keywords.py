from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db, limiter
from asr_pro.api.routes.auth import require_admin
from asr_pro.api.schemas.keywords import (
    KeywordRuleCreate,
    KeywordRuleOut,
    KeywordRuleUpdate,
    KeywordTestRequest,
    KeywordTestResponse,
    TopicOut,
)
from asr_pro.core.keyword_engine import RuleInput, evaluate_rule_on_text, hits_to_dict
from asr_pro.db.models import KeywordRule, Topic, new_uuid

router = APIRouter(prefix="/keyword-rules", tags=["keywords"])


@router.get("", response_model=list[KeywordRuleOut])
def list_rules(db: Session = Depends(get_db)):
    return db.query(KeywordRule).order_by(KeywordRule.name).all()


@router.post("", response_model=KeywordRuleOut, status_code=201, dependencies=[Depends(require_admin)])
@limiter.limit("20/minute")
def create_rule(request: Request, payload: KeywordRuleCreate, db: Session = Depends(get_db)):
    rule = KeywordRule(id=new_uuid(), **payload.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@router.get("/{rule_id}", response_model=KeywordRuleOut)
def get_rule(rule_id: str, db: Session = Depends(get_db)):
    rule = db.query(KeywordRule).filter(KeywordRule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Kural bulunamadı")
    return rule


@router.patch("/{rule_id}", response_model=KeywordRuleOut, dependencies=[Depends(require_admin)])
@limiter.limit("20/minute")
def update_rule(request: Request, rule_id: str, payload: KeywordRuleUpdate, db: Session = Depends(get_db)):
    rule = db.query(KeywordRule).filter(KeywordRule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Kural bulunamadı")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(rule, key, value)
    rule.version += 1
    db.commit()
    db.refresh(rule)
    return rule


@router.delete("/{rule_id}", status_code=204, dependencies=[Depends(require_admin)])
@limiter.limit("20/minute")
def delete_rule(request: Request, rule_id: str, db: Session = Depends(get_db)):
    rule = db.query(KeywordRule).filter(KeywordRule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Kural bulunamadı")
    db.delete(rule)
    db.commit()


@router.post("/test", response_model=KeywordTestResponse)
@limiter.limit("20/minute")
def test_keyword(request: Request, payload: KeywordTestRequest, db: Session = Depends(get_db)):
    if payload.rule_id:
        db_rule = db.query(KeywordRule).filter(KeywordRule.id == payload.rule_id).first()
        if not db_rule:
            raise HTTPException(404, "Kural bulunamadı")
        rule = RuleInput(
            id=db_rule.id,
            name=db_rule.name,
            keywords=tuple(db_rule.keywords or []),
            match_mode=db_rule.match_mode,
            fuzzy_threshold=db_rule.fuzzy_threshold,
            severity=db_rule.severity,
            topic_id=db_rule.topic_id,
        )
    elif payload.rule:
        rule = RuleInput(
            id="test",
            name=payload.rule.name,
            keywords=tuple(payload.rule.keywords),
            match_mode=payload.rule.match_mode,
            fuzzy_threshold=payload.rule.fuzzy_threshold,
            severity=payload.rule.severity,
        )
    else:
        raise HTTPException(400, "rule_id veya rule gerekli")

    hits = evaluate_rule_on_text(payload.text, rule)
    return KeywordTestResponse(hits=hits_to_dict(hits))


topics_router = APIRouter(prefix="/topics", tags=["topics"])


@topics_router.get("", response_model=list[TopicOut])
def list_topics(db: Session = Depends(get_db)):
    return db.query(Topic).order_by(Topic.label_tr).all()
