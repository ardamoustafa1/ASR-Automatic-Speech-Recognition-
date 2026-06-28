from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db, limiter
from asr_pro.api.routes.auth import require_admin
from asr_pro.api.schemas.alerts import AlertEventOut, AlertRuleCreate, AlertRuleOut
from asr_pro.core.alert_engine import evaluate_alerts
from asr_pro.db.models import AlertEvent, AlertRule, new_uuid

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/rules", response_model=list[AlertRuleOut])
def list_alert_rules(db: Session = Depends(get_db)):
    return db.query(AlertRule).order_by(AlertRule.name).all()


@router.post(
    "/rules", response_model=AlertRuleOut, status_code=201, dependencies=[Depends(require_admin)]
)
@limiter.limit("20/minute")
def create_alert_rule(request: Request, payload: AlertRuleCreate, db: Session = Depends(get_db)):
    rule = AlertRule(id=new_uuid(), **payload.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=204, dependencies=[Depends(require_admin)])
def delete_alert_rule(rule_id: str, db: Session = Depends(get_db)):
    rule = db.query(AlertRule).filter(AlertRule.id == rule_id).first()
    if not rule:
        raise HTTPException(404, "Kural bulunamadı")
    db.delete(rule)
    db.commit()
    return None


@router.get("", response_model=list[AlertEventOut])
def list_alerts(acknowledged: Optional[bool] = None, db: Session = Depends(get_db)):
    q = db.query(AlertEvent).order_by(AlertEvent.created_at.desc())
    if acknowledged is not None:
        q = q.filter(AlertEvent.acknowledged == acknowledged)
    events = q.limit(100).all()
    return [
        AlertEventOut(
            id=e.id,
            alert_rule_id=e.alert_rule_id,
            title=e.title,
            summary=e.summary,
            severity=e.severity,
            payload=e.payload or {},
            acknowledged=e.acknowledged,
            created_at=e.created_at.isoformat() if e.created_at else "",
        )
        for e in events
    ]


@router.patch("/{alert_id}/acknowledge", dependencies=[Depends(require_admin)])
@limiter.limit("20/minute")
def acknowledge_alert(request: Request, alert_id: str, db: Session = Depends(get_db)):
    event = db.query(AlertEvent).filter(AlertEvent.id == alert_id).first()
    if not event:
        raise HTTPException(404, "Uyarı bulunamadı")
    event.acknowledged = True
    db.commit()
    return {"ok": True}


@router.post("/evaluate", dependencies=[Depends(require_admin)])
@limiter.limit("5/minute")
def run_alert_evaluation(request: Request, db: Session = Depends(get_db)):
    events = evaluate_alerts(db)
    db.commit()
    return {"triggered": len(events), "events": [e.title for e in events]}
