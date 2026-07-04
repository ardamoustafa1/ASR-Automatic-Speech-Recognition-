# Triggers real-time notifications for supervisors when critical call thresholds are breached.
from __future__ import annotations

"""Alert evaluation and dispatch."""


from datetime import timedelta

from sqlalchemy.orm import Session

from asr_pro.core.trend_engine import compute_trend
from asr_pro.db.models import AlertEvent, AlertRule, new_uuid, utcnow


def _condition_matches(trend, condition: dict) -> bool:
    metric = condition.get("metric", "hit_count")
    operator = condition.get("operator", "pct_increase")
    threshold = float(condition.get("threshold", 30))
    min_count = int(condition.get("min_count", 5))

    value = trend.pct_change if metric == "pct_increase" else trend.current_count
    if trend.current_count < min_count:
        return False

    if operator == "pct_increase":
        return value is not None and value >= threshold
    if operator == "gte":
        return trend.current_count >= threshold
    return False


def evaluate_alerts(db: Session) -> list[AlertEvent]:
    events: list[AlertEvent] = []
    rules = db.query(AlertRule).filter(AlertRule.is_active.is_(True)).all()

    for rule in rules:
        if rule.last_triggered_at:
            cooldown = timedelta(minutes=rule.cooldown_minutes)
            if (
                utcnow() - rule.last_triggered_at.replace(tzinfo=rule.last_triggered_at.tzinfo)
                < cooldown
            ):
                continue

        condition = rule.condition or {}
        window = condition.get("window", "7d")

        if rule.target_type == "topic":
            trend = compute_trend(db, topic_id=rule.target_id, window=window)
        else:
            trend = compute_trend(db, rule_id=rule.target_id, window=window)

        if not _condition_matches(trend, condition):
            continue

        pct = trend.pct_change or 0
        title = f"'{trend.keyword or rule.name}' %{pct:.0f} arttı"
        summary = (
            f"Son {window} içinde {trend.current_count} eşleşme "
            f"(önceki dönem: {trend.previous_count})."
        )

        event = AlertEvent(
            id=new_uuid(),
            alert_rule_id=rule.id,
            title=title,
            summary=summary,
            severity="warning" if pct >= 40 else "info",
            payload={
                "pct_change": pct,
                "current_count": trend.current_count,
                "previous_count": trend.previous_count,
                "window": window,
            },
        )
        db.add(event)
        rule.last_triggered_at = utcnow()
        events.append(event)

        # Dispatch webhook if configured
        if "webhook" in (rule.channels or []) or "in_app" in (rule.channels or []):
            _dispatch_webhook(event)

    return events


def _dispatch_webhook(event: AlertEvent) -> None:
    """Send alert to external webhook URL if configured."""
    from loguru import logger

    from asr_pro.config import WEBHOOK_URL

    if not WEBHOOK_URL:
        return

    try:
        import httpx

        payload = {
            "title": event.title,
            "summary": event.summary,
            "severity": event.severity,
            "details": event.payload,
            "timestamp": event.created_at.isoformat() if event.created_at else None,
        }

        # Fire-and-forget in a background thread or synchronous if timeout is low
        # Using a low timeout for sync requests so we don't block the API
        with httpx.Client(timeout=3.0) as client:
            resp = client.post(WEBHOOK_URL, json=payload)
            if resp.status_code >= 400:
                logger.warning(f"Webhook delivery failed ({resp.status_code}): {resp.text}")
            else:
                logger.info(f"Webhook delivered: {event.title}")
    except Exception as e:
        logger.error(f"Webhook dispatch error: {str(e)}")
