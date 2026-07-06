# API route: audit-logs — read-only compliance/audit trail, admin & auditor only.
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from asr_pro.api.deps import get_db
from asr_pro.api.rbac import ADMIN, AUDITOR, require_roles
from asr_pro.api.schemas.audit import AuditLogOut
from asr_pro.db.models import AuditLog

router = APIRouter(
    prefix="/audit-logs",
    tags=["audit"],
    dependencies=[Depends(require_roles(ADMIN, AUDITOR))],
)


@router.get("", response_model=list[AuditLogOut])
def list_audit_logs(
    username: Optional[str] = None,
    action: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List audit trail entries — who did/viewed what, when. Filterable + paginated."""
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if username:
        q = q.filter(AuditLog.username == username)
    if action:
        q = q.filter(AuditLog.action == action)
    if date_from:
        q = q.filter(AuditLog.timestamp >= date_from)
    if date_to:
        q = q.filter(AuditLog.timestamp <= date_to)

    entries = q.offset(offset).limit(limit).all()
    return [
        AuditLogOut(
            id=e.id,
            user_id=e.user_id,
            username=e.username,
            action=e.action,
            target_resource=e.target_resource,
            ip_address=e.ip_address,
            details=e.details,
            timestamp=e.timestamp.isoformat() if e.timestamp else "",
        )
        for e in entries
    ]
