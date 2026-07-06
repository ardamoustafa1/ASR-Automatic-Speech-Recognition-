# Pydantic schema for audit log entries.
from typing import Optional

from pydantic import BaseModel


class AuditLogOut(BaseModel):
    id: str
    user_id: Optional[str]
    username: Optional[str]
    action: str
    target_resource: Optional[str]
    ip_address: Optional[str]
    details: Optional[dict]
    timestamp: str
