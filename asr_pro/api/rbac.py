# Role definitions, role-based authorization dependencies, and data-scoping helpers.
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Query, Session

from asr_pro.api.routes.auth import User, get_current_user
from asr_pro.db.models import Conversation
from asr_pro.db.models import User as DBUser

AGENT = "agent"
TEAM_LEAD = "team_lead"
QA = "qa"
ADMIN = "admin"
AUDITOR = "auditor"

ALL_ROLES = (AGENT, TEAM_LEAD, QA, ADMIN, AUDITOR)
# Roles with unrestricted read access to conversation data (QA review / compliance oversight).
FULL_VISIBILITY_ROLES = (QA, AUDITOR, ADMIN)


def require_roles(*allowed_roles: str):
    """Dependency factory: only lets the request through if the current user's
    (live, DB-backed) role is one of `allowed_roles`."""

    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return _check


def scope_conversations(query: Query, current_user: User, db: Session) -> Query:
    """Restrict a Conversation query to what `current_user`'s role is allowed to see.

    - agent: only conversations they personally recorded.
    - team_lead: conversations recorded by any agent on the same team.
    - qa / auditor / admin: unrestricted (full visibility roles).
    """
    if current_user.role in FULL_VISIBILITY_ROLES:
        return query

    if current_user.role == TEAM_LEAD:
        teammate_usernames = [
            u.username for u in db.query(DBUser.username).filter(DBUser.team == current_user.team).all()
        ]
        return query.filter(Conversation.agent_id.in_(teammate_usernames or [current_user.username]))

    # Default (agent, or any unrecognized role): strictly own recordings only.
    return query.filter(Conversation.agent_id == current_user.username)
