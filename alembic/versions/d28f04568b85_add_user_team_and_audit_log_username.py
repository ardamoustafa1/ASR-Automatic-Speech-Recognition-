"""Add user team and audit log username

Revision ID: d28f04568b85
Revises: 1869473b176c
Create Date: 2026-07-06 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d28f04568b85"
down_revision: Union[str, Sequence[str], None] = "1869473b176c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("users", sa.Column("team", sa.String(length=64), nullable=True))
    op.add_column("audit_logs", sa.Column("username", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_audit_logs_username"), "audit_logs", ["username"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_audit_logs_username"), table_name="audit_logs")
    op.drop_column("audit_logs", "username")
    op.drop_column("users", "team")
