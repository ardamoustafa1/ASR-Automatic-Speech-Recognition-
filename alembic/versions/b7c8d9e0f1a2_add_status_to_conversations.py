"""Add processing status lifecycle to conversations

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-07-12 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "conversations",
        sa.Column("status", sa.String(length=16), nullable=False, server_default="completed"),
    )
    op.add_column("conversations", sa.Column("error_message", sa.Text(), nullable=True))
    op.create_index("ix_conversations_status", "conversations", ["status"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_conversations_status", table_name="conversations")
    op.drop_column("conversations", "error_message")
    op.drop_column("conversations", "status")
