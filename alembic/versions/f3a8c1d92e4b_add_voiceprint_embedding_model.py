"""Add embedding_model to agent_voiceprints

Revision ID: f3a8c1d92e4b
Revises: d28f04568b85
Create Date: 2026-07-06 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a8c1d92e4b"
down_revision: Union[str, Sequence[str], None] = "d28f04568b85"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "agent_voiceprints",
        sa.Column(
            "embedding_model",
            sa.String(length=32),
            nullable=False,
            server_default="fft-legacy-v1",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("agent_voiceprints", "embedding_model")
