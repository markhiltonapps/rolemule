"""Add encrypted OpenRouter BYOK column to users.

Revision ID: 20261014_028
Revises: 20260715_027
Create Date: 2026-10-14
"""

from alembic import op
import sqlalchemy as sa

revision = "20261014_028"
down_revision = "20260715_027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("openrouter_api_key_encrypted", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "openrouter_api_key_encrypted")
