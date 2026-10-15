"""Add Find Jobs tables: job_finder_configs and job_leads.

Revision ID: 20261015_029
Revises: 20261014_028
Create Date: 2026-10-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20261015_029"
down_revision = "20261014_028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_finder_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "searches",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "companies",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("last_run", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )

    op.create_table(
        "job_leads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("company", sa.String(length=200), nullable=False),
        sa.Column("location", sa.String(length=300), nullable=True),
        sa.Column(
            "is_remote", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "matched_searches",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="new"
        ),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "user_id", "source", "external_id", name="uq_job_leads_user_source_external"
        ),
    )
    op.create_index(
        "ix_job_leads_user_status_seen",
        "job_leads",
        ["user_id", "status", "first_seen_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_leads_user_status_seen", table_name="job_leads")
    op.drop_table("job_leads")
    op.drop_table("job_finder_configs")
