"""
API endpoints for Find Jobs — saved searches, watched company careers pages,
search runs, and the resulting job leads.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from models.database import JobLead
from utils.auth import get_current_user
from utils.cache import check_rate_limit
from utils.database import get_database
from utils.error_responses import (
    APIError,
    ErrorCode,
    internal_error,
    not_found_error,
    rate_limit_error,
    validation_error,
)
from utils.job_finder import (
    LEAD_STATUSES,
    MAX_COMPANIES,
    MAX_SEARCH_CHARS,
    MAX_SEARCHES,
    claim_run_lock,
    get_or_create_config,
    is_run_active,
    release_run_lock,
    run_job_finder_background,
    stale_cutoff,
)
from utils.job_sources import board_page_url, default_company_name, parse_board_reference
from utils.logging_config import sanitize_log_value

logger: logging.Logger = logging.getLogger(__name__)
router: APIRouter = APIRouter()

RATE_LIMIT_RUNS_PER_HOUR = 12
MAX_LEADS_RETURNED = 300


# =============================================================================
# MODELS
# =============================================================================


def _clean_searches(value: List[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in value:
        text = " ".join(str(raw).split())
        if not text:
            continue
        if len(text) > MAX_SEARCH_CHARS:
            raise ValueError(f"Each search must be at most {MAX_SEARCH_CHARS} characters")
        if text.lower() not in (s.lower() for s in cleaned):
            cleaned.append(text)
    if len(cleaned) > MAX_SEARCHES:
        raise ValueError(f"At most {MAX_SEARCHES} searches")
    return cleaned


class JobFinderConfigUpdate(BaseModel):
    """Replace saved searches and watched companies."""

    searches: List[str] = Field(default_factory=list, max_length=200)
    companies: List[str] = Field(
        default_factory=list,
        max_length=200,
        description="Careers page links, optionally prefixed 'Name | link'",
    )

    @field_validator("searches")
    @classmethod
    def _validate_searches(cls, value: List[str]) -> List[str]:
        return _clean_searches(value)


class RunRequest(BaseModel):
    """Start a search run."""

    include_web: bool = Field(False, description="Also run AI web search (uses your AI key)")


class LeadStatusUpdate(BaseModel):
    """Change a lead's status."""

    status: str

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        if value not in LEAD_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(LEAD_STATUSES)}")
        return value


# =============================================================================
# HELPERS
# =============================================================================


def _user_uuid(current_user: Dict[str, Any]) -> uuid.UUID:
    user_id = current_user.get("id") or current_user.get("_id")
    return uuid.UUID(str(user_id))


def _parse_company_lines(lines: List[str], existing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Parse company lines (``link`` or ``Name | link``) into board entries.

    Raises:
        APIError: VAL_* listing every line that is not a supported careers page
    """
    known_names = {
        f"{c.get('platform')}:{c.get('token')}".lower(): c.get("name")
        for c in existing
        if isinstance(c, dict)
    }
    companies: List[Dict[str, str]] = []
    seen: set[str] = set()
    invalid: List[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line:
            continue
        name: Optional[str] = None
        link = line
        if "|" in line:
            name_part, link = (part.strip() for part in line.split("|", 1))
            name = " ".join(name_part.split())[:120] or None
        parsed = parse_board_reference(link)
        if not parsed:
            invalid.append(line[:200])
            continue
        platform, token = parsed
        key = f"{platform}:{token}".lower()
        if key in seen:
            continue
        seen.add(key)
        companies.append(
            {
                "platform": platform,
                "token": token,
                "name": name or known_names.get(key) or default_company_name(token),
            }
        )
    if invalid:
        raise validation_error(
            "These links aren't supported careers pages. Use the link to the page that "
            "lists all of a company's open roles: " + "; ".join(invalid[:5])
        )
    if len(companies) > MAX_COMPANIES:
        raise validation_error(f"At most {MAX_COMPANIES} companies")
    return companies


def _config_payload(config: Any, running: bool) -> Dict[str, Any]:
    last_run = config.last_run or None
    statuses = (last_run or {}).get("company_status") or {}
    companies = []
    for c in config.companies or []:
        if not isinstance(c, dict):
            continue
        key = f"{c.get('platform')}:{c.get('token')}"
        companies.append(
            {
                "platform": c.get("platform"),
                "token": c.get("token"),
                "name": c.get("name"),
                "page_url": board_page_url(str(c.get("platform")), str(c.get("token"))),
                "last_status": statuses.get(key),
            }
        )
    return {
        "searches": list(config.searches or []),
        "companies": companies,
        "last_run": last_run,
        "running": running,
    }


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("")
async def get_job_finder(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """Return saved searches, watched companies, last run summary, and running flag."""
    try:
        user_id = _user_uuid(current_user)
        config = await get_or_create_config(db, user_id)
        await db.commit()
        running = await is_run_active(str(user_id))
        return _config_payload(config, running)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load Find Jobs config: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to load Find Jobs settings")


@router.put("/config")
async def update_job_finder_config(
    request: JobFinderConfigUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """Replace saved searches and watched companies."""
    try:
        user_id = _user_uuid(current_user)
        config = await get_or_create_config(db, user_id)
        companies = _parse_company_lines(request.companies, list(config.companies or []))
        config.searches = request.searches
        config.companies = companies
        flag_modified(config, "searches")
        flag_modified(config, "companies")
        await db.commit()
        logger.info(
            "Find Jobs config saved for user %s (%s searches, %s companies)",
            sanitize_log_value(str(user_id)),
            len(request.searches),
            len(companies),
        )
        running = await is_run_active(str(user_id))
        return _config_payload(config, running)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save Find Jobs config: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to save Find Jobs settings")


@router.post("/run", status_code=202)
async def start_job_finder_run(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """Start a background search run (boards always; AI web search optional)."""
    try:
        user_id = _user_uuid(current_user)

        llm_provider: Optional[str] = None
        user_api_key: Optional[str] = None
        preferred_model: Optional[str] = None
        web_search = request.include_web
        if web_search:
            from utils.llm_context import require_user_llm_context
            from utils.llm_preferences import load_preferred_model

            _u, llm_ctx, _p = await require_user_llm_context(db, user_id)
            llm_provider = llm_ctx.provider
            user_api_key = llm_ctx.user_api_key
            preferred_model = await load_preferred_model(
                db, user_id, user_api_key, has_credentials=True
            )
            # Ollama has no web-search tools
            web_search = llm_provider != "ollama"

        is_allowed, _remaining = await check_rate_limit(
            identifier=f"{user_id}:job_finder_run",
            limit=RATE_LIMIT_RUNS_PER_HOUR,
            window_seconds=3600,
        )
        if not is_allowed:
            raise rate_limit_error(
                f"Rate limit exceeded. Maximum {RATE_LIMIT_RUNS_PER_HOUR} job searches per hour.",
                retry_after=3600,
            )

        if not await claim_run_lock(str(user_id)):
            raise APIError(
                ErrorCode.RESOURCE_CONFLICT,
                "A job search is already running.",
                status_code=409,
            )

        try:
            background_tasks.add_task(
                run_job_finder_background,
                user_id=str(user_id),
                include_web=web_search,
                llm_provider=llm_provider,
                user_api_key=user_api_key,
                preferred_model=preferred_model,
            )
        except Exception:
            await release_run_lock(str(user_id))
            raise

        logger.info(
            "Find Jobs run started for user %s (web=%s)",
            sanitize_log_value(str(user_id)),
            sanitize_log_value(web_search),
        )
        return {
            "status": "started",
            "web_search": web_search,
            "web_search_skipped_reason": (
                "Your AI provider (Ollama) can't search the web."
                if request.include_web and not web_search
                else None
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start Find Jobs run: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to start the job search")


@router.get("/leads")
async def list_job_leads(
    status: str = Query("new"),
    limit: int = Query(MAX_LEADS_RETURNED, ge=1, le=MAX_LEADS_RETURNED),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """List leads by status (``new`` hides leads not seen recently), with counts."""
    try:
        if status not in LEAD_STATUSES:
            raise validation_error(f"status must be one of: {', '.join(LEAD_STATUSES)}")
        user_id = _user_uuid(current_user)
        cutoff = stale_cutoff()
        fresh_or_not_new = or_(JobLead.status != "new", JobLead.last_seen_at >= cutoff)

        query = (
            select(JobLead)
            .where(and_(JobLead.user_id == user_id, JobLead.status == status, fresh_or_not_new))
            .order_by(
                JobLead.posted_at.desc().nulls_last(),
                JobLead.first_seen_at.desc(),
            )
            .limit(limit)
        )
        leads = (await db.execute(query)).scalars().all()

        counts_rows = await db.execute(
            select(JobLead.status, func.count())
            .where(and_(JobLead.user_id == user_id, fresh_or_not_new))
            .group_by(JobLead.status)
        )
        counts = {s: 0 for s in LEAD_STATUSES}
        for row_status, count in counts_rows.all():
            counts[row_status] = count

        return {"leads": [lead.to_dict() for lead in leads], "counts": counts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list job leads: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to load job leads")


async def _get_owned_lead(db: AsyncSession, user_id: uuid.UUID, lead_id: str) -> JobLead:
    try:
        lead_uuid = uuid.UUID(lead_id)
    except ValueError:
        raise not_found_error("Job lead not found")
    lead = (
        await db.execute(
            select(JobLead).where(and_(JobLead.id == lead_uuid, JobLead.user_id == user_id))
        )
    ).scalar_one_or_none()
    if lead is None:
        raise not_found_error("Job lead not found")
    return lead


@router.get("/leads/{lead_id}")
async def get_job_lead(
    lead_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """Return one lead including its full description (used by Analyze)."""
    try:
        lead = await _get_owned_lead(db, _user_uuid(current_user), lead_id)
        return lead.to_dict(include_description=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to load job lead: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to load the job lead")


@router.patch("/leads/{lead_id}")
async def update_job_lead(
    lead_id: str,
    request: LeadStatusUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_database),
) -> Dict[str, Any]:
    """Change a lead's status (new / saved / dismissed / analyzed)."""
    try:
        lead = await _get_owned_lead(db, _user_uuid(current_user), lead_id)
        lead.status = request.status
        await db.commit()
        return lead.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update job lead: %s", sanitize_log_value(e), exc_info=True)
        raise internal_error("Failed to update the job lead")
