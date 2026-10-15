"""Find Jobs — config defaults, run lock, and the background search run.

A run:

1. Loads the user's saved searches and watched company careers pages.
2. Fetches every watched board (free public listings) and keeps postings that
   match at least one saved search (:mod:`utils.job_matching`).
3. Optionally runs one AI web search per saved search with the user's BYOK
   provider (:mod:`agents.job_web_search`).
4. Upserts results into ``job_leads`` (unique per user + source + id) and
   stores a summary on ``job_finder_configs.last_run``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from models.database import JobFinderConfig, JobLead
from utils.cache import get_redis_or_none
from utils.database import get_session
from utils.error_reporting import report_exception
from utils.job_matching import matching_searches, parse_search
from utils.job_sources import JobPosting, fetch_boards
from utils.llm_client import user_facing_message_from_llm_exception
from utils.logging_config import sanitize_log_value

logger = logging.getLogger(__name__)

MAX_SEARCHES = 30
MAX_COMPANIES = 60
MAX_SEARCH_CHARS = 150
WEB_SEARCH_CONCURRENCY = 3
WEB_SEARCH_TIMEOUT_SECONDS = 180
RUN_LOCK_TTL_SECONDS = 15 * 60
# "new" leads not seen again for this long are treated as closed and hidden
STALE_LEAD_DAYS = 21

LEAD_STATUSES = ("new", "saved", "dismissed", "analyzed")

DEFAULT_SEARCHES: List[str] = [
    "AI solutions architect remote jobs",
    "AI solutions consultant remote",
    "presales engineer AI remote",
    "solutions engineer voice AI remote",
    "forward deployed engineer AI remote",
    "AI enablement lead remote jobs",
    "technical product marketing manager AI remote",
    "AI product manager voice remote",
    "developer advocate AI remote",
    "AI adoption consultant remote",
    "generative AI consultant remote contract",
    "AI coach remote",
    "AI solutions architect jobs Houston",
    "solutions engineer jobs The Woodlands TX",
]

# Starter watchlist of AI / voice-AI employers with public careers boards.
# Users can remove or add companies on the Find Jobs page.
DEFAULT_COMPANIES: List[Dict[str, str]] = [
    {"platform": "greenhouse", "token": "anthropic", "name": "Anthropic"},
    {"platform": "ashby", "token": "openai", "name": "OpenAI"},
    {"platform": "ashby", "token": "elevenlabs", "name": "ElevenLabs"},
    {"platform": "ashby", "token": "cohere", "name": "Cohere"},
    {"platform": "ashby", "token": "deepgram", "name": "Deepgram"},
    {"platform": "ashby", "token": "decagon", "name": "Decagon"},
    {"platform": "ashby", "token": "langchain", "name": "LangChain"},
    {"platform": "greenhouse", "token": "scaleai", "name": "Scale AI"},
    {"platform": "greenhouse", "token": "databricks", "name": "Databricks"},
    {"platform": "greenhouse", "token": "gleanwork", "name": "Glean"},
    {"platform": "greenhouse", "token": "twilio", "name": "Twilio"},
    {"platform": "lever", "token": "palantir", "name": "Palantir"},
    {"platform": "lever", "token": "mistral", "name": "Mistral AI"},
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Config
# =============================================================================


async def get_or_create_config(db: AsyncSession, user_id: uuid.UUID) -> JobFinderConfig:
    """Return the user's Find Jobs config, creating it with defaults on first use."""
    result = await db.execute(
        select(JobFinderConfig).where(JobFinderConfig.user_id == user_id)
    )
    config = result.scalar_one_or_none()
    if config is not None:
        return config
    try:
        async with db.begin_nested():
            config = JobFinderConfig(
                id=uuid.uuid4(),
                user_id=user_id,
                searches=list(DEFAULT_SEARCHES),
                companies=[dict(c) for c in DEFAULT_COMPANIES],
                last_run=None,
            )
            db.add(config)
    except IntegrityError:
        # Concurrent first request created it; savepoint rolled back only
        result = await db.execute(
            select(JobFinderConfig).where(JobFinderConfig.user_id == user_id)
        )
        config = result.scalar_one()
    return config


# =============================================================================
# Run lock (one run per user at a time)
# =============================================================================


def _lock_key(user_id: str) -> str:
    return f"job_finder:running:{user_id}"


async def claim_run_lock(user_id: str) -> bool:
    """Atomically claim the per-user run slot. True when the caller may run."""
    try:
        redis = await get_redis_or_none()
        if not redis:
            return True
        was_set = await redis.set(_lock_key(user_id), "1", nx=True, ex=RUN_LOCK_TTL_SECONDS)
        return was_set is not None and was_set is not False
    except Exception as exc:
        logger.warning("Find Jobs lock claim failed: %s", sanitize_log_value(str(exc)))
        return True


async def release_run_lock(user_id: str) -> None:
    """Release the per-user run slot."""
    try:
        redis = await get_redis_or_none()
        if redis:
            await redis.delete(_lock_key(user_id))
    except Exception as exc:
        logger.warning("Find Jobs lock release failed: %s", sanitize_log_value(str(exc)))


async def is_run_active(user_id: str) -> bool:
    """True while a run holds the lock."""
    try:
        redis = await get_redis_or_none()
        if not redis:
            return False
        return await redis.get(_lock_key(user_id)) is not None
    except Exception as exc:
        logger.warning("Find Jobs lock check failed: %s", sanitize_log_value(str(exc)))
        return False


# =============================================================================
# Leads
# =============================================================================


def stale_cutoff() -> datetime:
    """Leads with status "new" last seen before this are hidden as likely closed."""
    return _now() - timedelta(days=STALE_LEAD_DAYS)


async def upsert_leads(
    db: AsyncSession,
    user_id: uuid.UUID,
    leads: List[Tuple[JobPosting, List[str]]],
) -> int:
    """
    Insert new leads and refresh existing ones (status is never touched).

    Returns:
        Number of leads that were new for this user
    """
    new_count = 0
    seen_at = _now()
    for posting, matched in leads:
        values: Dict[str, Any] = {
            "id": uuid.uuid4(),
            "user_id": user_id,
            "source": posting.source,
            "external_id": posting.external_id[:255],
            "title": posting.title[:300],
            "company": posting.company[:200],
            "location": (posting.location or None) and posting.location[:300],
            "is_remote": posting.is_remote,
            "url": posting.url,
            "description": posting.description,
            "posted_at": posting.posted_at,
            "matched_searches": matched,
            "status": "new",
            "first_seen_at": seen_at,
            "last_seen_at": seen_at,
        }
        stmt = pg_insert(JobLead).values(**values)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_job_leads_user_source_external",
            set_={
                "title": stmt.excluded.title,
                "company": stmt.excluded.company,
                "location": stmt.excluded.location,
                "is_remote": stmt.excluded.is_remote,
                "url": stmt.excluded.url,
                "description": stmt.excluded.description,
                "posted_at": stmt.excluded.posted_at,
                "matched_searches": stmt.excluded.matched_searches,
                "last_seen_at": stmt.excluded.last_seen_at,
            },
        ).returning(literal_column("(xmax = 0)").label("inserted"))
        result = await db.execute(stmt)
        if result.scalar():
            new_count += 1
    return new_count


def collect_board_matches(
    postings: List[JobPosting], searches: List[str]
) -> List[Tuple[JobPosting, List[str]]]:
    """Keep board postings that match at least one saved search."""
    specs = [parse_search(q) for q in searches]
    specs = [s for s in specs if s.is_usable]
    matched: List[Tuple[JobPosting, List[str]]] = []
    for posting in postings:
        hits = matching_searches(posting, specs)
        if hits:
            matched.append((posting, hits))
    return matched


def merge_web_results(
    results: List[Tuple[str, List[JobPosting]]]
) -> List[Tuple[JobPosting, List[str]]]:
    """Merge per-search web results, combining searches that found the same posting."""
    merged: Dict[str, Tuple[JobPosting, List[str]]] = {}
    for query, postings in results:
        for posting in postings:
            if posting.external_id in merged:
                hits = merged[posting.external_id][1]
                if query not in hits:
                    hits.append(query)
            else:
                merged[posting.external_id] = (posting, [query])
    return list(merged.values())


# =============================================================================
# Background run
# =============================================================================

# Factory hook so tests can inject a fake web-search agent
WebAgentFactory = Callable[..., Any]


def _default_web_agent_factory(**kwargs: Any) -> Any:
    from agents.job_web_search import JobWebSearchAgent

    return JobWebSearchAgent(**kwargs)


web_agent_factory: WebAgentFactory = _default_web_agent_factory


async def _run_web_searches(
    searches: List[str],
    *,
    llm_provider: Optional[str],
    user_api_key: Optional[str],
    preferred_model: Optional[str],
) -> Tuple[List[Tuple[str, List[JobPosting]]], List[str]]:
    agent = web_agent_factory(
        llm_provider=llm_provider, user_api_key=user_api_key, model=preferred_model
    )
    semaphore = asyncio.Semaphore(WEB_SEARCH_CONCURRENCY)
    results: List[Tuple[str, List[JobPosting]]] = []
    errors: List[str] = []

    async def _one(query: str) -> None:
        async with semaphore:
            try:
                postings = await asyncio.wait_for(
                    agent.search(query), timeout=WEB_SEARCH_TIMEOUT_SECONDS
                )
                results.append((query, postings))
            except asyncio.TimeoutError:
                errors.append(f"{query}: web search timed out")
            except Exception as exc:
                logger.warning(
                    "Find Jobs web search failed for %s: %s",
                    sanitize_log_value(query),
                    sanitize_log_value(str(exc)),
                    exc_info=True,
                )
                errors.append(f"{query}: {user_facing_message_from_llm_exception(exc)[:200]}")

    await asyncio.gather(*(_one(q) for q in searches))
    return results, errors


async def _save_last_run(user_id: uuid.UUID, summary: Dict[str, Any]) -> None:
    async with get_session() as db:
        config = await get_or_create_config(db, user_id)
        config.last_run = summary
        flag_modified(config, "last_run")
        await db.commit()


async def run_job_finder_background(
    user_id: str,
    include_web: bool = False,
    llm_provider: Optional[str] = None,
    user_api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> None:
    """
    Background task: scan boards (+ optional web search) and save matching leads.

    The caller must already hold the run lock (:func:`claim_run_lock`); it is
    always released here.
    """
    uid = uuid.UUID(user_id)
    started_at = _now()
    summary: Dict[str, Any] = {"status": "running", "started_at": started_at.isoformat()}
    try:
        async with get_session() as db:
            config = await get_or_create_config(db, uid)
            searches = [s for s in (config.searches or []) if isinstance(s, str) and s.strip()]
            companies = [c for c in (config.companies or []) if isinstance(c, dict)]
            config.last_run = summary
            flag_modified(config, "last_run")
            await db.commit()

        postings, board_statuses = await fetch_boards(companies)
        board_matches = collect_board_matches(postings, searches)

        web_matches: List[Tuple[JobPosting, List[str]]] = []
        web_errors: List[str] = []
        web_ran = include_web and bool(llm_provider) and llm_provider != "ollama"
        if web_ran and searches:
            web_results, web_errors = await _run_web_searches(
                searches,
                llm_provider=llm_provider,
                user_api_key=user_api_key,
                preferred_model=preferred_model,
            )
            web_matches = merge_web_results(web_results)

        async with get_session() as db:
            new_count = await upsert_leads(db, uid, board_matches + web_matches)
            await db.commit()

        summary = {
            "status": "completed",
            "started_at": started_at.isoformat(),
            "finished_at": _now().isoformat(),
            "searches": len(searches),
            "companies_checked": len(companies),
            "companies_failed": sum(1 for s in board_statuses.values() if not s["ok"]),
            "board_jobs_scanned": len(postings),
            "board_matches": len(board_matches),
            "web_search": web_ran,
            "web_matches": len(web_matches),
            "web_errors": web_errors[:20],
            "new_leads": new_count,
            "company_status": board_statuses,
        }
        await _save_last_run(uid, summary)
        logger.info(
            "Find Jobs run complete for user %s: %s new leads",
            sanitize_log_value(user_id),
            new_count,
        )
    except Exception as exc:
        logger.error(
            "Find Jobs run failed for user %s: %s",
            sanitize_log_value(user_id),
            sanitize_log_value(str(exc)),
            exc_info=True,
        )
        await report_exception(exc, user_id=user_id)
        try:
            await _save_last_run(
                uid,
                {
                    "status": "failed",
                    "started_at": started_at.isoformat(),
                    "finished_at": _now().isoformat(),
                    "error": "The job search failed. Please try again.",
                },
            )
        except Exception as save_exc:
            logger.error(
                "Could not record failed Find Jobs run: %s",
                sanitize_log_value(str(save_exc)),
                exc_info=True,
            )
    finally:
        await release_run_lock(user_id)
