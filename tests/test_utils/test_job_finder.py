"""Tests for the Find Jobs run service (utils/job_finder.py) — real test DB."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import delete, select

from models.database import AuthMethod, JobFinderConfig, JobLead, User
from tests.test_api.conftest import _NullSessionLocal
from utils import job_finder
from utils.job_sources import JobPosting


@asynccontextmanager
async def _null_get_session():
    async with _NullSessionLocal() as db:
        yield db


async def _make_user() -> uuid.UUID:
    uid = uuid.uuid4()
    async with _NullSessionLocal() as db:
        db.add(
            User(
                id=uid,
                email=f"jobfinder_{uid.hex[:10]}@example.com",
                password_hash="$2b$12$placeholder",
                auth_method=AuthMethod.LOCAL.value,
                full_name="Job Finder User",
            )
        )
        await db.commit()
    return uid


async def _delete_user(uid: uuid.UUID) -> None:
    async with _NullSessionLocal() as db:
        await db.execute(delete(User).where(User.id == uid))
        await db.commit()


def _posting(ext: str, title: str = "AI Coach", source: str = "lever", **kw: Any) -> JobPosting:
    return JobPosting(
        source=source,
        external_id=ext,
        title=title,
        company="Acme",
        url=f"https://x/{ext}",
        location=kw.get("location", "Remote"),
        is_remote=kw.get("is_remote", True),
        description=kw.get("description", "We build AI tools."),
        posted_at=kw.get("posted_at"),
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_collect_board_matches_and_merge_web_results() -> None:
    postings = [_posting("1", "AI Coach"), _posting("2", "Recruiter")]
    matched = job_finder.collect_board_matches(postings, ["AI coach remote", "remote jobs"])
    assert [(p.external_id, hits) for p, hits in matched] == [("1", ["AI coach remote"])]

    a, b = _posting("w1", source="web"), _posting("w2", source="web")
    merged = job_finder.merge_web_results([("q1", [a, b]), ("q2", [a]), ("q2", [a])])
    by_id = {p.external_id: hits for p, hits in merged}
    assert by_id == {"w1": ["q1", "q2"], "w2": ["q1"]}


def test_defaults_are_parseable() -> None:
    from utils.job_matching import parse_search
    from utils.job_sources import SUPPORTED_PLATFORMS

    assert all(parse_search(q).is_usable for q in job_finder.DEFAULT_SEARCHES)
    assert all(c["platform"] in SUPPORTED_PLATFORMS for c in job_finder.DEFAULT_COMPANIES)


# ---------------------------------------------------------------------------
# Run lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_helpers_with_and_without_redis() -> None:
    fake = MagicMock()
    fake.set = AsyncMock(return_value=True)
    fake.get = AsyncMock(return_value="1")
    fake.delete = AsyncMock()
    with patch("utils.job_finder.get_redis_or_none", AsyncMock(return_value=fake)):
        assert await job_finder.claim_run_lock("u") is True
        assert await job_finder.is_run_active("u") is True
        await job_finder.release_run_lock("u")
        fake.delete.assert_awaited_once_with("job_finder:running:u")
        fake.set = AsyncMock(return_value=None)
        assert await job_finder.claim_run_lock("u") is False

    with patch("utils.job_finder.get_redis_or_none", AsyncMock(return_value=None)):
        assert await job_finder.claim_run_lock("u") is True
        assert await job_finder.is_run_active("u") is False
        await job_finder.release_run_lock("u")

    with patch("utils.job_finder.get_redis_or_none", AsyncMock(side_effect=RuntimeError("down"))):
        assert await job_finder.claim_run_lock("u") is True
        assert await job_finder.is_run_active("u") is False
        await job_finder.release_run_lock("u")


# ---------------------------------------------------------------------------
# Config + upsert (DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_config_seeds_defaults_once() -> None:
    uid = await _make_user()
    try:
        async with _NullSessionLocal() as db:
            config = await job_finder.get_or_create_config(db, uid)
            await db.commit()
            assert config.searches == job_finder.DEFAULT_SEARCHES
            assert len(config.companies) == len(job_finder.DEFAULT_COMPANIES)
            assert config.to_dict()["last_run"] is None
        async with _NullSessionLocal() as db:
            again = await job_finder.get_or_create_config(db, uid)
            assert again.id == config.id
    finally:
        await _delete_user(uid)


@pytest.mark.asyncio
async def test_get_or_create_config_handles_concurrent_insert() -> None:
    uid = await _make_user()
    try:
        async with _NullSessionLocal() as db:
            await job_finder.get_or_create_config(db, uid)
            await db.commit()
        async with _NullSessionLocal() as db:
            # First lookup misses (simulated race), insert hits the unique constraint
            real_execute = db.execute
            calls = {"n": 0}

            async def flaky_execute(stmt: Any, *args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                if calls["n"] == 1:
                    result = MagicMock()
                    result.scalar_one_or_none = MagicMock(return_value=None)
                    return result
                return await real_execute(stmt, *args, **kwargs)

            with patch.object(db, "execute", flaky_execute):
                config = await job_finder.get_or_create_config(db, uid)
            assert config.user_id == uid
    finally:
        await _delete_user(uid)


@pytest.mark.asyncio
async def test_upsert_leads_inserts_then_refreshes_without_touching_status() -> None:
    uid = await _make_user()
    try:
        posted = datetime(2026, 10, 1, tzinfo=timezone.utc)
        async with _NullSessionLocal() as db:
            n = await job_finder.upsert_leads(
                db, uid, [(_posting("1", posted_at=posted), ["AI coach remote"]), (_posting("2", location=None), ["q"])]
            )
            await db.commit()
        assert n == 2

        async with _NullSessionLocal() as db:
            lead = (await db.execute(select(JobLead).where(JobLead.user_id == uid, JobLead.external_id == "1"))).scalar_one()
            lead.status = "saved"
            await db.commit()

        async with _NullSessionLocal() as db:
            n = await job_finder.upsert_leads(db, uid, [(_posting("1", "AI Coach II"), ["new q"])])
            await db.commit()
        assert n == 0

        async with _NullSessionLocal() as db:
            lead = (await db.execute(select(JobLead).where(JobLead.user_id == uid, JobLead.external_id == "1"))).scalar_one()
            assert lead.status == "saved"
            assert lead.title == "AI Coach II"
            assert lead.matched_searches == ["new q"]
            data = lead.to_dict(include_description=True)
            assert data["description"] == "We build AI tools."
            assert data["has_description"] is True
    finally:
        await _delete_user(uid)


# ---------------------------------------------------------------------------
# Web searches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_web_searches_collects_results_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def search(self, query: str) -> List[JobPosting]:
            if query == "boom":
                raise RuntimeError("provider down")
            if query == "slow":
                await asyncio.sleep(1)
            return [_posting(f"w-{query}", source="web")]

    monkeypatch.setattr(job_finder, "web_agent_factory", FakeAgent)
    monkeypatch.setattr(job_finder, "WEB_SEARCH_TIMEOUT_SECONDS", 0.05)
    results, errors = await job_finder._run_web_searches(
        ["ok", "boom", "slow"], llm_provider="gemini", user_api_key="k", preferred_model=None
    )
    assert [q for q, _ in results] == ["ok"]
    assert any(e.startswith("boom:") for e in errors)
    assert "slow: web search timed out" in errors


def test_default_web_agent_factory_builds_agent() -> None:
    agent = job_finder._default_web_agent_factory(llm_provider="gemini", user_api_key="k", model=None)
    assert agent.llm_provider == "gemini"


# ---------------------------------------------------------------------------
# Full background run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_background_board_and_web(monkeypatch: pytest.MonkeyPatch) -> None:
    uid = await _make_user()
    try:
        async with _NullSessionLocal() as db:
            config = await job_finder.get_or_create_config(db, uid)
            config.searches = ["AI coach remote", "  "]
            config.companies = [{"platform": "lever", "token": "acme", "name": "Acme"}, "junk"]
            await db.commit()

        board = AsyncMock(
            return_value=(
                [_posting("b1", "AI Coach"), _posting("b2", "Recruiter")],
                {"lever:acme": {"ok": True, "jobs": 2, "error": None}},
            )
        )
        web = AsyncMock(return_value=([("AI coach remote", [_posting("w1", source="web")])], ["x: err"]))
        release = AsyncMock()
        monkeypatch.setattr(job_finder, "fetch_boards", board)
        monkeypatch.setattr(job_finder, "_run_web_searches", web)
        monkeypatch.setattr(job_finder, "get_session", _null_get_session)
        monkeypatch.setattr(job_finder, "release_run_lock", release)

        await job_finder.run_job_finder_background(
            str(uid), include_web=True, llm_provider="openrouter", user_api_key="k"
        )

        board.assert_awaited_once()
        assert board.call_args[0][0] == [{"platform": "lever", "token": "acme", "name": "Acme"}]
        web.assert_awaited_once()
        release.assert_awaited_once_with(str(uid))

        async with _NullSessionLocal() as db:
            config = (await db.execute(select(JobFinderConfig).where(JobFinderConfig.user_id == uid))).scalar_one()
            run = config.last_run
            leads = (await db.execute(select(JobLead).where(JobLead.user_id == uid))).scalars().all()
        assert run["status"] == "completed"
        assert run["new_leads"] == 2
        assert run["board_matches"] == 1 and run["web_matches"] == 1
        assert run["web_errors"] == ["x: err"]
        assert run["companies_failed"] == 0
        assert sorted(lead.external_id for lead in leads) == ["b1", "w1"]
    finally:
        await _delete_user(uid)


@pytest.mark.asyncio
async def test_run_background_skips_web_for_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    uid = await _make_user()
    try:
        monkeypatch.setattr(job_finder, "fetch_boards", AsyncMock(return_value=([], {})))
        web = AsyncMock()
        monkeypatch.setattr(job_finder, "_run_web_searches", web)
        monkeypatch.setattr(job_finder, "get_session", _null_get_session)
        monkeypatch.setattr(job_finder, "release_run_lock", AsyncMock())
        await job_finder.run_job_finder_background(str(uid), include_web=True, llm_provider="ollama")
        web.assert_not_awaited()
        async with _NullSessionLocal() as db:
            config = (await db.execute(select(JobFinderConfig).where(JobFinderConfig.user_id == uid))).scalar_one()
        assert config.last_run["web_search"] is False
    finally:
        await _delete_user(uid)


@pytest.mark.asyncio
async def test_run_background_failure_records_failed_status(monkeypatch: pytest.MonkeyPatch) -> None:
    uid = await _make_user()
    try:
        monkeypatch.setattr(job_finder, "fetch_boards", AsyncMock(side_effect=RuntimeError("boom")))
        monkeypatch.setattr(job_finder, "get_session", _null_get_session)
        release = AsyncMock()
        report = AsyncMock()
        monkeypatch.setattr(job_finder, "release_run_lock", release)
        monkeypatch.setattr(job_finder, "report_exception", report)
        await job_finder.run_job_finder_background(str(uid))
        report.assert_awaited_once()
        release.assert_awaited_once()
        async with _NullSessionLocal() as db:
            config = (await db.execute(select(JobFinderConfig).where(JobFinderConfig.user_id == uid))).scalar_one()
        assert config.last_run["status"] == "failed"
        assert "failed" in config.last_run["error"]
    finally:
        await _delete_user(uid)


@pytest.mark.asyncio
async def test_run_background_failure_even_saving_status_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_finder, "get_session", MagicMock(side_effect=RuntimeError("db down")))
    monkeypatch.setattr(job_finder, "report_exception", AsyncMock())
    release = AsyncMock()
    monkeypatch.setattr(job_finder, "release_run_lock", release)
    await job_finder.run_job_finder_background(str(uuid.uuid4()))
    release.assert_awaited_once()
