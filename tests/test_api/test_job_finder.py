"""API tests for Find Jobs (/api/v1/job-finder)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import update

from models.database import JobLead
from tests.test_api.conftest import _NullSessionLocal
from utils.error_responses import no_api_key_error
from utils.job_finder import DEFAULT_SEARCHES, upsert_leads
from utils.job_sources import JobPosting

BASE = "/api/v1/job-finder"


@pytest.fixture(autouse=True)
def _no_run_lock_or_rate_limit():
    with (
        patch("api.job_finder.check_rate_limit", AsyncMock(return_value=(True, 9))),
        patch("api.job_finder.is_run_active", AsyncMock(return_value=False)),
    ):
        yield


async def _user_id(client: AsyncClient) -> uuid.UUID:
    response = await client.get(BASE)
    assert response.status_code == 200
    from utils.auth import get_current_user
    from main import app

    user = await app.dependency_overrides[get_current_user]()
    return uuid.UUID(user["id"])


async def _seed_leads(uid: uuid.UUID) -> None:
    def posting(ext: str, source: str = "lever", description: str | None = "Full description") -> JobPosting:
        return JobPosting(
            source=source,
            external_id=ext,
            title=f"Role {ext}",
            company="Acme",
            url=f"https://x/{ext}",
            location="Remote",
            is_remote=True,
            description=description,
            posted_at=datetime(2026, 10, int(ext[-1]) + 1, tzinfo=timezone.utc),
        )

    async with _NullSessionLocal() as db:
        await upsert_leads(
            db,
            uid,
            [
                (posting("a1"), ["AI coach remote"]),
                (posting("a2"), ["AI coach remote"]),
                (posting("w3", source="web", description=None), ["q"]),
                (posting("a4"), ["q"]),
            ],
        )
        # a4 has not been seen for weeks → hidden from "new"
        await db.execute(
            update(JobLead)
            .where(JobLead.user_id == uid, JobLead.external_id == "a4")
            .values(last_seen_at=datetime.now(timezone.utc) - timedelta(days=60))
        )
        await db.commit()


@pytest.mark.asyncio
async def test_get_creates_default_config(authed_client_with_user: AsyncClient) -> None:
    response = await authed_client_with_user.get(BASE)
    assert response.status_code == 200
    data = response.json()
    assert data["searches"] == DEFAULT_SEARCHES
    assert data["running"] is False
    assert data["last_run"] is None
    first = data["companies"][0]
    assert first["page_url"].startswith("https://")
    assert first["last_status"] is None


@pytest.mark.asyncio
async def test_put_config_replaces_searches_and_companies(authed_client_with_user: AsyncClient) -> None:
    await authed_client_with_user.get(BASE)
    response = await authed_client_with_user.put(
        f"{BASE}/config",
        json={
            "searches": ["  AI coach   remote ", "ai coach remote", "", "developer advocate AI remote"],
            "companies": [
                "Anthropic | https://boards.greenhouse.io/anthropic",
                "https://jobs.lever.co/new-co",
                "https://jobs.lever.co/new-co/123",
                "",
            ],
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["searches"] == ["AI coach remote", "developer advocate AI remote"]
    assert [(c["name"], c["token"]) for c in data["companies"]] == [
        ("Anthropic", "anthropic"),
        ("New Co", "new-co"),
    ]


@pytest.mark.asyncio
async def test_put_config_keeps_known_company_names(authed_client_with_user: AsyncClient) -> None:
    await authed_client_with_user.get(BASE)
    response = await authed_client_with_user.put(
        f"{BASE}/config", json={"searches": [], "companies": ["https://jobs.lever.co/palantir"]}
    )
    assert response.json()["companies"][0]["name"] == "Palantir"


@pytest.mark.asyncio
async def test_put_config_rejects_unsupported_links(authed_client_with_user: AsyncClient) -> None:
    response = await authed_client_with_user.put(
        f"{BASE}/config", json={"searches": ["x"], "companies": ["https://example.com/careers"]}
    )
    assert response.status_code == 422
    assert "example.com" in response.json()["message"]


@pytest.mark.asyncio
async def test_put_config_validates_limits(authed_client_with_user: AsyncClient) -> None:
    too_long = await authed_client_with_user.put(f"{BASE}/config", json={"searches": ["x" * 200]})
    assert too_long.status_code == 422
    too_many = await authed_client_with_user.put(
        f"{BASE}/config", json={"searches": [f"search {i}" for i in range(31)]}
    )
    assert too_many.status_code == 422
    companies = [f"https://jobs.lever.co/co{i}" for i in range(61)]
    too_many_companies = await authed_client_with_user.put(
        f"{BASE}/config", json={"searches": [], "companies": companies}
    )
    assert too_many_companies.status_code == 422


@pytest.mark.asyncio
async def test_run_boards_only(authed_client_with_user: AsyncClient) -> None:
    background = AsyncMock()
    with (
        patch("api.job_finder.claim_run_lock", AsyncMock(return_value=True)),
        patch("api.job_finder.run_job_finder_background", background),
    ):
        response = await authed_client_with_user.post(f"{BASE}/run", json={"include_web": False})
    assert response.status_code == 202
    assert response.json() == {"status": "started", "web_search": False, "web_search_skipped_reason": None}
    assert background.await_args.kwargs["include_web"] is False


@pytest.mark.asyncio
async def test_run_with_web_on_ollama_is_skipped(authed_client_with_user: AsyncClient) -> None:
    background = AsyncMock()
    with (
        patch("api.job_finder.claim_run_lock", AsyncMock(return_value=True)),
        patch("api.job_finder.run_job_finder_background", background),
    ):
        response = await authed_client_with_user.post(f"{BASE}/run", json={"include_web": True})
    assert response.status_code == 202
    body = response.json()
    assert body["web_search"] is False
    assert "Ollama" in body["web_search_skipped_reason"]
    assert background.await_args.kwargs["llm_provider"] == "ollama"


@pytest.mark.asyncio
async def test_run_with_web_uses_byok(authed_client_with_user: AsyncClient) -> None:
    ctx = type("Ctx", (), {"provider": "openrouter", "user_api_key": "sk-or-x"})()
    background = AsyncMock()
    with (
        patch("utils.llm_context.require_user_llm_context", AsyncMock(return_value=(None, ctx, None))),
        patch("utils.llm_preferences.load_preferred_model", AsyncMock(return_value="x-ai/grok-4")),
        patch("api.job_finder.claim_run_lock", AsyncMock(return_value=True)),
        patch("api.job_finder.run_job_finder_background", background),
    ):
        response = await authed_client_with_user.post(f"{BASE}/run", json={"include_web": True})
    assert response.status_code == 202
    assert response.json()["web_search"] is True
    kwargs = background.await_args.kwargs
    assert kwargs["include_web"] is True
    assert kwargs["user_api_key"] == "sk-or-x"
    assert kwargs["preferred_model"] == "x-ai/grok-4"


@pytest.mark.asyncio
async def test_run_with_web_without_key_returns_cfg_6001(authed_client_with_user: AsyncClient) -> None:
    with patch(
        "utils.llm_context.require_user_llm_context", AsyncMock(side_effect=no_api_key_error())
    ):
        response = await authed_client_with_user.post(f"{BASE}/run", json={"include_web": True})
    assert response.status_code == 422
    assert response.json()["error_code"] == "CFG_6001"


@pytest.mark.asyncio
async def test_run_conflict_and_rate_limit(authed_client_with_user: AsyncClient) -> None:
    with patch("api.job_finder.claim_run_lock", AsyncMock(return_value=False)):
        busy = await authed_client_with_user.post(f"{BASE}/run", json={})
    assert busy.status_code == 409

    with patch("api.job_finder.check_rate_limit", AsyncMock(return_value=(False, 0))):
        limited = await authed_client_with_user.post(f"{BASE}/run", json={})
    assert limited.status_code == 429


@pytest.mark.asyncio
async def test_leads_list_detail_and_status(authed_client_with_user: AsyncClient) -> None:
    uid = await _user_id(authed_client_with_user)
    await _seed_leads(uid)

    listing = await authed_client_with_user.get(f"{BASE}/leads")
    assert listing.status_code == 200
    data = listing.json()
    # Stale a4 hidden; newest posted first
    assert [lead["title"] for lead in data["leads"]] == ["Role w3", "Role a2", "Role a1"]
    assert data["counts"] == {"new": 3, "saved": 0, "dismissed": 0, "analyzed": 0}
    assert "description" not in data["leads"][0]
    web = data["leads"][0]
    assert web["has_description"] is False

    lead_id = data["leads"][1]["id"]
    detail = await authed_client_with_user.get(f"{BASE}/leads/{lead_id}")
    assert detail.status_code == 200
    assert detail.json()["description"] == "Full description"

    patched = await authed_client_with_user.patch(f"{BASE}/leads/{lead_id}", json={"status": "saved"})
    assert patched.status_code == 200
    assert patched.json()["status"] == "saved"

    saved = (await authed_client_with_user.get(f"{BASE}/leads", params={"status": "saved"})).json()
    assert [lead["id"] for lead in saved["leads"]] == [lead_id]
    assert saved["counts"]["saved"] == 1 and saved["counts"]["new"] == 2


@pytest.mark.asyncio
async def test_leads_validation_and_not_found(authed_client_with_user: AsyncClient) -> None:
    bad_status = await authed_client_with_user.get(f"{BASE}/leads", params={"status": "archived"})
    assert bad_status.status_code == 422

    missing = await authed_client_with_user.get(f"{BASE}/leads/{uuid.uuid4()}")
    assert missing.status_code == 404
    not_uuid = await authed_client_with_user.get(f"{BASE}/leads/not-a-uuid")
    assert not_uuid.status_code == 404

    bad_patch = await authed_client_with_user.patch(
        f"{BASE}/leads/{uuid.uuid4()}", json={"status": "archived"}
    )
    assert bad_patch.status_code == 422
    missing_patch = await authed_client_with_user.patch(
        f"{BASE}/leads/{uuid.uuid4()}", json={"status": "saved"}
    )
    assert missing_patch.status_code == 404


@pytest.mark.asyncio
async def test_internal_errors_are_wrapped(authed_client_with_user: AsyncClient) -> None:
    boom = AsyncMock(side_effect=RuntimeError("db exploded"))
    with patch("api.job_finder.get_or_create_config", boom):
        assert (await authed_client_with_user.get(BASE)).status_code == 500
        assert (
            await authed_client_with_user.put(f"{BASE}/config", json={"searches": []})
        ).status_code == 500
    with patch("api.job_finder.claim_run_lock", boom):
        assert (await authed_client_with_user.post(f"{BASE}/run", json={})).status_code == 500
    with patch("api.job_finder.stale_cutoff", side_effect=RuntimeError("x")):
        assert (await authed_client_with_user.get(f"{BASE}/leads")).status_code == 500
    with patch("api.job_finder._get_owned_lead", boom):
        assert (await authed_client_with_user.get(f"{BASE}/leads/{uuid.uuid4()}")).status_code == 500
        assert (
            await authed_client_with_user.patch(f"{BASE}/leads/{uuid.uuid4()}", json={"status": "new"})
        ).status_code == 500


@pytest.mark.asyncio
async def test_run_releases_lock_when_scheduling_fails(authed_client_with_user: AsyncClient) -> None:
    release = AsyncMock()
    with (
        patch("api.job_finder.claim_run_lock", AsyncMock(return_value=True)),
        patch("api.job_finder.release_run_lock", release),
        patch("fastapi.BackgroundTasks.add_task", side_effect=RuntimeError("no")),
    ):
        response = await authed_client_with_user.post(f"{BASE}/run", json={})
    assert response.status_code == 500
    release.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_jobs_page_renders(api_client: AsyncClient) -> None:
    response = await api_client.get("/dashboard/jobs")
    assert response.status_code == 200
    assert "Find Jobs" in response.text
    assert "dashboard-jobs" in response.text
