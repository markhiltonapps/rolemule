"""Unit tests for Find Jobs careers-board sources (utils/job_sources.py)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from utils import job_sources
from utils.job_sources import (
    BoardFetchError,
    JobPosting,
    board_page_url,
    default_company_name,
    fetch_board,
    fetch_boards,
    html_to_text,
    normalize_ashby,
    normalize_greenhouse,
    normalize_lever,
    parse_board_reference,
)


# ---------------------------------------------------------------------------
# parse_board_reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://boards.greenhouse.io/anthropic", ("greenhouse", "anthropic")),
        ("https://job-boards.greenhouse.io/anthropic/jobs/123", ("greenhouse", "anthropic")),
        ("boards.greenhouse.io/scaleai", ("greenhouse", "scaleai")),
        ("https://boards-api.greenhouse.io/v1/boards/acme/jobs", ("greenhouse", "acme")),
        ("https://boards.greenhouse.io/embed/job_board?for=acme", ("greenhouse", "acme")),
        ("https://boards.greenhouse.io/?for=acme", ("greenhouse", "acme")),
        ("https://jobs.lever.co/palantir", ("lever", "palantir")),
        ("https://api.lever.co/v0/postings/mistral?mode=json", ("lever", "mistral")),
        ("https://jobs.ashbyhq.com/elevenlabs/abc-123", ("ashby", "elevenlabs")),
        ("https://api.ashbyhq.com/posting-api/job-board/openai", ("ashby", "openai")),
        ("Lever: palantir", ("lever", "palantir")),
        ("ashby:deepgram", ("ashby", "deepgram")),
    ],
)
def test_parse_board_reference_supported(raw: str, expected: Any) -> None:
    assert parse_board_reference(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "https://example.com/careers",
        "https://jobs.lever.co/",
        "https://boards.greenhouse.io/embed/job_board",
        "lever:bad token",
        "https://jobs.lever.co/%20bad",
        "http://[::1",
    ],
)
def test_parse_board_reference_rejects(raw: str) -> None:
    assert parse_board_reference(raw) is None


def test_board_page_url_and_default_name() -> None:
    assert board_page_url("greenhouse", "acme") == "https://job-boards.greenhouse.io/acme"
    assert board_page_url("lever", "acme") == "https://jobs.lever.co/acme"
    assert board_page_url("ashby", "acme") == "https://jobs.ashbyhq.com/acme"
    assert default_company_name("scale-ai") == "Scale Ai"
    assert default_company_name("acme_co.io") == "Acme Co Io"
    assert default_company_name("---") == "---"


# ---------------------------------------------------------------------------
# html_to_text
# ---------------------------------------------------------------------------


def test_html_to_text_handles_lists_scripts_and_escaped_html() -> None:
    raw = "<p>About&nbsp;us</p><script>alert(1)</script><style>p{}</style><ul><li>One</li><li>Two</li></ul>"
    text = html_to_text(raw)
    assert "About us" in text
    assert "alert" not in text and "p{}" not in text
    assert "- One" in text and "- Two" in text

    escaped = "&lt;p&gt;Hello &amp;amp; welcome&lt;/p&gt;"
    assert html_to_text(escaped) == "Hello & welcome"
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


def test_html_to_text_truncates() -> None:
    assert len(html_to_text("x" * (job_sources.MAX_DESCRIPTION_CHARS + 50))) == (
        job_sources.MAX_DESCRIPTION_CHARS
    )


def test_date_parsers() -> None:
    assert job_sources._parse_iso("2026-10-01T12:00:00Z") == datetime(
        2026, 10, 1, 12, tzinfo=timezone.utc
    )
    assert job_sources._parse_iso("2026-10-01T12:00:00").tzinfo is timezone.utc
    assert job_sources._parse_iso("not a date") is None
    assert job_sources._parse_iso(None) is None
    assert job_sources._parse_iso(123) is None
    assert job_sources._parse_epoch_ms(1_700_000_000_000).year == 2023
    assert job_sources._parse_epoch_ms(0) is None
    assert job_sources._parse_epoch_ms("x") is None
    assert job_sources._parse_epoch_ms(1e30) is None


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------


def test_normalize_greenhouse() -> None:
    payload = {
        "jobs": [
            {
                "id": 1,
                "title": " Solutions Architect ",
                "location": {"name": "Remote - US"},
                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
                "content": "&lt;p&gt;Build AI&lt;/p&gt;",
                "first_published": "2026-10-01T00:00:00Z",
            },
            {
                "id": 2,
                "title": "Engineer",
                "location": {"name": "Houston, TX"},
                "offices": [{"name": "Remote"}],
                "absolute_url": "https://x/2",
                "updated_at": "2026-10-02T00:00:00Z",
            },
            {"id": 3, "title": "No URL"},
            {"title": "No id"},
            "garbage",
        ]
    }
    postings = normalize_greenhouse(payload, "Acme")
    assert [p.external_id for p in postings] == ["1", "2"]
    first = postings[0]
    assert first.title == "Solutions Architect"
    assert first.is_remote is True
    assert first.description == "Build AI"
    assert first.posted_at == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert postings[1].is_remote is True
    assert postings[1].description is None
    assert normalize_greenhouse({}, "Acme") == []


def test_normalize_lever() -> None:
    payload = [
        {
            "id": "abc",
            "text": "Forward Deployed Engineer",
            "categories": {"location": "Remote", "commitment": "Contract", "allLocations": ["NYC"]},
            "workplaceType": "remote",
            "hostedUrl": "https://jobs.lever.co/acme/abc",
            "createdAt": 1_700_000_000_000,
            "descriptionPlain": "Intro text",
            "lists": [{"text": "What you'll do", "content": "<li>Ship</li>"}, "bad"],
            "additionalPlain": "Benefits",
        },
        {"id": "def", "text": "Onsite", "categories": {}, "applyUrl": "https://x/apply",
         "description": "<p>HTML body</p>"},
        {"id": "nourl", "text": "x"},
        {"text": "no id"},
    ]
    postings = normalize_lever(payload, "Acme")
    assert len(postings) == 2
    first = postings[0]
    assert first.is_remote is True
    assert first.extra_text == ["Contract"]
    assert "Intro text" in first.description
    assert "What you'll do" in first.description and "- Ship" in first.description
    assert "Benefits" in first.description
    assert first.posted_at is not None
    assert postings[1].is_remote is False
    assert postings[1].description == "HTML body"
    assert normalize_lever({"not": "a list"}, "Acme") == []


def test_normalize_ashby() -> None:
    payload = {
        "jobs": [
            {
                "id": "1",
                "title": "AI Product Manager",
                "location": "San Francisco",
                "secondaryLocations": [{"location": "Remote"}, {"location": ""}, "bad"],
                "jobUrl": "https://jobs.ashbyhq.com/acme/1",
                "descriptionPlain": "Voice AI product",
                "publishedAt": "2026-09-30T10:00:00+00:00",
                "employmentType": "FullTime",
            },
            {"id": "2", "title": "Hidden", "isListed": False, "jobUrl": "https://x"},
            {"id": "3", "title": "Remote flag", "isRemote": True, "applyUrl": "https://x/3",
             "descriptionHtml": "<p>Body</p>"},
            {"id": "4", "title": "No url"},
            {"title": "No id"},
        ]
    }
    postings = normalize_ashby(payload, "Acme")
    assert [p.external_id for p in postings] == ["1", "3"]
    assert postings[0].location == "San Francisco, Remote"
    assert postings[0].is_remote is True
    assert postings[0].extra_text == ["FullTime"]
    assert postings[1].is_remote is True
    assert postings[1].description == "Body"


# ---------------------------------------------------------------------------
# fetch_board / fetch_boards
# ---------------------------------------------------------------------------


def _client_returning(status: int = 200, json_body: Any = None, json_error: bool = False) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    if json_error:
        response.json = MagicMock(side_effect=ValueError("bad json"))
    else:
        response.json = MagicMock(return_value=json_body)
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_fetch_board_success_uses_platform_endpoint() -> None:
    client = _client_returning(json_body={"jobs": [{"id": 9, "title": "T", "absolute_url": "https://x"}]})
    postings = await fetch_board(client, "greenhouse", "acme", "Acme")
    assert postings[0].external_id == "9"
    url = client.get.call_args[0][0]
    assert url == "https://boards-api.greenhouse.io/v1/boards/acme/jobs?content=true"


@pytest.mark.asyncio
async def test_fetch_board_errors() -> None:
    with pytest.raises(BoardFetchError, match="Unsupported"):
        await fetch_board(_client_returning(), "workday", "acme", "Acme")
    with pytest.raises(BoardFetchError, match="Unsupported"):
        await fetch_board(_client_returning(), "lever", "bad token", "Acme")

    with pytest.raises(BoardFetchError) as not_found:
        await fetch_board(_client_returning(status=404), "lever", "acme", "Acme")
    assert not_found.value.not_found is True

    with pytest.raises(BoardFetchError, match="HTTP 500"):
        await fetch_board(_client_returning(status=500), "ashby", "acme", "Acme")

    with pytest.raises(BoardFetchError, match="unreadable"):
        await fetch_board(_client_returning(json_error=True), "ashby", "acme", "Acme")

    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(BoardFetchError, match="Network error: ConnectError"):
        await fetch_board(client, "ashby", "acme", "Acme")


@pytest.mark.asyncio
async def test_fetch_boards_collects_postings_and_statuses() -> None:
    async def fake_fetch(client: Any, platform: str, token: str, name: str) -> List[JobPosting]:
        if token == "missing":
            raise BoardFetchError("Careers page not found", not_found=True)
        if token == "explode":
            raise RuntimeError("unexpected")
        return [JobPosting(platform, f"{token}-1", "Role", name, "https://x")]

    postings, statuses = await fetch_boards(
        [
            {"platform": "lever", "token": "good"},
            {"platform": "ashby", "token": "missing", "name": "Missing Co"},
            {"platform": "ashby", "token": "explode"},
        ],
        fetcher=fake_fetch,
    )
    assert [p.external_id for p in postings] == ["good-1"]
    assert postings[0].company == "Good"
    assert statuses["lever:good"] == {"ok": True, "jobs": 1, "error": None}
    assert statuses["ashby:missing"]["ok"] is False
    assert statuses["ashby:missing"]["error"] == "Careers page not found"
    assert statuses["ashby:explode"]["error"] == "Unexpected error"


@pytest.mark.asyncio
async def test_fetch_boards_default_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = AsyncMock(return_value=[])
    monkeypatch.setattr(job_sources, "fetch_board", fake)
    postings, statuses = await fetch_boards([{"platform": "lever", "token": "acme", "name": "Acme"}])
    assert postings == []
    assert statuses == {"lever:acme": {"ok": True, "jobs": 0, "error": None}}
    assert fake.await_count == 1
