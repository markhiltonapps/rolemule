"""Unit tests for the Find Jobs AI web search agent (agents/job_web_search.py)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from agents.job_web_search import (
    MAX_RESULTS_PER_SEARCH,
    JobWebSearchAgent,
    _parse_date,
    _valid_posting_url,
    normalize_web_results,
    web_external_id,
)


def _job(**overrides):
    base = {
        "title": "AI Solutions Architect",
        "company": "Acme",
        "location": "Remote - US",
        "is_remote": False,
        "url": "https://acme.com/careers/123",
        "posted_date": "2026-10-01",
        "summary": "Design AI solutions   for customers.",
    }
    base.update(overrides)
    return base


def test_url_and_date_helpers() -> None:
    assert _valid_posting_url("https://acme.com/jobs/1")
    assert not _valid_posting_url("javascript:alert(1)")
    assert not _valid_posting_url("https://localhost/x")
    assert not _valid_posting_url("ftp://acme.com/x")
    assert not _valid_posting_url("http://[::1")
    assert _parse_date("2026-10-01").year == 2026
    assert _parse_date("yesterday") is None
    assert _parse_date(None) is None
    assert web_external_id("https://A.com/x/#frag") == web_external_id("https://a.com/x")


def test_normalize_web_results_filters_and_dedupes() -> None:
    parsed = {
        "jobs": [
            _job(),
            _job(),  # duplicate URL
            _job(url="javascript:alert(1)"),
            _job(title=""),
            _job(company=None),
            "garbage",
            _job(url="https://b.com/2", location=None, is_remote=True, posted_date=None, summary=None),
        ]
    }
    postings = normalize_web_results(parsed, "q")
    assert len(postings) == 2
    first = postings[0]
    assert first.source == "web"
    assert first.is_remote is True  # inferred from location text
    assert first.description is None
    assert first.extra_text == ["Design AI solutions for customers."]
    assert first.posted_at is not None
    second = postings[1]
    assert second.is_remote is True and second.location is None and second.extra_text == []
    assert normalize_web_results({"jobs": "nope"}, "q") == []
    assert normalize_web_results({}, "q") == []


def test_normalize_web_results_caps_count() -> None:
    parsed = {"jobs": [_job(url=f"https://acme.com/{i}") for i in range(30)]}
    assert len(normalize_web_results(parsed, "q")) == MAX_RESULTS_PER_SEARCH


@pytest.mark.asyncio
async def test_search_calls_grounded_llm_with_byok() -> None:
    client = AsyncMock()
    client.generate = AsyncMock(return_value={"response": json.dumps({"jobs": [_job()]})})
    agent = JobWebSearchAgent(
        llm_provider="openrouter", user_api_key="sk-or-x", model="google/gemini-2.5-flash", llm_client=client
    )
    postings = await agent.search('AI "coach" remote')
    assert len(postings) == 1
    kwargs = client.generate.call_args.kwargs
    assert kwargs["use_google_search_grounding"] is True
    assert kwargs["provider"] == "openrouter"
    assert kwargs["user_api_key"] == "sk-or-x"
    assert kwargs["model"] == "google/gemini-2.5-flash"
    assert "AI 'coach' remote" in kwargs["prompt"]


@pytest.mark.asyncio
async def test_search_filtered_and_unparseable_responses() -> None:
    client = AsyncMock()
    client.generate = AsyncMock(return_value={"response": "", "filtered": True})
    agent = JobWebSearchAgent(llm_provider="gemini", user_api_key="k", llm_client=client)
    assert await agent.search("x") == []

    client.generate = AsyncMock(return_value={"response": "no json here"})
    assert await agent.search("x") == []


@pytest.mark.asyncio
async def test_search_lazily_gets_llm_client() -> None:
    client = AsyncMock()
    client.generate = AsyncMock(return_value={"response": "{}"})
    with patch("agents.job_web_search.get_llm_client", AsyncMock(return_value=client)) as getter:
        agent = JobWebSearchAgent(llm_provider="gemini", user_api_key="k")
        assert await agent.search("x") == []
        assert getter.await_count == 1
