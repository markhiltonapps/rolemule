"""Unit tests for Find Jobs search parsing and matching (utils/job_matching.py)."""

from __future__ import annotations

from typing import List, Optional

import pytest

from utils.job_matching import (
    matching_searches,
    normalize_text,
    parse_search,
    posting_matches,
)
from utils.job_sources import JobPosting


def _posting(
    title: str,
    *,
    location: Optional[str] = "Remote",
    remote: bool = True,
    description: Optional[str] = "We build AI products for enterprises.",
    extra: Optional[List[str]] = None,
) -> JobPosting:
    return JobPosting(
        source="greenhouse",
        external_id="1",
        title=title,
        company="Acme",
        url="https://x",
        location=location,
        is_remote=remote,
        description=description,
        extra_text=extra or [],
    )


def test_normalize_text() -> None:
    assert normalize_text("Pre-Sales Engineer, A.I.") == "presales engineer ai"
    assert normalize_text("Gen AI / pre sales") == "genai presales"
    assert normalize_text(None) == ""


@pytest.mark.parametrize(
    "query,role,topic,remote,location,contract",
    [
        ("AI solutions architect remote jobs", ["solutions", "architect"], ["ai"], True, [], False),
        ("solutions engineer voice AI remote", ["solutions", "engineer"], ["voice", "ai"], True, [], False),
        ("generative AI consultant remote contract", ["consultant"], ["generative", "ai"], True, [], True),
        ("AI solutions architect jobs Houston", ["solutions", "architect"], ["ai"], False, ["houston"], False),
        ("solutions engineer jobs The Woodlands TX", ["solutions", "engineer"], [], False, ["woodlands", "tx"], False),
        ("AI coach in Austin remote", ["coach"], ["ai"], True, ["austin"], False),
        ("AI AI architect architect", ["architect"], ["ai"], False, [], False),
    ],
)
def test_parse_search(query, role, topic, remote, location, contract) -> None:
    spec = parse_search(query)
    assert spec.query == query
    assert spec.role_words == role
    assert spec.topic_words == topic
    assert spec.remote is remote
    assert spec.location_words == location
    assert spec.contract is contract


def test_unusable_search_matches_nothing() -> None:
    spec = parse_search("remote jobs")
    assert not spec.is_usable
    assert posting_matches(spec, _posting("Anything")) is False


def test_role_words_must_be_in_title_with_plural_folding() -> None:
    spec = parse_search("AI solutions architect remote")
    assert posting_matches(spec, _posting("Solutions Architect, Applied AI"))
    assert posting_matches(spec, _posting("Senior Solutions Architects"))
    assert not posting_matches(spec, _posting("Solutions Engineer"))


def test_presales_spelling_variants() -> None:
    spec = parse_search("presales engineer AI remote")
    assert posting_matches(spec, _posting("Pre-Sales Engineer"))
    assert posting_matches(spec, _posting("Pre Sales Engineer"))


def test_topic_words_from_title_or_description() -> None:
    spec = parse_search("solutions engineer voice AI remote")
    assert posting_matches(spec, _posting("Solutions Engineer", description="Voice AI agents"))
    assert posting_matches(spec, _posting("Voice AI Solutions Engineer", description=None))
    assert not posting_matches(spec, _posting("Solutions Engineer", description="AI chat only"))


def test_remote_required_when_requested() -> None:
    spec = parse_search("developer advocate AI remote")
    assert not posting_matches(spec, _posting("Developer Advocate", location="NYC", remote=False))
    assert posting_matches(spec, _posting("Developer Advocate"))


def test_location_matching_with_state_names() -> None:
    houston = parse_search("AI solutions architect jobs Houston")
    assert posting_matches(houston, _posting("AI Solutions Architect", location="Houston, TX", remote=False))
    assert not posting_matches(houston, _posting("AI Solutions Architect", location="Remote", remote=True))

    woodlands = parse_search("solutions engineer jobs The Woodlands TX")
    assert posting_matches(woodlands, _posting("Solutions Engineer", location="The Woodlands, Texas", remote=False))
    assert posting_matches(woodlands, _posting("Solutions Engineer", location="The Woodlands, TX", remote=False))
    assert not posting_matches(woodlands, _posting("Solutions Engineer", location="Houston, TX", remote=False))
    assert not posting_matches(woodlands, _posting("Solutions Engineer", location=None, remote=False))

    multi_word_state = parse_search("solutions engineer jobs Albany NY")
    assert posting_matches(multi_word_state, _posting("Solutions Engineer", location="Albany, New York"))


def test_contract_required_when_requested() -> None:
    spec = parse_search("generative AI consultant remote contract")
    assert posting_matches(spec, _posting("Generative AI Consultant (Contract)"))
    assert posting_matches(spec, _posting("Generative AI Consultant", extra=["Contractor"]))
    assert not posting_matches(spec, _posting("Generative AI Consultant", extra=["FullTime"]))


def test_matching_searches_returns_original_queries() -> None:
    specs = [parse_search(q) for q in ["AI coach remote", "AI solutions architect remote"]]
    assert matching_searches(_posting("AI Coach"), specs) == ["AI coach remote"]
    assert matching_searches(_posting("Recruiter"), specs) == []
