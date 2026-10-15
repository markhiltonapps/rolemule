"""Find Jobs — match postings against a user's free-text saved searches.

A saved search such as ``"solutions engineer voice AI remote"`` is parsed into:

- **role words** (``solutions``, ``engineer``) — must all appear in the job title
- **topic words** (``voice``, ``ai``) — must appear in the title or description
- **remote** flag — posting must be remote
- **location** (``"... jobs Houston"``) — posting location must mention it
- **contract** flag — posting must be a contract / freelance role

Matching is deterministic and cheap so it can run over thousands of postings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List, Optional, Set

from utils.job_sources import JobPosting

TOPIC_WORDS: FrozenSet[str] = frozenset(
    {"ai", "genai", "generative", "voice", "llm", "llms", "ml", "conversational"}
)
_FILLER_WORDS: FrozenSet[str] = frozenset(
    {"job", "jobs", "role", "roles", "position", "positions", "opening", "openings",
     "hiring", "for", "a", "an", "of", "and", "or", "the", "near", "me", "opportunities"}
)
_CONTRACT_WORDS: FrozenSet[str] = frozenset(
    {"contract", "contractor", "freelance", "consultancy", "temporary", "fixed-term"}
)

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# How much of a description is scanned for topic words (keeps matching fast)
_DESCRIPTION_SCAN_CHARS = 20_000


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, unify ``pre-sales`` / ``pre sales`` → ``presales``, drop punctuation."""
    t = (text or "").lower()
    t = re.sub(r"\bpre[\s-]+sales\b", "presales", t)
    t = re.sub(r"\bgen[\s-]+ai\b", "genai", t)
    t = re.sub(r"\ba\.i\.", "ai", t)
    t = re.sub(r"[^a-z0-9+#]+", " ", t)
    return " ".join(t.split())


def _word_set(text: Optional[str]) -> Set[str]:
    words = set(normalize_text(text).split())
    # Light plural folding so "architects" satisfies "architect"
    words |= {w[:-1] for w in words if len(w) > 3 and w.endswith("s")}
    return words


@dataclass
class SearchSpec:
    """A parsed saved search."""

    query: str
    role_words: List[str] = field(default_factory=list)
    topic_words: List[str] = field(default_factory=list)
    remote: bool = False
    contract: bool = False
    location_words: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A search needs at least one role or topic word to match anything."""
        return bool(self.role_words or self.topic_words)


def parse_search(query: str) -> SearchSpec:
    """
    Parse a free-text search into a :class:`SearchSpec`.

    Location is whatever follows ``jobs`` / ``in`` (``"AI architect jobs Houston"``).
    """
    spec = SearchSpec(query=(query or "").strip())
    words = normalize_text(query).split()

    location_start: Optional[int] = None
    for idx, word in enumerate(words):
        if word in ("jobs", "job", "in") and idx < len(words) - 1:
            rest = [w for w in words[idx + 1:] if w not in ("remote", "contract")]
            if rest:
                location_start = idx
                break

    head = words if location_start is None else words[:location_start]
    if location_start is not None:
        spec.location_words = [
            w for w in words[location_start + 1:]
            if w not in ("remote", "contract", "the", "area", "metro")
        ]

    for word in words:
        if word == "remote":
            spec.remote = True
        elif word in _CONTRACT_WORDS:
            spec.contract = True

    for word in head:
        if word in ("remote",) or word in _CONTRACT_WORDS or word in _FILLER_WORDS:
            continue
        if word in TOPIC_WORDS:
            if word not in spec.topic_words:
                spec.topic_words.append(word)
        elif word not in spec.role_words:
            spec.role_words.append(word)
    return spec


def _location_matches(location_words: List[str], location_text: str) -> bool:
    words = _word_set(location_text)
    for word in location_words:
        if word in words:
            continue
        full_state = US_STATES.get(word)
        if full_state and all(part in words for part in full_state.split()):
            continue
        return False
    return True


def posting_matches(spec: SearchSpec, posting: JobPosting) -> bool:
    """Return True when the posting satisfies every part of the search."""
    if not spec.is_usable:
        return False

    title_words = _word_set(posting.title)
    if any(word not in title_words for word in spec.role_words):
        return False

    if spec.topic_words:
        context_words = title_words | _word_set(
            (posting.description or "")[:_DESCRIPTION_SCAN_CHARS]
        )
        if any(word not in context_words for word in spec.topic_words):
            return False

    if spec.location_words:
        if not _location_matches(spec.location_words, posting.location or ""):
            return False
    elif spec.remote and not posting.is_remote:
        return False

    if spec.contract:
        contract_context = _word_set(
            " ".join([posting.title, *posting.extra_text, (posting.description or "")[:2000]])
        )
        if not contract_context & _CONTRACT_WORDS:
            return False

    return True


def matching_searches(posting: JobPosting, specs: Iterable[SearchSpec]) -> List[str]:
    """Return the original query text of every search the posting matches."""
    return [spec.query for spec in specs if posting_matches(spec, posting)]
