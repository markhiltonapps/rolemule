"""Find Jobs — public company careers-board sources.

Many companies publish their open roles through an applicant-tracking system
that exposes a free, unauthenticated JSON listing. This module parses a
careers-page link into ``(platform, board token)``, fetches the listing, and
normalizes each posting into :class:`JobPosting`.

Only fixed API hosts are ever contacted; user input supplies just the board
token (validated by ``_BOARD_TOKEN_RE``), so there is no SSRF surface here.
"""

from __future__ import annotations

import asyncio
import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from utils.logging_config import sanitize_log_value

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS: Tuple[str, ...] = ("greenhouse", "lever", "ashby")

MAX_DESCRIPTION_CHARS = 50_000
FETCH_TIMEOUT_SECONDS = 20.0
_BOARD_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")

# Careers-page hostnames → platform
_HOST_PLATFORMS: Dict[str, str] = {
    "boards.greenhouse.io": "greenhouse",
    "job-boards.greenhouse.io": "greenhouse",
    "boards-api.greenhouse.io": "greenhouse",
    "jobs.lever.co": "lever",
    "api.lever.co": "lever",
    "jobs.ashbyhq.com": "ashby",
    "api.ashbyhq.com": "ashby",
}

# Path segments that precede the board token on API-style URLs
_API_PATH_PREFIXES: Tuple[str, ...] = (
    "v1/boards/",
    "v0/postings/",
    "posting-api/job-board/",
)


@dataclass
class JobPosting:
    """One normalized open role from any source."""

    source: str
    external_id: str
    title: str
    company: str
    url: str
    location: Optional[str] = None
    is_remote: bool = False
    description: Optional[str] = None
    posted_at: Optional[datetime] = None
    extra_text: List[str] = field(default_factory=list)


class BoardFetchError(Exception):
    """A careers board could not be loaded (not found, network, bad payload)."""

    def __init__(self, message: str, *, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


# =============================================================================
# Careers-page link parsing
# =============================================================================


def parse_board_reference(raw: str) -> Optional[Tuple[str, str]]:
    """
    Parse a careers-page link (or ``platform:token``) into ``(platform, token)``.

    Args:
        raw: e.g. ``https://jobs.ashbyhq.com/acme``, ``boards.greenhouse.io/acme``,
            or ``lever:acme``

    Returns:
        ``(platform, token)`` or None when the link is not a supported board
    """
    text = (raw or "").strip()
    if not text:
        return None

    prefixed = re.match(r"^(greenhouse|lever|ashby)\s*:\s*([^\s/]+)$", text, re.I)
    if prefixed:
        platform, token = prefixed.group(1).lower(), prefixed.group(2)
        return (platform, token) if _BOARD_TOKEN_RE.match(token) else None

    if "://" not in text:
        text = f"https://{text}"
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    platform = _HOST_PLATFORMS.get(host)
    if not platform:
        return None

    path = parsed.path.strip("/")
    for prefix in _API_PATH_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    token = path.split("/", 1)[0] if path else ""
    if not token and platform == "greenhouse":
        # Embedded boards: boards.greenhouse.io/embed/job_board?for=acme
        match = re.search(r"(?:^|&)for=([^&]+)", parsed.query)
        token = match.group(1) if match else ""
    if token == "embed":
        match = re.search(r"(?:^|&)for=([^&]+)", parsed.query)
        token = match.group(1) if match else ""
    if not token or not _BOARD_TOKEN_RE.match(token):
        return None
    return platform, token


def board_page_url(platform: str, token: str) -> str:
    """Public careers page for a board (shown in the UI)."""
    if platform == "greenhouse":
        return f"https://job-boards.greenhouse.io/{token}"
    if platform == "lever":
        return f"https://jobs.lever.co/{token}"
    return f"https://jobs.ashbyhq.com/{token}"


def default_company_name(token: str) -> str:
    """Readable fallback name from a board token (``scale-ai`` → ``Scale Ai``)."""
    words = re.split(r"[-_.]+", token)
    return " ".join(w[:1].upper() + w[1:] for w in words if w) or token


# =============================================================================
# Text helpers
# =============================================================================


class _TextExtractor(HTMLParser):
    """Collect visible text, keeping paragraph and list breaks."""

    _BREAK_TAGS = {"p", "br", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1
        elif tag in self._BREAK_TAGS:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(raw_html: Optional[str]) -> str:
    """
    Convert posting HTML to readable plain text.

    Handles double-escaped HTML (``&lt;p&gt;``) as some boards return it.
    """
    if not raw_html:
        return ""
    text = raw_html
    if "&lt;" in text and "<" not in text:
        text = html.unescape(text)
    parser = _TextExtractor()
    parser.feed(text)
    parser.close()
    joined = "".join(parser.parts).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in joined.splitlines()]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned[:MAX_DESCRIPTION_CHARS]


def _looks_remote(*values: Optional[str]) -> bool:
    return any(v and "remote" in v.lower() for v in values)


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_epoch_ms(value: Any) -> Optional[datetime]:
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


# =============================================================================
# Per-platform normalizers
# =============================================================================


def normalize_greenhouse(payload: Dict[str, Any], company: str) -> List[JobPosting]:
    """Normalize a ``/v1/boards/{token}/jobs?content=true`` payload."""
    postings: List[JobPosting] = []
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict) or not job.get("id") or not job.get("title"):
            continue
        location = ((job.get("location") or {}).get("name") or "").strip() or None
        offices = [
            (o.get("name") or "").strip()
            for o in job.get("offices") or []
            if isinstance(o, dict)
        ]
        url = job.get("absolute_url") or ""
        if not url:
            continue
        postings.append(
            JobPosting(
                source="greenhouse",
                external_id=str(job["id"]),
                title=str(job["title"]).strip(),
                company=company,
                url=url,
                location=location,
                is_remote=_looks_remote(location, *offices),
                description=html_to_text(job.get("content")) or None,
                posted_at=_parse_iso(job.get("first_published") or job.get("updated_at")),
            )
        )
    return postings


def normalize_lever(payload: Any, company: str) -> List[JobPosting]:
    """Normalize a ``/v0/postings/{token}?mode=json`` payload (a JSON list)."""
    postings: List[JobPosting] = []
    if not isinstance(payload, list):
        return postings
    for job in payload:
        if not isinstance(job, dict) or not job.get("id") or not job.get("text"):
            continue
        categories = job.get("categories") or {}
        location = (categories.get("location") or "").strip() or None
        all_locations = [str(x) for x in categories.get("allLocations") or []]
        commitment = (categories.get("commitment") or "").strip()
        workplace = (job.get("workplaceType") or "").strip()
        sections = [job.get("descriptionPlain") or html_to_text(job.get("description"))]
        for block in job.get("lists") or []:
            if isinstance(block, dict):
                heading = (block.get("text") or "").strip()
                body = html_to_text(block.get("content"))
                sections.append(f"{heading}\n{body}".strip())
        sections.append(job.get("additionalPlain") or "")
        description = "\n\n".join(s.strip() for s in sections if s and s.strip())
        url = job.get("hostedUrl") or job.get("applyUrl") or ""
        if not url:
            continue
        postings.append(
            JobPosting(
                source="lever",
                external_id=str(job["id"]),
                title=str(job["text"]).strip(),
                company=company,
                url=url,
                location=location,
                is_remote=workplace.lower() == "remote"
                or _looks_remote(location, *all_locations),
                description=description[:MAX_DESCRIPTION_CHARS] or None,
                posted_at=_parse_epoch_ms(job.get("createdAt")),
                extra_text=[commitment] if commitment else [],
            )
        )
    return postings


def normalize_ashby(payload: Dict[str, Any], company: str) -> List[JobPosting]:
    """Normalize a ``/posting-api/job-board/{token}`` payload."""
    postings: List[JobPosting] = []
    for job in payload.get("jobs") or []:
        if not isinstance(job, dict) or not job.get("id") or not job.get("title"):
            continue
        if job.get("isListed") is False:
            continue
        location = (job.get("location") or "").strip() or None
        secondary = [
            (s.get("location") or "").strip()
            for s in job.get("secondaryLocations") or []
            if isinstance(s, dict)
        ]
        if secondary and location:
            location = ", ".join([location, *[s for s in secondary if s]])[:300]
        workplace = (job.get("workplaceType") or "").strip()
        url = job.get("jobUrl") or job.get("applyUrl") or ""
        if not url:
            continue
        description = job.get("descriptionPlain") or html_to_text(job.get("descriptionHtml"))
        employment = (job.get("employmentType") or "").strip()
        postings.append(
            JobPosting(
                source="ashby",
                external_id=str(job["id"]),
                title=str(job["title"]).strip(),
                company=company,
                url=url,
                location=location,
                is_remote=bool(job.get("isRemote"))
                or workplace.lower() == "remote"
                or _looks_remote(location),
                description=(description or "")[:MAX_DESCRIPTION_CHARS] or None,
                posted_at=_parse_iso(job.get("publishedAt")),
                extra_text=[employment] if employment else [],
            )
        )
    return postings


_ENDPOINTS: Dict[str, Tuple[str, Callable[[Any, str], List[JobPosting]]]] = {
    "greenhouse": (
        "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true",
        normalize_greenhouse,
    ),
    "lever": ("https://api.lever.co/v0/postings/{token}?mode=json", normalize_lever),
    "ashby": (
        "https://api.ashbyhq.com/posting-api/job-board/{token}",
        normalize_ashby,
    ),
}


async def fetch_board(
    client: httpx.AsyncClient, platform: str, token: str, company: str
) -> List[JobPosting]:
    """
    Fetch and normalize every open role on one careers board.

    Raises:
        BoardFetchError: board missing (404), HTTP error, or unreadable payload
    """
    if platform not in _ENDPOINTS or not _BOARD_TOKEN_RE.match(token or ""):
        raise BoardFetchError("Unsupported careers page")
    template, normalizer = _ENDPOINTS[platform]
    url = template.format(token=token)
    try:
        response = await client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise BoardFetchError(f"Network error: {exc.__class__.__name__}") from exc
    if response.status_code == 404:
        raise BoardFetchError("Careers page not found", not_found=True)
    if response.status_code >= 400:
        raise BoardFetchError(f"Careers page returned HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise BoardFetchError("Careers page returned unreadable data") from exc
    return normalizer(payload, company)


async def fetch_boards(
    companies: List[Dict[str, Any]],
    *,
    concurrency: int = 5,
    fetcher: Optional[
        Callable[[httpx.AsyncClient, str, str, str], Awaitable[List[JobPosting]]]
    ] = None,
) -> Tuple[List[JobPosting], Dict[str, Dict[str, Any]]]:
    """
    Fetch all watched boards concurrently.

    Args:
        companies: ``[{"platform", "token", "name"}]``
        concurrency: max simultaneous requests
        fetcher: injectable for tests (defaults to :func:`fetch_board`)

    Returns:
        ``(postings, status_by_key)`` where key is ``platform:token`` and status
        is ``{"ok": bool, "jobs": int, "error": str | None}``
    """
    fetch = fetcher or fetch_board
    semaphore = asyncio.Semaphore(max(1, concurrency))
    statuses: Dict[str, Dict[str, Any]] = {}
    all_postings: List[JobPosting] = []
    timeout = httpx.Timeout(FETCH_TIMEOUT_SECONDS, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:

        async def _one(company: Dict[str, Any]) -> None:
            platform = str(company.get("platform") or "")
            token = str(company.get("token") or "")
            name = str(company.get("name") or default_company_name(token))
            key = f"{platform}:{token}"
            async with semaphore:
                try:
                    postings = await fetch(client, platform, token, name)
                except BoardFetchError as exc:
                    statuses[key] = {"ok": False, "jobs": 0, "error": str(exc)}
                    logger.info(
                        "Find Jobs board %s failed: %s",
                        sanitize_log_value(key),
                        sanitize_log_value(str(exc)),
                    )
                    return
                except Exception as exc:
                    statuses[key] = {"ok": False, "jobs": 0, "error": "Unexpected error"}
                    logger.warning(
                        "Find Jobs board %s unexpected error: %s",
                        sanitize_log_value(key),
                        sanitize_log_value(str(exc)),
                        exc_info=True,
                    )
                    return
            statuses[key] = {"ok": True, "jobs": len(postings), "error": None}
            all_postings.extend(postings)

        await asyncio.gather(*(_one(c) for c in companies))

    return all_postings, statuses
