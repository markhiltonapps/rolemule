"""Find Jobs — AI web search for open roles matching a saved search.

Uses the user's BYOK provider with web-search grounding (Gemini Google Search,
OpenAI / Anthropic web search, OpenRouter ``web`` plugin). Ollama has no web
tools, so callers must skip this agent for Ollama.

Results come from a model reading search results, so they are leads to verify,
not authoritative data: only http(s) URLs are kept, and nothing is fetched
server-side from those URLs.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from utils.job_sources import JobPosting
from utils.llm_client import get_llm_client
from utils.llm_parsing import parse_json_from_llm_response
from utils.logging_config import sanitize_log_value

logger = logging.getLogger(__name__)

MAX_RESULTS_PER_SEARCH = 10
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4000

SYSTEM_CONTEXT = """You are a job-search assistant. You use web search to find job postings that are \
currently open. You only report postings you actually found in search results, with the exact URL \
of the posting page. You never invent companies, titles, or URLs. You answer with JSON only."""

PROMPT_TEMPLATE = """Today is {today}. Search the web for job postings that match this search:

"{query}"

Rules:
- Only include postings that appear to be open now (posted in roughly the last 30 days when a date is shown).
- Prefer the employer's own careers page URL for each posting; otherwise use the job board page URL.
- Each item must be a single specific job posting, not a search-results page or a list of jobs.
- Include at most {limit} postings. Fewer is fine; an empty list is fine.
- Respect the location / remote / contract wording in the search.

Return ONLY this JSON object:
{{
  "jobs": [
    {{
      "title": "exact job title",
      "company": "employer name",
      "location": "location as written in the posting, or Remote",
      "is_remote": true,
      "url": "https://... direct link to the posting",
      "posted_date": "YYYY-MM-DD or null",
      "summary": "2-3 sentences: what the role does and key requirements"
    }}
  ]
}}"""


def _clean_str(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:max_len]


def _valid_posting_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.hostname) and "." in (
        parsed.hostname or ""
    )


def _parse_date(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def web_external_id(url: str) -> str:
    """Stable dedupe id for a web-found posting (hash of the URL without fragment)."""
    normalized = url.split("#", 1)[0].rstrip("/").lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:40]


def normalize_web_results(parsed: Dict[str, Any], query: str) -> List[JobPosting]:
    """Turn the model's JSON into :class:`JobPosting` objects, dropping bad rows."""
    jobs = parsed.get("jobs") if isinstance(parsed, dict) else None
    if not isinstance(jobs, list):
        return []
    postings: List[JobPosting] = []
    seen: set[str] = set()
    for item in jobs[: MAX_RESULTS_PER_SEARCH * 2]:
        if not isinstance(item, dict):
            continue
        title = _clean_str(item.get("title"), 300)
        company = _clean_str(item.get("company"), 200)
        url = _clean_str(item.get("url"), 2000)
        if not title or not company or not _valid_posting_url(url):
            continue
        external_id = web_external_id(url)
        if external_id in seen:
            continue
        seen.add(external_id)
        location = _clean_str(item.get("location"), 300) or None
        summary = _clean_str(item.get("summary"), 2000)
        is_remote = item.get("is_remote") is True or bool(
            location and "remote" in location.lower()
        )
        postings.append(
            JobPosting(
                source="web",
                external_id=external_id,
                title=title,
                company=company,
                url=url,
                location=location,
                is_remote=is_remote,
                # Summaries are too short to analyze; Analyze opens the posting instead
                description=None,
                posted_at=_parse_date(item.get("posted_date")),
                extra_text=[summary] if summary else [],
            )
        )
        if len(postings) >= MAX_RESULTS_PER_SEARCH:
            break
    return postings


class JobWebSearchAgent:
    """Runs one grounded web search per saved search."""

    def __init__(
        self,
        *,
        llm_provider: Optional[str],
        user_api_key: Optional[str],
        model: Optional[str] = None,
        llm_client: Any = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.user_api_key = user_api_key
        self.model = model
        self._llm_client = llm_client

    async def search(self, query: str) -> List[JobPosting]:
        """
        Search the web for postings matching ``query``.

        Raises:
            LLMError: when the provider call fails (caller records the error)
        """
        if self._llm_client is None:
            self._llm_client = await get_llm_client()
        prompt = PROMPT_TEMPLATE.format(
            today=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            query=query.replace('"', "'"),
            limit=MAX_RESULTS_PER_SEARCH,
        )
        response = await self._llm_client.generate(
            prompt=prompt,
            system=SYSTEM_CONTEXT,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            user_api_key=self.user_api_key,
            model=self.model,
            provider=self.llm_provider,
            use_google_search_grounding=True,
        )
        if response.get("filtered"):
            logger.info("Find Jobs web search filtered for query %s", sanitize_log_value(query))
            return []
        parsed = parse_json_from_llm_response(response.get("response") or "")
        postings = normalize_web_results(parsed, query)
        logger.info(
            "Find Jobs web search %s returned %s postings",
            sanitize_log_value(query),
            len(postings),
        )
        return postings
