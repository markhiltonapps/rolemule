"""OpenRouter provider — OpenAI-compatible Chat Completions (httpx async).

OpenRouter (https://openrouter.ai) fronts many model vendors behind one key.
Model ids are vendor-prefixed slugs such as ``google/gemini-2.5-flash``.
Grounded calls use OpenRouter's ``web`` plugin instead of a vendor tool.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from config.settings import get_settings
from utils.llm.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    HTTP_CONNECT_TIMEOUT,
)
from utils.llm.errors import LLMError
from utils.llm.types import as_generate_result
from utils.logging_config import get_structured_logger, sanitize_log_value

logger = logging.getLogger(__name__)
structured_logger = get_structured_logger(__name__)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_CHAT_URL = f"{OPENROUTER_API_BASE}/chat/completions"
_APP_TITLE = "RoleMule"


def _chat_body(
    *,
    model: str,
    system: Optional[str],
    prompt: str,
    temperature: float,
    max_tokens: int,
    web_search: bool,
) -> Dict[str, Any]:
    """Build an OpenRouter Chat Completions body (unsupported params are ignored upstream)."""
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if web_search:
        payload["plugins"] = [{"id": "web"}]
    return payload


def _headers(api_key: str) -> Dict[str, str]:
    """Auth + optional OpenRouter app-attribution headers."""
    settings = get_settings()
    referer = getattr(settings, "base_url", None) or "http://localhost:8000"
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": str(referer),
        "X-Title": _APP_TITLE,
    }


def _error_from_body(data: Dict[str, Any]) -> Optional[str]:
    """Return an error message when OpenRouter embeds one in a 200 payload."""
    err = data.get("error")
    if not err:
        return None
    if isinstance(err, dict):
        code = err.get("code")
        msg = err.get("message") or "unknown error"
        return f"{code}: {msg}" if code is not None else str(msg)
    return str(err)


class OpenRouterProvider:
    """OpenRouter Chat Completions adapter with optional web plugin grounding."""

    name: str = "openrouter"

    def __init__(self) -> None:
        """Load OpenRouter settings."""
        settings = get_settings()
        self.api_key: Optional[str] = getattr(settings, "openrouter_api_key", None)
        self.default_model: str = getattr(
            settings, "openrouter_model", "google/gemini-2.5-flash"
        )
        self.timeout = DEFAULT_TIMEOUT
        logger.info(
            "[LLM] Ready  model=%s  backend=OpenRouter  key=%s",
            sanitize_log_value(self.default_model),
            sanitize_log_value("set" if self.api_key else "missing"),
        )

    def _require_key(self, user_api_key: Optional[str]) -> str:
        """Return the effective key or raise."""
        effective_key = user_api_key or self.api_key
        if not effective_key:
            raise LLMError(
                "No API key available. Configure OPENROUTER_API_KEY or provide a user key.",
                provider="openrouter",
            )
        return effective_key

    async def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        user_api_key: Optional[str] = None,
        use_google_search_grounding: bool = False,
    ) -> Dict[str, Any]:
        """Call OpenRouter chat completions (``web`` plugin when grounded)."""
        effective_key = self._require_key(user_api_key)
        model_to_use = model or self.default_model
        operation = (
            "chat.completions.web" if use_google_search_grounding else "chat.completions"
        )
        payload = _chat_body(
            model=model_to_use,
            system=system,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            web_search=use_google_search_grounding,
        )

        logger.info(
            "[LLM] OpenRouter  model=%s  prompt=%s chars  temp=%s  web=%s",
            sanitize_log_value(model_to_use),
            sanitize_log_value(len(prompt) + (len(system) if system else 0)),
            sanitize_log_value(temperature),
            sanitize_log_value(use_google_search_grounding),
        )

        api_start = perf_counter()
        try:
            timeout = httpx.Timeout(self.timeout, connect=HTTP_CONNECT_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    _OPENROUTER_CHAT_URL, json=payload, headers=_headers(effective_key)
                )
            api_duration_ms = (perf_counter() - api_start) * 1000

            if response.status_code >= 400:
                body = response.text[:500]
                raise LLMError(
                    f"OpenRouter generate failed ({response.status_code}): {body}",
                    status_code=response.status_code,
                    provider="openrouter",
                )

            data = response.json()
            embedded = _error_from_body(data)
            if embedded:
                raise LLMError(
                    f"OpenRouter generate failed: {embedded}",
                    provider="openrouter",
                )
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(
                    "OpenRouter generate failed: empty choices",
                    provider="openrouter",
                )
            text = (choices[0].get("message") or {}).get("content") or ""
            logger.info(
                "[LLM] Done  %sms  response=%s chars",
                sanitize_log_value(api_duration_ms),
                sanitize_log_value(len(text)),
            )
            structured_logger.log_external_api_call(
                service="openrouter",
                operation=operation,
                duration_ms=api_duration_ms,
                success=True,
            )
            return as_generate_result(
                response=text, model=str(data.get("model") or model_to_use)
            )

        except LLMError:
            raise
        except Exception as e:
            structured_logger.log_external_api_call(
                service="openrouter",
                operation=operation,
                duration_ms=0,
                success=False,
                error=str(e),
            )
            logger.error(
                "Error in OpenRouter generate: %s",
                sanitize_log_value(str(e)),
                exc_info=True,
            )
            raise LLMError(
                f"OpenRouter generate failed: {e}",
                original_error=e if isinstance(e, Exception) else None,
                provider="openrouter",
            ) from e

    async def generate_stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        user_api_key: Optional[str] = None,
        use_google_search_grounding: bool = False,
    ) -> AsyncIterator[str]:
        """Stream Chat Completions deltas (grounding falls back to non-stream)."""
        if use_google_search_grounding:
            full = await self.generate(
                prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                user_api_key=user_api_key,
                use_google_search_grounding=True,
            )
            text = full.get("response") or ""
            if text:
                yield text
            return

        effective_key = self._require_key(user_api_key)
        model_to_use = model or self.default_model
        payload = _chat_body(
            model=model_to_use,
            system=system,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            web_search=False,
        )
        payload["stream"] = True
        logger.info(
            "[LLM] OpenRouter stream  model=%s  prompt=%s chars",
            sanitize_log_value(model_to_use),
            sanitize_log_value(len(prompt) + (len(system) if system else 0)),
        )
        api_start = perf_counter()
        total_chars = 0
        try:
            timeout = httpx.Timeout(self.timeout, connect=HTTP_CONNECT_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    _OPENROUTER_CHAT_URL,
                    json=payload,
                    headers=_headers(effective_key),
                ) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="replace")[
                            :500
                        ]
                        raise LLMError(
                            f"OpenRouter stream failed ({response.status_code}): {body}",
                            status_code=response.status_code,
                            provider="openrouter",
                        )
                    # SSE: ": OPENROUTER PROCESSING" keep-alive comments are skipped
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug(
                                "OpenRouter stream skip bad JSON: %s",
                                sanitize_log_value(data_str[:80]),
                            )
                            continue
                        embedded = _error_from_body(data)
                        if embedded:
                            raise LLMError(
                                f"OpenRouter stream failed: {embedded}",
                                provider="openrouter",
                            )
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if delta:
                            total_chars += len(delta)
                            yield delta
            api_duration_ms = (perf_counter() - api_start) * 1000
            logger.info(
                "[LLM] Done stream  %sms  response=%s chars",
                sanitize_log_value(api_duration_ms),
                sanitize_log_value(total_chars),
            )
            structured_logger.log_external_api_call(
                service="openrouter",
                operation="chat.completions.stream",
                duration_ms=api_duration_ms,
                success=True,
            )
        except LLMError:
            raise
        except Exception as e:
            structured_logger.log_external_api_call(
                service="openrouter",
                operation="chat.completions.stream",
                duration_ms=0,
                success=False,
                error=str(e),
            )
            logger.error(
                "Error in OpenRouter stream: %s",
                sanitize_log_value(str(e)),
                exc_info=True,
            )
            raise LLMError(
                f"OpenRouter stream failed: {e}",
                original_error=e if isinstance(e, Exception) else None,
                provider="openrouter",
            ) from e

    async def health_check(self) -> bool:
        """Check the server key against /key; missing key is a no-op success."""
        if not self.api_key:
            return True
        try:
            timeout = httpx.Timeout(10.0, connect=HTTP_CONNECT_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(
                    f"{OPENROUTER_API_BASE}/key",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            if response.status_code == 429:
                logger.info("OpenRouter health check: rate limited but reachable")
                return True
            return response.status_code < 500
        except Exception as e:
            logger.warning(
                "OpenRouter health check failed: %s", sanitize_log_value(str(e))
            )
            return False
