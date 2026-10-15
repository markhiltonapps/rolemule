"""Unit tests for the OpenRouter provider adapter and its allowlist / key helpers."""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from utils.llm.availability import (
    encrypted_key_attr_for_provider,
    provider_requires_api_key,
    resolve_user_llm_context,
)
from utils.llm.errors import (
    LLMError,
    is_llm_quota_or_rate_limit_exception,
    user_facing_message_from_llm_exception,
)
from utils.llm.models import (
    default_model_for_provider,
    is_valid_model_for_provider,
    is_valid_openrouter_model_slug,
    model_list_for_provider,
    models_payload,
)
from utils.llm.providers.openrouter import OpenRouterProvider
from utils.llm.registry import create_provider
from utils.openrouter_api_key_format import (
    looks_like_openrouter_api_key,
    validate_openrouter_api_key,
)

# Not the real sk-or-v1-<64 hex> shape, so secret scanning ignores it
DUMMY_OPENROUTER_KEY = "sk-or-test-dummy-key-1234567890"


def _settings(**overrides: Any) -> MagicMock:
    base = dict(
        openrouter_api_key=DUMMY_OPENROUTER_KEY,
        openrouter_model="google/gemini-2.5-flash",
        base_url="http://localhost:8000",
    )
    base.update(overrides)
    return MagicMock(**base)


def _mock_post_client(status_code: int = 200, json_body: Any = None, text: str = "") -> AsyncMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body or {})
    resp.text = text
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


class _FakeStreamResponse:
    def __init__(self, lines: List[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b'{"error":{"code":402,"message":"Insufficient credits"}}'

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False


def _mock_stream_client(stream_resp: _FakeStreamResponse) -> AsyncMock:
    client = AsyncMock()
    client.stream = MagicMock(return_value=stream_resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _patches(client: AsyncMock, **settings_overrides: Any):
    return (
        patch(
            "utils.llm.providers.openrouter.get_settings",
            return_value=_settings(**settings_overrides),
        ),
        patch("utils.llm.providers.openrouter.httpx.AsyncClient", return_value=client),
    )


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_success_payload_and_headers() -> None:
    client = _mock_post_client(
        json_body={
            "model": "google/gemini-2.5-flash",
            "choices": [{"message": {"content": "hello from openrouter"}}],
        }
    )
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        result = await provider.generate(prompt="hi", system="sys", user_api_key="sk-or-user-key-abcdefghijk")

    assert result["response"] == "hello from openrouter"
    assert result["model"] == "google/gemini-2.5-flash"
    url = client.post.call_args[0][0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    payload = client.post.call_args[1]["json"]
    assert payload["model"] == "google/gemini-2.5-flash"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["max_tokens"] == 16000
    assert "plugins" not in payload
    headers = client.post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer sk-or-user-key-abcdefghijk"
    assert headers["X-Title"] == "RoleMule"
    assert headers["HTTP-Referer"] == "http://localhost:8000"


@pytest.mark.asyncio
async def test_generate_grounding_uses_web_plugin_and_custom_model() -> None:
    client = _mock_post_client(json_body={"choices": [{"message": {"content": "grounded"}}]})
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        result = await provider.generate(
            prompt="research acme",
            model="anthropic/claude-sonnet-4.5",
            use_google_search_grounding=True,
        )
    payload = client.post.call_args[1]["json"]
    assert payload["plugins"] == [{"id": "web"}]
    assert payload["model"] == "anthropic/claude-sonnet-4.5"
    # No "model" in the response body → falls back to the requested model
    assert result["model"] == "anthropic/claude-sonnet-4.5"


@pytest.mark.asyncio
async def test_generate_http_error_raises_llm_error() -> None:
    client = _mock_post_client(status_code=402, text='{"error":{"message":"Insufficient credits"}}')
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError) as exc:
            await provider.generate(prompt="hi")
    assert exc.value.status_code == 402
    assert exc.value.provider == "openrouter"
    assert "(402)" in str(exc.value)


@pytest.mark.asyncio
async def test_generate_embedded_error_in_200_body() -> None:
    client = _mock_post_client(json_body={"error": {"code": 502, "message": "upstream down"}})
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="502: upstream down"):
            await provider.generate(prompt="hi")


@pytest.mark.asyncio
async def test_generate_embedded_string_error() -> None:
    client = _mock_post_client(json_body={"error": "boom"})
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="boom"):
            await provider.generate(prompt="hi")


@pytest.mark.asyncio
async def test_generate_empty_choices_raises() -> None:
    client = _mock_post_client(json_body={"choices": []})
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="empty choices"):
            await provider.generate(prompt="hi")


@pytest.mark.asyncio
async def test_generate_network_error_is_wrapped() -> None:
    client = _mock_post_client()
    client.post = AsyncMock(side_effect=RuntimeError("connection reset"))
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="connection reset") as exc:
            await provider.generate(prompt="hi")
    assert isinstance(exc.value.original_error, RuntimeError)


@pytest.mark.asyncio
async def test_generate_missing_key_raises() -> None:
    with patch(
        "utils.llm.providers.openrouter.get_settings",
        return_value=_settings(openrouter_api_key=None),
    ):
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="No API key"):
            await provider.generate(prompt="hi")


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_deltas_and_skips_keepalives() -> None:
    lines = [
        ": OPENROUTER PROCESSING",
        "",
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        "data: not-json",
        'data: {"choices":[]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"ignored"}}]}',
    ]
    client = _mock_stream_client(_FakeStreamResponse(lines))
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        chunks = [c async for c in provider.generate_stream(prompt="hi", system="sys")]
    assert chunks == ["Hel", "lo"]
    assert client.stream.call_args[1]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_stream_http_error_raises() -> None:
    client = _mock_stream_client(_FakeStreamResponse([], status_code=402))
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="Insufficient credits"):
            async for _ in provider.generate_stream(prompt="hi"):
                pass


@pytest.mark.asyncio
async def test_stream_midstream_error_chunk_raises() -> None:
    lines = [
        'data: {"choices":[{"delta":{"content":"partial"}}]}',
        'data: {"error":{"code":"server_error","message":"provider died"}}',
    ]
    client = _mock_stream_client(_FakeStreamResponse(lines))
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="provider died"):
            async for _ in provider.generate_stream(prompt="hi"):
                pass


@pytest.mark.asyncio
async def test_stream_network_error_is_wrapped() -> None:
    client = AsyncMock()
    client.stream = MagicMock(side_effect=RuntimeError("socket closed"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    p1, p2 = _patches(client)
    with p1, p2:
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="socket closed"):
            async for _ in provider.generate_stream(prompt="hi"):
                pass


@pytest.mark.asyncio
async def test_stream_missing_key_raises() -> None:
    with patch(
        "utils.llm.providers.openrouter.get_settings",
        return_value=_settings(openrouter_api_key=None),
    ):
        provider = OpenRouterProvider()
        with pytest.raises(LLMError, match="No API key"):
            async for _ in provider.generate_stream(prompt="hi"):
                pass


@pytest.mark.asyncio
async def test_stream_grounding_falls_back_to_generate() -> None:
    with patch(
        "utils.llm.providers.openrouter.get_settings", return_value=_settings()
    ):
        provider = OpenRouterProvider()
        provider.generate = AsyncMock(return_value={"response": "full text"})  # type: ignore[method-assign]
        chunks = [
            c
            async for c in provider.generate_stream(
                prompt="hi", use_google_search_grounding=True
            )
        ]
    assert chunks == ["full text"]
    assert provider.generate.call_args[1]["use_google_search_grounding"] is True


@pytest.mark.asyncio
async def test_stream_grounding_empty_text_yields_nothing() -> None:
    with patch(
        "utils.llm.providers.openrouter.get_settings", return_value=_settings()
    ):
        provider = OpenRouterProvider()
        provider.generate = AsyncMock(return_value={"response": ""})  # type: ignore[method-assign]
        chunks = [
            c
            async for c in provider.generate_stream(
                prompt="hi", use_google_search_grounding=True
            )
        ]
    assert chunks == []


# ---------------------------------------------------------------------------
# health_check()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_without_server_key_is_noop() -> None:
    with patch(
        "utils.llm.providers.openrouter.get_settings",
        return_value=_settings(openrouter_api_key=None),
    ):
        assert await OpenRouterProvider().health_check() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(200, True), (429, True), (503, False)])
async def test_health_check_status_codes(status: int, expected: bool) -> None:
    client = _mock_post_client(status_code=status)
    p1, p2 = _patches(client)
    with p1, p2:
        assert await OpenRouterProvider().health_check() is expected
    assert client.get.call_args[0][0] == "https://openrouter.ai/api/v1/key"


@pytest.mark.asyncio
async def test_health_check_exception_returns_false() -> None:
    client = _mock_post_client()
    client.get = AsyncMock(side_effect=RuntimeError("dns"))
    p1, p2 = _patches(client)
    with p1, p2:
        assert await OpenRouterProvider().health_check() is False


# ---------------------------------------------------------------------------
# Registry, models, keys, errors, availability
# ---------------------------------------------------------------------------


def test_registry_creates_openrouter_provider() -> None:
    with patch("utils.llm.providers.openrouter.get_settings", return_value=_settings()):
        assert isinstance(create_provider("OpenRouter"), OpenRouterProvider)


def test_models_curated_list_and_default() -> None:
    models = model_list_for_provider("openrouter")
    assert models[0] == "google/gemini-2.5-flash"
    assert "openrouter/auto" in models
    assert "openrouter" in models_payload()
    assert default_model_for_provider("openrouter", _settings()) == "google/gemini-2.5-flash"


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/auto",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct:free",
        "x-ai/grok-4",
        None,
        "  ",
    ],
)
def test_openrouter_accepts_curated_and_custom_slugs(model: Any) -> None:
    assert is_valid_model_for_provider("openrouter", model) is True


@pytest.mark.parametrize(
    "model",
    ["gpt-5", "no-slash", "/leading", "vendor/", "vendor/has space", "a/" + "b" * 63],
)
def test_openrouter_rejects_malformed_slugs(model: str) -> None:
    assert is_valid_model_for_provider("openrouter", model) is False


def test_custom_slugs_only_allowed_for_openrouter() -> None:
    assert is_valid_model_for_provider("openai", "deepseek/deepseek-chat") is False
    assert is_valid_openrouter_model_slug("deepseek/deepseek-chat") is True


@pytest.mark.parametrize(
    "key,expected",
    [
        (DUMMY_OPENROUTER_KEY, True),
        ("  " + DUMMY_OPENROUTER_KEY + "  ", True),
        ("sk-proj-abcdefghijklmnopqrstuvwxyz", False),
        ("sk-or-short", False),
        ("sk-or-has space in the key 123", False),
        ("sk-or-bad!chars-1234567890abc", False),
        ("sk-or-" + "a" * 600, False),
        ("", False),
        ("   ", False),
    ],
)
def test_validate_openrouter_api_key(key: str, expected: bool) -> None:
    assert validate_openrouter_api_key(key) is expected


def test_looks_like_openrouter_api_key() -> None:
    assert looks_like_openrouter_api_key(DUMMY_OPENROUTER_KEY) is True
    assert looks_like_openrouter_api_key("sk-proj-abc") is False
    assert looks_like_openrouter_api_key("") is False


@pytest.mark.parametrize(
    "message",
    [
        "OpenRouter generate failed (402): Insufficient credits",
        "OpenRouter generate failed (429): slow down",
        "OpenRouter stream failed: 429: Rate limit exceeded",
    ],
)
def test_openrouter_quota_errors_get_friendly_message(message: str) -> None:
    exc = LLMError(message, provider="openrouter")
    assert is_llm_quota_or_rate_limit_exception(exc) is True
    assert "OpenRouter" in user_facing_message_from_llm_exception(exc)


def test_openrouter_other_errors_pass_through() -> None:
    exc = LLMError("OpenRouter generate failed (400): bad model", provider="openrouter")
    assert is_llm_quota_or_rate_limit_exception(exc) is False
    assert "bad model" in user_facing_message_from_llm_exception(exc)


def test_availability_resolves_openrouter_key() -> None:
    assert encrypted_key_attr_for_provider("openrouter") == "openrouter_api_key_encrypted"
    assert provider_requires_api_key("openrouter") is True

    user = MagicMock(openrouter_api_key_encrypted="encrypted-blob")
    prefs = {"preferred_provider": "openrouter", "preferred_model": "x-ai/grok-4"}
    cfg = MagicMock(use_vertex_ai=False)
    with patch("utils.encryption.decrypt_api_key", return_value=DUMMY_OPENROUTER_KEY):
        ctx = resolve_user_llm_context(user, prefs, settings=cfg)
    assert ctx.ready is True
    assert ctx.provider == "openrouter"
    assert ctx.user_api_key == DUMMY_OPENROUTER_KEY
    # Custom slug survives the stale-model filter
    assert ctx.preferred_model == "x-ai/grok-4"


def test_availability_openrouter_without_key_not_ready() -> None:
    user = MagicMock(openrouter_api_key_encrypted=None)
    ctx = resolve_user_llm_context(
        user,
        {"preferred_provider": "openrouter"},
        settings=MagicMock(use_vertex_ai=False),
    )
    assert ctx.ready is False
    assert ctx.reason == "no_api_key"
