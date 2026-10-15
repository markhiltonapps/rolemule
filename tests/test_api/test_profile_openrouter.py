"""Profile API tests for OpenRouter BYOK — save, status, validate, custom model slugs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.profile import (
    ApiKeyRequest,
    ApplicationPreferencesRequest,
    get_api_key_status,
    set_api_key,
    update_application_preferences,
    validate_api_key_endpoint,
)
from tests.test_api.conftest import _NullSessionLocal
from tests.test_api.test_profile_coverage import _current_user
from tests.test_api.test_profile_extended import (
    _create_user_with_password,
    _delete_user_data,
)

# Not the real sk-or-v1-<64 hex> shape, so secret scanning ignores it
OPENROUTER_KEY = "sk-or-test-dummy-key-1234567890"


def _mock_httpx_get(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = None
    client.get = AsyncMock(return_value=resp)
    return client


class TestOpenRouterApiKey:
    @pytest.mark.asyncio
    async def test_set_key_activates_provider_and_status_reports_it(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            async with _NullSessionLocal() as db:
                result = await set_api_key(
                    ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openrouter"),
                    _current_user(uid, email),
                    db,
                )
                assert result["provider"] == "openrouter"

            async with _NullSessionLocal() as db:
                status = await get_api_key_status(_current_user(uid, email), db)
            assert status.preferred_provider == "openrouter"
            assert status.has_credentials is True
            assert status.providers["openrouter"].has_key is True
            assert status.key_preview == f"{OPENROUTER_KEY[:4]}...{OPENROUTER_KEY[-4:]}"
            assert status.models["openrouter"][0] == "google/gemini-2.5-flash"
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_set_key_invalid_format(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            async with _NullSessionLocal() as db:
                with pytest.raises(Exception) as exc:
                    await set_api_key(
                        ApiKeyRequest(api_key="sk-proj-notopenrouter123", provider="openrouter"),
                        _current_user(uid, email),
                        db,
                    )
            assert exc.value.status_code == 422
            assert "OpenRouter" in exc.value.message
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_openrouter_key_saved_as_openai_is_redirected(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            async with _NullSessionLocal() as db:
                with pytest.raises(Exception) as exc:
                    await set_api_key(
                        ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openai"),
                        _current_user(uid, email),
                        db,
                    )
            assert exc.value.status_code == 422
            assert "OpenRouter" in exc.value.message
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_validate_key_success(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            client = _mock_httpx_get(200)
            with patch("api.profile.check_rate_limit", AsyncMock(return_value=(True, 9))), patch(
                "httpx.AsyncClient", return_value=client
            ):
                result = await validate_api_key_endpoint(
                    ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openrouter"),
                    _current_user(uid, email),
                )
            assert result["valid"] is True
            assert result["provider"] == "openrouter"
            assert client.get.call_args[0][0] == "https://openrouter.ai/api/v1/key"
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_validate_key_rejected_by_openrouter(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            with patch("api.profile.check_rate_limit", AsyncMock(return_value=(True, 9))), patch(
                "httpx.AsyncClient", return_value=_mock_httpx_get(401)
            ):
                with pytest.raises(Exception) as exc:
                    await validate_api_key_endpoint(
                        ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openrouter"),
                        _current_user(uid, email),
                    )
            assert exc.value.status_code == 422
            assert "OpenRouter" in exc.value.message
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_validate_key_network_error(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            client = _mock_httpx_get(200)
            client.get = AsyncMock(side_effect=RuntimeError("dns failure"))
            with patch("api.profile.check_rate_limit", AsyncMock(return_value=(True, 9))), patch(
                "httpx.AsyncClient", return_value=client
            ):
                with pytest.raises(Exception) as exc:
                    await validate_api_key_endpoint(
                        ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openrouter"),
                        _current_user(uid, email),
                    )
            assert exc.value.status_code == 422
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_validate_key_invalid_format(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            with patch("api.profile.check_rate_limit", AsyncMock(return_value=(True, 9))):
                with pytest.raises(Exception) as exc:
                    await validate_api_key_endpoint(
                        ApiKeyRequest(api_key="not-an-openrouter-key", provider="openrouter"),
                        _current_user(uid, email),
                    )
            assert exc.value.status_code == 422
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_validate_openrouter_key_as_openai_is_redirected(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            with patch("api.profile.check_rate_limit", AsyncMock(return_value=(True, 9))):
                with pytest.raises(Exception) as exc:
                    await validate_api_key_endpoint(
                        ApiKeyRequest(api_key=OPENROUTER_KEY, provider="openai"),
                        _current_user(uid, email),
                    )
            assert exc.value.status_code == 422
            assert "OpenRouter" in exc.value.message
        finally:
            await _delete_user_data(uid)


class TestOpenRouterPreferences:
    @pytest.mark.asyncio
    async def test_custom_model_slug_accepted_for_openrouter(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            async with _NullSessionLocal() as db:
                resp = await update_application_preferences(
                    ApplicationPreferencesRequest(
                        preferred_provider="openrouter",
                        preferred_model="deepseek/deepseek-chat",
                    ),
                    _current_user(uid, email),
                    db,
                )
            assert resp.preferred_provider == "openrouter"
            assert resp.preferred_model == "deepseek/deepseek-chat"
        finally:
            await _delete_user_data(uid)

    @pytest.mark.asyncio
    async def test_malformed_slug_rejected_for_openrouter(self) -> None:
        uid, email = await _create_user_with_password()
        try:
            async with _NullSessionLocal() as db:
                with pytest.raises(Exception) as exc:
                    await update_application_preferences(
                        ApplicationPreferencesRequest(
                            preferred_provider="openrouter",
                            preferred_model="not a slug",
                        ),
                        _current_user(uid, email),
                        db,
                    )
            assert exc.value.status_code == 422
        finally:
            await _delete_user_data(uid)
