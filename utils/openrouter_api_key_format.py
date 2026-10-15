"""Loose format check for OpenRouter API keys (sk-or-...)."""

from typing import Final

_MIN_LEN: Final[int] = 20
_MAX_LEN: Final[int] = 512
OPENROUTER_KEY_PREFIX: Final[str] = "sk-or-"


def validate_openrouter_api_key(api_key: str) -> bool:
    """
    Return True if the string plausibly looks like an OpenRouter API key.

    Args:
        api_key: Raw key string from the user

    Returns:
        True when the format looks acceptable
    """
    if not api_key or not api_key.strip():
        return False
    k = api_key.strip()
    if any(c.isspace() for c in k):
        return False
    if len(k) < _MIN_LEN or len(k) > _MAX_LEN:
        return False
    if not k.startswith(OPENROUTER_KEY_PREFIX):
        return False
    return all(c.isalnum() or c in "-_" for c in k)


def looks_like_openrouter_api_key(api_key: str) -> bool:
    """Return True when a key carries the OpenRouter prefix (used to catch wrong-provider pastes)."""
    return bool(api_key) and api_key.strip().startswith(OPENROUTER_KEY_PREFIX)
