"""OpenAI clients with round-robin API keys and sticky failover on rate limits."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any

from db import load_dotenv
from llm_throttle import throttle_llm_async, throttle_llm_sync

RETRIES_PER_KEY = 3
BACKOFF_BASE_SEC = 0.5
BACKOFF_MAX_SEC = 8.0

OPENAI_PARTIAL_STATUS = "partial_openai_keys_exhausted"

_state_lock = threading.Lock()
_call_counter = 0
_blocked_key_indices: set[int] = set()


class AllKeysExhaustedError(RuntimeError):
    """All configured OpenAI API keys are rate-limited or unavailable."""

    def __init__(
        self,
        *,
        blocked_indices: set[int],
        n_keys: int,
        last_exc: Exception | None = None,
    ) -> None:
        self.blocked_indices = blocked_indices
        self.n_keys = n_keys
        self.last_exc = last_exc
        blocked = ", ".join(str(i) for i in sorted(blocked_indices)) or "none"
        detail = f" ({last_exc})" if last_exc else ""
        super().__init__(
            f"All {n_keys} OpenAI API key(s) exhausted (blocked: {blocked}){detail}"
        )


def reset_openai_key_state() -> None:
    """Clear round-robin / failover state (for tests)."""
    global _call_counter
    with _state_lock:
        _call_counter = 0
        _blocked_key_indices.clear()


def get_openai_api_keys() -> list[str]:
    """Return distinct OpenAI API keys from env (primary then fallbacks)."""
    load_dotenv()
    keys: list[str] = []
    for name in ("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3"):
        value = os.environ.get(name, "").strip()
        if value and value not in keys:
            keys.append(value)
    if not keys:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY "
            "(and optionally OPENAI_API_KEY_2, OPENAI_API_KEY_3) in .env"
        )
    return keys


def get_foodon_openai_api_key() -> str:
    """Dedicated OpenAI key for FoodOn batch runs (separate org quota)."""
    load_dotenv()
    for name in ("OPENAI_API_KEY_FOODON", "OPENAI_API_KEY_3"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "No FoodOn OpenAI key configured. Set OPENAI_API_KEY_FOODON or OPENAI_API_KEY_3 in .env"
    )


BATCH_KEY_ENV_NAMES: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_API_KEY_2",
    "OPENAI_API_KEY_3",
    "OPENAI_API_KEY_FOODON",
)


def load_named_openai_keys() -> dict[str, str]:
    """Return env-name -> API key for all configured batch keys."""
    load_dotenv()
    out: dict[str, str] = {}
    for name in BATCH_KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            out[name] = value
    return out


def batch_api_key_chain(*prefer_env_names: str) -> list[tuple[str, str]]:
    """Ordered (env_name, key) pairs for batch submit/poll, preferring listed env vars first."""
    named = load_named_openai_keys()
    chain: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in prefer_env_names:
        value = named.get(name)
        if value and value not in seen:
            chain.append((name, value))
            seen.add(value)
    for name in BATCH_KEY_ENV_NAMES:
        value = named.get(name)
        if value and value not in seen:
            chain.append((name, value))
            seen.add(value)
    if not chain:
        raise RuntimeError(
            "No OpenAI API key configured. Set OPENAI_API_KEY "
            "(and optionally OPENAI_API_KEY_2, OPENAI_API_KEY_3, OPENAI_API_KEY_FOODON) in .env"
        )
    return chain


def foodon_batch_api_key_chain() -> list[tuple[str, str]]:
    """FoodOn batches prefer the dedicated org, then fall back to other keys."""
    return batch_api_key_chain("OPENAI_API_KEY_FOODON", "OPENAI_API_KEY_3")


def v7_batch_api_key_chain() -> list[tuple[str, str]]:
    """V7 batches prefer the primary org, then fall back to other keys."""
    return batch_api_key_chain("OPENAI_API_KEY", "OPENAI_API_KEY_2", "OPENAI_API_KEY_3")


def api_key_for_batch_record(
    batch_rec: dict[str, Any],
    chain: list[tuple[str, str]],
) -> str:
    """Resolve the API key that owns a batch (for poll/ingest/download)."""
    from openai_batch_client import retrieve_batch

    named = load_named_openai_keys()
    env = batch_rec.get("openai_key_env")
    if env and env in named:
        return named[env]

    bid = str(batch_rec.get("id") or "").strip()
    if not bid:
        raise RuntimeError("batch record has no id")

    last_exc: Exception | None = None
    for env_name, key in chain:
        try:
            retrieve_batch(bid, api_key=key)
            if env_name != "explicit":
                batch_rec["openai_key_env"] = env_name
            return key
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Batch {bid} not found on any configured OpenAI org") from last_exc


def get_key_pool_status() -> dict[str, Any]:
    """Snapshot of round-robin / blocked-key state for run manifests."""
    with _state_lock:
        blocked = sorted(_blocked_key_indices)
        return {
            "n_keys": len(get_openai_api_keys()),
            "blocked_indices": blocked,
            "call_counter": _call_counter,
            "all_exhausted": len(blocked) >= len(get_openai_api_keys()),
        }


def _is_daily_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "requests per day" in msg or "rpd" in msg


def _is_rate_limit(exc: Exception) -> bool:
    return "rate_limit" in str(exc).lower() or type(exc).__name__ == "RateLimitError"


def _should_switch_key(exc: Exception) -> bool:
    if _is_daily_limit(exc):
        return True
    if _is_rate_limit(exc):
        return True
    msg = str(exc).lower()
    if "insufficient_quota" in msg or "billing" in msg:
        return True
    status = getattr(exc, "status_code", None)
    if status in (401, 403, 429):
        return True
    return False


def _is_retryable_same_key(exc: Exception) -> bool:
    if _should_switch_key(exc):
        return not _is_daily_limit(exc)
    msg = str(exc).lower()
    if any(s in msg for s in ("timeout", "timed out", "connection", "server error", "503", "502")):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and int(status) >= 500


def _available_key_indices(n_keys: int) -> list[int]:
    with _state_lock:
        blocked = set(_blocked_key_indices)
    return [i for i in range(n_keys) if i not in blocked]


def count_available_openai_keys() -> int:
    """Number of API keys not marked exhausted for this process."""
    return len(_available_key_indices(len(get_openai_api_keys())))


def _pick_key_index(n_keys: int) -> int | None:
    """Round-robin among non-blocked keys; None when all keys are blocked."""
    global _call_counter
    with _state_lock:
        available = [i for i in range(n_keys) if i not in _blocked_key_indices]
        if not available:
            return None
        if len(available) == 1:
            return available[0]
        for _ in range(n_keys):
            candidate = _call_counter % n_keys
            _call_counter += 1
            if candidate in _blocked_key_indices:
                continue
            return candidate
        return available[0]


def _mark_key_exhausted(failed_idx: int, n_keys: int) -> None:
    """Block a rate-limited key; remaining traffic uses other key(s) only."""
    with _state_lock:
        _blocked_key_indices.add(failed_idx)


def _backoff_sleep(attempt: int) -> None:
    time.sleep(min(BACKOFF_BASE_SEC * (2**attempt), BACKOFF_MAX_SEC))


async def _backoff_sleep_async(attempt: int) -> None:
    await asyncio.sleep(min(BACKOFF_BASE_SEC * (2**attempt), BACKOFF_MAX_SEC))


def _raise_if_all_blocked(n_keys: int, last_exc: Exception | None) -> None:
    if _available_key_indices(n_keys):
        return
    with _state_lock:
        blocked = set(_blocked_key_indices)
    raise AllKeysExhaustedError(
        blocked_indices=blocked,
        n_keys=n_keys,
        last_exc=last_exc,
    )


def _execute_with_fallback(
    call_fn: Any,
    *,
    throttle_fn: Any,
) -> Any:
    from openai import OpenAI

    keys = get_openai_api_keys()
    last_exc: Exception | None = None

    while True:
        key_idx = _pick_key_index(len(keys))
        if key_idx is None:
            _raise_if_all_blocked(len(keys), last_exc)

        client = OpenAI(api_key=keys[key_idx])
        switched = False
        for attempt in range(RETRIES_PER_KEY):
            try:
                throttle_fn(key_idx)
                return call_fn(client)
            except Exception as exc:
                last_exc = exc
                if _should_switch_key(exc):
                    _mark_key_exhausted(key_idx, len(keys))
                    switched = True
                    break
                if _is_retryable_same_key(exc) and attempt < RETRIES_PER_KEY - 1:
                    _backoff_sleep(attempt)
                    continue
                raise
        if switched:
            continue


async def _execute_with_fallback_async(call_fn: Any) -> Any:
    from openai import AsyncOpenAI

    keys = get_openai_api_keys()
    last_exc: Exception | None = None

    while True:
        key_idx = _pick_key_index(len(keys))
        if key_idx is None:
            _raise_if_all_blocked(len(keys), last_exc)

        client = AsyncOpenAI(api_key=keys[key_idx])
        switched = False
        for attempt in range(RETRIES_PER_KEY):
            try:
                await throttle_llm_async(key_idx)
                return await call_fn(client)
            except Exception as exc:
                last_exc = exc
                if _should_switch_key(exc):
                    _mark_key_exhausted(key_idx, len(keys))
                    switched = True
                    break
                if _is_retryable_same_key(exc) and attempt < RETRIES_PER_KEY - 1:
                    await _backoff_sleep_async(attempt)
                    continue
                raise
        if switched:
            continue


class _FallbackAsyncCompletions:
    async def create(self, **kwargs: Any) -> Any:
        async def _call(client: Any) -> Any:
            return await client.chat.completions.create(**kwargs)

        return await _execute_with_fallback_async(_call)


class _FallbackAsyncChat:
    def __init__(self) -> None:
        self.completions = _FallbackAsyncCompletions()


class AsyncOpenAIFallback:
    """Drop-in async client: `await client.chat.completions.create(...)`."""

    def __init__(self) -> None:
        self.chat = _FallbackAsyncChat()


class _FallbackSyncCompletions:
    def create(self, **kwargs: Any) -> Any:
        def _call(client: Any) -> Any:
            return client.chat.completions.create(**kwargs)

        return _execute_with_fallback(_call, throttle_fn=throttle_llm_sync)


class _FallbackSyncChat:
    def __init__(self) -> None:
        self.completions = _FallbackSyncCompletions()


class SyncOpenAIFallback:
    """Drop-in sync client: `client.chat.completions.create(...)`."""

    def __init__(self) -> None:
        self.chat = _FallbackSyncChat()


def get_async_openai_client() -> AsyncOpenAIFallback:
    return AsyncOpenAIFallback()


def get_sync_openai_client() -> SyncOpenAIFallback:
    return SyncOpenAIFallback()
