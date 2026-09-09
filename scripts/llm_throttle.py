"""Per-key spacing between OpenAI calls to stay under 500 RPM per API key."""

from __future__ import annotations

import asyncio
import threading
import time
import weakref

# 500 requests per 60 seconds per key → minimum gap between calls on that key
LLM_CALL_INTERVAL_SEC = 60 / 500
MAX_KEY_SLOTS = 8


def effective_llm_call_interval_sec() -> float:
    """Per-key spacing; halved when two or more API keys are available (round-robin)."""
    try:
        from openai_fallback import count_available_openai_keys, get_openai_api_keys

        n_keys = len(get_openai_api_keys())
        available = count_available_openai_keys()
    except Exception:
        return LLM_CALL_INTERVAL_SEC
    if n_keys >= 2 and available >= 2:
        return LLM_CALL_INTERVAL_SEC / 2
    return LLM_CALL_INTERVAL_SEC

# Per-event-loop lock (module-level asyncio.Lock breaks across asyncio.run() calls)
_loop_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    weakref.WeakKeyDictionary()
)
_async_last: dict[int, float] = {}
_async_last_lock = threading.Lock()

_sync_lock = threading.Lock()
_sync_last: dict[int, float] = {}


def _async_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _loop_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _loop_locks[loop] = lock
    return lock


async def throttle_llm_async(key_index: int = 0) -> None:
    """Await until at least effective_llm_call_interval_sec() since the previous call on this key."""
    interval = effective_llm_call_interval_sec()
    slot = int(key_index) % MAX_KEY_SLOTS
    async with _async_lock():
        with _async_last_lock:
            now = time.monotonic()
            last = _async_last.get(slot, 0.0)
            wait = interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            _async_last[slot] = time.monotonic()


def throttle_llm_sync(key_index: int = 0) -> None:
    """Block until at least effective_llm_call_interval_sec() since the previous call on this key."""
    interval = effective_llm_call_interval_sec()
    slot = int(key_index) % MAX_KEY_SLOTS
    with _sync_lock:
        now = time.monotonic()
        last = _sync_last.get(slot, 0.0)
        wait = interval - (now - last)
        if wait > 0:
            time.sleep(wait)
        _sync_last[slot] = time.monotonic()
