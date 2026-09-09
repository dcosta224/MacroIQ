"""Eager-load FoodOn / MiniLM / dequant so the first /api/run is not cold."""

from __future__ import annotations

import resource
import sys
from typing import Any

_WARM_STATUS: dict[str, Any] = {"warmed": False}


def _rss_mb() -> float:
    """Resident set size in MiB (Linux: ru_maxrss is KiB; macOS: bytes)."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def warm_status() -> dict[str, Any]:
    return dict(_WARM_STATUS)


def warm_runtime_caches() -> dict[str, Any]:
    """Load process-wide caches. Safe to call multiple times."""
    global _WARM_STATUS
    status: dict[str, Any] = {
        "warmed": False,
        "rss_mb_before": round(_rss_mb(), 1),
        "foodon": None,
        "embedding": None,
        "dequant": None,
        "rss_mb_after": None,
        "errors": [],
    }
    try:
        from canonical_optimization import _get_hierarchy, _get_index

        _get_index()
        _get_hierarchy()
        status["foodon"] = {"ok": True}
    except Exception as exc:
        status["foodon"] = {"ok": False, "error": str(exc)}
        status["errors"].append(f"foodon: {exc}")

    try:
        from recipe_opt_agent.embedding_model import warm_embedding_model

        status["embedding"] = warm_embedding_model()
        if not status["embedding"].get("ok"):
            status["errors"].append(f"embedding: {status['embedding'].get('error')}")
    except Exception as exc:
        status["embedding"] = {"ok": False, "error": str(exc)}
        status["errors"].append(f"embedding: {exc}")

    try:
        from eval_fdc_grounding_ui.draft_cache import warm_dequant_cache

        status["dequant"] = warm_dequant_cache()
    except Exception as exc:
        status["dequant"] = {"ok": False, "error": str(exc)}
        status["errors"].append(f"dequant: {exc}")

    status["rss_mb_after"] = round(_rss_mb(), 1)
    status["warmed"] = not status["errors"]
    _WARM_STATUS = status
    return status
