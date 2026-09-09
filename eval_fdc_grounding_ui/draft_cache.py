"""Mass-aware dequant-cache lookups for LLM draft ingredients (grams-first)."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from amount_kind import COUNT_UNITS, AmountKind, normalize_count_unit  # noqa: E402
from build_dequant_norm_cache import (  # noqa: E402
    load_dequant_norm_cache,
    resolve_dequant_cache_path,
)
from dequant_volume_anchor import (  # noqa: E402
    is_volume_stem_cache_key,
    lookup_dequant_cache_entry,
    volume_stem,
    volume_stem_cache_key,
)
from unit_convert import UnitConversionError, unit_kind  # noqa: E402

AmountKindOrMass = AmountKind
DEFAULT_MASS_MIN_SCORE = 0.55


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_text(value: str | None) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: str) -> set[str]:
    stop = {
        "and",
        "the",
        "with",
        "from",
        "for",
        "raw",
        "fresh",
        "or",
        "of",
        "a",
        "an",
        "to",
        "in",
    }
    return {t for t in _norm_text(text).split() if len(t) > 2 and t not in stop}


# Extra tokens allowed when a shorter stem is contained in a longer query (or vice versa).
# Identity-changing words (noodle, flour, sauce, …) are intentionally excluded.
_FORM_MODIFIER_TOKENS = {
    "raw",
    "fresh",
    "frozen",
    "cooked",
    "dried",
    "dry",
    "diced",
    "minced",
    "chopped",
    "sliced",
    "boneless",
    "skinless",
    "whole",
    "large",
    "small",
    "medium",
    "organic",
    "plain",
    "unsalted",
    "salted",
    "ground",
    "grated",
    "peeled",
    "trimmed",
    "lean",
    "fat",
    "free",
    "low",
    "reduced",
    "extra",
    "virgin",
    "fine",
    "coarse",
    "white",
    "brown",
    "yellow",
    "green",
    "red",
    "black",
    "long",
    "grain",
    "short",
    "medium",
    "flat",  # shape alone is weak; paired with noodle still blocked by "noodle"
}


def _containment_is_safe(query: str, stem: str) -> bool:
    """Allow phrase containment only when leftovers look like form modifiers."""
    q = _norm_text(query)
    s = _norm_text(stem)
    if not q or not s or q == s:
        return True
    if s not in q and q not in s:
        return False
    longer, shorter = (q, s) if len(q) >= len(s) else (s, q)
    # Require shorter to appear as a contiguous token phrase in longer.
    long_toks = longer.split()
    short_toks = shorter.split()
    if not short_toks:
        return False
    joined = False
    for i in range(0, len(long_toks) - len(short_toks) + 1):
        if long_toks[i : i + len(short_toks)] == short_toks:
            joined = True
            break
    if not joined:
        return False
    leftover = _content_tokens(longer) - _content_tokens(shorter)
    return leftover.issubset(_FORM_MODIFIER_TOKENS)


def _token_score(query: str, stem: str) -> float:
    q = _norm_text(query)
    s = _norm_text(stem)
    if not q or not s:
        return 0.0
    if q == s:
        return 1.0
    if (q in s or s in q) and _containment_is_safe(q, s):
        return 0.9
    tq = _content_tokens(q)
    ts = _content_tokens(s)
    if not tq or not ts:
        return 0.0
    return len(tq & ts) / len(tq | ts)


def strip_leading_unit(cache_key: str) -> str | None:
    """Return food stem with leading mass/volume/count unit removed."""
    key = str(cache_key or "").strip().lower()
    if not key:
        return None
    if is_volume_stem_cache_key(key):
        stem = key[len("vol:") :].strip()
        return stem or None
    tokens = key.split()
    if not tokens:
        return None
    head = tokens[0]
    try:
        kind = unit_kind(head)
        if kind in {"mass", "volume"} and len(tokens) >= 2:
            return " ".join(tokens[1:]).strip() or None
    except UnitConversionError:
        pass
    count = normalize_count_unit(head)
    if count and count in COUNT_UNITS and len(tokens) >= 2:
        return " ".join(tokens[1:]).strip() or None
    return key


def classify_draft_ingredient_kind(
    name: str,
    *,
    grams: float | None = None,
    unit: str | None = None,
) -> AmountKind:
    """Classify draft ingredient text for cache lookup strategy."""
    if unit:
        try:
            return unit_kind(str(unit))  # type: ignore[return-value]
        except UnitConversionError:
            count = normalize_count_unit(unit)
            if count in COUNT_UNITS:
                return "count"
    tokens = _norm_text(name).split()
    if tokens:
        head = tokens[0]
        try:
            kind = unit_kind(head)
            if kind in {"mass", "volume"}:
                return kind  # type: ignore[return-value]
        except UnitConversionError:
            pass
        count = normalize_count_unit(head)
        if count in COUNT_UNITS:
            return "count"
    if grams is not None:
        try:
            if float(grams) > 0:
                return "mass"
        except (TypeError, ValueError):
            pass
    return "unknown"


def default_cache_path() -> Path:
    return Path(resolve_dequant_cache_path(None))


@dataclass
class DraftCacheHit:
    query_name: str
    amount_kind: AmountKind
    match_mode: Literal["exact", "volume_stem", "mass_stem"]
    cache_key: str
    stem: str | None
    score: float
    fdc_id: int
    description: str | None
    portion_id: int | None
    entry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_name": self.query_name,
            "amount_kind": self.amount_kind,
            "match_mode": self.match_mode,
            "cache_key": self.cache_key,
            "stem": self.stem,
            "score": self.score,
            "fdc_id": self.fdc_id,
            "description": self.description,
            "portion_id": self.portion_id,
        }


_SHARED_DEQUANT: "DraftDequantCache | None" = None


def get_shared_dequant_cache(cache_path: Path | str | None = None) -> "DraftDequantCache":
    """Process-wide dequant cache (load JSON once; reuse across grounding calls)."""
    global _SHARED_DEQUANT
    if _SHARED_DEQUANT is None:
        _SHARED_DEQUANT = DraftDequantCache(cache_path)
    return _SHARED_DEQUANT


def warm_dequant_cache(cache_path: Path | str | None = None) -> dict[str, Any]:
    """Eager-load dequant entries for server startup."""
    cache = get_shared_dequant_cache(cache_path)
    return {
        "ok": True,
        "n_entries": len(cache.entries),
        "path": str(cache.cache_path),
    }


class DraftDequantCache:
    """Lookup + write helpers for draft-ingredient cache hits."""

    def __init__(self, cache_path: Path | str | None = None):
        self.cache_path = Path(cache_path) if cache_path else default_cache_path()
        self.entries: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        self._stem_index: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.reload()

    def reload(self) -> None:
        if self.cache_path.is_file():
            self.entries, self.meta = load_dequant_norm_cache(self.cache_path)
        else:
            self.entries, self.meta = {}, {}
        self._rebuild_stem_index()

    def _rebuild_stem_index(self) -> None:
        index: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for key, entry in self.entries.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("resolution_class") in {"unknowable", "unmeasurable"}:
                continue
            fdc = entry.get("llm_fdc_id")
            if fdc is None:
                continue
            stem = strip_leading_unit(str(key))
            if not stem:
                continue
            index.setdefault(stem, []).append((str(key), entry))
        self._stem_index = index

    def lookup(
        self,
        name: str,
        *,
        grams: float | None = None,
        unit: str | None = None,
        min_score: float = DEFAULT_MASS_MIN_SCORE,
    ) -> DraftCacheHit | None:
        kind = classify_draft_ingredient_kind(name, grams=grams, unit=unit)
        if kind == "mass":
            return self._lookup_mass_stem(name, kind=kind, min_score=min_score)
        # Non-mass: exact / volume-stem dequant lookup as usual.
        norm = _norm_text(name)
        if not norm:
            return None
        entry = lookup_dequant_cache_entry(self.entries, norm)
        if entry is None and volume_stem(norm):
            entry = lookup_dequant_cache_entry(self.entries, norm)
        if entry is None:
            # Also try name with leading unit preserved if present
            return None
        fdc = entry.get("llm_fdc_id")
        if fdc is None:
            return None
        portion = entry.get("portion_id")
        if portion is None:
            portion = entry.get("matched_portion_id")
        mode: Literal["exact", "volume_stem", "mass_stem"] = (
            "volume_stem" if volume_stem(norm) and norm not in self.entries else "exact"
        )
        return DraftCacheHit(
            query_name=name,
            amount_kind=kind,
            match_mode=mode,
            cache_key=norm if norm in self.entries else volume_stem_cache_key(volume_stem(norm) or ""),
            stem=volume_stem(norm) or norm,
            score=1.0,
            fdc_id=int(fdc),
            description=entry.get("llm_description"),
            portion_id=int(portion) if portion is not None else None,
            entry=entry,
        )

    def _lookup_mass_stem(
        self,
        name: str,
        *,
        kind: AmountKind,
        min_score: float,
    ) -> DraftCacheHit | None:
        query = _norm_text(name)
        if not query:
            return None
        # Prefer exact stem / exact key first
        if query in self.entries:
            entry = self.entries[query]
            fdc = entry.get("llm_fdc_id")
            if fdc is not None and entry.get("resolution_class") not in {
                "unknowable",
                "unmeasurable",
            }:
                portion = entry.get("portion_id") or entry.get("matched_portion_id")
                return DraftCacheHit(
                    query_name=name,
                    amount_kind=kind,
                    match_mode="exact",
                    cache_key=query,
                    stem=query,
                    score=1.0,
                    fdc_id=int(fdc),
                    description=entry.get("llm_description"),
                    portion_id=int(portion) if portion is not None else None,
                    entry=entry,
                )

        best: DraftCacheHit | None = None
        for stem, items in self._stem_index.items():
            score = _token_score(query, stem)
            if score < min_score:
                continue
            for cache_key, entry in items:
                fdc = entry.get("llm_fdc_id")
                if fdc is None:
                    continue
                portion = entry.get("portion_id") or entry.get("matched_portion_id")
                hit = DraftCacheHit(
                    query_name=name,
                    amount_kind=kind,
                    match_mode="mass_stem",
                    cache_key=cache_key,
                    stem=stem,
                    score=float(score),
                    fdc_id=int(fdc),
                    description=entry.get("llm_description"),
                    portion_id=int(portion) if portion is not None else None,
                    entry=entry,
                )
                if best is None or hit.score > best.score:
                    best = hit
                elif best is not None and hit.score == best.score:
                    # Prefer entries that share more exact stem equality
                    if stem == query and best.stem != query:
                        best = hit
        return best

    def upsert_mass_fdc(
        self,
        *,
        name: str,
        fdc_id: int,
        description: str | None = None,
        portion_id: int | None = None,
        curator_note: str | None = None,
        example_ingredient: str | None = None,
        source: str = "eval_fdc_grounding_ui",
    ) -> dict[str, Any]:
        """Write / update a mass-style cache entry (FDC required; portion optional)."""
        key = _norm_text(name)
        if not key:
            raise ValueError("empty cache key")
        entry: dict[str, Any] = {
            "resolution_class": "resolved",
            "llm_fdc_id": int(fdc_id),
            "llm_description": description,
            "llm_certainty": 1.0,
            "llm_negligible_calories": False,
            "portion_id": int(portion_id) if portion_id is not None else None,
            "matched_portion_id": int(portion_id) if portion_id is not None else None,
            "n_sample_lines": int((self.entries.get(key) or {}).get("n_sample_lines") or 1),
            "example_ingredient": example_ingredient or name,
            "curator_source": source,
            "curator_mass_fdc_only": portion_id is None,
            "updated_at": _utc_now(),
        }
        if curator_note:
            entry["curator_note"] = curator_note
        self.entries[key] = entry
        self._persist()
        self._rebuild_stem_index()
        return {"cache_key": key, "entry": entry, "path": str(self.cache_path)}

    def upsert_non_mass(
        self,
        *,
        dequant_norm: str,
        fdc_id: int,
        description: str | None = None,
        portion_id: int | None = None,
        curator_note: str | None = None,
        example_ingredient: str | None = None,
        source: str = "eval_fdc_grounding_ui",
    ) -> dict[str, Any]:
        """Write a standard dequant cache entry (portion recommended for volume/count)."""
        key = _norm_text(dequant_norm)
        if not key:
            raise ValueError("empty cache key")
        if portion_id is None:
            raise ValueError("portion_id required for non-mass cache writes")
        entry: dict[str, Any] = {
            "resolution_class": "resolved",
            "llm_fdc_id": int(fdc_id),
            "llm_description": description,
            "llm_certainty": 1.0,
            "llm_negligible_calories": False,
            "portion_id": int(portion_id),
            "matched_portion_id": int(portion_id),
            "n_sample_lines": int((self.entries.get(key) or {}).get("n_sample_lines") or 1),
            "example_ingredient": example_ingredient or dequant_norm,
            "curator_source": source,
            "updated_at": _utc_now(),
        }
        stem = volume_stem(key)
        if stem:
            entry["volume_portion_anchor"] = True
            entry["volume_stem"] = stem
            # Also pin stem anchor for cross-unit fan-in
            self.entries[volume_stem_cache_key(stem)] = {
                **entry,
                "volume_portion_anchor": True,
                "volume_stem": stem,
            }
        if curator_note:
            entry["curator_note"] = curator_note
        self.entries[key] = entry
        self._persist()
        self._rebuild_stem_index()
        return {"cache_key": key, "entry": entry, "path": str(self.cache_path)}

    def _persist(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta or {})
        meta["updated_at"] = _utc_now()
        meta["n_entries"] = len(self.entries)
        meta.setdefault("source", "eval_fdc_grounding_ui")
        payload = {"meta": meta, "entries": self.entries}
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.cache_path)
