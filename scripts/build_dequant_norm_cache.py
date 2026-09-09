"""Build dequant_norm → LLM FDC + modal portion cache from pipeline_matches.

Cache entries are eligible when, for a normalized dequant string:
  - n_lines >= min_count (default 10)
  - modal share of llm_fdc_id is 100%

Each entry pins the modal FDC and modal portion_id (among same-FDC lines). Gram
resolution at runtime scales quantity against that portion; lines that previously
picked a non-modal portion are still cache hits.

Example:
  uv run python scripts/build_dequant_norm_cache.py \\
    --matches scratch/EDA/portion_feasibility_1000_v6/pipeline_matches.parquet \\
    --out scratch/EDA/portion_feasibility_1000_v6/dequant_norm_llm_cache.json
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from ingredient_query_cache import RECIPE_SEMANTIC_EMBEDDING_VERSION, dequantified_text
from ingredient_match_staged import normalize_text
from dequant_volume_anchor import lookup_dequant_cache_entry
from resolved_recipe_portion import SENTINEL_FDC_ID, WATER_SENTINEL_FDC_ID, extract_portion_id

SENTINEL_IDS = {SENTINEL_FDC_ID, WATER_SENTINEL_FDC_ID}


def norm_field(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return normalize_text(str(value))


def prepare_dequant_norm(pm: pd.DataFrame) -> pd.DataFrame:
    out = pm.copy()
    # Always recompute so cache keys match live dequant normalization (e.g. unit aliases).
    out["dequantified"] = out.apply(
        lambda r: dequantified_text(r.to_dict(), raw=str(r["ingredient"])),
        axis=1,
    )
    out["dequant_norm"] = out["dequantified"].fillna(out["ingredient"]).map(norm_field)
    out["portion_id"] = out.apply(extract_portion_id, axis=1)
    out["resolution_sig"] = (
        out["llm_fdc_id"].astype("Int64").astype(str)
        + "|"
        + out["grams_status"].fillna("").astype(str)
        + "|"
        + out["grams_method"].fillna("").astype(str)
    )
    return out


def _modal_portion_stats(same_fdc: pd.DataFrame) -> tuple[int | None, float | None, int]:
    """Return (modal_portion_id, modal_portion_share, n_distinct_portion_id)."""
    with_portion = same_fdc[same_fdc["portion_id"].notna()]
    if with_portion.empty:
        return None, None, 0
    portion_counts = with_portion["portion_id"].value_counts()
    modal_portion_id = int(portion_counts.index[0])
    modal_portion_share = float(portion_counts.iloc[0] / len(with_portion))
    n_distinct = int(with_portion["portion_id"].nunique())
    return modal_portion_id, modal_portion_share, n_distinct


def summarize_dequant_norms(pm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dequant_norm, g in pm.groupby("dequant_norm", dropna=False):
        with_fdc = g[g["llm_fdc_id"].notna()]
        n_lines = len(g)
        n_with_fdc = len(with_fdc)
        if n_with_fdc == 0:
            continue
        fdc_counts = with_fdc["llm_fdc_id"].value_counts()
        sig_counts = with_fdc["resolution_sig"].value_counts()
        modal_fdc = int(fdc_counts.index[0])
        same_fdc = with_fdc[with_fdc["llm_fdc_id"] == modal_fdc]
        modal_portion_id, modal_portion_share, n_distinct_portion = _modal_portion_stats(same_fdc)
        rows.append(
            {
                "dequant_norm": dequant_norm,
                "n_lines": n_lines,
                "n_with_fdc": n_with_fdc,
                "n_recipes": g["recipe_id"].nunique(),
                "modal_fdc_id": modal_fdc,
                "modal_fdc_share": float(fdc_counts.iloc[0] / n_with_fdc),
                "modal_portion_id": modal_portion_id,
                "modal_portion_share": modal_portion_share,
                "modal_resolution_share": float(sig_counts.iloc[0] / n_with_fdc),
                "n_distinct_fdc": int(with_fdc["llm_fdc_id"].nunique()),
                "n_distinct_portion_id": n_distinct_portion,
                "n_distinct_resolution_sig": int(with_fdc["resolution_sig"].nunique()),
                "eligible": (
                    n_lines >= 0  # filled by caller filter
                    and fdc_counts.iloc[0] / n_with_fdc == 1.0
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["n_lines", "dequant_norm"], ascending=[False, True])


def _pick_modal_row(same_fdc: pd.DataFrame, modal_portion_id: int | None) -> pd.Series:
    if modal_portion_id is not None:
        modal_rows = same_fdc[same_fdc["portion_id"] == modal_portion_id]
        if not modal_rows.empty:
            return modal_rows.iloc[0]
    return same_fdc.iloc[0]


def build_cache_entries(
    pm: pd.DataFrame,
    eligible_norms: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sub = pm[pm["dequant_norm"].isin(eligible_norms)]
    for dequant_norm, g in sub.groupby("dequant_norm", sort=True):
        with_fdc = g[g["llm_fdc_id"].notna()]
        modal_fdc = int(with_fdc["llm_fdc_id"].mode().iloc[0])
        same_fdc = with_fdc[with_fdc["llm_fdc_id"] == modal_fdc]
        modal_portion_id, modal_portion_share, n_distinct_portion = _modal_portion_stats(same_fdc)
        modal = _pick_modal_row(same_fdc, modal_portion_id)
        matched_portion_id = modal_portion_id if modal_portion_id is not None else modal.get("matched_portion_id")
        rows.append(
            {
                "dequant_norm": dequant_norm,
                "n_sample_lines": int(len(g)),
                "n_sample_recipes": int(g["recipe_id"].nunique()),
                "llm_fdc_id": modal_fdc,
                "llm_description": modal.get("llm_description"),
                "llm_certainty": modal.get("llm_certainty"),
                "llm_negligible_calories": bool(modal.get("llm_negligible_calories", False)),
                "llm_rationale": modal.get("llm_rationale"),
                "matched_portion_id": matched_portion_id,
                "modal_portion_id": modal_portion_id,
                "modal_portion_share": modal_portion_share,
                "n_distinct_portion_id": n_distinct_portion,
                "grams_status": modal.get("grams_status"),
                "grams_method": modal.get("grams_method"),
                "example_ingredient": g["ingredient"].iloc[0],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dequant_norm",
                "n_sample_lines",
                "n_sample_recipes",
                "llm_fdc_id",
                "llm_description",
                "llm_certainty",
                "llm_negligible_calories",
                "llm_rationale",
                "matched_portion_id",
                "modal_portion_id",
                "modal_portion_share",
                "n_distinct_portion_id",
                "grams_status",
                "grams_method",
                "example_ingredient",
            ]
        )
    return pd.DataFrame(rows).sort_values("n_sample_lines", ascending=False)


def build_dequant_norm_cache(
    pm: pd.DataFrame,
    *,
    min_count: int = 10,
    min_modal_share: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pm = prepare_dequant_norm(pm)
    stats = summarize_dequant_norms(pm)
    stats["eligible"] = (
        (stats["n_lines"] >= min_count)
        & (stats["modal_fdc_share"] >= min_modal_share)
        & (~stats["modal_fdc_id"].isin(SENTINEL_IDS))
    )
    eligible_norms = set(stats.loc[stats["eligible"], "dequant_norm"])
    cache_df = build_cache_entries(pm, eligible_norms)

    hit_mask = pm["dequant_norm"].isin(eligible_norms)
    meta = {
        "dequant_embedding_version": RECIPE_SEMANTIC_EMBEDDING_VERSION,
        "criteria": {
            "min_count": min_count,
            "min_modal_fdc_share": min_modal_share,
            "key": "dequant_norm",
            "pinned_fields": "modal llm_fdc_id + modal portion_id (among same-FDC lines)",
        },
        "n_source_lines": int(len(pm)),
        "n_source_recipes": int(pm["recipe_id"].nunique()),
        "n_cache_terms": int(len(cache_df)),
        "n_cache_hits": int(hit_mask.sum()),
        "cache_hit_rate": round(float(hit_mask.mean()), 6),
    }
    return stats, cache_df, meta


def resolve_dequant_cache_path(explicit: Path | str | None) -> Path | None:
    """Resolve cache JSON path: explicit arg, then standard locations."""
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    root = Path(__file__).resolve().parents[1]
    for candidate in (
        root / "data" / "dequant_norm_llm_cache.json",
        root / "scratch" / "EDA" / "portion_feasibility_1000_v6" / "dequant_norm_llm_cache.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def load_dequant_norm_cache(path: Path | str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load runtime cache JSON; returns (entries, meta)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries") or {}
    meta = payload.get("meta") or {}
    return entries, meta


def dequant_norm_from_payload(payload: dict[str, Any]) -> str:
    """Normalized dequant key for a judge payload (must match cache build)."""
    raw = payload.get("dequantified")
    if not raw:
        row = payload.get("parsed_row") or payload
        raw = dequantified_text(row, raw=str(payload.get("ingredient") or ""))
    return norm_field(raw)


def build_cached_judge_from_entry(entry: dict[str, Any], *, dequant_norm: str) -> dict[str, Any]:
    """Synthetic judge row from a cache entry — skips the LLM."""
    resolution_class = entry.get("resolution_class")
    if resolution_class in ("unknowable", "unmeasurable"):
        return {
            "fdc_id": None,
            "certainty": 0.0,
            "rationale": f"dequant_norm cache ({dequant_norm!r}): {resolution_class}",
            "matched_portion_id": None,
            "negligible_calories": False,
            "response": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": None,
            "hardcoded": True,
            "dequant_cache": True,
            "dequant_norm": dequant_norm,
            "curator_resolution_class": resolution_class,
        }

    portion_id = entry.get("portion_id")
    if portion_id is None:
        portion_id = entry.get("matched_portion_id")
    portion_int = int(portion_id) if portion_id is not None else None
    certainty = entry.get("llm_certainty")
    fdc_raw = entry.get("llm_fdc_id")
    if fdc_raw is None or (isinstance(fdc_raw, float) and pd.isna(fdc_raw)):
        raise ValueError(f"cache entry for {dequant_norm!r} missing llm_fdc_id")
    judge = {
        "fdc_id": int(fdc_raw),
        "certainty": None if certainty is None else float(certainty),
        "rationale": (
            f"dequant_norm cache ({dequant_norm!r}): "
            f"{entry.get('llm_description') or entry.get('example_ingredient', '')}"
        ),
        "matched_portion_id": portion_int,
        "negligible_calories": bool(entry.get("llm_negligible_calories", False)),
        "response": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
        "hardcoded": True,
        "dequant_cache": True,
        "dequant_norm": dequant_norm,
        "curator_resolution_class": resolution_class or "resolved",
    }
    if entry.get("curator_scale_quantity") is not None and entry.get("curator_scale_unit"):
        judge["curator_scale_quantity"] = float(entry["curator_scale_quantity"])
        judge["curator_scale_unit"] = str(entry["curator_scale_unit"])
        scale_pid = entry.get("curator_scale_portion_id")
        if scale_pid is not None:
            judge["curator_scale_portion_id"] = int(scale_pid)
        scale_ref = entry.get("curator_scale_portion_ref_amount")
        if scale_ref is None:
            scale_ref = entry.get("portion_ref_amount")
        if scale_ref is not None:
            judge["curator_scale_portion_ref_amount"] = float(scale_ref)
        scale_ref_unit = entry.get("curator_scale_portion_ref_unit")
        if scale_ref_unit is None:
            scale_ref_unit = entry.get("portion_ref_unit")
        if scale_ref_unit:
            judge["curator_scale_portion_ref_unit"] = str(scale_ref_unit)
        if entry.get("curator_scale_label"):
            judge["curator_scale_label"] = str(entry["curator_scale_label"])
    if entry.get("curator_fixed_scale") or entry.get("curator_scale_quantity") is not None:
        judge["curator_fixed_scale"] = True
    if entry.get("curator_manual_volume"):
        judge["curator_manual_volume"] = True
    if entry.get("volume_portion_anchor") and not judge.get("curator_fixed_scale"):
        judge["volume_portion_anchor"] = True
    if entry.get("portion_ref_unit"):
        judge["portion_ref_unit"] = str(entry["portion_ref_unit"])
    if entry.get("portion_ref_amount") is not None:
        judge["portion_ref_amount"] = float(entry["portion_ref_amount"])
    if entry.get("portion_gram_weight") is not None:
        judge["portion_gram_weight"] = float(entry["portion_gram_weight"])
    return judge


DEFAULT_MIN_COUNT = 10


def entry_from_observations(
    dequant_norm: str,
    observations: list[dict[str, Any]],
    *,
    modal_fdc: int,
    modal_portion_id: int | None,
) -> dict[str, Any]:
    """Build a cache entry dict from runtime observations."""
    same_fdc = [o for o in observations if o.get("llm_fdc_id") == modal_fdc]
    portion_counts = Counter(
        o["portion_id"] for o in same_fdc if o.get("portion_id") is not None
    )
    if modal_portion_id is None and portion_counts:
        modal_portion_id = int(portion_counts.most_common(1)[0][0])
    modal_rows = [o for o in same_fdc if o.get("portion_id") == modal_portion_id]
    if not modal_rows and same_fdc:
        modal_rows = [same_fdc[0]]
    modal = modal_rows[0] if modal_rows else observations[0]
    return {
        "llm_fdc_id": int(modal_fdc),
        "llm_description": modal.get("llm_description"),
        "llm_certainty": modal.get("llm_certainty"),
        "llm_negligible_calories": bool(modal.get("llm_negligible_calories", False)),
        "portion_id": modal_portion_id,
        "matched_portion_id": modal_portion_id,
        "modal_portion_share": (
            float(portion_counts[modal_portion_id] / len(same_fdc))
            if modal_portion_id is not None and same_fdc and portion_counts
            else None
        ),
        "n_distinct_portion_id": len(portion_counts),
        "grams_status": modal.get("grams_status"),
        "grams_method": modal.get("grams_method"),
        "n_sample_lines": len(observations),
        "example_ingredient": modal.get("ingredient") or observations[0].get("ingredient"),
    }


class DequantNormCacheRuntime:
    """Mutable dequant_norm cache with live stats and same eligibility rules as offline build."""

    def __init__(
        self,
        entries: dict[str, dict[str, Any]],
        *,
        write_path: Path | None = None,
        min_count: int = DEFAULT_MIN_COUNT,
        meta: dict[str, Any] | None = None,
        initial_term_keys: set[str] | None = None,
    ) -> None:
        self.entries = dict(entries)
        self.write_path = Path(write_path) if write_path else None
        self.min_count = min_count
        self.meta = dict(meta or {})
        self._initial_terms = set(initial_term_keys or entries.keys())
        self._observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "initial_terms": len(self._initial_terms),
            "initial_hits": 0,
            "runtime_growth_hits": 0,
            "llm_calls": 0,
            "generic_hits": 0,
            "terms_added_during_run": 0,
            "promotions": [],
            "n_observations": 0,
        }

    def lookup(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        dequant_norm = dequant_norm_from_payload(payload)
        with self._lock:
            entry = lookup_dequant_cache_entry(self.entries, dequant_norm)
            if entry is None:
                return None
            if dequant_norm in self._initial_terms:
                self.stats["initial_hits"] += 1
            else:
                self.stats["runtime_growth_hits"] += 1
        return build_cached_judge_from_entry(entry, dequant_norm=dequant_norm)

    def record_completion(
        self,
        payload: dict[str, Any],
        judge: dict[str, Any],
        exp: dict[str, Any],
    ) -> str | None:
        """Record a completed line; promote to cache when eligible. Returns new dequant_norm if promoted."""
        if judge.get("generic_default_key"):
            self.stats["generic_hits"] += 1
            return None
        if not judge.get("dequant_cache") and not judge.get("hardcoded"):
            self.stats["llm_calls"] += 1

        dequant_norm = dequant_norm_from_payload(payload)
        fdc_id = exp.get("llm_fdc_id")
        if fdc_id is None:
            fdc_id = judge.get("fdc_id")
        portion_id = exp.get("matched_portion_id")
        if portion_id is None:
            portion_id = judge.get("matched_portion_id")

        obs = {
            "ingredient": payload.get("ingredient"),
            "llm_fdc_id": int(fdc_id) if fdc_id is not None and pd.notna(fdc_id) else None,
            "llm_description": exp.get("llm_description"),
            "llm_certainty": exp.get("llm_certainty"),
            "llm_negligible_calories": bool(
                exp.get("llm_negligible_calories") or judge.get("negligible_calories")
            ),
            "portion_id": int(portion_id) if portion_id is not None and pd.notna(portion_id) else None,
            "grams_status": exp.get("grams_status"),
            "grams_method": exp.get("grams_method"),
        }

        with self._lock:
            if dequant_norm in self.entries:
                self._observations[dequant_norm].append(obs)
                self.stats["n_observations"] += 1
                return None
            self._observations[dequant_norm].append(obs)
            self.stats["n_observations"] += 1
            return self._maybe_promote_locked(dequant_norm)

    def _maybe_promote_locked(self, dequant_norm: str) -> str | None:
        if dequant_norm in self.entries:
            return None
        obs = self._observations[dequant_norm]
        if len(obs) < self.min_count:
            return None
        with_fdc = [o for o in obs if o.get("llm_fdc_id") is not None]
        if len(with_fdc) < self.min_count:
            return None
        fdc_counts = Counter(o["llm_fdc_id"] for o in with_fdc)
        modal_fdc, modal_n = fdc_counts.most_common(1)[0]
        if modal_n / len(with_fdc) < 1.0:
            return None
        if int(modal_fdc) in SENTINEL_IDS:
            return None

        same_fdc = [o for o in with_fdc if o["llm_fdc_id"] == modal_fdc]
        portion_counts = Counter(o["portion_id"] for o in same_fdc if o.get("portion_id") is not None)
        modal_portion_id = int(portion_counts.most_common(1)[0][0]) if portion_counts else None

        entry = entry_from_observations(
            dequant_norm,
            obs,
            modal_fdc=int(modal_fdc),
            modal_portion_id=modal_portion_id,
        )
        self.entries[dequant_norm] = entry
        self.stats["terms_added_during_run"] += 1
        self.stats["promotions"].append(
            {
                "dequant_norm": dequant_norm,
                "n_lines": len(obs),
                "llm_fdc_id": int(modal_fdc),
                "portion_id": modal_portion_id,
                "after_n_observations": len(obs),
            }
        )
        return dequant_norm

    def _summary_locked(self) -> dict[str, Any]:
        """Build summary dict; caller must hold ``self._lock``."""
        total_hits = self.stats["initial_hits"] + self.stats["runtime_growth_hits"]
        return {
            **self.stats,
            "final_terms": len(self.entries),
            "total_cache_hits": total_hits,
            "terms_added_during_run": self.stats["terms_added_during_run"],
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_locked()

    def save(self) -> Path | None:
        if self.write_path is None:
            return None
        with self._lock:
            stats_snapshot = self._summary_locked()
            meta = {
                **self.meta,
                "dequant_embedding_version": RECIPE_SEMANTIC_EMBEDDING_VERSION,
                "criteria": {
                    "min_count": self.min_count,
                    "min_modal_fdc_share": 1.0,
                    "key": "dequant_norm",
                    "pinned_fields": "modal llm_fdc_id + modal portion_id (among same-FDC lines)",
                },
                "runtime_stats": stats_snapshot,
                "n_cache_terms": len(self.entries),
            }
            payload = {"meta": meta, "entries": dict(self.entries)}
            self.write_path.parent.mkdir(parents=True, exist_ok=True)
            self.write_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            stats_path = self.write_path.with_name("dequant_cache_runtime_stats.json")
            stats_path.write_text(json.dumps(stats_snapshot, indent=2), encoding="utf-8")
            return self.write_path


def create_dequant_cache_runtime(
    *,
    load_path: Path | None,
    write_path: Path,
    min_count: int = DEFAULT_MIN_COUNT,
) -> DequantNormCacheRuntime:
    entries: dict[str, dict[str, Any]] = {}
    meta: dict[str, Any] = {}
    initial_keys: set[str] = set()
    if load_path is not None and load_path.is_file():
        entries, meta = load_dequant_norm_cache(load_path)
        initial_keys = set(entries.keys())
    if write_path.is_file() and (load_path is None or write_path.resolve() != load_path.resolve()):
        run_entries, run_meta = load_dequant_norm_cache(write_path)
        entries.update(run_entries)
        meta = {**meta, **run_meta}
    return DequantNormCacheRuntime(
        entries,
        write_path=write_path,
        min_count=min_count,
        meta=meta,
        initial_term_keys=initial_keys,
    )


def apply_dequant_cache_to_payloads(
    payloads: list[dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    *,
    runtime: DequantNormCacheRuntime | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Attach hardcoded judges for cache hits; return (n_hits, payloads_sorted)."""
    hits = 0
    for payload in payloads:
        if payload.get("hardcoded_judge") is not None:
            continue
        judge = None
        if runtime is not None:
            judge = runtime.lookup(payload)
        if judge is None:
            dequant_norm = dequant_norm_from_payload(payload)
            entry = lookup_dequant_cache_entry(entries, dequant_norm)
            if entry is not None:
                judge = build_cached_judge_from_entry(entry, dequant_norm=dequant_norm)
        if judge is None:
            continue
        payload["hardcoded_judge"] = judge
        payload["dequant_norm"] = dequant_norm_from_payload(payload)
        hits += 1

    ordered = sorted(payloads, key=lambda p: 0 if p.get("hardcoded_judge") is not None else 1)
    return hits, ordered


def cache_to_json(cache_df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    entries = {}
    for row in cache_df.itertuples(index=False):
        portion_id = row.modal_portion_id
        if portion_id is None and not pd.isna(row.matched_portion_id):
            portion_id = int(row.matched_portion_id)
        entries[row.dequant_norm] = {
            "llm_fdc_id": int(row.llm_fdc_id),
            "llm_description": row.llm_description,
            "llm_certainty": None if pd.isna(row.llm_certainty) else float(row.llm_certainty),
            "llm_negligible_calories": bool(row.llm_negligible_calories),
            "portion_id": None if portion_id is None or pd.isna(portion_id) else int(portion_id),
            "matched_portion_id": None
            if portion_id is None or pd.isna(portion_id)
            else int(portion_id),
            "modal_portion_share": None
            if row.modal_portion_share is None or pd.isna(row.modal_portion_share)
            else float(row.modal_portion_share),
            "n_distinct_portion_id": int(row.n_distinct_portion_id),
            "grams_status": row.grams_status,
            "grams_method": row.grams_method,
            "n_sample_lines": int(row.n_sample_lines),
            "example_ingredient": row.example_ingredient,
        }
    return {"meta": meta, "entries": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dequant_norm LLM resolution cache")
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--min-modal-share", type=float, default=1.0)
    args = parser.parse_args()

    pm = pd.read_parquet(args.matches)
    stats, cache_df, meta = build_dequant_norm_cache(
        pm,
        min_count=args.min_count,
        min_modal_share=args.min_modal_share,
    )

    payload = cache_to_json(cache_df, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    cache_df.to_csv(csv_path, index=False)
    stats_path = args.out.with_name(args.out.stem + "_stats.csv")
    stats.to_csv(stats_path, index=False)

    print(
        f"Wrote {len(cache_df)} cache terms covering {meta['n_cache_hits']:,}/"
        f"{meta['n_source_lines']:,} lines ({meta['cache_hit_rate']:.1%})"
    )
    print(f"  json: {args.out}")
    print(f"  csv:  {csv_path}")
    print(f"  stats: {stats_path}")


if __name__ == "__main__":
    main()
