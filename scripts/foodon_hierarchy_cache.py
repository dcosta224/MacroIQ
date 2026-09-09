"""Precomputed FoodOn ancestor/descendant closures for fast rollup queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from foodon_index import FOODON_OWL_URL, FoodOnIndex

DEFAULT_INDEX_CACHE = Path(__file__).resolve().parents[1] / "foodon_web" / "cache" / "foodon_index.json"
DEFAULT_HIERARCHY_CACHE = Path(__file__).resolve().parents[1] / "foodon_web" / "cache" / "foodon_hierarchy.json"


def norm_foodon_id(node_id: Any) -> str | None:
    if node_id is None:
        return None
    text = str(node_id).strip()
    return text or None


class FoodOnHierarchyCache:
    """Memoized ancestor/descendant closures over a FoodOnIndex."""

    def __init__(
        self,
        index: FoodOnIndex,
        *,
        ancestors: dict[str, list[str]],
        descendants: dict[str, list[str]],
        leaves: list[str],
    ):
        self.index = index
        self.ancestors = ancestors
        self.descendants = descendants
        self._leaves = set(leaves)
        self.leaves = leaves

    @classmethod
    def from_index(
        cls,
        index: FoodOnIndex,
        cache_path: Path | None = DEFAULT_HIERARCHY_CACHE,
        *,
        force_rebuild: bool = False,
    ) -> FoodOnHierarchyCache:
        if cache_path and cache_path.exists() and not force_rebuild:
            return cls.from_json(cache_path.read_text(encoding="utf-8"), index=index)

        ancestors, descendants, leaves = _build_closures(index)
        cache = cls(index, ancestors=ancestors, descendants=descendants, leaves=leaves)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(cache.to_json(), encoding="utf-8")
        return cache

    @classmethod
    def from_json(cls, payload: str, *, index: FoodOnIndex) -> FoodOnHierarchyCache:
        data = json.loads(payload)
        return cls(
            index,
            ancestors={k: list(v) for k, v in data["ancestors"].items()},
            descendants={k: list(v) for k, v in data["descendants"].items()},
            leaves=list(data["leaves"]),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "node_count": len(self.index.labels),
                "leaf_count": len(self.leaves),
                "ancestors": self.ancestors,
                "descendants": self.descendants,
                "leaves": self.leaves,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def is_leaf(self, node_id: str) -> bool:
        return node_id in self._leaves

    def ancestry_path(self, node_id: str) -> list[str]:
        return self.index.ancestry_path(node_id)

    def rollup_for_leaf_mappings(
        self,
        leaf_grams: dict[str, float],
        leaf_lines: dict[str, int],
        *,
        leaf_fdc: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        """Roll leaf grams/lines up to every ancestor (inclusive)."""
        grams_acc: dict[str, float] = {}
        lines_acc: dict[str, int] = {}
        fdc_acc: dict[str, int] = {}

        for leaf_id, grams in leaf_grams.items():
            lines = int(leaf_lines.get(leaf_id, 0))
            n_fdc = int((leaf_fdc or {}).get(leaf_id, 0))
            for ancestor_id in self.ancestors.get(leaf_id, [leaf_id]):
                grams_acc[ancestor_id] = grams_acc.get(ancestor_id, 0.0) + float(grams)
                lines_acc[ancestor_id] = lines_acc.get(ancestor_id, 0) + lines
                if leaf_fdc is not None:
                    fdc_acc[ancestor_id] = fdc_acc.get(ancestor_id, 0) + n_fdc

        rows: list[dict[str, Any]] = []
        for node_id in sorted(grams_acc, key=lambda nid: self.index.labels.get(nid, nid).lower()):
            row: dict[str, Any] = {
                "foodon_id": node_id,
                "label": self.index.labels.get(node_id, node_id),
                "is_leaf": self.is_leaf(node_id),
                "total_grams": grams_acc[node_id],
                "n_lines": lines_acc[node_id],
            }
            if leaf_fdc is not None:
                row["n_fdc"] = fdc_acc.get(node_id, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_grams", ascending=False).reset_index(drop=True)
        return df

    def metrics_for_node(
        self,
        node_id: str,
        leaf_grams: dict[str, float],
        leaf_lines: dict[str, int],
        total_grams: float,
        total_lines: int,
        *,
        leaf_fdc: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Exact, subtree, parent, and per-ancestor metrics for one FoodOn class."""
        node_id = norm_foodon_id(node_id) or ""
        if node_id not in self.index.labels:
            return {
                "node_id": node_id,
                "found": False,
                "exact": {},
                "subtree": {},
                "parent_subtree": {},
                "ancestry_table": pd.DataFrame(),
            }

        def _pct(part: float | int, whole: float | int) -> float:
            if not whole:
                return 0.0
            return 100.0 * float(part) / float(whole)

        def _subtree_metrics(target_id: str) -> dict[str, Any]:
            desc = set(self.descendants.get(target_id, [target_id]))
            grams = sum(float(leaf_grams.get(leaf_id, 0.0)) for leaf_id in desc)
            lines = sum(int(leaf_lines.get(leaf_id, 0)) for leaf_id in desc)
            n_fdc = 0
            if leaf_fdc:
                n_fdc = sum(int(leaf_fdc.get(leaf_id, 0)) for leaf_id in desc)
            mapped_leaves = sorted(leaf_id for leaf_id in desc if leaf_id in leaf_grams)
            return {
                "foodon_id": target_id,
                "label": self.index.labels.get(target_id, target_id),
                "total_grams": grams,
                "n_lines": lines,
                "n_fdc": n_fdc,
                "pct_grams": round(_pct(grams, total_grams), 4),
                "pct_lines": round(_pct(lines, total_lines), 4),
                "mapped_leaf_count": len(mapped_leaves),
            }

        exact_grams = float(leaf_grams.get(node_id, 0.0))
        exact_lines = int(leaf_lines.get(node_id, 0))
        exact_fdc = int((leaf_fdc or {}).get(node_id, 0))
        exact = {
            "foodon_id": node_id,
            "label": self.index.labels.get(node_id, node_id),
            "total_grams": exact_grams,
            "n_lines": exact_lines,
            "n_fdc": exact_fdc,
            "pct_grams": round(_pct(exact_grams, total_grams), 4),
            "pct_lines": round(_pct(exact_lines, total_lines), 4),
        }

        subtree = _subtree_metrics(node_id)

        path = self.ancestry_path(node_id)
        parent_id: str | None = None
        if len(path) >= 2:
            parent_id = path[-2]
        parent_subtree = _subtree_metrics(parent_id) if parent_id else {}

        ancestry_rows: list[dict[str, Any]] = []
        for level, ancestor_id in enumerate(reversed(path)):
            metrics = _subtree_metrics(ancestor_id)
            ancestry_rows.append(
                {
                    "level": level,
                    "foodon_id": ancestor_id,
                    "label": metrics["label"],
                    "is_leaf": self.is_leaf(ancestor_id),
                    "n_lines": metrics["n_lines"],
                    "n_fdc": metrics["n_fdc"],
                    "total_grams": metrics["total_grams"],
                    "pct_of_all_grams": metrics["pct_grams"],
                    "pct_of_all_lines": metrics["pct_lines"],
                    "mapped_leaf_count": metrics["mapped_leaf_count"],
                }
            )

        ancestry_table = pd.DataFrame(ancestry_rows)

        return {
            "node_id": node_id,
            "found": True,
            "label": self.index.labels.get(node_id, node_id),
            "is_leaf": self.is_leaf(node_id),
            "ancestry_path": path,
            "exact": exact,
            "subtree": subtree,
            "parent_subtree": parent_subtree,
            "ancestry_table": ancestry_table,
        }


def _build_closures(index: FoodOnIndex) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    ancestors: dict[str, list[str]] = {}
    descendants: dict[str, list[str]] = {}
    leaves: list[str] = []

    for node_id in index.labels:
        if not index.children.get(node_id):
            leaves.append(node_id)

        seen_anc: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in seen_anc:
                continue
            seen_anc.add(current)
            for parent_id in index.parents.get(current, []):
                if parent_id in index.labels:
                    stack.append(parent_id)
        ancestors[node_id] = sorted(seen_anc)

        descendants[node_id] = sorted(index.descendants_of(node_id, include_root=True))

    leaves.sort(key=lambda nid: index.labels.get(nid, nid).lower())
    return ancestors, descendants, leaves


def build_cache(
    *,
    index_cache: Path = DEFAULT_INDEX_CACHE,
    hierarchy_cache: Path = DEFAULT_HIERARCHY_CACHE,
    force_rebuild: bool = False,
    owl_url: str = FOODON_OWL_URL,
) -> FoodOnHierarchyCache:
    index = FoodOnIndex.from_owl(owl_url=owl_url, cache_path=index_cache)
    return FoodOnHierarchyCache.from_index(
        index,
        cache_path=hierarchy_cache,
        force_rebuild=force_rebuild,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build FoodOn hierarchy closure cache.")
    parser.add_argument(
        "--index-cache",
        type=Path,
        default=DEFAULT_INDEX_CACHE,
        help="Path to foodon_index.json (OWL parsed once if missing).",
    )
    parser.add_argument(
        "--hierarchy-cache",
        type=Path,
        default=DEFAULT_HIERARCHY_CACHE,
        help="Output path for foodon_hierarchy.json.",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore existing hierarchy cache.")
    parser.add_argument("--owl-url", default=FOODON_OWL_URL, help="FoodOn OWL URL when index cache is absent.")
    args = parser.parse_args(argv)

    cache = build_cache(
        index_cache=args.index_cache,
        hierarchy_cache=args.hierarchy_cache,
        force_rebuild=args.force_rebuild,
        owl_url=args.owl_url,
    )
    print(f"FoodOn classes: {len(cache.index.labels):,}")
    print(f"Leaves: {len(cache.leaves):,}")
    print(f"Hierarchy cache written: {args.hierarchy_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
