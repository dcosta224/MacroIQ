"""In-memory + on-disk cache for the 106-recipe MVP demo corpus."""

from __future__ import annotations

import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from db import connect
from mvp_data import (
    fetch_food_nutrients_for_recipe,
    fetch_mvp_recipe_ids,
    fetch_recipe_embeddings,
    fetch_recipe_features,
    fetch_recipe_nutrients,
    fetch_resolved_ingredients,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "mvp_web" / "cache"
CACHE_FILE = CACHE_DIR / "mvp_corpus.pkl"
LOCAL_EMB_DIR = ROOT / "Data" / "recipes" / "embeddings"

_cache: dict[str, Any] | None = None
_cache_lock = threading.Lock()


def _embeddings_from_local_export(recipe_ids: list[int]) -> tuple[np.ndarray, list[int]] | None:
    """Subset rows from exported RecipeNLG embedding memmap if available."""
    ids_path = LOCAL_EMB_DIR / "recipe_ids.npy"
    emb_path = LOCAL_EMB_DIR / "embeddings.f32.memmap"
    manifest_path = LOCAL_EMB_DIR / "manifest.json"
    if not (ids_path.is_file() and emb_path.is_file()):
        return None

    all_ids = np.load(ids_path)
    id_to_idx = {int(rid): i for i, rid in enumerate(all_ids)}
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    dims = int(manifest.get("dims", 384))
    count = int(manifest.get("count", len(all_ids)))
    memmap = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(count, dims))

    found_ids: list[int] = []
    rows: list[np.ndarray] = []
    for rid in recipe_ids:
        idx = id_to_idx.get(int(rid))
        if idx is not None:
            found_ids.append(int(rid))
            rows.append(np.array(memmap[idx], dtype=np.float32))

    if not rows:
        return None
    return np.vstack(rows), found_ids


def _encode_missing_from_features(
    features: dict[int, dict[str, Any]],
    missing_ids: list[int],
) -> tuple[np.ndarray, list[int]]:
    """Compute MiniLM embeddings for recipes not in memmap or Supabase."""
    if not missing_ids:
        return np.zeros((0, 384), dtype=np.float32), []

    from sentence_transformers import SentenceTransformer

    from mvp_recipe_ranker import EMBEDDING_MODEL

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts: list[str] = []
    ids: list[int] = []
    for rid in missing_ids:
        feat = features.get(int(rid), {})
        text = (feat.get("semantic_text") or feat.get("title_clean") or "").strip()
        if not text:
            text = f"Recipe {rid}"
        texts.append(text)
        ids.append(int(rid))

    embs = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32), ids


def _resolve_embeddings(
    cur,
    recipe_ids: list[int],
    features: dict[int, dict[str, Any]],
) -> tuple[dict[int, np.ndarray], str, list[int]]:
    """Load embeddings: local memmap first, then Supabase, then on-the-fly encode."""
    emb_by_id: dict[int, np.ndarray] = {}
    source_parts: list[str] = []

    local = _embeddings_from_local_export(recipe_ids)
    if local is not None:
        local_embs, local_ids = local
        for i, rid in enumerate(local_ids):
            emb_by_id[int(rid)] = local_embs[i]
        source_parts.append(f"local:{len(local_ids)}")

    missing = [int(rid) for rid in recipe_ids if int(rid) not in emb_by_id]
    if missing:
        db_embs, db_ids = fetch_recipe_embeddings(cur, missing)
        for i, rid in enumerate(db_ids):
            emb_by_id[int(rid)] = db_embs[i]
        if db_ids:
            source_parts.append(f"supabase:{len(db_ids)}")

    missing = [int(rid) for rid in recipe_ids if int(rid) not in emb_by_id]
    if missing:
        enc_embs, enc_ids = _encode_missing_from_features(features, missing)
        for i, rid in enumerate(enc_ids):
            emb_by_id[int(rid)] = enc_embs[i]
        if enc_ids:
            source_parts.append(f"encoded:{len(enc_ids)}")

    still_missing = [int(rid) for rid in recipe_ids if int(rid) not in emb_by_id]
    emb_source = "+".join(source_parts) if source_parts else "none"
    return emb_by_id, emb_source, still_missing


def _fetch_all_ingredients(cur, recipe_ids: list[int]) -> dict[int, pd.DataFrame]:
    if not recipe_ids:
        return {}
    cur.execute(
        """
        SELECT recipe_id, ingredient_idx, ingredient, fdc_id, fdc_description,
               portion_id, portion_label, quantity, unit, gram_weight
        FROM recipe.resolved_recipes
        WHERE recipe_id = ANY(%s)
        ORDER BY recipe_id, ingredient_idx
        """,
        (recipe_ids,),
    )
    cols = [
        "recipe_id",
        "ingredient_idx",
        "ingredient",
        "fdc_id",
        "fdc_description",
        "portion_id",
        "portion_label",
        "quantity",
        "unit",
        "gram_weight",
    ]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    return {
        int(rid): grp.reset_index(drop=True)
        for rid, grp in df.groupby("recipe_id")
    }


def _build_corpus_from_db() -> dict[str, Any]:
    t0 = time.monotonic()
    conn = connect()
    try:
        with conn.cursor() as cur:
            recipe_ids = fetch_mvp_recipe_ids(cur)
            nutrients = fetch_recipe_nutrients(cur, recipe_ids)
            features = fetch_recipe_features(cur, recipe_ids)
            emb_by_id, emb_source, missing_emb = _resolve_embeddings(
                cur, recipe_ids, features
            )
            ingredients_by_recipe = _fetch_all_ingredients(cur, recipe_ids)

            all_fdc: set[int] = set()
            for ing_df in ingredients_by_recipe.values():
                for val in ing_df["fdc_id"].dropna():
                    all_fdc.add(int(val))
            food_nutrients = fetch_food_nutrients_for_recipe(cur, sorted(all_fdc))
    finally:
        conn.close()

    aligned_embs: list[np.ndarray] = []
    aligned_ids: list[int] = []
    aligned_names: list[str] = []
    aligned_nutrients: list[dict[str, Any]] = []

    for _, row in nutrients.iterrows():
        rid = int(row["recipe_id"])
        emb = emb_by_id.get(rid)
        if emb is None:
            continue
        aligned_ids.append(rid)
        aligned_embs.append(emb)
        aligned_names.append(str(row["recipe_name"]))
        aligned_nutrients.append(row.to_dict())

    corpus = {
        "recipe_ids": aligned_ids,
        "recipe_names": aligned_names,
        "embeddings": np.vstack(aligned_embs) if aligned_embs else np.zeros((0, 384), dtype=np.float32),
        "nutrient_rows": aligned_nutrients,
        "features": features,
        "ingredients_by_recipe": ingredients_by_recipe,
        "food_nutrients": food_nutrients,
        "cached_at": time.time(),
        "emb_source": emb_source,
        "n_mvp_ids": len(recipe_ids),
        "n_recipes": len(aligned_ids),
        "missing_embedding_ids": missing_emb,
        "build_ms": int((time.monotonic() - t0) * 1000),
    }
    return corpus


def save_corpus_to_disk(corpus: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(corpus)
    with CACHE_FILE.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cached_at": corpus.get("cached_at"),
        "n_mvp_ids": corpus.get("n_mvp_ids"),
        "n_recipes": corpus.get("n_recipes"),
        "emb_source": corpus.get("emb_source"),
        "missing_embedding_ids": corpus.get("missing_embedding_ids", []),
        "build_ms": corpus.get("build_ms"),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(meta, indent=2))
    return CACHE_FILE


def _disk_cache_complete(disk: dict[str, Any]) -> bool:
    """True if on-disk cache covers all MVP recipe ids with embeddings."""
    n_recipes = int(disk.get("n_recipes") or 0)
    n_mvp = int(disk.get("n_mvp_ids") or n_recipes)
    if n_mvp <= 0:
        return n_recipes > 0
    if disk.get("missing_embedding_ids"):
        return False
    return n_recipes >= n_mvp


def load_corpus_from_disk() -> dict[str, Any] | None:
    if not CACHE_FILE.is_file():
        return None
    with CACHE_FILE.open("rb") as f:
        return pickle.load(f)


def warm_mvp_corpus(*, force_refresh: bool = False) -> dict[str, Any]:
    """Load corpus into memory (disk cache first, else Supabase)."""
    global _cache
    with _cache_lock:
        if _cache is not None and not force_refresh:
            return _cache

        if not force_refresh:
            disk = None
            try:
                disk = load_corpus_from_disk()
            except Exception:
                disk = None
            if disk is not None and _disk_cache_complete(disk):
                disk["load_source"] = "disk"
                _cache = disk
                return _cache

        corpus = _build_corpus_from_db()
        corpus["load_source"] = "database"
        save_corpus_to_disk(corpus)
        _cache = corpus
        return _cache


def get_mvp_corpus() -> dict[str, Any]:
    """Return warmed corpus; builds cache on first call."""
    if _cache is not None:
        return _cache
    return warm_mvp_corpus()


def corpus_status() -> dict[str, Any]:
    if _cache is None:
        disk = None
        try:
            disk = load_corpus_from_disk()
        except Exception:
            disk = None
        if disk is not None:
            return {
                "ready": False,
                "on_disk": True,
                "n_mvp_ids": disk.get("n_mvp_ids"),
                "n_recipes": disk.get("n_recipes"),
                "emb_source": disk.get("emb_source"),
                "cache_complete": _disk_cache_complete(disk),
            }
        return {"ready": False, "on_disk": False}
    return {
        "ready": True,
        "on_disk": CACHE_FILE.is_file(),
        "n_mvp_ids": _cache.get("n_mvp_ids"),
        "n_recipes": _cache.get("n_recipes"),
        "emb_source": _cache.get("emb_source"),
        "load_source": _cache.get("load_source"),
        "cache_complete": _disk_cache_complete(_cache),
        "build_ms": _cache.get("build_ms"),
    }


def get_cached_ingredients(corpus: dict[str, Any], recipe_id: int) -> pd.DataFrame:
    by_recipe = corpus.get("ingredients_by_recipe") or {}
    if int(recipe_id) in by_recipe:
        return by_recipe[int(recipe_id)].copy()
    conn = connect()
    try:
        with conn.cursor() as cur:
            return fetch_resolved_ingredients(cur, int(recipe_id))
    finally:
        conn.close()


def get_cached_food_nutrients(corpus: dict[str, Any], fdc_ids: list[int]) -> pd.DataFrame:
    fn = corpus.get("food_nutrients")
    if isinstance(fn, pd.DataFrame) and not fn.empty and fdc_ids:
        return fn[fn["fdc_id"].isin(fdc_ids)].copy()
    conn = connect()
    try:
        with conn.cursor() as cur:
            return fetch_food_nutrients_for_recipe(cur, fdc_ids)
    finally:
        conn.close()
