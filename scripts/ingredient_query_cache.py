"""One-shot parse + triple embedding cache for recipe lines and food_4macro catalog."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from db import load_dotenv

import numpy as np
import pandas as pd

from parse_recipe_ingredient import parse_ingredient_fields, strip_quantities_from_text
from unit_aliases import normalize_units_in_text
from progress_utils import iter_progress, progress_enabled_for_count, map_progress

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = ROOT / "scratch" / "recipe_matching_10k"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
# Bump when recipe dequant embedding text changes (invalidates recipe_cache dequant .npy).
RECIPE_SEMANTIC_EMBEDDING_VERSION = "v3_unit_alias_dequant"

# Recipe query artifacts
RECIPE_PARSED = "recipe_ingredients_parsed.parquet"
RECIPE_NAME_EMB = "recipe_name_embeddings.npy"
RECIPE_PREP_EMB = "recipe_prep_embeddings.npy"
RECIPE_DEQUANT_EMB = "recipe_dequant_embeddings.npy"
# Single vector: recipe lines with empty preparation use this for prep semantic match
UNPREPARED_PREP_TEXT = "unprepared"
UNPREPARED_PREP_EMB = "unprepared_prep_embedding.npy"

# Food catalog artifacts (parsed USDA descriptions)
FOOD_PARSED = "food_4macro_parsed.parquet"
FOOD_NAME_EMB = "food_4macro_name_embeddings.npy"
FOOD_PREP_EMB = "food_4macro_prep_embeddings.npy"
FOOD_DEQUANT_EMB = "food_4macro_dequant_embeddings.npy"

EMBEDDINGS_META = "embeddings_meta.json"

# Back-compat alias (legacy single-vector food cache)
FOOD_DESC_EMBEDDINGS = FOOD_DEQUANT_EMB

# All parse + embedding artifacts managed by this module
EMBEDDING_CACHE_FILENAMES = (
    RECIPE_PARSED,
    RECIPE_NAME_EMB,
    RECIPE_PREP_EMB,
    RECIPE_DEQUANT_EMB,
    FOOD_PARSED,
    FOOD_NAME_EMB,
    FOOD_PREP_EMB,
    FOOD_DEQUANT_EMB,
    EMBEDDINGS_META,
    UNPREPARED_PREP_EMB,
    "food_4macro_desc_embeddings.npy",
    "recipe_embeddings_meta.json",
)

DEFAULT_RECIPE_CSV = ROOT / "Data" / "recipes" / "RecipeNLG.csv"
DEFAULT_RECIPE_NROWS = 10_000


def _field_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def dequantified_text(row: pd.Series | dict[str, Any], *, raw: str = "") -> str:
    """Original ingredient text with parsed numeric quantities stripped (units and prep kept)."""
    if isinstance(row, dict):
        row = pd.Series(row)
    text = _field_text(raw)
    if not text:
        text = _field_text(row.get("ingredient") or row.get("description"))
    if not text:
        return ""
    return normalize_units_in_text(strip_quantities_from_text(text))


def semantic_embedding_text(row: pd.Series | dict[str, Any], *, raw: str = "") -> str:
    """Recipe text for global semantic retrieval: dequantified line plus parsed preparation."""
    dequant = dequantified_text(row, raw=raw)
    prep = _field_text(row.get("preparation") if isinstance(row, dict) else row.get("preparation"))
    if not prep:
        return dequant
    if not dequant:
        return prep
    prep_low = prep.lower()
    if prep_low in dequant.lower():
        return dequant
    return f"{dequant}, {prep}"


def ensure_hf_token() -> bool:
    """Load repo `.env` and expose HF token to huggingface_hub / sentence-transformers."""
    load_dotenv()
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return True


def load_encoder(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    if ensure_hf_token():
        print("Using HF_TOKEN from .env for Hugging Face Hub downloads.", flush=True)
    else:
        print(
            "Warning: HF_TOKEN not set — unauthenticated Hub requests (slower rate limits).",
            flush=True,
        )
    return SentenceTransformer(model_name)


def has_preparation(value: Any) -> bool:
    return bool(_field_text(value))


def encode_unprepared_vector(
    model,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """One shared prep embedding for implicit 'unprepared' recipe ingredients."""
    return _encode_texts(
        model,
        [UNPREPARED_PREP_TEXT],
        batch_size=batch_size,
        show_progress=False,
        label="unprepared (once)",
    )[0]


def load_or_build_unprepared_embedding(
    work_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    model: Any | None = None,
    force: bool = False,
) -> np.ndarray:
    work_dir = Path(work_dir)
    path = work_dir / UNPREPARED_PREP_EMB
    if not force and path.is_file():
        return np.load(path)
    encoder = model if model is not None else load_encoder(model_name)
    vec = encode_unprepared_vector(encoder)
    work_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vec)
    print(f"Saved {UNPREPARED_PREP_TEXT!r} embedding → {path}", flush=True)
    return vec


def _embed_recipe_prep_embeddings(
    parsed: pd.DataFrame,
    model,
    unprepared_vector: np.ndarray,
    *,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """
    Recipe prep vectors: empty preparation → shared *unprepared* embedding;
    non-empty → embed parsed preparation text. Food catalog uses full parsed prep instead.
    """
    n = len(parsed)
    unprepared_vector = np.asarray(unprepared_vector, dtype=np.float32).reshape(-1)
    prep_emb = np.tile(unprepared_vector, (n, 1))
    nonempty_idx = [i for i in range(n) if has_preparation(parsed["preparation"].iloc[i])]
    if nonempty_idx:
        texts = [str(parsed["preparation"].iloc[i]) for i in nonempty_idx]
        encoded = _encode_texts(
            model,
            texts,
            batch_size=batch_size,
            show_progress=show_progress,
            label="recipe prep (specified)",
        )
        for j, i in enumerate(nonempty_idx):
            prep_emb[i] = encoded[j]
    n_unprepared = n - len(nonempty_idx)
    print(
        f"Recipe prep: {n_unprepared:,} lines use {UNPREPARED_PREP_TEXT!r} proxy; "
        f"{len(nonempty_idx):,} embed parsed preparation",
        flush=True,
    )
    return prep_emb


def _encode_texts(
    model,
    texts: list[str],
    *,
    batch_size: int,
    show_progress: bool,
    label: str,
) -> np.ndarray:
    encode_kw: dict[str, Any] = {"batch_size": batch_size, "normalize_embeddings": True}
    if show_progress:
        encode_kw["show_progress_bar"] = True
        print(f"Embedding {label} ({len(texts):,} texts)…", flush=True)
    return model.encode(texts, **encode_kw).astype(np.float32)


def _parse_table(
    df: pd.DataFrame,
    text_col: str,
    *,
    id_cols: list[str],
    show_progress: bool = True,
) -> pd.DataFrame:
    # id_cols often includes text_col (e.g. recipe "ingredient"); avoid duplicate columns
    cols = list(dict.fromkeys([*id_cols, text_col]))
    base = df[cols].copy()
    n = len(base)
    use_progress = show_progress and progress_enabled_for_count(n, threshold=100)
    texts = df[text_col].astype(str).tolist()
    parsed_rows = map_progress(
        parse_ingredient_fields,
        texts,
        desc="Parsing ingredients",
        enabled=use_progress,
        total=n,
    )
    parsed_df = pd.DataFrame(parsed_rows)
    out = pd.concat([base.reset_index(drop=True), parsed_df], axis=1)
    out["name"] = out["name"].map(_field_text)
    out["preparation"] = out["preparation"].map(_field_text)
    out["size"] = out["size"].map(_field_text)
    dequantified: list[str] = []
    for i in iter_progress(
        range(n),
        total=n,
        desc="Dequantifying",
        enabled=use_progress,
    ):
        dequantified.append(dequantified_text(out.iloc[i], raw=str(out.iloc[i][text_col])))
    out["dequantified"] = dequantified
    return out


def _embed_parsed_table(
    parsed: pd.DataFrame,
    *,
    text_col: str,
    model_name: str,
    batch_size: int,
    show_progress: bool,
    labels: tuple[str, str, str],
    recipe_unprepared_proxy: bool = False,
    unprepared_vector: np.ndarray | None = None,
    model: Any | None = None,
    food_catalog: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder = model if model is not None else load_encoder(model_name)
    raw_col = text_col if text_col in parsed.columns else "ingredient"
    if food_catalog:
        # Full description (quantity stripped) — already computed in _parse_table as dequantified.
        full_texts = parsed["dequantified"].astype(str).tolist()
        names = full_texts
        dequants = full_texts
    else:
        names = parsed["name"].where(parsed["name"].astype(bool), parsed[raw_col]).tolist()
        dequants = [
            semantic_embedding_text(parsed.iloc[i], raw=str(parsed.iloc[i][raw_col]))
            for i in range(len(parsed))
        ]

    name_emb = _encode_texts(
        encoder, names, batch_size=batch_size, show_progress=show_progress, label=labels[0]
    )
    if recipe_unprepared_proxy:
        if unprepared_vector is None:
            unprepared_vector = encode_unprepared_vector(encoder, batch_size=batch_size)
        prep_emb = _embed_recipe_prep_embeddings(
            parsed,
            encoder,
            unprepared_vector,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    else:
        preps = [p if p else "[no preparation]" for p in parsed["preparation"].tolist()]
        prep_emb = _encode_texts(
            encoder, preps, batch_size=batch_size, show_progress=show_progress, label=labels[1]
        )
    dequant_emb = _encode_texts(
        encoder, dequants, batch_size=batch_size, show_progress=show_progress, label=labels[2]
    )
    return name_emb, prep_emb, dequant_emb


def _recipe_paths(work_dir: Path) -> tuple[Path, Path, Path, Path]:
    d = Path(work_dir)
    return (
        d / RECIPE_PARSED,
        d / RECIPE_NAME_EMB,
        d / RECIPE_PREP_EMB,
        d / RECIPE_DEQUANT_EMB,
    )


def _food_paths(work_dir: Path) -> tuple[Path, Path, Path, Path]:
    d = Path(work_dir)
    return (
        d / FOOD_PARSED,
        d / FOOD_NAME_EMB,
        d / FOOD_PREP_EMB,
        d / FOOD_DEQUANT_EMB,
    )


def _save_triple(
    paths: tuple[Path, Path, Path, Path],
    parsed: pd.DataFrame,
    name_emb: np.ndarray,
    prep_emb: np.ndarray,
    dequant_emb: np.ndarray,
) -> None:
    parsed_path, name_path, prep_path, dequant_path = paths
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_parquet(parsed_path, index=False)
    np.save(name_path, name_emb)
    np.save(prep_path, prep_emb)
    np.save(dequant_path, dequant_emb)


def _triple_exists(paths: tuple[Path, Path, Path, Path]) -> bool:
    return all(p.is_file() for p in paths)


def _load_triple(
    paths: tuple[Path, Path, Path, Path],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    parsed_path, name_path, prep_path, dequant_path = paths
    return (
        pd.read_parquet(parsed_path),
        np.load(name_path),
        np.load(prep_path),
        np.load(dequant_path),
    )


def build_recipe_artifacts(
    recipe_ingredients: pd.DataFrame,
    work_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    id_cols = ["recipe_id", "ingredient_idx", "ingredient"]
    missing = [c for c in id_cols if c not in recipe_ingredients.columns]
    if missing:
        raise ValueError(f"recipe_ingredients missing columns: {missing}")

    parsed = _parse_table(
        recipe_ingredients, "ingredient", id_cols=id_cols, show_progress=show_progress
    )
    parsed["prep_used_unprepared"] = ~parsed["preparation"].map(has_preparation)
    model = load_encoder(model_name)
    unprepared_vec = load_or_build_unprepared_embedding(
        work_dir, model_name=model_name, model=model
    )
    name_emb, prep_emb, dequant_emb = _embed_parsed_table(
        parsed,
        text_col="ingredient",
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
        labels=("recipe name", "recipe prep", "recipe dequant"),
        recipe_unprepared_proxy=True,
        unprepared_vector=unprepared_vec,
        model=model,
    )
    _save_triple(_recipe_paths(work_dir), parsed, name_emb, prep_emb, dequant_emb)
    parsed["semantic_embedding_text"] = [
        semantic_embedding_text(parsed.iloc[i], raw=str(parsed.iloc[i]["ingredient"]))
        for i in range(len(parsed))
    ]
    meta_bit = {
        "n_rows": len(parsed),
        "name_shape": list(name_emb.shape),
        "prep_shape": list(prep_emb.shape),
        "dequant_shape": list(dequant_emb.shape),
        "unprepared_prep_proxy_n": int(parsed["prep_used_unprepared"].sum()),
        "unprepared_prep_text": UNPREPARED_PREP_TEXT,
        "semantic_embedding_version": RECIPE_SEMANTIC_EMBEDDING_VERSION,
    }
    return parsed, name_emb, prep_emb, dequant_emb, meta_bit


def build_food_artifacts(
    food_df: pd.DataFrame,
    work_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    food_lines = food_df[["fdc_id", "data_type", "description", "food_category_id", "publication_date"]].copy()
    food_lines = food_lines.rename(columns={"description": "ingredient"})
    parsed = _parse_table(
        food_lines,
        "ingredient",
        id_cols=["fdc_id", "data_type", "food_category_id", "publication_date"],
        show_progress=show_progress,
    )
    parsed = parsed.rename(columns={"ingredient": "description"})
    print("Parsing complete; embedding food catalog (3 matrices)…", flush=True)

    name_emb, prep_emb, dequant_emb = _embed_parsed_table(
        parsed,
        text_col="description",
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
        labels=("food name", "food prep", "food dequant"),
        food_catalog=True,
    )
    _save_triple(_food_paths(work_dir), parsed, name_emb, prep_emb, dequant_emb)
    meta_bit = {
        "n_rows": len(parsed),
        "name_shape": list(name_emb.shape),
        "prep_shape": list(prep_emb.shape),
        "dequant_shape": list(dequant_emb.shape),
    }
    return parsed, name_emb, prep_emb, dequant_emb, meta_bit


def _read_meta(work_dir: Path) -> dict[str, Any]:
    path = Path(work_dir) / EMBEDDINGS_META
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def _write_meta(work_dir: Path, meta: dict[str, Any]) -> None:
    (Path(work_dir) / EMBEDDINGS_META).write_text(json.dumps(meta, indent=2) + "\n")


def load_or_build_recipe_artifacts(
    recipe_ingredients: pd.DataFrame,
    work_dir: Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    force: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    n_expected = len(recipe_ingredients)

    recipe_paths = _recipe_paths(work_dir)
    unprepared_path = work_dir / UNPREPARED_PREP_EMB
    if not force and _triple_exists(recipe_paths) and unprepared_path.is_file():
        parsed, name_emb, prep_emb, dequant_emb = _load_triple(recipe_paths)
        meta = _read_meta(work_dir).get("recipe", {})
        if (
            meta.get("n_rows") == n_expected
            and len(parsed) == n_expected
            and meta.get("semantic_embedding_version") == RECIPE_SEMANTIC_EMBEDDING_VERSION
        ):
            print(f"Loaded cached recipe parse + embeddings ({n_expected:,} rows) → {work_dir}")
            return parsed, name_emb, prep_emb, dequant_emb, meta
        if meta.get("semantic_embedding_version") != RECIPE_SEMANTIC_EMBEDDING_VERSION:
            print(
                f"Recipe embedding cache stale "
                f"({meta.get('semantic_embedding_version')!r} != {RECIPE_SEMANTIC_EMBEDDING_VERSION!r}); rebuilding…",
                flush=True,
            )

    print(f"Building recipe parse + 3× embeddings ({n_expected:,} rows) → {work_dir}")
    parsed, name_emb, prep_emb, dequant_emb, meta_bit = build_recipe_artifacts(
        recipe_ingredients,
        work_dir,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    full_meta = _read_meta(work_dir)
    full_meta["model_name"] = model_name
    full_meta["embedding_dim"] = int(name_emb.shape[1])
    full_meta["recipe"] = meta_bit
    _write_meta(work_dir, full_meta)
    return parsed, name_emb, prep_emb, dequant_emb, meta_bit


def embed_adhoc_recipe_queries(
    ingredients: list[str],
    work_dir: Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Parse + embed a small list of recipe-style ingredient lines (uses unprepared prep proxy)."""
    work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    rows = [
        {"recipe_id": i, "ingredient_idx": 0, "ingredient": text}
        for i, text in enumerate(ingredients)
    ]
    df = pd.DataFrame(rows)
    parsed = _parse_table(df, "ingredient", id_cols=["recipe_id", "ingredient_idx"], show_progress=False)
    parsed["prep_used_unprepared"] = ~parsed["preparation"].map(has_preparation)
    for i in range(len(parsed)):
        row = parsed.iloc[i]
        raw = str(row["ingredient"])
        parsed.loc[i, "dequantified"] = dequantified_text(row, raw=raw)
        parsed.loc[i, "semantic_embedding_text"] = semantic_embedding_text(row, raw=raw)
    model = load_encoder(model_name)
    unprepared_vec = load_or_build_unprepared_embedding(work_dir, model_name=model_name, model=model)
    name_emb, prep_emb, dequant_emb = _embed_parsed_table(
        parsed,
        text_col="ingredient",
        model_name=model_name,
        batch_size=min(32, max(1, len(parsed))),
        show_progress=False,
        labels=("adhoc name", "adhoc prep", "adhoc dequant"),
        recipe_unprepared_proxy=True,
        unprepared_vector=unprepared_vec,
        model=model,
    )
    return parsed, name_emb, prep_emb, dequant_emb


def load_or_build_food_artifacts(
    food_df: pd.DataFrame,
    work_dir: Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    force: bool = False,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    n_expected = len(food_df)

    food_paths = _food_paths(work_dir)
    if not force and _triple_exists(food_paths):
        parsed, name_emb, prep_emb, dequant_emb = _load_triple(food_paths)
        meta = _read_meta(work_dir).get("food_4macro", {})
        if meta.get("n_rows") == n_expected and len(parsed) == n_expected:
            print(f"Loaded cached food_4macro parse + embeddings ({n_expected:,} rows) → {work_dir}")
            return parsed, name_emb, prep_emb, dequant_emb, meta

    print(f"Building food_4macro parse + 3× embeddings ({n_expected:,} rows) → {work_dir}")
    parsed, name_emb, prep_emb, dequant_emb, meta_bit = build_food_artifacts(
        food_df,
        work_dir,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    full_meta = _read_meta(work_dir)
    full_meta["model_name"] = model_name
    full_meta["embedding_dim"] = int(name_emb.shape[1])
    full_meta["food_4macro"] = meta_bit
    _write_meta(work_dir, full_meta)
    return parsed, name_emb, prep_emb, dequant_emb, meta_bit


def embedding_cache_status(work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Report whether recipe and food_4macro parse + embedding triples exist."""
    work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    out: dict[str, dict[str, Any]] = {}
    for label, paths_fn in (
        ("recipe", _recipe_paths),
        ("food_4macro", _food_paths),
    ):
        paths = paths_fn(work_dir)
        complete = _triple_exists(paths)
        n_rows: int | None = None
        if complete:
            n_rows = len(pd.read_parquet(paths[0]))
        out[label] = {
            "complete": complete,
            "n_rows": n_rows,
            "files": [p.name for p in paths],
        }
    return out


def print_embedding_cache_summary(work_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    status = embedding_cache_status(work_dir)
    print("\n=== Embedding cache summary ===", flush=True)
    for label, info in status.items():
        mark = "OK" if info["complete"] else "MISSING"
        rows = f"{info['n_rows']:,} rows" if info["n_rows"] is not None else "—"
        print(f"  [{mark}] {label}: {rows}", flush=True)
        if not info["complete"]:
            print(f"         expected: {', '.join(info['files'])}", flush=True)
    return status


def load_or_build_all_embedding_artifacts(
    recipe_ingredients: pd.DataFrame,
    food_df: pd.DataFrame,
    work_dir: Path | None = None,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build/load recipe + food parse and all six embedding matrices."""
    print("=== Recipe ingredients ===", flush=True)
    recipe = load_or_build_recipe_artifacts(
        recipe_ingredients,
        work_dir,
        model_name=model_name,
        batch_size=batch_size,
        force=force,
        show_progress=show_progress,
    )
    print("\n=== food_4macro catalog (database) ===", flush=True)
    print(f"food_4macro rows: {len(food_df):,}", flush=True)
    food = load_or_build_food_artifacts(
        food_df,
        work_dir,
        model_name=model_name,
        batch_size=batch_size,
        force=force,
        show_progress=show_progress,
    )
    print_embedding_cache_summary(work_dir)
    meta = _read_meta(work_dir or DEFAULT_WORK_DIR)
    return {
        "recipe_parsed": recipe[0],
        "recipe_name_emb": recipe[1],
        "recipe_prep_emb": recipe[2],
        "recipe_dequant_emb": recipe[3],
        "food_parsed": food[0],
        "food_name_emb": food[1],
        "food_prep_emb": food[2],
        "food_dequant_emb": food[3],
        "meta": meta,
    }


# Back-compat alias used by notebook during transition
def artifacts_exist(work_dir: Path) -> bool:
    return _triple_exists(_recipe_paths(work_dir))


def load_or_build_artifacts(
    recipe_ingredients: pd.DataFrame,
    work_dir: Path | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Legacy API: returns name + prep only. Prefer load_or_build_recipe_artifacts."""
    parsed, name_emb, prep_emb, dequant_emb, meta = load_or_build_recipe_artifacts(
        recipe_ingredients, work_dir, **kwargs
    )
    legacy_meta = {"model_name": kwargs.get("model_name", DEFAULT_MODEL), **meta}
    return parsed, name_emb, prep_emb, legacy_meta


def load_artifacts(work_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    parsed, name_emb, prep_emb, _dequant_emb = _load_triple(_recipe_paths(work_dir))
    meta = _read_meta(work_dir).get("recipe", {})
    return parsed, name_emb, prep_emb, meta


def clear_embedding_cache(work_dir: Path | None = None) -> list[Path]:
    """Remove cached parse + embedding files (does not delete match result CSVs)."""
    work_dir = Path(work_dir or DEFAULT_WORK_DIR)
    removed: list[Path] = []
    for name in EMBEDDING_CACHE_FILENAMES:
        path = work_dir / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    if removed:
        print(f"Cleared {len(removed)} embedding cache file(s) from {work_dir}")
    else:
        print(f"No embedding cache files found in {work_dir}")
    return removed


def _parse_ingredient_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    text = text.strip()
    if not text:
        return []
    if '", "' not in text:
        item = text.strip('"')
        return [item] if item else []
    parts = text.split('", "')
    out: list[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if i == 0:
            part = part.removeprefix('["').removeprefix('"')
        if i == len(parts) - 1:
            part = part.removesuffix('"]').removesuffix('"')
        if part:
            out.append(part)
    return out


def load_recipe_ingredients(
    *,
    recipe_csv: Path = DEFAULT_RECIPE_CSV,
    nrows: int = DEFAULT_RECIPE_NROWS,
) -> pd.DataFrame:
    """Explode RecipeNLG into one row per ingredient line (notebook §1 logic)."""
    recipes = pd.read_csv(recipe_csv, nrows=nrows)
    recipes["ingredients_list"] = recipes["ingredients"].map(_parse_ingredient_list)
    recipes["recipe_id"] = recipes.index
    recipe_ingredients = (
        recipes[["recipe_id", "ingredients_list"]]
        .explode("ingredients_list")
        .rename(columns={"ingredients_list": "ingredient"})
        .reset_index(drop=True)
    )
    recipe_ingredients["ingredient_idx"] = recipe_ingredients.groupby("recipe_id").cumcount()
    return recipe_ingredients


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Parse and embed recipe lines AND food_4macro catalog (cached under work dir). "
            "Default builds both; use --food-only if recipe artifacts already exist."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Delete cached parse/embedding files and rebuild all matrices",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Cache directory (default: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--recipe-csv",
        type=Path,
        default=DEFAULT_RECIPE_CSV,
        help="RecipeNLG CSV path",
    )
    parser.add_argument(
        "--recipe-nrows",
        type=int,
        default=DEFAULT_RECIPE_NROWS,
        help="Number of recipes to load from CSV (default: 10000)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recipe-only", action="store_true", help="Skip food_4macro embeddings")
    parser.add_argument("--food-only", action="store_true", help="Skip recipe embeddings")
    args = parser.parse_args()

    if args.recipe_only and args.food_only:
        raise SystemExit("Cannot use --recipe-only and --food-only together")

    ensure_hf_token()
    work_dir = args.work_dir
    force = args.rerun
    if args.rerun:
        clear_embedding_cache(work_dir)

    need_recipe = not args.food_only
    need_food = not args.recipe_only

    if need_recipe:
        if not args.recipe_csv.is_file():
            raise SystemExit(f"Recipe CSV not found: {args.recipe_csv}")
        print("=== Recipe ingredients (RecipeNLG) ===", flush=True)
        recipe_ingredients = load_recipe_ingredients(
            recipe_csv=args.recipe_csv,
            nrows=args.recipe_nrows,
        )
        print(f"Recipe ingredient lines: {len(recipe_ingredients):,}", flush=True)
        load_or_build_recipe_artifacts(
            recipe_ingredients,
            work_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            force=force,
            show_progress=True,
        )

    if need_food:
        from load_food_4macro import load_food_4macro

        print("\n=== food_4macro catalog (USDA database) ===", flush=True)
        food_df = load_food_4macro()
        print(f"food_4macro rows: {len(food_df):,}", flush=True)
        load_or_build_food_artifacts(
            food_df,
            work_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            force=force,
            show_progress=True,
        )

    status = print_embedding_cache_summary(work_dir)
    if need_recipe and not status["recipe"]["complete"]:
        raise SystemExit("Recipe embedding cache incomplete.")
    if need_food and not status["food_4macro"]["complete"]:
        raise SystemExit(
            "food_4macro embedding cache incomplete. "
            "Re-run with --food-only if recipe artifacts are already built."
        )

    print(f"\nDone → {work_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
