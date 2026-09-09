"""Process-wide SentenceTransformer singleton (load once, reuse every request)."""

from __future__ import annotations

from typing import Any

_MODEL = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_LOAD_ERROR: str | None = None


def get_embedding_model():
    """Return the cached MiniLM model, or None if unavailable."""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _LOAD_ERROR is not None:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_MODEL_NAME)
        return _MODEL
    except Exception as exc:  # pragma: no cover - env-dependent
        _LOAD_ERROR = str(exc)
        return None


def warm_embedding_model() -> dict[str, Any]:
    """Eager load for server startup. Returns status dict."""
    model = get_embedding_model()
    if model is None:
        return {"ok": False, "model": _MODEL_NAME, "error": _LOAD_ERROR}
    # Tiny encode to finish any lazy init / device placement.
    try:
        model.encode(["warmup"], normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "model": _MODEL_NAME, "error": str(exc)}
    return {"ok": True, "model": _MODEL_NAME}


def encode_texts(texts: list[str], *, batch_size: int = 64):
    """Encode texts with the shared model. Returns ndarray or None."""
    model = get_embedding_model()
    if model is None or not texts:
        return None
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
