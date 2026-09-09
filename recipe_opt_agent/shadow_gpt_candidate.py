"""Silent GPT draft → ground → LP candidate for the agent pool / judge.

The optimized entry is parked alongside loop candidates. Public fields must not
advertise model authorship; keep any model marker private (`_shadow_model`).

The draft+ground+LP work runs asynchronously (background thread) so the diagnose
loop can proceed in parallel; callers join via ``collect_shadow_gpt_job`` before
finalists / the final arbiter.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from typing import Any

import numpy as np

SHADOW_CANDIDATE_ID = "pool_shadow_0"
SHADOW_SOURCE = "shadow_optimized_draft"
SHADOW_BRANCH = "in_distribution"

_log = logging.getLogger(__name__)

# job_id → {thread, event, result, error, model, started_at, finished_at}
_SHADOW_JOBS: dict[str, dict[str, Any]] = {}
_SHADOW_JOBS_LOCK = threading.Lock()


def is_shadow_candidate(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    cid = str(entry.get("candidate_id") or "")
    if cid.startswith("pool_shadow"):
        return True
    return entry.get("source") == SHADOW_SOURCE


def force_include_shadow(
    entries: list[dict[str, Any]],
    *,
    max_n: int,
) -> list[dict[str, Any]]:
    """Keep at most one shadow entry, always reserved in the finalist / arbiter set."""
    max_n = max(1, int(max_n))
    shadow = [e for e in entries if is_shadow_candidate(e)][:1]
    rest = [e for e in entries if not is_shadow_candidate(e)]
    room = max(0, max_n - len(shadow))
    return shadow + rest[:room]


def _snapshot_state_for_shadow(state: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields the shadow builder needs (safe for a background thread)."""
    keys = (
        "config",
        "user_request",
        "taste_text",
        "title",
        "problem",
        "requirement_tags",
        "candidate_pool",
        "interesting_candidates",
        "iteration",
    )
    snap: dict[str, Any] = {}
    for k in keys:
        if k not in state:
            continue
        try:
            snap[k] = copy.deepcopy(state.get(k))
        except Exception:
            snap[k] = state.get(k)
    return snap


def start_shadow_gpt_job(
    state: dict[str, Any],
    *,
    model: str | None = None,
    enabled: bool = True,
) -> str | None:
    """Kick off draft→ground→LP on a daemon thread. Returns job_id or None."""
    if not enabled:
        return None
    pool = list(state.get("candidate_pool") or [])
    interesting = list(state.get("interesting_candidates") or [])
    if any(is_shadow_candidate(e) for e in pool + interesting):
        return None
    if state.get("shadow_job_id"):
        return str(state["shadow_job_id"])

    from recipe_opt_agent.config import AgentConfig

    raw_cfg = state.get("config") or {}
    cfg = AgentConfig(**{k: v for k, v in raw_cfg.items() if k in AgentConfig.__dataclass_fields__})
    model = model or getattr(cfg, "shadow_draft_model", None) or "gpt-5.5"
    job_id = str(uuid.uuid4())
    snap = _snapshot_state_for_shadow(state)
    done = threading.Event()
    job: dict[str, Any] = {
        "event": done,
        "result": None,
        "error": None,
        "model": model,
        "started_at": time.time(),
        "finished_at": None,
    }

    def _run() -> None:
        try:
            entry = build_shadow_gpt_candidate(snap, model=model, enabled=True)
            job["result"] = entry
        except Exception as exc:
            job["error"] = str(exc)
            _log.warning("shadow gpt job %s failed: %s", job_id, exc)
        finally:
            job["finished_at"] = time.time()
            done.set()

    thread = threading.Thread(target=_run, name=f"shadow-gpt-{job_id[:8]}", daemon=True)
    job["thread"] = thread
    with _SHADOW_JOBS_LOCK:
        _SHADOW_JOBS[job_id] = job
    thread.start()
    _log.info("shadow gpt job %s started (model=%s, async)", job_id, model)
    return job_id


def collect_shadow_gpt_job(
    job_id: str | None,
    *,
    timeout: float | None = 180.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Wait for a shadow job; return (entry_or_None, backend_meta)."""
    meta: dict[str, Any] = {
        "job_id": job_id,
        "collected": False,
        "timed_out": False,
        "error": None,
        "model": None,
        "elapsed_s": None,
        "has_entry": False,
    }
    if not job_id:
        return None, meta
    with _SHADOW_JOBS_LOCK:
        job = _SHADOW_JOBS.get(job_id)
    if job is None:
        meta["error"] = "unknown_job"
        return None, meta
    meta["model"] = job.get("model")
    wait_s = None if timeout is None else max(0.0, float(timeout))
    finished = job["event"].wait(timeout=wait_s)
    if not finished:
        meta["timed_out"] = True
        meta["error"] = "timeout"
        _log.warning("shadow gpt job %s timed out after %ss", job_id, wait_s)
        return None, meta
    started = float(job.get("started_at") or time.time())
    finished_at = float(job.get("finished_at") or time.time())
    meta["elapsed_s"] = round(finished_at - started, 3)
    meta["collected"] = True
    meta["error"] = job.get("error")
    entry = job.get("result")
    meta["has_entry"] = isinstance(entry, dict) and bool(entry.get("candidate_id"))
    with _SHADOW_JOBS_LOCK:
        _SHADOW_JOBS.pop(job_id, None)
    _log.info(
        "shadow gpt job %s collected (has_entry=%s, elapsed_s=%s, model=%s)",
        job_id,
        meta["has_entry"],
        meta["elapsed_s"],
        meta["model"],
    )
    return (entry if meta["has_entry"] else None), meta


def merge_shadow_into_pools(
    state: dict[str, Any],
    *,
    timeout: float | None = 180.0,
) -> dict[str, Any]:
    """Join async shadow job (if any) and append the entry to pool / interesting."""
    job_id = state.get("shadow_job_id")
    pool = list(state.get("candidate_pool") or [])
    interesting = list(state.get("interesting_candidates") or [])
    if any(is_shadow_candidate(e) for e in pool + interesting):
        return {
            "candidate_pool": pool,
            "interesting_candidates": interesting,
            "shadow_job_id": None,
            "shadow_collect_meta": {"skipped": "already_present", "job_id": job_id},
        }
    entry, meta = collect_shadow_gpt_job(job_id, timeout=timeout)
    if entry:
        pool.append(entry)
        interesting.append(dict(entry))
    return {
        "candidate_pool": pool,
        "interesting_candidates": interesting[-24:],
        "shadow_job_id": None,
        "shadow_collect_meta": meta,
    }


def shadow_arbiter_consideration(
    *,
    candidates: list[dict[str, Any]],
    winner_id: str | None,
    collect_meta: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Backend-only audit: was the GPT-5.5 shadow entry in the arbiter set?"""
    shadow_ids = [str(c.get("candidate_id")) for c in candidates if is_shadow_candidate(c)]
    all_ids = [str(c.get("candidate_id")) for c in candidates if c.get("candidate_id")]
    shadow_models = sorted(
        {
            str(c.get("_shadow_model"))
            for c in candidates
            if is_shadow_candidate(c) and c.get("_shadow_model")
        }
    )
    return {
        "shadow_model": model or (shadow_models[0] if shadow_models else "gpt-5.5"),
        "shadow_models_seen": shadow_models,
        "async_collect": dict(collect_meta or {}),
        "shadow_in_arbiter_set": bool(shadow_ids),
        "shadow_candidate_ids": shadow_ids,
        "arbiter_candidate_ids": all_ids,
        "winner_is_shadow": bool(winner_id) and str(winner_id) in set(shadow_ids),
        "winner_id": winner_id,
    }


def optimize_grounded_problem(
    problem: dict[str, Any],
    box: dict[str, float],
    *,
    nutrition_slack_weight: float | None = 1.0,
) -> dict[str, Any]:
    """LP-optimize a grounded problem; return problem/opt/chosen with x_opt grams."""
    from weighted_empirical_opt import (
        MARGINAL_COLUMN_NODES,
        optimize_weighted_empirical_obj,
        pfc_fractions_from_portions,
        term_losses,
    )

    x0 = np.asarray(problem["x0"], dtype=float)
    M = np.asarray(problem["M"], dtype=float)
    basis_samples = {
        str(k): np.asarray(v, dtype=float) for k, v in (problem.get("basis_samples") or {}).items()
    }
    ratio_samples = np.asarray(problem.get("ratio_samples") or [], dtype=float)
    preferred = list(problem.get("marginal_nodes") or [])
    if preferred:
        hit = [n for n in preferred if n in basis_samples and len(basis_samples.get(n) or []) > 0]
        marginal = hit or [str(k) for k, v in basis_samples.items() if v is not None and len(v) > 0]
    else:
        marginal = [str(k) for k, v in basis_samples.items() if v is not None and len(v) > 0]
    if not marginal:
        marginal = [nid for _, nid in MARGINAL_COLUMN_NODES]
    ingredient_basis = list(problem.get("ingredient_basis") or [])
    for nid in ingredient_basis:
        if nid and str(nid) not in marginal:
            marginal.append(str(nid))
    weights_raw = problem.get("basis_sample_weights") or {}
    basis_sample_weights = (
        {str(k): np.asarray(v, dtype=float) for k, v in weights_raw.items()} if weights_raw else None
    )
    total_mass = float(problem.get("total_mass") or float(x0.sum()))
    kcal_target = float(problem.get("kcal_target") or 0.0) or None
    opt = optimize_weighted_empirical_obj(
        x0,
        M,
        marginal_nodes=marginal,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        ingredient_basis=ingredient_basis,
        kcal_target=kcal_target,
        protein_frac_min=box["protein_min"],
        protein_frac_max=box["protein_max"],
        carb_frac_min=box["carb_min"],
        carb_frac_max=box["carb_max"],
        fat_frac_min=box["fat_min"],
        fat_frac_max=box["fat_max"],
        total_mass=total_mass,
        basis_sample_weights=basis_sample_weights,
        nutrition_slack_weight=nutrition_slack_weight,
    )
    x_opt = np.asarray(opt["x_opt"], dtype=float)
    tl = term_losses(
        x_opt,
        marginal_nodes=marginal,
        basis_samples=basis_samples,
        ratio_samples=ratio_samples,
        total_mass=float(x_opt.sum()),
        ingredient_basis=ingredient_basis,
        basis_sample_weights=basis_sample_weights,
    )
    p, c, f = pfc_fractions_from_portions(x_opt, M)
    chosen = dict(problem.get("chosen_recipe") or {})
    ings = list(chosen.get("ingredients") or [])
    for i, row in enumerate(ings):
        if i < len(x_opt):
            row = dict(row)
            row["grams"] = float(x_opt[i])
            ings[i] = row
    chosen["ingredients"] = ings
    problem = dict(problem)
    problem["chosen_recipe"] = chosen
    problem["x_opt"] = x_opt.tolist()
    opt_pub = {
        "status": opt.get("status"),
        "objective": float(opt.get("objective") or 0.0),
        "nutrient_slack": float(opt.get("nutrient_slack") or 0.0),
        "feasible": bool(opt.get("feasible")),
        "x_opt": x_opt.tolist(),
        "term_losses": {k: float(v) for k, v in tl.items()},
        "pfc_after": {"protein": float(p), "carbs": float(c), "fat": float(f)},
    }
    return {"problem": problem, "opt": opt_pub, "chosen_recipe": chosen}


def _overlay_neighborhood_geometry(problem: dict[str, Any], stub: dict[str, Any]) -> dict[str, Any]:
    problem = dict(problem)
    if stub.get("basis_samples"):
        problem["basis_samples"] = stub["basis_samples"]
    if stub.get("basis_sample_weights"):
        problem["basis_sample_weights"] = stub["basis_sample_weights"]
    if stub.get("ratio_samples") is not None:
        problem["ratio_samples"] = stub["ratio_samples"]
    if stub.get("marginal_nodes"):
        problem["marginal_nodes"] = list(stub["marginal_nodes"])
    if stub.get("foodon_basis_report") and not problem.get("foodon_basis_report"):
        problem["foodon_basis_report"] = stub["foodon_basis_report"]
    if stub.get("neighborhood_hull_context") and not problem.get("neighborhood_hull_context"):
        problem["neighborhood_hull_context"] = stub["neighborhood_hull_context"]
    return problem


def _light_diagnosis(opt: dict[str, Any]) -> dict[str, Any]:
    """Compact diagnosis fields so finalist scoring / arbiter cards stay populated."""
    feasible = bool(opt.get("feasible"))
    tl = opt.get("term_losses") or {}
    share_vals = [float(v) for k, v in tl.items() if str(k).endswith("__share")]
    l_max = max(share_vals) if share_vals else float(opt.get("objective") or 0.0)
    band = "accept" if feasible else "must_retry"
    return {
        "diagnosis": "shadow_optimized",
        "meaning": "Optimized alternate draft",
        "L_max_norm": float(l_max),
        "L_total": float(opt.get("objective") or 0.0),
        "n_red": 0 if feasible else 1,
        "fidelity_band": band,
        "binding_macros": [],
        "retry_triggers": [],
    }


def build_shadow_gpt_candidate(
    state: dict[str, Any],
    *,
    model: str | None = None,
    enabled: bool = True,
) -> dict[str, Any] | None:
    """Draft with the shadow model, ground, LP-optimize, return a pool entry (or None)."""
    if not enabled:
        return None
    pool = list(state.get("candidate_pool") or [])
    interesting = list(state.get("interesting_candidates") or [])
    if any(is_shadow_candidate(e) for e in pool + interesting):
        return None

    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.grounding import ground_draft_to_problem
    from recipe_opt_agent.kcal_utils import resolve_kcal_target, restore_kcal_target
    from recipe_opt_agent.llm import llm_draft_recipe
    from recipe_opt_agent.requirement_tags import RequirementTag, deduce_requirement_tags

    raw_cfg = state.get("config") or {}
    cfg = AgentConfig(**{k: v for k, v in raw_cfg.items() if k in AgentConfig.__dataclass_fields__})
    model = model or getattr(cfg, "shadow_draft_model", None) or "gpt-5.5"
    request = str(state.get("user_request") or state.get("taste_text") or "").strip()
    title = str(state.get("title") or "").strip()
    if not request and not title:
        return None
    box = cfg.target_box_dict()
    stub = dict(state.get("problem") or {})
    kcal_target = resolve_kcal_target(cfg, stub)
    ctx = dict(stub.get("retrieval_context") or {})
    if stub.get("basis_nodes") and not ctx.get("basis_nodes"):
        ctx["basis_nodes"] = list(stub["basis_nodes"])
    if stub.get("rollup_chains") and not ctx.get("rollup_chains"):
        ctx["rollup_chains"] = stub["rollup_chains"]
    if stub.get("fdc_basis") and not ctx.get("fdc_basis"):
        ctx["fdc_basis"] = stub["fdc_basis"]
    nb_cat = list(ctx.get("fdc_catalog") or [])

    draft_temp: float | None = None if str(model).startswith("gpt-5") else 0.2
    example = stub.get("example_recipe") if isinstance(stub.get("example_recipe"), dict) else None
    try:
        draft, _trace = llm_draft_recipe(
            request or title,
            macro_box=box,
            example_recipe=example,
            model=model,
            temperature=draft_temp,
            canonical_title=title or None,
            kcal_target=kcal_target,
        )
    except Exception:
        return None
    if not (draft.get("ingredients") or []):
        return None

    tags_raw = state.get("requirement_tags") or []
    tags: list[RequirementTag] = []
    for r in tags_raw:
        if isinstance(r, RequirementTag):
            tags.append(r)
        elif isinstance(r, dict) and r.get("tag_id"):
            tags.append(
                RequirementTag(
                    tag_id=str(r.get("tag_id") or ""),
                    kind=str(r.get("kind") or "preference"),
                    polarity=str(r.get("polarity") or "require"),
                    source_text=str(r.get("source_text") or ""),
                )
            )
    if not tags:
        try:
            tags = deduce_requirement_tags(
                request or title,
                draft_tags=draft.get("requirement_tags"),
                force_llm=False,
            )
        except Exception:
            tags = []

    try:
        problem, _report, chosen = ground_draft_to_problem(
            draft,
            requirement_tags=tags,
            neighborhood_catalog=nb_cat,
            broader_catalog=nb_cat,
            basis_samples=stub.get("basis_samples"),
            ratio_samples=stub.get("ratio_samples"),
            retrieval_context=ctx,
            offline=bool(stub.get("grounding_offline") or stub.get("creative_offline")),
            use_dequant_cache=True,
        )
        problem = _overlay_neighborhood_geometry(problem, stub)
        problem = restore_kcal_target(problem, cfg, stub)
        scored = optimize_grounded_problem(
            problem,
            box,
            nutrition_slack_weight=cfg.nutrition_slack_weight,
        )
    except Exception:
        return None

    opt = scored["opt"]
    chosen = scored["chosen_recipe"]
    problem = scored["problem"]
    diag = _light_diagnosis(opt)
    foodon_report = (
        problem.get("foodon_basis_report")
        or stub.get("foodon_basis_report")
        or chosen.get("foodon_basis_report")
    )
    return {
        "candidate_id": SHADOW_CANDIDATE_ID,
        "iteration": int(state.get("iteration") or 0),
        "branch": SHADOW_BRANCH,
        "source": SHADOW_SOURCE,
        "objective": opt.get("objective"),
        "L_total": diag.get("L_total"),
        "L_max_norm": diag.get("L_max_norm"),
        "n_red": diag.get("n_red"),
        "x_opt": opt.get("x_opt"),
        "pfc_after": opt.get("pfc_after"),
        "diagnosis": diag.get("diagnosis"),
        "fidelity_band": diag.get("fidelity_band"),
        "ingredients": chosen.get("ingredients"),
        "foodon_basis_report": foodon_report,
        "opt": opt,
        "diagnosis_full": diag,
        "problem": {
            "x0": problem.get("x0"),
            "M": problem.get("M"),
            "ingredient_basis": problem.get("ingredient_basis"),
            "kcal_target": problem.get("kcal_target"),
            "chosen_recipe": chosen,
        },
        "_shadow_model": model,
    }
