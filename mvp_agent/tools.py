"""Strands @tool wrappers around the MVP pipeline phases."""

from __future__ import annotations

from typing import Any

from strands import tool

from mvp_corpus_cache import get_mvp_corpus
from mvp_data import parse_nlg_ingredients
from mvp_nutrient_fit import kcal_target_midpoint
from mvp_pipeline import encode_query, optimize_recipe
from mvp_recipe_judge import FinalPickCandidate, select_final_candidate
from mvp_recipe_ranker import EMBEDDING_MODEL, rank_recipes

from mvp_agent.context import Phase, get_active_session


def _ranked_to_dict(r) -> dict[str, Any]:
    return {
        "recipe_id": r.recipe_id,
        "recipe_name": r.recipe_name,
        "semantic_sim": r.semantic_sim,
        "semantic_score": r.semantic_score,
        "nutrient_fit": r.nutrient_fit,
        "nutrient_score": r.nutrient_score,
        "combined_score": r.combined_score,
        "rank": r.rank,
        "pfc_in_range": r.pfc_in_range,
        "kcal_target": r.kcal_target,
        "recipe_kcal": r.recipe_kcal,
    }


def _tool_result(ok: bool, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": ok, "message": message, **extra}


@tool
def embed_taste_query(taste_text: str) -> dict[str, Any]:
    """Encode the user's taste preference into an embedding vector (step 1)."""
    session = get_active_session()
    err = session.require_phase("embed_taste_query")
    if err:
        return _tool_result(False, err)

    session.emit(
        "embed_query",
        {
            "message": "Encoding taste preference…",
            "model": EMBEDDING_MODEL,
            "status": "running",
        },
    )
    session.query_emb = encode_query(taste_text)
    session.emit(
        "embed_query",
        {
            "message": "Taste embedding complete",
            "model": EMBEDDING_MODEL,
            "status": "done",
        },
        log=False,
    )
    session.advance("embed_taste_query")
    return _tool_result(True, "Taste query embedded", phase=session.phase.value)


@tool
def rank_recipes_by_fit() -> dict[str, Any]:
    """Rank MVP corpus recipes by semantic similarity and PFC nutrient fit (step 2)."""
    session = get_active_session()
    err = session.require_phase("rank_recipes_by_fit")
    if err:
        return _tool_result(False, err)
    if session.query_emb is None:
        return _tool_result(False, "Missing query embedding; call embed_taste_query first")

    query = session.query
    session.kcal_target = kcal_target_midpoint(query.kcal_min, query.kcal_max)
    if session.corpus is None:
        session.corpus = get_mvp_corpus()

    session.emit(
        "stage1_rank",
        {
            "message": "Ranking recipes by semantic + PFC fit…",
            "status": "running",
            "kcal_target": session.kcal_target,
        },
    )
    ranked = rank_recipes(
        session.corpus["recipe_ids"],
        session.corpus["recipe_names"],
        session.corpus["embeddings"],
        session.query_emb,
        session.corpus["nutrient_rows"],
        kcal_min=query.kcal_min,
        kcal_max=query.kcal_max,
        fat_frac_min=query.fat_frac_min,
        fat_frac_max=query.fat_frac_max,
        carb_frac_min=query.carb_frac_min,
        carb_frac_max=query.carb_frac_max,
        protein_frac_min=query.protein_frac_min,
        protein_frac_max=query.protein_frac_max,
        w_semantic=query.w_semantic,
        w_nutrient=query.w_nutrient,
    )
    session.ranked = ranked
    session.emit(
        "stage1_rank",
        {
            "message": "Ranking complete",
            "status": "done",
            "n_recipes": len(ranked),
            "kcal_target": session.kcal_target,
            "top_20": [_ranked_to_dict(r) for r in ranked[:20]],
        },
        log=False,
    )
    session.advance("rank_recipes_by_fit")
    return _tool_result(
        True,
        f"Ranked {len(ranked)} recipes",
        n_recipes=len(ranked),
        phase=session.phase.value,
    )


@tool
def optimize_top_candidates(top_k: int | None = None) -> dict[str, Any]:
    """Optimize macro portions for top-K ranked recipes (step 3)."""
    session = get_active_session()
    err = session.require_phase("optimize_top_candidates")
    if err:
        return _tool_result(False, err)
    if not session.ranked:
        return _tool_result(False, "No ranked recipes; call rank_recipes_by_fit first")

    query = session.query
    k = top_k if top_k is not None else query.top_k
    top_ids = [r.recipe_id for r in session.ranked[:k]]
    session.emit(
        "optimize",
        {
            "message": f"Optimizing top {len(top_ids)} recipes…",
            "status": "running",
            "kcal_target": session.kcal_target,
            "total": len(top_ids),
            "completed": 0,
            "candidates": [],
        },
    )
    optimized: list[dict[str, Any]] = []
    for idx, rid in enumerate(top_ids):
        candidate = optimize_recipe(rid, query, corpus=session.corpus)
        optimized.append(candidate)
        session.emit(
            "optimize_progress",
            {
                "index": idx + 1,
                "total": len(top_ids),
                "candidate": candidate,
                "status": "running",
            },
        )
    session.optimized = optimized
    n_infeasible = sum(1 for o in optimized if not o.get("macro_feasible", True))
    n_fallback = sum(1 for o in optimized if o.get("used_fallback"))
    n_already = sum(1 for o in optimized if o.get("already_feasible"))
    session.emit(
        "optimize",
        {
            "message": "Optimization complete",
            "status": "done",
            "kcal_target": session.kcal_target,
            "candidates": optimized,
            "n_infeasible": n_infeasible,
            "n_fallback": n_fallback,
            "n_already_feasible": n_already,
            "completed": len(optimized),
            "total": len(top_ids),
            "infeasible_note": (
                f"{n_infeasible} of {len(optimized)} recipes cannot reach PFC targets "
                f"at {session.kcal_target:.0f} kcal."
                if n_infeasible
                else None
            ),
        },
        log=False,
    )
    session.advance("optimize_top_candidates")
    return _tool_result(
        True,
        f"Optimized {len(optimized)} candidates",
        n_candidates=len(optimized),
        phase=session.phase.value,
    )


@tool
def judge_final_recipe() -> dict[str, Any]:
    """LLM judge picks the best candidate for taste fit (step 4; OpenAI)."""
    session = get_active_session()
    err = session.require_phase("judge_final_recipe")
    if err:
        return _tool_result(False, err)
    if not session.optimized:
        return _tool_result(False, "No optimized candidates; call optimize_top_candidates first")

    session.emit(
        "judge",
        {"message": "Reranking for taste fit…", "status": "running"},
    )
    rank_by_id = {r.recipe_id: r.rank for r in session.ranked}
    sim_by_id = {r.recipe_id: r.semantic_sim for r in session.ranked}
    corpus = session.corpus or get_mvp_corpus()
    pick_candidates: list[FinalPickCandidate] = []
    for opt in session.optimized:
        rid = opt["recipe_id"]
        feat = corpus["features"].get(rid, {})
        semantic_text = feat.get("semantic_text", "")
        nlg_ingredients = feat.get("nlg_ingredients") or parse_nlg_ingredients(
            semantic_text
        )
        ing_names = ", ".join(i["ingredient"] for i in opt["ingredients"])
        pick_candidates.append(
            FinalPickCandidate(
                recipe_id=rid,
                title_clean=feat.get("title_clean", f"Recipe {rid}"),
                semantic_text=semantic_text,
                nlg_ingredients=nlg_ingredients,
                ingredient_summary=ing_names,
                semantic_sim=float(sim_by_id.get(rid, 0.0)),
                portion_score=float(opt["portion_score"]),
                avg_pct_change=float(opt.get("avg_pct_change", 0.0)),
                macro_feasible=bool(opt.get("macro_feasible", True)),
                used_fallback=bool(opt.get("used_fallback", False)),
                already_feasible=bool(opt.get("already_feasible", False)),
                stage1_rank=int(rank_by_id.get(rid, 999)),
            )
        )
    session.pick_candidates = pick_candidates
    judge_result = select_final_candidate(session.query.taste_text, pick_candidates)
    session.judge_result = judge_result

    chosen_id = judge_result.chosen_recipe_id
    chosen_feat = corpus["features"].get(chosen_id, {})
    chosen_name = chosen_feat.get("title_clean") or f"Recipe {chosen_id}"
    session.emit(
        "judge",
        {
            "message": f"Selected {chosen_name}",
            "status": "done",
            "chosen_recipe_id": chosen_id,
            "rationale_preview": judge_result.rationale[:200],
        },
        log=False,
    )
    session.advance("judge_final_recipe")
    return _tool_result(
        True,
        f"Selected recipe {chosen_id}",
        chosen_recipe_id=chosen_id,
        phase=session.phase.value,
    )


@tool
def finalize_recommendation() -> dict[str, Any]:
    """Assemble the final recommendation payload for the UI (step 5)."""
    session = get_active_session()
    err = session.require_phase("finalize_recommendation")
    if err:
        return _tool_result(False, err)
    if session.judge_result is None or not session.optimized:
        return _tool_result(False, "Missing judge result; call judge_final_recipe first")

    judge_result = session.judge_result
    chosen_id = judge_result.chosen_recipe_id
    chosen_opt = next(o for o in session.optimized if o["recipe_id"] == chosen_id)
    corpus = session.corpus or get_mvp_corpus()
    chosen_feat = corpus["features"].get(chosen_id, {})

    session.emit(
        "finalize",
        {"message": "Assembling final recommendation…", "status": "running"},
    )

    if chosen_opt.get("already_feasible"):
        opt_note = chosen_opt.get("feasibility_message")
    elif chosen_opt.get("used_fallback"):
        opt_note = chosen_opt.get("feasibility_message")
    else:
        opt_note = None

    final_payload = {
        "run_id": session.run_id,
        "chosen_recipe_id": chosen_id,
        "recipe_name": chosen_feat.get("title_clean", f"Recipe {chosen_id}"),
        "semantic_text": chosen_feat.get("semantic_text", ""),
        "ingredients": chosen_opt["ingredients"],
        "macros_before": chosen_opt["macros_before"],
        "macros_after": chosen_opt["macros_after"],
        "portion_score": chosen_opt["portion_score"],
        "avg_pct_change": chosen_opt.get("avg_pct_change", 0.0),
        "max_pct_change": chosen_opt.get("max_pct_change", 0.0),
        "kcal_target": session.kcal_target,
        "macro_feasible": chosen_opt.get("macro_feasible", True),
        "already_feasible": chosen_opt.get("already_feasible", False),
        "feasibility_message": chosen_opt.get("feasibility_message"),
        "used_fallback": chosen_opt.get("used_fallback", False),
        "optimization_note": opt_note,
        "judge": {
            "rationale": judge_result.rationale,
            "portion_summary": judge_result.portion_summary,
            "runner_up_notes": judge_result.runner_up_notes,
        },
        "stage1_top_20": [_ranked_to_dict(r) for r in session.ranked[:20]],
        "optimizer_candidates": session.optimized,
    }
    session.final_payload = final_payload
    session.emit(
        "format_result",
        {**final_payload, "status": "done"},
    )
    session.emit("done", {"status": "done"}, log=False)
    session.advance("finalize_recommendation")

    from mvp_log import finish_run

    if session.log_to_db and session.run_id:
        try:
            finish_run(session.run_id, status="done", chosen_recipe_id=chosen_id)
        except Exception:
            pass

    return _tool_result(
        True,
        f"Recommendation ready: {final_payload['recipe_name']}",
        chosen_recipe_id=chosen_id,
        phase=session.phase.value,
    )


TOOL_REGISTRY: dict[str, Any] = {
    "embed_taste_query": embed_taste_query,
    "rank_recipes_by_fit": rank_recipes_by_fit,
    "optimize_top_candidates": optimize_top_candidates,
    "judge_final_recipe": judge_final_recipe,
    "finalize_recommendation": finalize_recommendation,
}

ALL_TOOLS = list(TOOL_REGISTRY.values())
