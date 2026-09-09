"""FastAPI server for recipe optimization agent playground."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from db import load_dotenv

    load_dotenv()
except Exception:
    pass

from recipe_opt_agent.config import AgentConfig, fast_demo_from_env
from recipe_opt_agent.creative_loader import load_creative_problem
from recipe_opt_agent.graph import (
    CREATIVE_FLOW_EDGES,
    CREATIVE_FLOW_NODES,
    FLOW_COMPUTE_KINDS,
    FLOW_EDGES,
    FLOW_NODE_DOCS,
    FLOW_NODES,
)
from recipe_opt_agent.problem_loader import (
    count_canonical_dishes,
    list_canonical_dishes,
    load_canonical_problem,
    search_canonical_dishes,
)
from recipe_opt_agent.runner import run_recipe_opt_agent
from recipe_opt_agent.runtime_warm import warm_runtime_caches, warm_status

app = FastAPI(title="MacroIQ", version="0.2.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _warm_foodon_caches() -> None:
    """Load FoodOn + MiniLM + dequant once so first request isn't cold."""
    warm_runtime_caches()


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


class RunRequest(BaseModel):
    mode: str = Field(default="neighborhood", description="neighborhood | creative")
    user_request: str = Field(default="", description="Optional free-text request / taste notes")
    taste_text: str = Field(default="", description="Optional; defaults to selected recipe title")
    title: str = Field(default="")
    canonical_id: int | None = Field(default=None, description="Canonical dish id (required for MacroIQ)")
    start_metric: str = Field(
        default="l1_pfc",
        description="Neighborhood start pick: l1_pfc | loss_projection",
    )
    use_macro_targets: bool = Field(
        default=True,
        description="When false, ignore PFC box and minimize ratio/empirical loss only",
    )
    kcal_target: float | None = Field(
        default=None,
        ge=100,
        le=8000,
        description="Optional calorie target; scales total mass to match",
    )
    protein_min: float = Field(default=0.19, ge=0, le=1)
    protein_max: float = Field(default=0.23, ge=0, le=1)
    carb_min: float = Field(default=0.345, ge=0, le=1)
    carb_max: float = Field(default=0.545, ge=0, le=1)
    fat_min: float = Field(default=0.245, ge=0, le=1)
    fat_max: float = Field(default=0.445, ge=0, le=1)
    F_accept: float = Field(default=1.0, gt=0)
    F_max: float = Field(default=1.5, gt=0)
    max_iterations: int = Field(
        default=2 if fast_demo_from_env() else 3,
        ge=1,
        le=10,
    )


def _apply_kcal_target(problem: dict, kcal_target: float | None) -> dict:
    """Override problem kcal (and scale mass/x0) when the user sets a calorie target."""
    from recipe_opt_agent.kcal_utils import apply_kcal_target

    return apply_kcal_target(problem, kcal_target)


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"


def _cache_busted_html(filename: str, asset_names: tuple[str, ...]) -> HTMLResponse:
    """Serve HTML with mtime query params on linked static assets."""
    html_path = STATIC_DIR / filename
    html = html_path.read_text(encoding="utf-8")
    for name in asset_names:
        path = STATIC_DIR / name
        ver = int(path.stat().st_mtime) if path.is_file() else 0
        html = html.replace(f"/static/{name}", f"/static/{name}?v={ver}")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
def index():
    """MacroIQ product UI (semantic ask + macros + canonical menu + live progress)."""
    return _cache_busted_html("macroiq.html", ("macroiq.css", "macroiq.js"))


@app.get("/playground")
def playground():
    """Developer playground: flow graph, transcript, and node inspector."""
    return _cache_busted_html("index.html", ("styles.css", "app.js"))


@app.get("/loop-demo")
def loop_demo():
    """Lightweight presentation demo of the online agent loop (separate from playground)."""
    return _cache_busted_html(
        "loop_demo.html",
        ("loop_demo.css", "loop_demo.js"),
    )


@app.get("/animated-demo")
def animated_demo():
    """Timed ~20s graph walkthrough of the agent loop (no LLM / API calls)."""
    return _cache_busted_html(
        "animated_demo.html",
        ("animated_demo.css", "animated_demo.js"),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY")),
        "fast_demo": fast_demo_from_env(),
        "warm": warm_status(),
        "flow_nodes": list(FLOW_NODES),
        "mode": "live_canonical_end_to_end",
    }


class FlowSummaryRequest(BaseModel):
    mode: str = Field(default="neighborhood")
    user_request: str = Field(default="")
    title: str = Field(default="")
    final: dict = Field(default_factory=dict)
    steps: list[dict] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    decision_outcomes: list[dict] = Field(default_factory=list)
    llm_calls: list[dict] = Field(default_factory=list)
    run_telemetry: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str = Field(description="user | assistant")
    content: str = Field(default="")


class RunChatRequest(FlowSummaryRequest):
    messages: list[ChatMessage] = Field(default_factory=list)


def _compact_for_summary(payload: dict, *, max_chars: int = 48000) -> str:
    """Serialize run briefing; truncate if huge so gpt-4o stays focused."""
    text = json.dumps(payload, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80] + "\n… [truncated for summary prompt]"


def _build_run_briefing(req: FlowSummaryRequest) -> dict:
    return {
        "mode": req.mode,
        "user_request": req.user_request,
        "title": req.title,
        "final_status": (req.final or {}).get("status"),
        "final": req.final,
        "path_finals": (req.final or {}).get("path_finals"),
        "display_scores": (req.final or {}).get("display_scores"),
        "final_judgment": (req.final or {}).get("final_judgment"),
        "final_evaluation": (req.final or {}).get("final_evaluation"),
        "neighborhood_hull_context": ((req.final or {}).get("problem") or {}).get(
            "neighborhood_hull_context"
        ),
        "run_telemetry": req.run_telemetry or (req.final or {}).get("run_telemetry"),
        "history": req.history or (req.final or {}).get("history"),
        "decision_outcomes": req.decision_outcomes or (req.final or {}).get("decision_outcomes"),
        "steps": req.steps,
        "llm_calls": req.llm_calls,
        "logical_flow_note": (
            "Neighborhood edges: propose always goes to decide (never directly to finalize). "
            "finalize is reached from diagnose (already accept / max iters), decide (accept), "
            "or apply (after accept)."
        ),
    }


@app.post("/api/flow_summary")
def flow_summary(req: FlowSummaryRequest):
    """UI-only holistic run review with gpt-4o (OPENAI_API_KEY only)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY is required for flow summary")

    from recipe_opt_agent.observability import get_openai_client

    briefing = _build_run_briefing(req)
    system = (
        "You are reviewing a finished recipe-optimization agent run for a human designer. "
        "Be concrete and brief. Use markdown with these sections:\n"
        "1. Verdict — did the run meet the user query? (success / partial / failed) + one sentence why\n"
        "2. Step-by-step — for each graph step in order: what happened (1–2 sentences); "
        "if there was an LLM call, contextualize its intermediate output (what it chose and why it mattered)\n"
        "3. Tradeoffs — ratio fidelity vs nutrient fit vs dish similarity\n"
        "4. Reasoning improvements — specific prompt/policy/auto-apply/retrieval changes worth trying\n"
        "Do not invent tools or nodes that are not in the briefing. Do not dump raw JSON."
    )
    user = (
        f"User query / taste:\n{req.user_request or req.title or '(none)'}\n\n"
        f"Run briefing:\n{_compact_for_summary(briefing)}"
    )
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
    except Exception as exc:
        raise HTTPException(502, f"Flow summary LLM failed: {exc}") from exc
    return {
        "model": "gpt-4o",
        "summary_markdown": content,
        "usage": usage,
    }


@app.post("/api/run_chat")
def run_chat(req: RunChatRequest):
    """Interactive Q&A about a finished run with gpt-4o (OPENAI_API_KEY only)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY is required for run chat")

    if not req.messages:
        raise HTTPException(400, "messages must include at least one user message")

    from recipe_opt_agent.observability import get_openai_client

    briefing = _build_run_briefing(req)
    system = (
        "You answer specific questions about a finished recipe-optimization agent run. "
        "You have a JSON briefing with steps, decisions, telemetry, final recipe(s), "
        "path_finals (in-distribution vs OOD champions), scores, final_judgment, "
        "final_evaluation (GPT-4o holistic review with dietary_violation_flag), "
        "neighborhood_hull_context (how far macros sit outside neighbor hulls), and LLM traces.\n\n"
        "Rules:\n"
        "- Ground answers in the briefing only; say when data is missing.\n"
        "- Be concise and direct; use markdown for lists or emphasis when helpful.\n"
        "- Quote concrete numbers (ratio loss, nutrient slack, ΔL*, ingredient grams) when relevant.\n"
        "- Do not invent tools, nodes, or outcomes not present in the briefing.\n"
        "- If asked to compare ID vs OOD paths, use path_finals when available.\n\n"
        f"Run briefing:\n{_compact_for_summary(briefing)}"
    )
    chat_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for msg in req.messages:
        role = (msg.role or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = (msg.content or "").strip()
        if not content:
            continue
        chat_messages.append({"role": role, "content": content})

    if not any(m["role"] == "user" for m in chat_messages):
        raise HTTPException(400, "messages must include at least one user message")

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.25,
            messages=chat_messages,
        )
        content = (resp.choices[0].message.content or "").strip()
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
    except Exception as exc:
        raise HTTPException(502, f"Run chat LLM failed: {exc}") from exc

    return {
        "model": "gpt-4o",
        "reply_markdown": content,
        "usage": usage,
    }


@app.get("/api/flow")
def flow_meta(mode: str = Query(default="neighborhood")):
    creative = mode == "creative"
    nodes = list(CREATIVE_FLOW_NODES if creative else FLOW_NODES)
    edges = list(CREATIVE_FLOW_EDGES if creative else FLOW_EDGES)
    # Docs enriched per-mode with tools (+ prompt templates), incoming/outgoing.
    from recipe_opt_agent.prompts import prompts_for_tool

    docs = {}
    for name, doc in FLOW_NODE_DOCS.items():
        tools = []
        for tool in doc.get("tools") or []:
            t = dict(tool)
            t.setdefault("detail", t.get("purpose") or "")
            t["prompts"] = prompts_for_tool(str(t.get("name") or ""))
            tools.append(t)
        docs[name] = {
            **doc,
            "compute": str(doc.get("compute") or "deterministic"),
            "tools": tools,
            "incoming": [a for a, b in edges if b == name],
            "outgoing": [b for a, b in edges if a == name],
        }
    return {
        "nodes": nodes,
        "edges": [{"from": a, "to": b} for a, b in edges],
        "docs": docs,
        "compute_kinds": FLOW_COMPUTE_KINDS,
        "mode": mode,
        "optimizer_note": (
            "The weighted empirical optimizer (optimize_weighted_empirical_obj) runs inside the "
            "diagnose node, not as its own LangGraph step. diagnose = hull geometry + LP solve + fidelity bands."
        ),
    }


@app.get("/api/canonicals")
def canonicals(
    limit: int | None = Query(default=None, ge=0, le=10000),
    min_neighborhood: int = Query(default=5, ge=1, le=100),
    q: str | None = Query(default=None, description="Optional title filter (ILIKE)"),
    count_only: bool = Query(default=False),
):
    """List canonical dishes. No default limit — returns the full catalog unless ``limit`` is set."""
    try:
        if count_only:
            total = count_canonical_dishes(min_neighborhood=min_neighborhood, q=q)
            return {"count": total, "min_neighborhood": min_neighborhood, "q": q or ""}
        effective_limit = None if limit in (None, 0) else int(limit)
        dishes = list_canonical_dishes(
            limit=effective_limit,
            min_neighborhood=min_neighborhood,
            q=q,
        )
        total = count_canonical_dishes(min_neighborhood=min_neighborhood, q=q)
    except Exception as exc:
        raise HTTPException(503, f"Could not list canonical dishes (DB?): {exc}") from exc
    return {
        "dishes": dishes,
        "count": len(dishes),
        "total": total,
        "limited": effective_limit is not None,
        "min_neighborhood": min_neighborhood,
        "q": q or "",
    }


@app.get("/api/canonicals/search")
def canonicals_search(
    q: str = Query(default="", description="Title search string"),
    limit: int = Query(default=40, ge=1, le=200),
    min_neighborhood: int = Query(default=5, ge=1, le=100),
):
    """Search the full canonical catalog by title (preferred UI entry point)."""
    try:
        dishes = search_canonical_dishes(q, min_neighborhood=min_neighborhood, limit=limit)
        total = count_canonical_dishes(min_neighborhood=min_neighborhood, q=q or None)
    except Exception as exc:
        raise HTTPException(503, f"Could not search canonical dishes (DB?): {exc}") from exc
    return {
        "dishes": dishes,
        "count": len(dishes),
        "total": total,
        "q": q,
        "min_neighborhood": min_neighborhood,
    }


class MacroTargetSuggestRequest(BaseModel):
    canonical_id: int = Field(..., description="Canonical dish whose neighborhood hulls to probe")
    protein_half: float = Field(default=0.02, ge=0.005, le=0.15)
    carb_half: float = Field(default=0.05, ge=0.005, le=0.25)
    fat_half: float = Field(default=0.05, ge=0.005, le=0.25)


@app.post("/api/macro_targets/suggest")
def macro_targets_suggest(req: MacroTargetSuggestRequest):
    """Suggest a neighborhood coverage macro box (≤10pp bands covering most recipes)."""
    from recipe_opt_agent.macro_target_suggestions import suggest_macro_targets_for_canonical

    try:
        result = suggest_macro_targets_for_canonical(
            int(req.canonical_id),
            half_widths={
                "protein": float(req.protein_half),
                "carb": float(req.carb_half),
                "fat": float(req.fat_half),
            },
            fast_neighborhood=True,
            # Prefer cache; live rebuild on miss so dish selection still works.
            require_cache=False,
        )
    except Exception as exc:
        raise HTTPException(503, f"Could not suggest macro targets: {exc}") from exc
    if result.get("error"):
        raise HTTPException(404, result["error"])
    return result


def validate_macro_box(
    protein_min: float,
    protein_max: float,
    carb_min: float,
    carb_max: float,
    fat_min: float,
    fat_max: float,
) -> list[str]:
    """Check the macro box admits a P+C+F=100% point (nonempty simplex intersection)."""
    errors: list[str] = []
    for name, lo, hi in (
        ("protein", protein_min, protein_max),
        ("carb", carb_min, carb_max),
        ("fat", fat_min, fat_max),
    ):
        if lo > hi:
            errors.append(f"{name} min ({lo:.0%}) exceeds {name} max ({hi:.0%})")
    sum_min = protein_min + carb_min + fat_min
    sum_max = protein_max + carb_max + fat_max
    if sum_max < 1.0 - 1e-9:
        errors.append(
            f"macro maxes sum to {sum_max:.0%} — they must sum to at least 100% "
            "so protein+carbs+fat can reach 100%"
        )
    if sum_min > 1.0 + 1e-9:
        errors.append(
            f"macro mins sum to {sum_min:.0%} — they must sum to at most 100% "
            "so protein+carbs+fat can total 100%"
        )
    return errors


@app.post("/api/run")
def run(req: RunRequest):
    use_macros = bool(req.use_macro_targets)
    if use_macros:
        box_errors = validate_macro_box(
            req.protein_min,
            req.protein_max,
            req.carb_min,
            req.carb_max,
            req.fat_min,
            req.fat_max,
        )
        if box_errors:
            raise HTTPException(400, "Infeasible macro targets: " + "; ".join(box_errors))
        protein_min, protein_max = req.protein_min, req.protein_max
        carb_min, carb_max = req.carb_min, req.carb_max
        fat_min, fat_max = req.fat_min, req.fat_max
        nutrition_slack_weight: float | None = 1.0
    else:
        # Unconstrained PFC: keep fractions in the simplex but do not penalize
        # deviation from a user box — minimize empirical / ratio loss only.
        protein_min, protein_max = 0.0, 1.0
        carb_min, carb_max = 0.0, 1.0
        fat_min, fat_max = 0.0, 1.0
        nutrition_slack_weight = 0.0
    cfg = AgentConfig.for_request(
        protein_min=protein_min,
        protein_max=protein_max,
        carb_min=carb_min,
        carb_max=carb_max,
        fat_min=fat_min,
        fat_max=fat_max,
        F_accept=req.F_accept,
        F_max=req.F_max,
        max_iterations=req.max_iterations,
        nutrition_slack_weight=nutrition_slack_weight,
        kcal_target=float(req.kcal_target) if req.kcal_target is not None else None,
    )

    def event_stream():
        q: queue.Queue = queue.Queue()
        run_id_holder: dict[str, str | None] = {"id": None}

        def emit(etype: str, data: dict) -> None:
            q.put(_sse_event(etype, data))

        def on_event(ev: dict) -> None:
            etype = ev.get("type", "step")
            emit(etype, ev)

        def worker() -> None:
            from recipe_opt_web.run_log import finish_macroiq_run, start_macroiq_run

            mode = (req.mode or "neighborhood").strip().lower()
            request_snapshot = {
                "mode": mode,
                "canonical_id": req.canonical_id,
                "title": req.title,
                "taste_text": req.taste_text,
                "user_request": req.user_request,
                "kcal_target": req.kcal_target,
                "use_macro_targets": use_macros,
                "start_metric": req.start_metric,
                "protein_min": req.protein_min,
                "protein_max": req.protein_max,
                "carb_min": req.carb_min,
                "carb_max": req.carb_max,
                "fat_min": req.fat_min,
                "fat_max": req.fat_max,
                "F_accept": req.F_accept,
                "F_max": req.F_max,
                "max_iterations": req.max_iterations,
            }
            run_id_holder["id"] = start_macroiq_run(
                mode=mode,
                request=request_snapshot,
                config={
                    "protein_min": protein_min,
                    "protein_max": protein_max,
                    "carb_min": carb_min,
                    "carb_max": carb_max,
                    "fat_min": fat_min,
                    "fat_max": fat_max,
                    "kcal_target": cfg.kcal_target,
                    "nutrition_slack_weight": nutrition_slack_weight,
                    "max_iterations": cfg.max_iterations,
                },
            )
            try:
                creative = mode == "creative"
                if not creative and req.canonical_id is None:
                    raise ValueError("canonical_id is required for neighborhood mode")

                if creative:
                    emit(
                        "load",
                        {
                            "type": "load",
                            "phase": "creative_start",
                            "message": (
                                "Creative mode: loading neighborhood context "
                                "(start recipe uses nutrition proximity when a family is selected)…"
                            ),
                            "user_request": req.user_request,
                        },
                    )
                    problem = load_creative_problem(
                        user_request=req.user_request or req.taste_text,
                        canonical_id=req.canonical_id,
                        protein_min=protein_min,
                        protein_max=protein_max,
                        carb_min=carb_min,
                        carb_max=carb_max,
                        fat_min=fat_min,
                        fat_max=fat_max,
                        offline=req.canonical_id is None,
                        # FoodOn cache is only for the neighbor set; start pick is L1 PFC.
                        require_cache=False,
                    )
                    problem = _apply_kcal_target(problem, req.kcal_target)
                    title = req.title or problem.get("title") or "Creative Recipe"
                    taste_text = (req.user_request or req.taste_text or "").strip() or title
                    cache_hit = bool(problem.get("neighborhood_from_cache"))
                    if req.canonical_id is None:
                        nb_note = " · offline stub (no dish selected; neighborhood cache N/A)"
                        nb_flag = None
                    elif cache_hit:
                        nb_note = " · neighborhood set from FoodOn cache"
                        nb_flag = True
                    else:
                        nb_note = " · neighborhood set rebuilt live"
                        nb_flag = False
                    emit(
                        "load",
                        {
                            "type": "load",
                            "phase": "agent_start",
                            "message": (
                                "Starting creative LangGraph (draft → ground → diagnose loop → finalists)…"
                                + nb_note
                            ),
                            "neighborhood_from_cache": nb_flag,
                            "dequant_cache_expected": req.canonical_id is not None,
                        },
                    )
                    from recipe_opt_agent.observability import run_config as lg_run_config

                    lg_cfg = lg_run_config(
                        case_name="web_ui",
                        agent_mode="creative",
                        config=cfg,
                        user_request=req.user_request or taste_text,
                        taste_text=taste_text,
                        title=title,
                        canonical_id=int(req.canonical_id or 0) if req.canonical_id else None,
                        problem=problem,
                        tags=["web_ui"],
                    )
                    result = run_recipe_opt_agent(
                        problem=problem,
                        taste_text=taste_text,
                        title=title,
                        canonical_id=int(req.canonical_id or 0),
                        config=cfg,
                        on_event=on_event,
                        agent_mode="creative",
                        user_request=req.user_request or taste_text,
                        langgraph_config=lg_cfg,
                    )
                    finish_macroiq_run(
                        run_id_holder["id"],
                        status="done",
                        final=result if isinstance(result, dict) else None,
                        mode="creative",
                    )
                    emit("result", {"type": "result", "final": result, "mode": "creative"})
                    return

                emit(
                    "load",
                    {
                        "type": "load",
                        "phase": "build_neighborhood",
                        "message": (
                            f"Loading FoodOn neighborhood for canonical_id={req.canonical_id} "
                            f"(start recipe = nutrition proximity / {req.start_metric or 'l1_pfc'})…"
                        ),
                        "canonical_id": req.canonical_id,
                        "start_metric": req.start_metric,
                    },
                )
                problem = load_canonical_problem(
                    int(req.canonical_id),
                    protein_min=protein_min,
                    protein_max=protein_max,
                    carb_min=carb_min,
                    carb_max=carb_max,
                    fat_min=fat_min,
                    fat_max=fat_max,
                    # Always pick the starting NLG recipe by nutrition proximity
                    # (L1 PFC to the macro box), not by FoodOn Jaccard hit-count.
                    prefer_nutrition_start=True,
                    start_metric=req.start_metric or "l1_pfc",
                    fast_neighborhood=True,
                    # FoodOn Jaccard cache is only for the neighbor *set*; rebuild on miss.
                    require_cache=False,
                )
                problem = _apply_kcal_target(problem, req.kcal_target)
                title = req.title or problem.get("title") or f"canonical {req.canonical_id}"
                taste_text = (req.taste_text or req.user_request or "").strip() or problem.get("taste_text") or title
                chosen = problem.get("chosen_recipe") or {}
                sel = (chosen.get("selection") or {}) if isinstance(chosen, dict) else {}
                metric_note = sel.get("method") or sel.get("selection_mode") or req.start_metric
                fallback = sel.get("fallback_reason")
                from_cache = bool(
                    problem.get("neighborhood_from_cache")
                    or sel.get("neighborhood_from_cache")
                )
                cache_note = (
                    "neighborhood set from FoodOn cache"
                    if from_cache
                    else "neighborhood set rebuilt live"
                )
                macro_note = (
                    " · macro targets on"
                    if use_macros
                    else " · ratio-loss only (no macro targets)"
                )
                kcal_note = (
                    f" · {int(req.kcal_target)} kcal target"
                    if req.kcal_target is not None
                    else ""
                )
                emit(
                    "load",
                    {
                        "type": "load",
                        "phase": "selected_start",
                        "message": (
                            f"Selected starting recipe {chosen.get('recipe_nlg_id')} "
                            f"by nutrition proximity ({metric_note})"
                            + (f" (fallback: {fallback})" if fallback else "")
                            + f" · {cache_note}{macro_note}{kcal_note} · "
                            f"{len(chosen.get('ingredients') or [])} ingredients · "
                            f"{len(problem.get('neighborhood_recipes') or [])} neighborhood recipes."
                        ),
                        "chosen_recipe": chosen,
                        "n_neighborhood": len(problem.get("neighborhood_recipes") or []),
                        "taste_text": taste_text,
                        "start_metric": req.start_metric,
                        "selection": sel,
                        "neighborhood_from_cache": from_cache,
                        "use_macro_targets": use_macros,
                        "kcal_target": req.kcal_target,
                    },
                )
                emit(
                    "load",
                    {
                        "type": "load",
                        "phase": "agent_start",
                        "message": "Neighborhood ready — starting LangGraph agent (diagnose → propose retrieves candidates live)…",
                    },
                )
                from recipe_opt_agent.observability import run_config as lg_run_config

                lg_cfg = lg_run_config(
                    case_name="web_ui",
                    agent_mode="neighborhood",
                    config=cfg,
                    user_request=taste_text,
                    taste_text=taste_text,
                    title=title,
                    canonical_id=int(req.canonical_id),
                    problem=problem,
                    tags=["web_ui"],
                )
                result = run_recipe_opt_agent(
                    problem=problem,
                    taste_text=taste_text,
                    title=title,
                    canonical_id=int(req.canonical_id),
                    config=cfg,
                    on_event=on_event,
                    agent_mode="neighborhood",
                    langgraph_config=lg_cfg,
                )
                finish_macroiq_run(
                    run_id_holder["id"],
                    status="done",
                    final=result if isinstance(result, dict) else None,
                    mode="neighborhood",
                )
                emit("result", {"type": "result", "final": result, "mode": "neighborhood"})
            except Exception as exc:
                finish_macroiq_run(
                    run_id_holder["id"],
                    status="error",
                    error_message=str(exc),
                    mode=(req.mode or "neighborhood").strip().lower(),
                )
                emit("error", {"type": "error", "error": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                break
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


class RecomputeRequest(BaseModel):
    problem: dict = Field(default_factory=dict)
    ingredients: list[dict] = Field(default_factory=list)
    grams: list[float] = Field(default_factory=list)
    macro_targets: dict = Field(default_factory=dict)
    score_history: list[dict] = Field(default_factory=list)
    baseline_ratio: float | None = Field(default=None)


@app.post("/api/recipe/recompute")
def recipe_recompute(req: RecomputeRequest):
    """Recompute macros / cookability after an interactive gram edit."""
    from recipe_opt_agent.score_display import recompute_recipe_at_grams

    if not req.grams:
        raise HTTPException(400, "grams is required")
    try:
        return recompute_recipe_at_grams(
            problem=req.problem,
            ingredients=req.ingredients,
            grams=[float(g) for g in req.grams],
            macro_targets=req.macro_targets or None,
            score_history=req.score_history,
            baseline_ratio=req.baseline_ratio,
        )
    except Exception as exc:
        raise HTTPException(502, f"Recompute failed: {exc}") from exc


@app.get("/api/portions/{fdc_id}")
def portions_for_fdc(fdc_id: int, grams: float = Query(default=100.0, gt=0)):
    """Suggest a kitchen amount for an FDC food at a given gram weight."""
    from recipe_opt_agent.portion_display import kitchen_amount_from_usda_portions

    try:
        from db import connect
        from portion_gram import load_portion_rows_cache

        with connect() as conn:
            cache = load_portion_rows_cache(conn, {int(fdc_id)})
        suggestion = kitchen_amount_from_usda_portions(float(grams), cache.get(int(fdc_id)) or [])
    except Exception as exc:
        raise HTTPException(503, f"Portion lookup failed: {exc}") from exc
    return {"fdc_id": int(fdc_id), "grams": float(grams), "suggestion": suggestion}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8010)
