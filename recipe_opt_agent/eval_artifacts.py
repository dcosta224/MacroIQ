"""Full eval artifact capture: steps, tools, LLM traces, retrieval, flow, metrics."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = ROOT / "scratch" / "recipe_opt_runs" / "eval_suites"


def _jsonable(obj: Any, *, _depth: int = 0) -> Any:
    """Best-effort JSON conversion; strip huge next_problem payloads."""
    if _depth > 12:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "next_problem" and v is not None:
                report = (v or {}).get("foodon_basis_report") if isinstance(v, dict) else None
                out[k] = {
                    "_present": True,
                    "n_x0": len((v or {}).get("x0") or []) if isinstance(v, dict) else None,
                    "n_basis": len((v or {}).get("ingredient_basis") or []) if isinstance(v, dict) else None,
                    "foodon_basis_report": report,
                }
                continue
            if k in {"M", "basis_samples", "ratio_samples", "rollup_chains"} and isinstance(v, (list, dict)):
                out[k] = f"<omitted shape={_shape_hint(v)}>"
                continue
            out[str(k)] = _jsonable(v, _depth=_depth + 1)
        return out
    if isinstance(obj, (list, tuple)):
        if len(obj) > 200:
            return [_jsonable(x, _depth=_depth + 1) for x in obj[:50]] + [f"<…{len(obj) - 50} more>"]
        return [_jsonable(x, _depth=_depth + 1) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return str(obj)


def _shape_hint(v: Any) -> str:
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"dict[{len(v)}]"
    return type(v).__name__


@dataclass
class EvalRunRecorder:
    """Collects every stream event + final metrics for one agent run."""

    case_name: str
    agent_mode: str
    suite_dir: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    transcripts: list[dict[str, Any]] = field(default_factory=list)
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    flow_meta: dict[str, Any] | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    t0: float = field(default_factory=time.perf_counter)
    result: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    langsmith_run_id: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    request: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.suite_dir / "runs" / f"{self.case_name}__{self.run_id}"

    def on_event(self, event: dict[str, Any]) -> None:
        ev = _jsonable(event)
        self.events.append(ev)  # type: ignore[arg-type]
        et = event.get("type")
        if et == "graph_meta":
            self.flow_meta = {
                "nodes": event.get("nodes"),
                "edges": event.get("edges"),
                "creative": event.get("creative"),
            }
        elif et == "step":
            payload = event.get("payload") or {}
            step = {
                "seq": event.get("seq"),
                "node": event.get("node"),
                "iteration": event.get("iteration"),
                "fidelity_band": event.get("fidelity_band"),
                "payload": _jsonable(payload),
            }
            self.steps.append(step)
            for tool in payload.get("tools_used") or []:
                tool_rec = {
                    "seq": event.get("seq"),
                    "node": event.get("node"),
                    **{k: tool.get(k) for k in ("name", "purpose", "mode", "model", "output_summary")},
                    "output": _jsonable(tool.get("output")),
                }
                self.tools.append(tool_rec)
                if tool.get("name") in {"retrieve_slots", "plan_slots", "score_bundles"}:
                    self.retrieval.append(tool_rec)
                trace = tool.get("llm_trace")
                if trace:
                    self.llm_calls.append(
                        {
                            "seq": event.get("seq"),
                            "node": event.get("node"),
                            "tool": tool.get("name"),
                            "mode": trace.get("mode"),
                            "model": trace.get("model"),
                            "messages": _jsonable(trace.get("messages")),
                            "raw_response": trace.get("raw_response"),
                            "parsed": _jsonable(trace.get("parsed")),
                            "usage": trace.get("usage"),
                            "rationale": trace.get("rationale"),
                        }
                    )
            # Also capture top-level llm_trace on the step payload
            if payload.get("llm_trace") and not (payload.get("tools_used") or []):
                tr = payload["llm_trace"]
                self.llm_calls.append(
                    {
                        "seq": event.get("seq"),
                        "node": event.get("node"),
                        "tool": None,
                        "mode": tr.get("mode"),
                        "model": tr.get("model"),
                        "messages": _jsonable(tr.get("messages")),
                        "raw_response": tr.get("raw_response"),
                        "usage": tr.get("usage"),
                    }
                )
        elif et == "transcript":
            self.transcripts.append(_jsonable(event))  # type: ignore[arg-type]

    def finish(
        self,
        result: dict[str, Any],
        *,
        langsmith_run_id: str | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> Path:
        from recipe_opt_agent.observability import metrics_from_result, post_run_feedback

        elapsed = time.perf_counter() - self.t0
        self.result = result
        self.langsmith_run_id = langsmith_run_id
        self.metrics = metrics_from_result(result, elapsed_s=round(elapsed, 3))
        if extra_metrics:
            self.metrics.update(extra_metrics)

        if langsmith_run_id:
            posted = post_run_feedback(run_id=langsmith_run_id, metrics=self.metrics)
            self.metrics["langsmith_feedback_keys"] = posted

        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "case_name": self.case_name,
            "agent_mode": self.agent_mode,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": self.metrics.get("elapsed_s"),
            "request": self.request,
            "config": self.config_snapshot,
            "run_metadata": self.run_metadata,
            "langsmith_run_id": langsmith_run_id,
            "metrics": self.metrics,
            "n_events": len(self.events),
            "n_steps": len(self.steps),
            "n_tools": len(self.tools),
            "n_llm_calls": len(self.llm_calls),
            "flow": self.flow_meta,
            "files": {
                "manifest": "manifest.json",
                "events": "events.jsonl",
                "steps": "steps.json",
                "tools": "tools.json",
                "llm_calls": "llm_calls.json",
                "retrieval": "retrieval.json",
                "transcripts": "transcripts.jsonl",
                "final": "final.json",
                "metrics": "metrics.json",
                "run_metadata": "run_metadata.json",
                "flow": "flow.json",
            },
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        (self.run_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2, default=str))
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(self.run_metadata or {}, indent=2, default=str)
        )
        (self.run_dir / "final.json").write_text(json.dumps(_jsonable(result), indent=2, default=str))
        (self.run_dir / "steps.json").write_text(json.dumps(self.steps, indent=2, default=str))
        (self.run_dir / "tools.json").write_text(json.dumps(self.tools, indent=2, default=str))
        (self.run_dir / "llm_calls.json").write_text(json.dumps(self.llm_calls, indent=2, default=str))
        (self.run_dir / "retrieval.json").write_text(json.dumps(self.retrieval, indent=2, default=str))
        (self.run_dir / "flow.json").write_text(json.dumps(self.flow_meta or {}, indent=2, default=str))
        with (self.run_dir / "events.jsonl").open("w") as f:
            for ev in self.events:
                f.write(json.dumps(ev, default=str) + "\n")
        with (self.run_dir / "transcripts.jsonl").open("w") as f:
            for tr in self.transcripts:
                f.write(json.dumps(tr, default=str) + "\n")
        return self.run_dir


@dataclass
class EvalSuiteRecorder:
    """One directory per suite: suite_manifest + per-run folders + summary.csv/json."""

    name: str = "eval"
    root: Path = field(default_factory=lambda: DEFAULT_EVAL_ROOT)
    suite_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    runs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def suite_dir(self) -> Path:
        return self.root / f"{self.name}_{self.suite_id}"

    def start(self) -> Path:
        self.suite_dir.mkdir(parents=True, exist_ok=True)
        (self.suite_dir / "runs").mkdir(exist_ok=True)
        (self.suite_dir / "suite_meta.json").write_text(
            json.dumps(
                {
                    "name": self.name,
                    "suite_id": self.suite_id,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "dir": str(self.suite_dir),
                },
                indent=2,
            )
        )
        return self.suite_dir

    def new_run(
        self,
        *,
        case_name: str,
        agent_mode: str,
        request: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ) -> EvalRunRecorder:
        rec = EvalRunRecorder(
            case_name=case_name,
            agent_mode=agent_mode,
            suite_dir=self.suite_dir,
            request=request,
            config_snapshot=config_snapshot or {},
        )
        return rec

    def add_run_summary(self, recorder: EvalRunRecorder) -> None:
        self.runs.append(
            {
                "case_name": recorder.case_name,
                "agent_mode": recorder.agent_mode,
                "run_id": recorder.run_id,
                "run_dir": str(recorder.run_dir),
                "langsmith_run_id": recorder.langsmith_run_id,
                "metrics": recorder.metrics,
            }
        )

    def finish(self) -> Path:
        summary = {
            "name": self.name,
            "suite_id": self.suite_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_runs": len(self.runs),
            "runs": self.runs,
            "aggregates": _aggregate(self.runs),
        }
        path = self.suite_dir / "suite_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        # Flat CSV-ish JSON lines for quick grep
        with (self.suite_dir / "suite_metrics.jsonl").open("w") as f:
            for r in self.runs:
                row = {"case_name": r["case_name"], "agent_mode": r["agent_mode"], **(r.get("metrics") or {})}
                f.write(json.dumps(row, default=str) + "\n")
        return path


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    keys = (
        "final_status_ok",
        "tag_violations_final",
        "n_auto_applies",
        "n_llm_calls",
        "lp_agreement_rate",
        "escalate_rate",
        "elapsed_s",
        "final_L_max_norm",
    )
    out: dict[str, Any] = {}
    for k in keys:
        vals = []
        for r in runs:
            v = (r.get("metrics") or {}).get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[k] = {"mean": sum(vals) / len(vals), "n": len(vals), "sum": sum(vals)}
    return out


def run_agent_with_artifacts(
    *,
    problem: dict[str, Any],
    case_name: str,
    agent_mode: str,
    suite: EvalSuiteRecorder,
    taste_text: str = "",
    title: str = "",
    user_request: str = "",
    canonical_id: int | None = None,
    config: Any = None,
    extra_tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], EvalRunRecorder]:
    """Stream a full agent run, persist artifacts, optionally attach LangSmith feedback."""
    from recipe_opt_agent.config import AgentConfig
    from recipe_opt_agent.observability import (
        build_run_metadata,
        ensure_tracing_env,
        get_current_run_id,
        models_by_stage,
        run_config,
        tracing_enabled,
    )
    from recipe_opt_agent.runner import run_recipe_opt_agent

    cfg = config or AgentConfig()
    ensure_tracing_env()

    run_meta = build_run_metadata(
        config=cfg,
        agent_mode=agent_mode,
        user_request=user_request or taste_text,
        taste_text=taste_text,
        title=title,
        case_name=case_name,
        suite_id=suite.suite_id,
        canonical_id=canonical_id,
        problem=problem,
        extra=metadata,
    )
    rec = suite.new_run(
        case_name=case_name,
        agent_mode=agent_mode,
        request=user_request or taste_text,
        config_snapshot={
            "models": models_by_stage(cfg),
            "model": cfg.model,
            "model_escalate": cfg.model_escalate,
            "creative_model": cfg.creative_model,
            "tags_model": cfg.tags_model,
            "judge_model": cfg.judge_model,
            "max_iterations": cfg.max_iterations,
            "F_accept": cfg.F_accept,
            "F_max": cfg.F_max,
            "macro_targets": cfg.target_box_dict(),
            "auto_apply_delta_eps": cfg.auto_apply_delta_eps,
            "auto_apply_margin": cfg.auto_apply_margin,
            "graph": run_meta.get("graph"),
            "agent_mode": agent_mode,
        },
    )
    rec.run_metadata = run_meta
    lg_config = run_config(
        case_name=case_name,
        agent_mode=agent_mode,
        suite_id=suite.suite_id,
        tags=extra_tags,
        metadata=metadata,
        config=cfg,
        user_request=user_request or taste_text,
        taste_text=taste_text,
        title=title,
        canonical_id=canonical_id,
        problem=problem,
    )

    captured: dict[str, str | None] = {"run_id": None}

    def _on_event(event: dict[str, Any]) -> None:
        rec.on_event(event)
        if event.get("type") == "step" and tracing_enabled() and not captured["run_id"]:
            captured["run_id"] = get_current_run_id()

    if tracing_enabled():
        try:
            from langsmith import traceable

            @traceable(
                name=f"recipe_opt:{case_name}",
                run_type="chain",
                metadata=lg_config.get("metadata") or {},
                tags=lg_config.get("tags") or [],
            )
            def _traced() -> dict[str, Any]:
                captured["run_id"] = get_current_run_id()
                return run_recipe_opt_agent(
                    problem=problem,
                    taste_text=taste_text,
                    title=title,
                    canonical_id=canonical_id,
                    config=cfg,
                    on_event=_on_event,
                    agent_mode=agent_mode,
                    user_request=user_request or taste_text,
                    langgraph_config=lg_config,
                )

            result = _traced()
        except Exception:
            result = run_recipe_opt_agent(
                problem=problem,
                taste_text=taste_text,
                title=title,
                canonical_id=canonical_id,
                config=cfg,
                on_event=_on_event,
                agent_mode=agent_mode,
                user_request=user_request or taste_text,
                langgraph_config=lg_config,
            )
    else:
        result = run_recipe_opt_agent(
            problem=problem,
            taste_text=taste_text,
            title=title,
            canonical_id=canonical_id,
            config=cfg,
            on_event=_on_event,
            agent_mode=agent_mode,
            user_request=user_request or taste_text,
            langgraph_config=lg_config,
        )

    rec.finish(result, langsmith_run_id=captured.get("run_id"))
    suite.add_run_summary(rec)
    return result, rec
