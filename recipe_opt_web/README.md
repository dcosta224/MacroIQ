# MacroIQ web UI

Product UI and developer playground for the LangGraph recipe optimization agent.

Partner setup (branch, env, local store): see the **Recipe optimization agent** section in the [project README](../README.md).

## Run

```bash
# from repo root
PYTHONPATH=scripts:. uv run python -m recipe_opt_web --reload
# → http://127.0.0.1:8010             MacroIQ product UI
# → http://127.0.0.1:8010/playground  flow graph + transcript playground
# → http://127.0.0.1:8010/loop-demo   simple agent-loop presentation demo
```

Or:

```bash
PYTHONPATH=scripts:. uv run uvicorn recipe_opt_web.server:app --reload --port 8010
```

### MacroIQ product (`/`)

Semantic ask, macro ranges, searchable menu of indexed canonical dishes, and a live
pipeline stage that surfaces load phases, optimizer status, nutrient position vs the
box, proposed edits, and LLM inference calls as the agent runs.

### Loop demo (`/loop-demo`)

A separate, simpler page for talk tracks: query + macro sliders, then step readouts
(neighborhood → draft → diagnose → propose / decide / apply → expand → compare → done).
Readouts look static but are editable (`contenteditable`) so you can tweak numbers live.
No agent backend calls — demo content only.

## Inputs

| Field | Meaning |
|-------|---------|
| Mode | **Neighborhood** (canonical dish) or **Creative** (free-text request) |
| Canonical recipe dropdown | List from the local cap40 store by default (`RECIPE_DATA_SOURCE=local`), ordered by `n_matches`. Defines the FoodOn neighborhood; `taste_text` = dish title. |
| Creative request | Free-text dish idea; agent drafts then grounds to FDC before the diagnose loop. |
| Macro % targets | Calorie-share percents → 0–1 fractions. Starting NLG instance chosen closest to this box during load. |
| F_accept / F_max / Max iterations | Fidelity bands + loop budget |

On **Run**, the UI streams live load phases (build neighborhood → pick start recipe, or creative draft/ground) then the LangGraph loop. **propose** retrieves modification candidates live (not from a JSON fixture).

There is no “fixture” or “title override” in this playground — those were only for offline unit tests.

## Related

- Agent package: [`recipe_opt_agent/`](../recipe_opt_agent/)
- Notebook sandbox: [`notebooks/recipe_opt_agent_sandbox.ipynb`](../notebooks/recipe_opt_agent_sandbox.ipynb)
- Design notes: [`docs/recipe_opt_agent.md`](../docs/recipe_opt_agent.md)
