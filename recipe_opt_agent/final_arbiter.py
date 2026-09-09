"""Final LLM arbitration across saved intermediate recipe candidates.

The loop saves feasible snapshots (candidate_pool) and branch champions
(interesting_candidates / path finalists) as it iterates. Before finalize
returns, this module asks an LLM to make the LAST call: are the ratio /
nutrient gains of an edited recipe large enough — in RELATIVE terms — to
justify how far its ingredient list strays from the dish's original intent?

The briefing gives the LLM, per candidate:
- the full ingredient list and an explicit diff vs the original recipe
- absolute losses (ratio loss, nutrient slack, L_total, L_max_norm, n_red)
- percentage-vs-best framing for each loss ("+3.2% worse than best")
- neighborhood evidence for every added ingredient (how many recipes in the
  expanded neighborhood actually contain it)

The system prompt encodes the decision policy: small relative gains never buy
identity-clashing ingredients; culinary plausibility (from the model's own
knowledge + neighborhood evidence) sets how large a gain must be.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

FINAL_ARBITER_SYSTEM_PROMPT = """You are the final judge for a recipe optimization run.
The numeric loop is done. You receive 2-6 saved recipe candidates (intermediate snapshots and
branch champions) with their FULL ingredient lists, ingredient diffs vs the original recipe,
loss metrics, and neighborhood evidence. Pick the single recipe to present to the user.

What the numbers mean:
- ratio_loss: deviation of ingredient mass-shares from real recipes of this dish (lower = closer
  to how this dish is actually made). This is the fidelity-to-neighborhood number.
- nutrient_slack: total violation of the user's macro box (0.0 = macros satisfied; >0 = the
  recipe MISSES the user's nutrition targets — this is close to disqualifying).
- L_total / L_max_norm / n_red: optimizer objective, worst IQR-normalized term, and count of
  ingredients far outside their typical share range.
- pct_vs_best: each loss expressed relative to the best candidate on that metric
  ("+4%" = 4 percent worse than the best candidate). USE THESE, not raw differences.
- neighborhood_evidence: for each ADDED ingredient, how many recipes in the expanded
  neighborhood contain it (n_recipes_containing / n_shell_recipes). This is external proof that
  the ingredient co-occurs with this dish family in the real world.

Decision policy — apply in this order:
1. Macro box first. A candidate with nutrient_slack ≈ 0 beats one that misses the box, unless
   the only way it hit the box was an ingredient that ruins the dish.
2. Classify every added/swapped ingredient using YOUR culinary knowledge plus the evidence:
   - "canonical": belongs to the dish as commonly made.
   - "plausible_extension": not traditional but a recognized, appetizing variant
     (e.g. chicken in a carbonara-style pasta — chicken carbonara is a real dish; strong
     neighborhood evidence supports this class).
   - "clash": flattens or distorts the dish's identity/flavor profile (e.g. ricotta or milk
     dumped into a carbonara that already has egg and hard cheese; syrup, wine, or lemon added
     purely to move macros). Weak neighborhood evidence + your own judgment both matter.
   Evidence override: if an added ingredient appears in ≥20% of the expanded-neighborhood shell
   recipes (n_shell_recipes_containing / n_shell_recipes_total), real cooks demonstrably combine
   it with this dish family — classify it AT WORST as "plausible_extension", never "clash".
   Conversely, an ingredient with ~0 evidence that also strikes you as odd is a "clash".
3. Price the classes in relative-loss terms:
   - Ties (losses within ~10% of each other) are decided ONLY by culinary fidelity — take the
     cleaner ingredient list.
   - A "plausible_extension" needs a MODERATE relative advantage to win: ≥ ~15% better ratio_loss
     or L_total than the cleanest candidate, or it fixes a macro-box miss (nutrient_slack > 0 → 0).
   - A "clash" ingredient needs an OVERWHELMING case: ≥ ~40% relative loss advantage AND it must
     be the only candidate that satisfies the macro box. A clash is NEVER worth a 1-5% loss edge —
     "its losses are 1% lower in each case" is not a reason to serve a strange dish.
4. Fewer, better-justified edits beat many small ones. Between two candidates with similar
   numbers, prefer the one with fewer additions.
5. Never pick a candidate that violates hard requirement_tags.

Return ONLY JSON:
{
  "winner_id": "...",
  "ranking": ["candidate ids best→worst"],
  "verdicts": {
    "<candidate_id>": {
      "culinary_fit": "canonical" | "plausible_extension" | "clash",
      "numeric_case": "strong" | "moderate" | "weak",
      "odd_ingredients": ["labels you judge as clashes, [] if none"],
      "note": "one sentence"
    }
  },
  "rationale": "2-4 sentences: name the decisive relative-loss numbers and the culinary call",
  "holistic_0_10": 0-10
}
"""


WEIRDNESS_FLAGGER_SYSTEM_PROMPT = """You screen recipe candidates for culinary disasters ONLY.

You receive several recipe candidates for the same dish (ingredient lists, diffs vs the original,
and neighborhood co-occurrence evidence). Your job is NOT to rank them on macros or fidelity.

Flag a candidate as weird ONLY when it has truly identity-destroying or absurd ingredients for
this dish — e.g. syrup/jam in a carbonara, onion rings in a stew, coffee grounds in bread,
random dessert toppings in a savory entrée, or stacked soft dairy that collapses the dish.

Do NOT flag:
- plausible extensions (chicken carbonara, turkey chili, tofu stir-fry variants)
- ordinary seasoning, oil, aromatics, or garnish that cooks commonly use
- modest swaps that keep the dish recognizable
- recipes that are merely imperfect on macros or proportions

When neighborhood evidence shows an added ingredient in ≥20% of shell recipes, do NOT flag it
as weird. Prefer false negatives over false positives — only catch the obvious disasters.

Return ONLY JSON:
{
  "flags": {
    "<candidate_id>": {
      "is_weird": true|false,
      "odd_ingredients": ["labels that make it weird, else []"],
      "note": "one short sentence if weird, else empty string"
    }
  }
}
"""


_EPS = 1e-9


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2}


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _ratio_loss_from_opt(opt: dict[str, Any] | None) -> float | None:
    """Sum of share-term losses (matches final_ratio_source=share_losses_sum).

    A bare ratio_surrogate of 0 with NO share terms is a false perfect (the LP
    surrogate is 0 whenever within bounds); report None instead so the pct
    framing marks it unknown rather than "best".
    """
    tl = (opt or {}).get("term_losses") or {}
    rs = tl.get("ratio_surrogate")
    shares = [float(v) for k, v in tl.items() if str(k).endswith("__share")]
    if shares:
        total = sum(shares)
        if rs is not None and float(rs) > 0:
            total += float(rs)
        return float(total)
    if rs is not None and float(rs) > 0:
        return float(rs)
    return None


def _nutrient_slack(pfc: dict[str, Any] | None, box: dict[str, Any]) -> float | None:
    if not pfc:
        return None
    try:
        p, c, f = float(pfc.get("protein")), float(pfc.get("carbs")), float(pfc.get("fat"))
    except (TypeError, ValueError):
        return None

    def _slack(v: float, lo: Any, hi: Any) -> float:
        lo = float(lo) if lo is not None else 0.0
        hi = float(hi) if hi is not None else 1.0
        return max(lo - v, 0.0) + max(v - hi, 0.0)

    return float(
        _slack(p, box.get("protein_min"), box.get("protein_max"))
        + _slack(c, box.get("carb_min"), box.get("carb_max"))
        + _slack(f, box.get("fat_min"), box.get("fat_max"))
    )


def _ingredient_diff(
    ingredients: list[dict[str, Any]],
    original: list[dict[str, Any]],
) -> dict[str, list[str]]:
    cur = {str(r.get("label") or r.get("name") or "").lower(): str(r.get("label") or r.get("name") or "") for r in ingredients or []}
    orig = {str(r.get("label") or r.get("name") or "").lower(): str(r.get("label") or r.get("name") or "") for r in original or []}
    added = [v for k, v in cur.items() if k and k not in orig]
    removed = [v for k, v in orig.items() if k and k not in cur]
    return {"added": added, "removed": removed}


def _neighborhood_evidence(
    added_labels: list[str],
    problem: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """How often each added ingredient appears in the (expanded) neighborhood."""
    ctx = (problem or {}).get("retrieval_context") or {}
    label_sets = [set(map(str, s)) for s in (ctx.get("neighbor_label_sets") or [])]
    shell = ctx.get("query_shell_recipes") or []
    out: dict[str, dict[str, Any]] = {}
    for label in added_labels:
        toks = _tokens(label)
        # Core noun heuristic: drop prep words so "Chicken breast, raw, skinless" → chicken/breast
        toks -= {"raw", "fresh", "cooked", "skinless", "boneless", "dried", "grated", "whole", "table", "part", "skim"}
        n_sets = 0
        for s in label_sets:
            flat = _tokens(" ".join(s))
            if toks & flat:
                n_sets += 1
        n_shell = 0
        for row in shell:
            flat = _tokens(str(row.get("title") or "")) | _tokens(" ".join(map(str, row.get("labels") or [])))
            if toks & flat:
                n_shell += 1
        # Prefer structure-verified shell counts when available
        verified_ids = {
            int(x)
            for x in (ctx.get("structure_verified_shell_ids") or [])
            if str(x).isdigit()
        }
        n_shell_verified = 0
        if verified_ids:
            for row in shell:
                rid = row.get("recipe_id")
                if rid is None or int(rid) not in verified_ids:
                    continue
                flat = _tokens(str(row.get("title") or "")) | _tokens(
                    " ".join(map(str, row.get("labels") or []))
                )
                if toks & flat:
                    n_shell_verified += 1
        struct_meta = ctx.get("neighborhood_structure_meta") or {}
        out[label] = {
            "n_neighbor_recipes_containing": n_sets,
            "n_neighbor_recipes_total": len(label_sets),
            "n_shell_recipes_containing": n_shell,
            "n_shell_recipes_total": len(shell),
            "n_structure_verified_shell_containing": n_shell_verified if verified_ids else None,
            "n_structure_verified_shell_total": len(verified_ids) if verified_ids else None,
            "structure_rejected_wrong_dominance": struct_meta.get("n_rejected_wrong_dominance"),
            "anchor_share_median": struct_meta.get("anchor_share_median"),
            "stretch_share_median": struct_meta.get("stretch_share_median"),
            "dish_structure": ctx.get("dish_structure") or struct_meta.get("dish_structure"),
        }
    return out


def _candidate_card(
    cand: dict[str, Any],
    *,
    original: list[dict[str, Any]],
    problem: dict[str, Any],
    box: dict[str, Any],
) -> dict[str, Any] | None:
    ingredients = cand.get("ingredients") or []
    if not ingredients:
        return None
    opt = cand.get("opt") or {}
    diag = cand.get("diagnosis_full") or {}
    diff = _ingredient_diff(ingredients, original)
    ratio = _ratio_loss_from_opt(opt)
    if ratio is None:
        rs = cand.get("ratio_surrogate")
        ratio = float(rs) if rs is not None and float(rs) > 0 else None
    slack = _nutrient_slack(opt.get("pfc_after") or cand.get("pfc_after"), box)
    if slack is None and cand.get("nutrient_slack") is not None:
        slack = float(cand["nutrient_slack"])
    return {
        "candidate_id": str(cand.get("candidate_id") or cand.get("bundle_id") or "unknown"),
        "branch": cand.get("branch") or "in_distribution",
        "iteration": cand.get("iteration"),
        "ingredients": [
            {"label": r.get("label") or r.get("name"), "grams": round(float(r.get("grams") or 0.0), 1)}
            for r in ingredients
        ],
        "diff_vs_original": diff,
        "n_edits": len(diff["added"]) + len(diff["removed"]),
        "losses": {
            "ratio_loss": ratio,
            "nutrient_slack": slack,
            "L_total": cand.get("L_total") if cand.get("L_total") is not None else diag.get("L_total"),
            "L_max_norm": cand.get("L_max_norm") if cand.get("L_max_norm") is not None else diag.get("L_max_norm"),
            "n_red": cand.get("n_red") if cand.get("n_red") is not None else diag.get("n_red"),
        },
        "neighborhood_evidence": _neighborhood_evidence(diff["added"], problem),
    }


def _attach_pct_vs_best(cards: list[dict[str, Any]]) -> None:
    """Annotate each loss with its percentage gap to the best candidate."""
    for metric in ("ratio_loss", "nutrient_slack", "L_total", "L_max_norm"):
        vals = [c["losses"].get(metric) for c in cards]
        known = [float(v) for v in vals if v is not None]
        if not known:
            continue
        best = min(known)
        for c in cards:
            v = c["losses"].get(metric)
            if v is None:
                c["losses"][f"{metric}_pct_vs_best"] = None
            elif best <= _EPS:
                c["losses"][f"{metric}_pct_vs_best"] = (
                    "best" if float(v) <= _EPS else f"worse (best is ~0, this is {float(v):.4g})"
                )
            else:
                pct = 100.0 * (float(v) - best) / best
                c["losses"][f"{metric}_pct_vs_best"] = "best" if pct <= 0.01 else f"+{pct:.1f}% worse than best"


def collect_arbiter_candidates(state: dict[str, Any], *, max_candidates: int = 4) -> list[dict[str, Any]]:
    """Distinct saved recipe snapshots: pool entries + path-family champions + current."""
    from recipe_opt_agent.score_display import select_path_finalists
    from recipe_opt_agent.shadow_gpt_candidate import force_include_shadow, is_shadow_candidate

    cfg = state.get("config") or {}
    max_candidates = int(cfg.get("max_finalists") or max_candidates or 4)
    raw: list[dict[str, Any]] = []
    for entry in state.get("candidate_pool") or []:
        raw.append(dict(entry))

    finalists = select_path_finalists(
        state, ood_handicap=float(cfg.get("ood_delta_handicap") or 0.015)
    )
    for fam, payload in (finalists or {}).items():
        if not payload:
            continue
        chosen = payload.get("chosen") or {}
        entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else {}
        # L_total proxy: joint-LP objective after the edit (comparable within the run).
        l_total = entry.get("L_total")
        if l_total is None:
            l_total = entry.get("L_star_after")
        if l_total is None:
            l_total = (chosen.get("opt") or {}).get("objective")
        raw.append(
            {
                "candidate_id": f"path_{fam}",
                "branch": fam if fam != "in_distribution" else "in_distribution",
                "iteration": payload.get("iteration"),
                "ingredients": chosen.get("ingredients"),
                "opt": chosen.get("opt") or payload.get("opt"),
                "diagnosis_full": {},
                "L_total": l_total,
                "L_max_norm": entry.get("L_max_norm"),
                "n_red": entry.get("n_red"),
                "nutrient_slack": entry.get("nutrient_slack"),
            }
        )

    cr = state.get("chosen_recipe") or (state.get("problem") or {}).get("chosen_recipe") or {}
    if cr.get("ingredients"):
        raw.append(
            {
                "candidate_id": "current_state",
                "branch": "current",
                "iteration": state.get("iteration"),
                "ingredients": cr.get("ingredients"),
                "opt": state.get("opt") or {},
                "diagnosis_full": state.get("diagnosis") or {},
            }
        )

    # Dedup by ingredient label set (keep the first, which has richer metadata).
    # Always preserve the silent optimized draft even if label sets collide.
    seen: set[frozenset[str]] = set()
    shadow: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []
    for cand in raw:
        if is_shadow_candidate(cand):
            if not shadow:
                shadow.append(cand)
            continue
        labels = frozenset(
            str(r.get("label") or r.get("name") or "").lower()
            for r in (cand.get("ingredients") or [])
        )
        if not labels or labels in seen:
            continue
        seen.add(labels)
        out.append(cand)
    return force_include_shadow(shadow + out, max_n=max_candidates)


def _heuristic_verdict(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline fallback: feasible box first, then fewest edits, then lowest L_total."""

    def _key(c: dict[str, Any]) -> tuple:
        losses = c["losses"]
        slack = losses.get("nutrient_slack")
        lt = losses.get("L_total")
        return (
            0 if (slack is not None and slack <= 1e-6) else 1,
            int(c.get("n_edits") or 0),
            float(lt) if lt is not None else 99.0,
        )

    ranked = sorted(cards, key=_key)
    return {
        "winner_id": ranked[0]["candidate_id"],
        "ranking": [c["candidate_id"] for c in ranked],
        "verdicts": {},
        "rationale": "Heuristic (no API key): macro-feasible first, then fewest edits, then lowest L_total.",
        "holistic_0_10": None,
    }


def arbitrate_final_recipe(
    state: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Run the final judgment. Returns `{winner_id, ranking, verdicts, rationale, comparison, _llm_trace}`."""
    import logging

    from recipe_opt_agent.shadow_gpt_candidate import (
        is_shadow_candidate,
        shadow_arbiter_consideration,
    )

    _log = logging.getLogger(__name__)
    cfg = state.get("config") or {}
    problem = state.get("problem") or {}
    original = list(state.get("original_ingredients") or [])
    if not original:
        original = list(
            ((problem.get("retrieval_context") or {}).get("starting_ingredients"))
            or (problem.get("chosen_recipe") or {}).get("ingredients")
            or []
        )
    box = {
        k: cfg.get(k)
        for k in ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max")
    }

    candidates = collect_arbiter_candidates(state)
    cards: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        card = _candidate_card(cand, original=original, problem=problem, box=box)
        if card is None:
            continue
        cards.append(card)
        by_id[card["candidate_id"]] = cand
    if len(cards) < 2:
        consideration = shadow_arbiter_consideration(
            candidates=candidates,
            winner_id=None,
            collect_meta=state.get("_shadow_collect_meta") or state.get("shadow_collect_meta"),
            model=str(cfg.get("shadow_draft_model") or "gpt-5.5"),
        )
        _log.info(
            "final_arbiter: skipped (need ≥2 cards); shadow consideration=%s",
            consideration,
        )
        return None
    _attach_pct_vs_best(cards)

    briefing = {
        "title": state.get("title"),
        "user_request": state.get("user_request") or state.get("taste_text"),
        "requirement_tags": state.get("requirement_tags") or [],
        "identity_roles": state.get("identity_roles") or [],
        "macro_box": box,
        "original_ingredients": [
            {"label": r.get("label") or r.get("name"), "grams": r.get("grams")} for r in original
        ],
        "candidates": cards,
        "note": (
            "pct_vs_best values are relative loss gaps. Apply the decision policy: "
            "clash ingredients need >=~40% relative advantage AND a macro-box fix; "
            "plausible extensions need >=~15%; within ~10% is a tie decided by culinary fidelity. "
            "A loss of null/None means the metric was not measured for that candidate — treat it "
            "as unknown (do NOT reward it as zero) and lean on the measured metrics."
        ),
    }
    messages = [
        {"role": "system", "content": FINAL_ARBITER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(briefing, indent=2, default=str)},
    ]

    def _with_consideration(result: dict[str, Any]) -> dict[str, Any]:
        consideration = shadow_arbiter_consideration(
            candidates=list(by_id.values()),
            winner_id=result.get("winner_id"),
            collect_meta=state.get("_shadow_collect_meta") or state.get("shadow_collect_meta"),
            model=str(cfg.get("shadow_draft_model") or "gpt-5.5"),
        )
        # Also note whether any arbiter card came from the shadow model.
        consideration["shadow_card_in_briefing"] = any(
            is_shadow_candidate(by_id.get(c["candidate_id"]) or {}) for c in cards
        )
        result["_shadow_consideration"] = consideration
        _log.info(
            "final_arbiter: GPT-5.5 shadow in set=%s ids=%s winner_is_shadow=%s",
            consideration.get("shadow_in_arbiter_set"),
            consideration.get("shadow_candidate_ids"),
            consideration.get("winner_is_shadow"),
        )
        return result

    if not os.environ.get("OPENAI_API_KEY"):
        result = _heuristic_verdict(cards)
        result["comparison"] = cards
        result["winner_entry"] = by_id.get(result["winner_id"])
        result["_llm_trace"] = {"mode": "heuristic", "model": None, "messages": messages}
        return _with_consideration(result)

    model = model or str(cfg.get("judge_model") or "gpt-4.1-mini")
    try:
        from recipe_opt_agent.observability import get_openai_client

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.15,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
    except Exception as exc:
        result = _heuristic_verdict(cards)
        result["comparison"] = cards
        result["winner_entry"] = by_id.get(result["winner_id"])
        result["_llm_trace"] = {
            "mode": "heuristic_fallback",
            "model": model,
            "error": str(exc),
            "messages": messages,
        }
        return _with_consideration(result)

    known_ids = {c["candidate_id"] for c in cards}
    winner_id = str(data.get("winner_id") or "")
    if winner_id not in known_ids:
        winner_id = _heuristic_verdict(cards)["winner_id"]
    return _with_consideration(
        {
            "winner_id": winner_id,
            "ranking": [r for r in (data.get("ranking") or []) if r in known_ids],
            "verdicts": data.get("verdicts") or {},
            "rationale": data.get("rationale"),
            "holistic_0_10": data.get("holistic_0_10"),
            "comparison": cards,
            "winner_entry": by_id.get(winner_id),
            "_llm_trace": {
                "mode": "openai",
                "model": model,
                "messages": messages,
                "raw_response": content,
                "usage": {
                    "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
                    "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
                },
            },
        }
    )


def weird_ids_from_judgment(judgment: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Normalize LLM weirdness flags / clash verdicts → ``{candidate_id: flag_dict}``."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(judgment, dict):
        return out
    flags = judgment.get("flags")
    if isinstance(flags, dict):
        for cid, raw in flags.items():
            if not isinstance(raw, dict):
                continue
            odd = [str(x) for x in (raw.get("odd_ingredients") or []) if str(x).strip()]
            is_weird = bool(raw.get("is_weird")) or bool(odd)
            if not is_weird:
                continue
            out[str(cid)] = {
                "is_weird": True,
                "odd_ingredients": odd,
                "note": str(raw.get("note") or "").strip(),
            }
        return out
    verdicts = judgment.get("verdicts")
    if isinstance(verdicts, dict):
        for cid, raw in verdicts.items():
            if not isinstance(raw, dict):
                continue
            fit = str(raw.get("culinary_fit") or "").strip().lower()
            odd = [str(x) for x in (raw.get("odd_ingredients") or []) if str(x).strip()]
            # Full-arbiter path: only demote explicit clashes, not plausible extensions.
            if fit != "clash":
                continue
            out[str(cid)] = {
                "is_weird": True,
                "odd_ingredients": odd,
                "note": str(raw.get("note") or "").strip(),
                "culinary_fit": fit or "clash",
            }
    return out


def flag_weird_candidates(
    state: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any] | None:
    """LLM screen: flag only obviously-bad ingredient lists. Does not rank winners."""
    import logging

    _log = logging.getLogger(__name__)
    cfg = state.get("config") or {}
    problem = state.get("problem") or {}
    original = list(state.get("original_ingredients") or [])
    if not original:
        original = list(
            ((problem.get("retrieval_context") or {}).get("starting_ingredients"))
            or (problem.get("chosen_recipe") or {}).get("ingredients")
            or []
        )
    box = {
        k: cfg.get(k)
        for k in ("protein_min", "protein_max", "carb_min", "carb_max", "fat_min", "fat_max")
    }

    candidates = collect_arbiter_candidates(state)
    cards: list[dict[str, Any]] = []
    for cand in candidates:
        card = _candidate_card(cand, original=original, problem=problem, box=box)
        if card is not None:
            cards.append(card)
    if len(cards) < 1:
        return None
    if len(cards) == 1:
        cid = cards[0]["candidate_id"]
        return {
            "flags": {cid: {"is_weird": False, "odd_ingredients": [], "note": ""}},
            "weird_candidate_ids": [],
            "rationale": "Single candidate — nothing to demote.",
            "comparison": cards,
            "_llm_trace": {"mode": "skip_single", "model": None},
        }

    # Compact briefing: ingredients + diffs + evidence only (no loss ranking task).
    briefing = {
        "title": state.get("title"),
        "user_request": state.get("user_request") or state.get("taste_text"),
        "requirement_tags": state.get("requirement_tags") or [],
        "identity_roles": state.get("identity_roles") or [],
        "original_ingredients": [
            {"label": r.get("label") or r.get("name"), "grams": r.get("grams")} for r in original
        ],
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "branch": c.get("branch"),
                "ingredients": c.get("ingredients"),
                "diff_vs_original": c.get("diff_vs_original"),
                "neighborhood_evidence": c.get("neighborhood_evidence"),
            }
            for c in cards
        ],
        "note": (
            "Flag only culinary disasters. Do not rank. Plausible extensions are fine. "
            "Prefer missing a subtle oddity over demoting a normal recipe."
        ),
    }
    messages = [
        {"role": "system", "content": WEIRDNESS_FLAGGER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(briefing, indent=2, default=str)},
    ]

    known_ids = {c["candidate_id"] for c in cards}
    empty_flags = {
        cid: {"is_weird": False, "odd_ingredients": [], "note": ""} for cid in known_ids
    }

    if not os.environ.get("OPENAI_API_KEY"):
        # Offline: denylist-style heuristic on added labels when available.
        flags = dict(empty_flags)
        try:
            from recipe_opt_agent.clash_gates import clash_reason_for_label

            title = str(state.get("title") or "")
            for c in cards:
                current = [
                    str(r.get("label") or r.get("name") or "")
                    for r in (c.get("ingredients") or [])
                ]
                odd: list[str] = []
                for label in current:
                    reason = clash_reason_for_label(label, current_labels=current, title=title)
                    if reason:
                        odd.append(label)
                if odd:
                    flags[c["candidate_id"]] = {
                        "is_weird": True,
                        "odd_ingredients": odd[:6],
                        "note": "Heuristic denylist / clash-family hit",
                    }
        except Exception:
            pass
        weird_ids = [cid for cid, f in flags.items() if f.get("is_weird")]
        return {
            "flags": flags,
            "weird_candidate_ids": weird_ids,
            "rationale": (
                f"Heuristic weirdness screen flagged {len(weird_ids)} candidate(s)."
                if weird_ids
                else "Heuristic weirdness screen found no disasters."
            ),
            "comparison": cards,
            "_llm_trace": {"mode": "heuristic", "model": None, "messages": messages},
        }

    model = model or str(cfg.get("judge_model") or "gpt-4.1-mini")
    try:
        from recipe_opt_agent.observability import get_openai_client

        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_json(content)
    except Exception as exc:
        _log.warning("flag_weird_candidates LLM failed: %s", exc)
        return {
            "flags": empty_flags,
            "weird_candidate_ids": [],
            "rationale": "Weirdness screen failed; left ranking unchanged.",
            "comparison": cards,
            "_llm_trace": {
                "mode": "error",
                "model": model,
                "error": str(exc),
                "messages": messages,
            },
        }

    flags = dict(empty_flags)
    raw_flags = data.get("flags") if isinstance(data, dict) else None
    if isinstance(raw_flags, dict):
        for cid, raw in raw_flags.items():
            if str(cid) not in known_ids or not isinstance(raw, dict):
                continue
            odd = [str(x) for x in (raw.get("odd_ingredients") or []) if str(x).strip()]
            is_weird = bool(raw.get("is_weird")) or bool(odd)
            flags[str(cid)] = {
                "is_weird": is_weird,
                "odd_ingredients": odd if is_weird else [],
                "note": str(raw.get("note") or "").strip() if is_weird else "",
            }
    weird_ids = [cid for cid, f in flags.items() if f.get("is_weird")]
    n_weird = len(weird_ids)
    rationale = (
        f"Demoted {n_weird} candidate(s) with odd ingredients to the end."
        if n_weird
        else "No culinary disasters flagged; order left to proportion quality."
    )
    return {
        "flags": flags,
        "weird_candidate_ids": weird_ids,
        "rationale": rationale,
        "comparison": cards,
        "_llm_trace": {
            "mode": "openai",
            "model": model,
            "messages": messages,
            "raw_response": content,
            "usage": {
                "prompt_tokens": getattr(getattr(resp, "usage", None), "prompt_tokens", None),
                "completion_tokens": getattr(getattr(resp, "usage", None), "completion_tokens", None),
            },
        },
    }
