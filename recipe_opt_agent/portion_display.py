"""Kitchen-unit display helpers: prefer RecipeNLG units, else USDA portions."""

from __future__ import annotations

from typing import Any


def _short_label(text: str | None, *, max_words: int = 3) -> str:
    raw = " ".join(str(text or "").strip().split())
    if not raw:
        return "ingredient"
    # Prefer the last noun-ish chunk for USDA-style "Cheese, cheddar, ..."
    if "," in raw:
        head = raw.split(",")[0].strip()
        if head:
            raw = head
    words = raw.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).lower()
    return raw.lower()


def edit_phrase(edit: dict[str, Any] | None) -> str | None:
    """Human phrase like 'added mayo' / 'swapped ham' / 'removed onion'."""
    if not isinstance(edit, dict):
        return None
    action = str(edit.get("action") or "").strip().lower()
    label = _short_label(edit.get("label"))
    replaced = _short_label(edit.get("replace_label") or edit.get("swap_out_label"))
    if action == "add":
        return f"added {label}"
    if action == "remove":
        return f"removed {label}"
    if action == "swap":
        if edit.get("replace_label") or edit.get("swap_out_label"):
            return f"swapped {replaced}"
        return f"swapped in {label}"
    if action:
        return f"{action} {label}"
    return None


def collect_applied_edits(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten applied edits from decision outcomes / last candidate / chosen."""
    payload = payload or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(edit: dict[str, Any]) -> None:
        phrase = edit_phrase(edit)
        if not phrase:
            return
        key = phrase.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "action": edit.get("action"),
                "label": edit.get("label"),
                "replace_label": edit.get("replace_label")
                or edit.get("swap_out_label"),
                "phrase": phrase,
            }
        )

    for outcome in payload.get("decision_outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        decision = outcome.get("decision") or {}
        for edit in decision.get("edits") or []:
            if isinstance(edit, dict):
                _push(edit)
    last = payload.get("last_applied_candidate") or {}
    if isinstance(last, dict):
        for edit in last.get("edits") or []:
            if isinstance(edit, dict):
                _push(edit)
    chosen = payload.get("chosen") or {}
    if isinstance(chosen, dict):
        for edit in chosen.get("edits") or []:
            if isinstance(edit, dict):
                _push(edit)
        entry = chosen.get("entry") if isinstance(chosen.get("entry"), dict) else None
        if entry:
            for edit in entry.get("edits") or []:
                if isinstance(edit, dict):
                    _push(edit)
    return out


def annotate_ingredient_edits(
    ingredients: list[dict[str, Any]],
    edits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach edit_note to ingredients that were added/swapped in."""
    if not edits:
        return ingredients
    by_label: dict[str, dict[str, Any]] = {}
    for edit in edits:
        lab = str(edit.get("label") or "").strip().lower()
        if lab:
            by_label[lab] = edit
    out: list[dict[str, Any]] = []
    for row in ingredients:
        item = dict(row)
        lab = str(item.get("label") or item.get("name") or "").strip().lower()
        hit = by_label.get(lab)
        if hit is None:
            # Fuzzy: label contains edit label or vice versa
            for key, edit in by_label.items():
                if key and (key in lab or lab in key):
                    hit = edit
                    break
        if hit is not None:
            item["edit_note"] = hit.get("phrase") or edit_phrase(hit)
            item["edit_action"] = hit.get("action")
        out.append(item)
    return out


def _norm_label(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _ingredient_keys(row: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    fid = row.get("fdc_id")
    if fid is not None:
        try:
            keys.add(f"fdc:{int(fid)}")
        except (TypeError, ValueError):
            pass
    for field in ("label", "name", "source_text", "fdc_description"):
        lab = _norm_label(row.get(field))
        if lab:
            keys.add(f"label:{lab}")
            # Also index first comma-segment for USDA-style names
            if "," in lab:
                keys.add(f"label:{lab.split(',', 1)[0].strip()}")
    return keys


def mark_novel_ingredients(
    ingredients: list[dict[str, Any]],
    *,
    original_ingredients: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Flag ingredients added in draft/loop or OOD vs the reference starting recipe.

    Hover copy: ``added_during_process_title``.
    """
    original = list(original_ingredients or [])
    original_keys: set[str] = set()
    for row in original:
        if isinstance(row, dict):
            original_keys |= _ingredient_keys(row)
    has_reference = bool(original_keys)

    out: list[dict[str, Any]] = []
    for row in ingredients:
        item = dict(row)
        keys = _ingredient_keys(item)
        basis = str(item.get("basis_node_id") or item.get("basis_node") or "")
        is_ood = basis.lower().startswith("ood") or "ood_" in basis.lower()
        edit_action = str(item.get("edit_action") or "").lower()
        edit_note = str(item.get("edit_note") or "").lower()
        explicitly_added = edit_action == "add" or edit_note.startswith("added ")
        in_reference = bool(keys & original_keys) if has_reference else False

        added = False
        reason = None
        if is_ood:
            added = True
            reason = "out_of_distribution"
        elif explicitly_added:
            added = True
            reason = "loop_add"
        elif has_reference and not in_reference:
            added = True
            reason = "draft_or_loop"
        elif not has_reference:
            # No frozen reference recipe (typical free-form creative warm-start):
            # the ingredient list itself came from the LLM draft / loop.
            added = True
            reason = "llm_draft"

        item["added_during_process"] = bool(added)
        item["added_during_process_reason"] = reason
        item["added_during_process_title"] = (
            "Added during the process (not in the reference recipe)"
            if added
            else None
        )
        out.append(item)
    return out


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def loss_improvement_pct(first: float | None, final: float | None) -> int | None:
    """Relative improvement % when final < first; never reports regressions."""
    if first is None or final is None or first <= 1e-12:
        return None
    if final >= first - 1e-12:
        return None
    pct_i = int(round((first - final) / first * 100.0))
    return pct_i if pct_i >= 1 else None


def _first_and_last_loss(
    score_history: list[dict[str, Any]] | None,
    key: str,
    *,
    final_override: float | None = None,
) -> tuple[float | None, float | None]:
    hist = [h for h in (score_history or []) if isinstance(h, dict)]
    diagnose = [h for h in hist if h.get("source") == "diagnose" and h.get(key) is not None]
    first = None
    if diagnose:
        first = _as_float(diagnose[0].get(key))
    if first is None:
        for h in hist:
            first = _as_float(h.get(key))
            if first is not None:
                break
    final = final_override
    if final is None and diagnose:
        final = _as_float(diagnose[-1].get(key))
    return first, final


def cookability_from_score_history(
    score_history: list[dict[str, Any]] | None,
    *,
    final_ratio: float | None,
) -> dict[str, Any]:
    """Only report improvements (never regressions) in ratio / cookability."""
    first, final = _first_and_last_loss(
        score_history, "ratio_loss", final_override=final_ratio
    )
    pct_i = loss_improvement_pct(first, final)
    if pct_i is None:
        return {
            "improved": False,
            "improved_pct": None,
            "summary": None,
            "first_ratio_loss": first,
            "final_ratio_loss": final,
        }
    return {
        "improved": True,
        "improved_pct": pct_i,
        "summary": f"Improved cookability by {pct_i}%",
        "first_ratio_loss": first,
        "final_ratio_loss": final,
    }


def nutrient_fit_from_score_history(
    score_history: list[dict[str, Any]] | None,
    *,
    final_nutrient: float | None,
) -> dict[str, Any]:
    """Only report improvements (never regressions) in nutrient loss / slack."""
    first, final = _first_and_last_loss(
        score_history, "nutrient_loss", final_override=final_nutrient
    )
    pct_i = loss_improvement_pct(first, final)
    if pct_i is None:
        return {
            "improved": False,
            "improved_pct": None,
            "summary": None,
            "first_nutrient_loss": first,
            "final_nutrient_loss": final,
        }
    return {
        "improved": True,
        "improved_pct": pct_i,
        "summary": f"Improved nutrient fit by {pct_i}%",
        "first_nutrient_loss": first,
        "final_nutrient_loss": final,
    }


def _collect_step_edits(update: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Gather edit dicts from a LangGraph step update / SSE payload."""
    update = update or {}
    found: list[dict[str, Any]] = []

    def _push(edit: Any) -> None:
        if isinstance(edit, dict) and (edit.get("action") or edit.get("label")):
            found.append(edit)

    decision = update.get("decision") if isinstance(update.get("decision"), dict) else {}
    for edit in decision.get("edits") or []:
        _push(edit)

    lac = update.get("last_applied_candidate")
    if isinstance(lac, dict):
        if lac.get("edits"):
            for edit in lac.get("edits") or []:
                _push(edit)
        elif lac.get("action"):
            _push(lac)

    for tool in update.get("tools_used") or []:
        if not isinstance(tool, dict):
            continue
        for blob in (tool.get("output_summary"), tool.get("output")):
            if not isinstance(blob, dict):
                continue
            for edit in blob.get("edits") or []:
                _push(edit)
            if blob.get("action") and blob.get("label") and not blob.get("edits"):
                _push(blob)

    bid = decision.get("chosen_bundle_id")
    if bid is not None and not found:
        for b in update.get("bundles") or []:
            if not isinstance(b, dict):
                continue
            if str(b.get("bundle_id")) != str(bid):
                continue
            for edit in b.get("edits") or []:
                _push(edit)
            break

    for outcome in update.get("decision_outcomes") or []:
        if not isinstance(outcome, dict):
            continue
        od = outcome.get("decision") if isinstance(outcome.get("decision"), dict) else {}
        for edit in od.get("edits") or []:
            _push(edit)

    return found


def edit_phrases_from_update(update: dict[str, Any] | None) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for edit in _collect_step_edits(update):
        phrase = edit_phrase(edit)
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


def build_progress_detail(node: str, update: dict[str, Any] | None) -> str | None:
    """Human progress-line detail for MacroIQ (edits + loss % improvements)."""
    update = update or {}
    node = str(node or "").strip()
    phrases = edit_phrases_from_update(update)

    if node in {"decide", "apply", "apply_or_expand"} and phrases:
        return "; ".join(phrases)

    if node == "diagnose":
        live = update.get("live_scores") if isinstance(update.get("live_scores"), dict) else {}
        ratio_now = None
        nut_now = None
        if isinstance(live.get("ratio_loss"), dict):
            ratio_now = _as_float(live["ratio_loss"].get("value"))
        else:
            ratio_now = _as_float(live.get("ratio_loss"))
        if isinstance(live.get("nutrient_loss"), dict):
            nut_now = _as_float(live["nutrient_loss"].get("value"))
        else:
            nut_now = _as_float(live.get("nutrient_loss"))

        cook = cookability_from_score_history(
            update.get("score_history"),
            final_ratio=ratio_now,
        )
        nut = nutrient_fit_from_score_history(
            update.get("score_history"),
            final_nutrient=nut_now,
        )
        # Only the most recent applied decision — avoid repeating the whole trail.
        recent_phrases: list[str] = []
        outcomes = update.get("decision_outcomes") or []
        if outcomes and isinstance(outcomes[-1], dict):
            od = outcomes[-1].get("decision") if isinstance(outcomes[-1].get("decision"), dict) else {}
            seen: set[str] = set()
            for edit in od.get("edits") or []:
                phrase = edit_phrase(edit if isinstance(edit, dict) else None)
                if not phrase:
                    continue
                key = phrase.lower()
                if key in seen:
                    continue
                seen.add(key)
                recent_phrases.append(phrase)
        parts: list[str] = []
        if recent_phrases:
            parts.append("; ".join(recent_phrases))
        if cook.get("summary"):
            parts.append(str(cook["summary"]))
        if nut.get("summary"):
            parts.append(str(nut["summary"]))
        if parts:
            return " · ".join(parts)
        return None

    if node in {"propose"} and update.get("bundles"):
        n = len(update.get("bundles") or [])
        if n:
            return f"Scored {n} edit bundle{'s' if n != 1 else ''}"

    return None


def _format_qty(value: float, unit: str) -> str:
    unit = (unit or "g").strip()
    if abs(value - round(value)) < 0.05:
        qty_s = str(int(round(value)))
        whole = int(round(value))
    else:
        qty_s = f"{value:.1f}".rstrip("0").rstrip(".")
        whole = None
    if unit in {"g", "gram", "grams"}:
        return f"{qty_s} g"
    unit_out = unit
    if whole is not None and abs(whole) != 1:
        if unit.endswith(("spoon", "ounce", "cup", "clove", "slice", "piece")) and not unit.endswith(
            "s"
        ):
            unit_out = unit + "s"
    return f"{qty_s} {unit_out}"


def _duplicate_merge_key(row: dict[str, Any]) -> str | None:
    """Stable key for duplicate ingredient lines (prefer FDC id, else label)."""
    fid = row.get("fdc_id")
    if fid is not None:
        try:
            return f"fdc:{int(fid)}"
        except (TypeError, ValueError):
            pass
    lab = _norm_label(row.get("label") or row.get("name") or row.get("source_text"))
    if not lab:
        return None
    return f"label:{lab}"


def _unit_gram_weight(row: dict[str, Any]) -> float | None:
    gw = _as_float(row.get("portion_gram_weight"))
    if gw is not None and gw > 0:
        return gw
    qty = _as_float(row.get("quantity"))
    orig = _as_float(row.get("original_grams"))
    if qty is not None and qty > 0 and orig is not None and orig > 0:
        return orig / qty
    # Already in kitchen units with amount_value reflecting current grams/unit weight.
    amount = _as_float(row.get("amount_value"))
    grams = _as_float(row.get("grams"))
    unit = str(row.get("amount_unit") or row.get("unit") or "").strip().lower()
    if (
        amount is not None
        and amount > 0
        and grams is not None
        and grams > 0
        and unit
        and unit not in {"g", "gram", "grams"}
    ):
        return grams / amount
    return None


_PORTION_SOURCE_RANK = {
    "scaled_portion": 0,
    "usda_count": 1,
    "usda_volume": 2,
    "usda_mass": 3,
    "usda_scaled": 4,
    "grams": 8,
    "missing": 9,
}


def _portion_source_rank(row: dict[str, Any]) -> tuple[int, int]:
    src = str(row.get("amount_source") or "missing")
    rank = _PORTION_SOURCE_RANK.get(src, 7)
    has_unit = 0 if _unit_gram_weight(row) else 1
    return (rank, has_unit)


def _apply_portion_for_grams(row: dict[str, Any], grams: float) -> dict[str, Any]:
    """Set amount_* fields for ``grams`` using the row's chosen kitchen unit."""
    out = dict(row)
    out["grams"] = float(grams)
    out["grams_rounded"] = int(round(float(grams)))
    gw = _unit_gram_weight(row)
    unit = (
        str(row.get("amount_unit") or row.get("unit") or "").strip()
        or ("g" if gw is None else "portion")
    )
    if gw is not None and gw > 0 and unit.lower() not in {"g", "gram", "grams"}:
        qty = float(grams) / float(gw)
        out["amount_value"] = qty
        out["amount_unit"] = unit
        out["amount_display"] = _format_qty(qty, unit)
        out["portion_gram_weight"] = float(gw)
        if out.get("amount_source") in {None, "grams", "missing"}:
            out["amount_source"] = "merged_portion"
        # Keep recipe-style quantity/original_grams coherent for later edits.
        out["quantity"] = qty
        out["unit"] = unit
        out["original_grams"] = float(gw)  # per 1 quantity unit
        return out
    out["amount_value"] = float(round(float(grams)))
    out["amount_unit"] = "g"
    out["amount_display"] = _format_qty(float(grams), "g")
    out["amount_source"] = "grams"
    return out


def consolidate_duplicate_ingredients(
    ingredients: list[dict[str, Any]] | None,
    *,
    problem: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Merge duplicate ingredient lines into one row with summed amounts.

    Matching prefers identical ``fdc_id``, else identical normalized label.
    Picks the best kitchen portion among duplicates (scaled → count → volume → mass)
    and converts the combined grams into that unit.

    When ``problem`` is provided and merges occur, returns an updated problem whose
    ``M`` / ``ingredient_basis`` / ``x0`` columns match the consolidated list so
    interactive recompute stays consistent.
    """
    rows = [dict(r) for r in (ingredients or []) if isinstance(r, dict)]
    if len(rows) <= 1:
        return rows, None

    groups: dict[str, list[int]] = {}
    order: list[str] = []
    singles: list[int] = []
    for i, row in enumerate(rows):
        key = _duplicate_merge_key(row)
        if key is None:
            singles.append(i)
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    # Nothing to merge
    if not any(len(idxs) > 1 for idxs in groups.values()):
        return rows, None

    merged_rows: list[dict[str, Any]] = []
    index_groups: list[list[int]] = []

    def _flush_group(idxs: list[int]) -> None:
        if len(idxs) == 1:
            merged_rows.append(dict(rows[idxs[0]]))
            index_groups.append(idxs)
            return
        members = [rows[i] for i in idxs]
        # Prefer the strongest kitchen-portion template.
        template = min(members, key=_portion_source_rank)
        total_grams = 0.0
        for m in members:
            g = _as_float(m.get("grams"))
            if g is not None:
                total_grams += g
        total_kcal = 0.0
        kcal_n = 0
        share_sum = 0.0
        share_n = 0
        loss_vals: list[float] = []
        notes: list[str] = []
        for m in members:
            if m.get("calories") is not None:
                try:
                    total_kcal += float(m["calories"])
                    kcal_n += 1
                except (TypeError, ValueError):
                    pass
            sh = _as_float(m.get("recipe_share"))
            if sh is not None:
                share_sum += sh
                share_n += 1
            lc = _as_float(m.get("loss_contribution"))
            if lc is not None:
                loss_vals.append(lc)
            note = m.get("edit_note")
            if note and str(note) not in notes:
                notes.append(str(note))

        combined = dict(template)
        combined = _apply_portion_for_grams(combined, total_grams)
        if kcal_n:
            combined["calories"] = int(round(total_kcal))
        if share_n:
            combined["recipe_share"] = share_sum
        if loss_vals:
            combined["loss_contribution"] = max(loss_vals)
            # Keep the worst band among members when present.
            band_rank = {"good": 0, "warn": 1, "bad": 2, "unknown": -1}
            best_band = None
            best_r = -2
            for m in members:
                b = m.get("loss_band")
                r = band_rank.get(str(b), -1)
                if r > best_r:
                    best_r = r
                    best_band = b
            if best_band is not None:
                combined["loss_band"] = best_band
        if notes:
            combined["edit_note"] = "; ".join(notes)
        if any(m.get("added_during_process") for m in members):
            combined["added_during_process"] = True
            combined["added_during_process_title"] = next(
                (
                    m.get("added_during_process_title")
                    for m in members
                    if m.get("added_during_process_title")
                ),
                "Added during the process (not in the reference recipe)",
            )
        if any(m.get("edit_action") for m in members):
            combined["edit_action"] = next(
                (m.get("edit_action") for m in members if m.get("edit_action")), None
            )
        combined["merged_from_count"] = len(idxs)
        combined["merged_from_indices"] = list(idxs)
        # Prefer a non-empty IQR from any member (same neighborhood basis typically).
        if not combined.get("share_iqr"):
            for m in members:
                if m.get("share_iqr"):
                    combined["share_iqr"] = m.get("share_iqr")
                    break
        merged_rows.append(combined)
        index_groups.append(idxs)

    for key in order:
        _flush_group(groups[key])
    for i in singles:
        _flush_group([i])

    if len(merged_rows) == len(rows):
        return rows, None

    # Reindex for UI editing
    for i, row in enumerate(merged_rows):
        row["index"] = i

    problem_out = None
    if isinstance(problem, dict) and problem:
        problem_out = _remap_problem_columns(problem, index_groups, merged_rows)

    return merged_rows, problem_out


def _remap_problem_columns(
    problem: dict[str, Any],
    index_groups: list[list[int]],
    merged_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild M / basis / x vectors so columns align with consolidated ingredients."""
    import numpy as np

    out = dict(problem)
    M_raw = problem.get("M") or []
    try:
        M = np.asarray(M_raw, dtype=float)
        if M.ndim != 2:
            M = np.zeros((0, 0), dtype=float)
    except (TypeError, ValueError):
        M = np.zeros((0, 0), dtype=float)

    basis = list(problem.get("ingredient_basis") or [])
    x_raw = problem.get("x_opt") or problem.get("x0") or []
    try:
        x = np.asarray(x_raw, dtype=float).ravel()
    except (TypeError, ValueError):
        x = np.zeros(0, dtype=float)

    new_cols: list[Any] = []
    new_basis: list[Any] = []
    new_x: list[float] = []

    for idxs, row in zip(index_groups, merged_rows):
        grams = _as_float(row.get("grams")) or 0.0
        new_x.append(float(grams))
        new_basis.append(basis[idxs[0]] if idxs and idxs[0] < len(basis) else None)
        if M.size and M.ndim == 2 and all(i < M.shape[1] for i in idxs):
            weights = []
            for i in idxs:
                w = float(x[i]) if i < x.size and float(x[i]) > 0 else 0.0
                weights.append(w)
            if sum(weights) <= 0:
                weights = [1.0] * len(idxs)
            w = np.asarray(weights, dtype=float)
            w = w / float(w.sum())
            col = M[:, idxs] @ w
            new_cols.append(col)
        elif M.size and M.ndim == 2 and idxs and idxs[0] < M.shape[1]:
            new_cols.append(M[:, idxs[0]])

    if new_cols:
        out["M"] = np.column_stack(new_cols).tolist()
    out["ingredient_basis"] = new_basis
    out["x0"] = list(new_x)
    out["x_opt"] = list(new_x)
    out["total_mass"] = float(sum(new_x)) if new_x else problem.get("total_mass")
    chosen = dict(problem.get("chosen_recipe") or {})
    chosen["ingredients"] = [
        {
            "label": r.get("label"),
            "name": r.get("label"),
            "grams": r.get("grams"),
            "fdc_id": r.get("fdc_id"),
            "quantity": r.get("quantity"),
            "unit": r.get("unit"),
            "original_grams": r.get("original_grams"),
            "portion_gram_weight": r.get("portion_gram_weight"),
            "amount_source": r.get("amount_source"),
            "foodon_id": r.get("foodon_leaf_id"),
        }
        for r in merged_rows
    ]
    out["chosen_recipe"] = chosen
    return out


def kitchen_amount_from_usda_portions(
    grams: float,
    portion_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer count, then volume, then mass USDA portions for a gram weight."""
    from portion_gram import classify_food_portion_row

    if grams is None or float(grams) <= 0 or not portion_rows:
        return None

    grams = float(grams)
    volume: list[dict[str, Any]] = []
    count: list[dict[str, Any]] = []
    mass: list[dict[str, Any]] = []
    for row in portion_rows:
        kind = classify_food_portion_row(row)
        gw = float(row.get("gram_weight") or 0)
        if gw <= 0:
            continue
        bucket = {"volume": volume, "count": count, "mass": mass}.get(kind)
        if bucket is not None:
            bucket.append(row)

    def _pick(rows: list[dict[str, Any]], *, kind: str) -> dict[str, Any] | None:
        if not rows:
            return None
        # Prefer portions whose reference weight is closest to a "nice" kitchen scale.
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                abs(float(r["gram_weight"]) - 15.0),  # ~tbsp-ish preference
                float(r["gram_weight"]),
            ),
        )
        best = rows_sorted[0]
        gw = float(best["gram_weight"])
        amount_ref = float(best.get("amount") or 1.0) or 1.0
        qty = grams / gw * amount_ref
        unit = (
            str(best.get("measure_unit_name") or "").strip()
            or str(best.get("modifier") or "").strip()
            or str(best.get("portion_description") or "").strip()
            or ("piece" if kind == "count" else "portion")
        )
        # Clean undetermined measure names
        if unit.lower() in {"undetermined", "racc", ""}:
            unit = "piece" if kind == "count" else "portion"
        return {
            "amount_value": qty,
            "amount_unit": unit,
            "amount_display": _format_qty(qty, unit),
            "amount_source": f"usda_{kind}",
            "portion_id": best.get("id"),
            "portion_gram_weight": gw,
        }

    return _pick(count, kind="count") or _pick(volume, kind="volume") or _pick(mass, kind="mass")


def enrich_ingredients_with_usda_portions(
    ingredients: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For rows still in grams, try USDA count/volume portions by fdc_id."""
    need: list[tuple[int, int]] = []
    for i, row in enumerate(ingredients):
        src = str(row.get("amount_source") or "")
        if src in {"scaled_portion"}:
            continue
        if row.get("quantity") is not None and row.get("unit"):
            continue
        fid = row.get("fdc_id")
        grams = row.get("grams")
        if fid is None or grams is None:
            continue
        try:
            need.append((i, int(fid)))
        except (TypeError, ValueError):
            continue
    if not need:
        return ingredients

    try:
        from db import connect
        from portion_gram import load_portion_rows_cache
    except Exception:
        return ingredients

    try:
        with connect() as conn:
            cache = load_portion_rows_cache(conn, {fid for _, fid in need})
    except Exception:
        return ingredients

    out = [dict(r) for r in ingredients]
    for i, fid in need:
        rows = cache.get(fid) or []
        grams = out[i].get("grams")
        try:
            g = float(grams)
        except (TypeError, ValueError):
            continue
        suggestion = kitchen_amount_from_usda_portions(g, rows)
        if not suggestion:
            continue
        out[i].update(suggestion)
        # Keep a mass toggle baseline
        out[i].setdefault("grams_rounded", int(round(g)))
    return out
