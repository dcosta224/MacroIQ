/* MacroIQ product UI — compose + simplified live progress */

// Fast-demo production profile caps the diagnose→propose→apply loop at 2.
const MAX_LOOP_ROUNDS = 2;
// Rough graph budget for remaining-step estimates (diagnose→propose→decide→apply).
const STEPS_PER_LOOP_ROUND = 4;
const WRAP_UP_STEPS = 3;

const STEP_COPY = {
  load: { title: "Loading context", detail: "Gathering similar recipes and starting ingredients." },
  init: { title: "Starting", detail: "Setting up the recipe." },
  deduce_tags: { title: "Reading your ask", detail: "Picking up dietary and macro constraints." },
  llm_draft: { title: "Drafting a recipe", detail: "Writing a starting ingredient list." },
  ground_recipe: { title: "Matching ingredients", detail: "Linking each line to a food in the database." },
  diagnose: { title: "Balancing the recipe", detail: "Adjusting amounts toward your targets." },
  save_candidate: { title: "Saved a candidate", detail: "Kept a workable version for comparison." },
  save_moderate: { title: "Saved a candidate", detail: "Kept a workable version for comparison." },
  propose: { title: "Looking for edits", detail: "Finding ingredient swaps or additions." },
  decide: { title: "Choosing an edit", detail: "Picking the next change to try." },
  apply: { title: "Applying an edit", detail: "Updating the recipe with the chosen change." },
  build_finalists: { title: "Comparing options", detail: "Narrowing to the strongest candidates." },
  pareto_and_rank: { title: "Ranking options", detail: "Sorting by dish proportion quality." },
  judge_final: { title: "Wrapping up", detail: "Preparing the final recipe." },
  finalize: { title: "Wrapping up", detail: "Preparing the final recipe." },
};

// SSE fires after a node finishes. While we wait for the next event, show the
// likely follow-up so a fast step (e.g. save_candidate) doesn't look stuck.
const NEXT_HINT = {
  load: { title: "Starting", detail: "Setting up the recipe." },
  init: { title: "Reading your ask", detail: "Picking up dietary and macro constraints." },
  deduce_tags: { title: "Drafting a recipe", detail: "Writing a starting ingredient list." },
  llm_draft: { title: "Matching ingredients", detail: "Linking each line to a food in the database." },
  ground_recipe: { title: "Balancing the recipe", detail: "Adjusting amounts toward your targets." },
  diagnose: { title: "Looking for edits", detail: "Finding ingredient swaps or additions." },
  save_candidate: { title: "Looking for edits", detail: "Finding ingredient swaps or additions." },
  save_moderate: { title: "Looking for edits", detail: "Finding ingredient swaps or additions." },
  propose: { title: "Choosing an edit", detail: "Picking the next change to try." },
  decide: { title: "Applying an edit", detail: "Updating the recipe with the chosen change." },
  apply: { title: "Balancing the recipe", detail: "Re-checking macros after the edit." },
  build_finalists: { title: "Ranking options", detail: "Sorting by dish proportion quality." },
  pareto_and_rank: { title: "Wrapping up", detail: "Preparing the final recipe." },
  judge_final: { title: "Wrapping up", detail: "Preparing the final recipe." },
};

const $ = (id) => document.getElementById(id);

const els = {
  page: $("page"),
  form: $("compose-form"),
  semantic: $("semantic"),
  kcalTarget: $("kcal_target"),
  runBtn: $("run-btn"),
  runStatus: $("run-status"),
  macroWarn: $("macro-warn"),
  macroSuggest: $("macro-suggest"),
  macroClear: $("macro-clear"),
  macroGrid: $("macro-grid"),
  macroHint: $("macro-hint"),
  macroTypicality: $("macro-typicality"),
  menuSearch: $("menu-search"),
  dishMenu: $("dish-menu"),
  menuCount: $("menu-count"),
  menuSelected: $("menu-selected"),
  stageIdle: $("stage-idle"),
  stageLive: $("stage-live"),
  phaseKicker: $("phase-kicker"),
  phaseTitle: $("phase-title"),
  phaseDetail: $("phase-detail"),
  progressRemaining: $("progress-remaining"),
  feed: $("activity-feed"),
  feedCount: $("feed-count"),
  resultBlock: $("result-block"),
  resultStatus: $("result-status"),
  resultMacros: $("result-macros"),
  resultVerdict: $("result-verdict"),
  resultIngs: $("result-ings"),
  revertAmounts: $("revert-amounts"),
  candidatePager: $("candidate-pager"),
  candidatePrev: $("candidate-prev"),
  candidateNext: $("candidate-next"),
  candidatePagerTitle: $("candidate-pager-title"),
  candidatePagerCount: $("candidate-pager-count"),
  judgeRationale: $("judge-rationale"),
  judgeRationaleBody: $("judge-rationale-body"),
  macroTip: $("macro-tip"),
};

const MACRO_KEYS = ["protein", "carb", "fat"];
const G_PER_OZ = 28.3495;
const G_PER_LB = 453.592;
const state = {
  dishes: [],
  selected: null,
  macrosEnabled: false,
  macroNeighborhood: null, // distribution + presets for selected dish
  eventCount: 0,
  running: false,
  iteration: 0,
  maxIterations: MAX_LOOP_ROUNDS,
  lastNode: null,
  firstRatio: null,
  firstNutrient: null,
  lastFinal: null,
  liveIngredients: [],
  originalGrams: null,
  baselineSnapshot: null, // exact scores/ingredients at load — restore when grams return
  unitMode: {}, // index -> kitchen|g|oz|lb
  editingAmountIdx: null,
  openMenu: null,
  macroFocus: null, // { key, idxs: number[] }
  activeMacroTip: null, // protein|carb|fat
  browseCandidates: [],
  browseIndex: 0,
  calorieScaleTarget: null, // draft kcal in the editable calories field
};

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function pct(id) {
  const v = Number($(id).value);
  return Number.isFinite(v) ? v : 0;
}

function frac(id) {
  return pct(id) / 100;
}

function setMacrosEnabled(on, { sync = true } = {}) {
  state.macrosEnabled = Boolean(on);
  if (els.macroGrid) els.macroGrid.classList.toggle("macro-inactive", !state.macrosEnabled);
  if (els.macroClear) els.macroClear.classList.toggle("hidden", !state.macrosEnabled);
  if (els.macroHint) {
    els.macroHint.textContent = state.macrosEnabled
      ? "Drag either end of each range (max 10% wide). Maxes must sum to at least 100%."
      : "Optional. Leave unset to optimize for dish proportions only. Drag a range or use Pick for me to target macros.";
  }
  if (!state.macrosEnabled) clearMacroTypicality();
  if (sync) syncMacroVisuals();
}

function clearMacroTypicality() {
  if (!els.macroTypicality) return;
  els.macroTypicality.textContent = "";
  els.macroTypicality.classList.add("hidden");
  els.macroTypicality.classList.remove("typical", "atypical");
}

function currentMacroBoxFracs() {
  return {
    protein_min: frac("protein_min"),
    protein_max: frac("protein_max"),
    carb_min: frac("carb_min"),
    carb_max: frac("carb_max"),
    fat_min: frac("fat_min"),
    fat_max: frac("fat_max"),
  };
}

function assessMacrosVsNeighborhood(box, distribution, dishTitle) {
  if (!box || !distribution) return null;
  const dish = (dishTitle || state.selected?.title || "this dish").trim() || "this dish";
  const axes = [
    { key: "protein", label: "Protein", distKey: "protein" },
    { key: "carb", label: "Carbs", distKey: "carbs" },
    { key: "fat", label: "Fat", distKey: "fat" },
  ];
  const highs = [];
  const lows = [];
  const SLIGHTLY_FRAC = 0.8;
  const VERY_MAX_IN_RANGE = 2;

  for (const ax of axes) {
    const stats = distribution[ax.distKey] || {};
    const vals = Array.isArray(stats.values)
      ? stats.values.map(Number).filter((v) => Number.isFinite(v))
      : null;
    let lo = Number(box[`${ax.key}_min`]);
    let hi = Number(box[`${ax.key}_max`]);
    if (![lo, hi].every(Number.isFinite)) continue;
    if (hi < lo) {
      const tmp = lo;
      lo = hi;
      hi = tmp;
    }

    let status = "typical";
    if (vals?.length) {
      const n = vals.length;
      let inRange = 0;
      let nBelow = 0;
      let nAbove = 0;
      for (const v of vals) {
        if (v + 1e-12 >= lo && v - 1e-12 <= hi) inRange += 1;
        else if (v < lo - 1e-12) nBelow += 1;
        else if (v > hi + 1e-12) nAbove += 1;
      }
      const fracBelow = nBelow / n;
      const fracAbove = nAbove / n;
      const sorted = [...vals].sort((a, b) => a - b);
      const median =
        n % 2 === 1 ? sorted[(n - 1) >> 1] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
      const mid = 0.5 * (lo + hi);

      if (inRange <= VERY_MAX_IN_RANGE) {
        if (median < lo - 1e-12 || (nBelow > nAbove && !(median > hi + 1e-12))) {
          status = "very_high";
        } else if (median > hi + 1e-12 || nAbove > nBelow) {
          status = "very_low";
        } else {
          status = mid >= median ? "very_high" : "very_low";
        }
      } else if (fracBelow >= SLIGHTLY_FRAC) {
        status = "slightly_high";
      } else if (fracAbove >= SLIGHTLY_FRAC) {
        status = "slightly_low";
      }
    } else {
      // Legacy fallback when recipe values are missing: IQR midpoint rule.
      const q1 = Number(stats.p25);
      const q3 = Number(stats.p75);
      if (![q1, q3].every(Number.isFinite)) continue;
      const mid = 0.5 * (lo + hi);
      const edge = 0.01;
      if (mid > q3 + edge) status = "slightly_high";
      else if (mid < q1 - edge) status = "slightly_low";
    }

    const pretty = ax.label.toLowerCase();
    if (status === "very_high" || status === "slightly_high") {
      highs.push({
        name: pretty,
        intensity: status.startsWith("very_") ? "very" : "slightly",
      });
    } else if (status === "very_low" || status === "slightly_low") {
      lows.push({
        name: pretty,
        intensity: status.startsWith("very_") ? "very" : "slightly",
      });
    }
  }

  const phrase = (items, side) => {
    const parts = items.map((it) => `${it.name} is ${it.intensity} ${side}`);
    if (parts.length === 1) return parts[0];
    if (parts.length === 2) return `${parts[0]} and ${parts[1]}`;
    return `${parts.slice(0, -1).join(", ")}, and ${parts[parts.length - 1]}`;
  };

  if (!highs.length && !lows.length) {
    const medP = Number(distribution.protein?.median);
    const medC = Number(distribution.carbs?.median);
    const medF = Number(distribution.fat?.median);
    const summary =
      Number.isFinite(medP) && Number.isFinite(medC) && Number.isFinite(medF)
        ? `Macros look typical for ${dish} (neighborhood median ~${Math.round(
            medP * 100
          )}% P / ${Math.round(medC * 100)}% C / ${Math.round(medF * 100)}% F).`
        : `Macros look typical for ${dish}.`;
    return { overall: "typical", summary };
  }
  const bits = [];
  if (highs.length) bits.push(phrase(highs, "high"));
  if (lows.length) bits.push(phrase(lows, "low"));
  return {
    overall: "atypical",
    summary: `For ${dish}, ${bits.join(" · ")} versus typical recipes of this type.`,
  };
}

function updateMacroTypicality() {
  if (!els.macroTypicality) return;
  if (!state.macrosEnabled || !state.macroNeighborhood?.distribution) {
    clearMacroTypicality();
    return;
  }
  const assessment = assessMacrosVsNeighborhood(
    currentMacroBoxFracs(),
    state.macroNeighborhood.distribution,
    state.selected?.title || state.macroNeighborhood.title
  );
  if (!assessment?.summary) {
    clearMacroTypicality();
    return;
  }
  els.macroTypicality.textContent = assessment.summary;
  els.macroTypicality.classList.remove("hidden", "typical", "atypical");
  els.macroTypicality.classList.add(assessment.overall === "typical" ? "typical" : "atypical");
}

function applyMacroBox(box) {
  const set = (id, fracVal) => {
    if (fracVal == null) return;
    $(id).value = String(Math.round(Number(fracVal) * 100));
  };
  set("protein_min", box.protein_min);
  set("protein_max", box.protein_max);
  set("carb_min", box.carb_min);
  set("carb_max", box.carb_max);
  set("fat_min", box.fat_min);
  set("fat_max", box.fat_max);
  setMacrosEnabled(true);
}

async function loadMacroNeighborhood(canonicalId, { quiet = false } = {}) {
  if (!canonicalId) {
    state.macroNeighborhood = null;
    clearMacroTypicality();
    return null;
  }
  if (
    state.macroNeighborhood?.canonical_id === Number(canonicalId) &&
    state.macroNeighborhood?.distribution
  ) {
    updateMacroTypicality();
    return state.macroNeighborhood;
  }
  if (!quiet) els.runStatus.textContent = "Loading dish macro profile…";
  const res = await fetch("/api/macro_targets/suggest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ canonical_id: Number(canonicalId) }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  state.macroNeighborhood = data;
  updateMacroTypicality();
  return data;
}

function syncMacroVisuals(changedId) {
  const MAX_MACRO_WIDTH_PP = 10;
  for (const key of MACRO_KEYS) {
    const minEl = $(`${key}_min`);
    const maxEl = $(`${key}_max`);
    let lo = Number(minEl.value);
    let hi = Number(maxEl.value);
    if (changedId === minEl.id && lo > hi) {
      hi = lo;
      maxEl.value = String(hi);
    } else if (changedId === maxEl.id && hi < lo) {
      lo = hi;
      minEl.value = String(lo);
    }
    // Cap each macro range at 10 percentage points.
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi - lo > MAX_MACRO_WIDTH_PP) {
      if (changedId === minEl.id) {
        hi = lo + MAX_MACRO_WIDTH_PP;
        maxEl.value = String(hi);
      } else if (changedId === maxEl.id) {
        lo = hi - MAX_MACRO_WIDTH_PP;
        minEl.value = String(lo);
      } else {
        const mid = 0.5 * (lo + hi);
        lo = Math.round(mid - MAX_MACRO_WIDTH_PP / 2);
        hi = lo + MAX_MACRO_WIDTH_PP;
        minEl.value = String(lo);
        maxEl.value = String(hi);
      }
    }
    const fill = $(`${key}_fill`);
    const readout = $(`${key}_readout`);
    if (fill) {
      fill.style.left = `${Math.min(lo, hi)}%`;
      fill.style.width = `${Math.max(Math.abs(hi - lo), 0.8)}%`;
    }
    if (readout) {
      readout.textContent = state.macrosEnabled
        ? `${Math.round(lo)}–${Math.round(hi)}%`
        : "—";
    }
  }
  validateMacros();
  updateMacroTypicality();
}

function validateMacros() {
  if (!state.macrosEnabled) {
    if (els.macroWarn) {
      els.macroWarn.textContent = "";
      els.macroWarn.classList.add("hidden");
    }
    clearMacroTypicality();
    return [];
  }
  const pLo = frac("protein_min");
  const pHi = frac("protein_max");
  const cLo = frac("carb_min");
  const cHi = frac("carb_max");
  const fLo = frac("fat_min");
  const fHi = frac("fat_max");
  const errors = [];
  for (const [name, lo, hi] of [
    ["protein", pLo, pHi],
    ["carbs", cLo, cHi],
    ["fat", fLo, fHi],
  ]) {
    if (lo > hi) errors.push(`${name} min exceeds max`);
  }
  if (pHi + cHi + fHi < 1 - 1e-9) errors.push("macro maxes must sum to at least 100%");
  if (pLo + cLo + fLo > 1 + 1e-9) errors.push("macro mins must sum to at most 100%");
  if (errors.length) {
    els.macroWarn.textContent = errors.join("; ");
    els.macroWarn.classList.remove("hidden");
  } else {
    els.macroWarn.textContent = "";
    els.macroWarn.classList.add("hidden");
  }
  return errors;
}

function renderDishMenu(dishes) {
  els.dishMenu.innerHTML = "";
  if (!dishes.length) {
    const li = document.createElement("li");
    li.innerHTML = `<button type="button" disabled><span class="dish-title">No recipe families found</span></button>`;
    els.dishMenu.appendChild(li);
    return;
  }
  for (const d of dishes) {
    const li = document.createElement("li");
    if (state.selected && Number(state.selected.canonical_id) === Number(d.canonical_id)) {
      li.classList.add("selected");
    }
    const n = d.n_matches ?? d.n_recipes ?? "—";
    li.innerHTML = `
      <button type="button" role="option" data-id="${escapeHtml(d.canonical_id)}">
        <span class="dish-title">${escapeHtml(d.title || `Dish ${d.canonical_id}`)}</span>
        <span class="dish-meta">${escapeHtml(n)} recipes</span>
      </button>`;
    li.querySelector("button").addEventListener("click", () => selectDish(d));
    els.dishMenu.appendChild(li);
  }
}

function selectDish(d) {
  state.selected = d;
  state.macroNeighborhood = null;
  clearMacroTypicality();
  els.menuSelected.textContent = `Selected: ${d.title}`;
  els.macroSuggest.disabled = !d?.canonical_id;
  renderDishMenu(state.dishes);
  if (d?.canonical_id) {
    loadMacroNeighborhood(d.canonical_id, { quiet: true }).catch(() => {
      /* typicality stays hidden until Pick for me / retry */
    });
  }
}

async function loadDishes(q = "") {
  els.menuCount.textContent = "Loading…";
  const params = new URLSearchParams({ min_neighborhood: "5" });
  let url = "/api/canonicals";
  if (q.trim()) {
    url = "/api/canonicals/search";
    params.set("q", q.trim());
    params.set("limit", "80");
  } else {
    params.set("limit", "200");
  }
  try {
    const res = await fetch(`${url}?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    state.dishes = data.dishes || [];
    els.menuCount.textContent =
      data.total != null
        ? `${state.dishes.length} shown · ${data.total} indexed`
        : `${state.dishes.length} dishes`;
    renderDishMenu(state.dishes);
  } catch (err) {
    els.menuCount.textContent = "Unavailable";
    els.dishMenu.innerHTML = `<li><button type="button" disabled><span class="dish-title">${escapeHtml(err.message)}</span></button></li>`;
  }
}

let searchTimer = null;
els.menuSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadDishes(els.menuSearch.value), 220);
});

els.macroSuggest.addEventListener("click", async () => {
  if (!state.selected?.canonical_id) return;
  els.macroSuggest.disabled = true;
  els.runStatus.textContent = "Picking macro ranges for this dish…";
  els.runStatus.classList.remove("error");
  try {
    const data = await loadMacroNeighborhood(state.selected.canonical_id);
    const preset =
      data?.presets?.neighborhood_coverage || data?.presets?.neighborhood_mean || {};
    const box = preset.box || {};
    if (box.protein_min == null || box.protein_max == null) {
      throw new Error("No neighborhood macro ranges available for this dish.");
    }
    applyMacroBox(box);
    const covPct =
      preset.coverage_frac != null ? Math.round(Number(preset.coverage_frac) * 100) : null;
    const n = data.n_recipes || data.n_neighborhood_recipes;
    els.runStatus.textContent =
      covPct != null
        ? `Applied ≤10% ranges covering ~${covPct}% of ${n || "neighborhood"} resolved recipes.`
        : "Applied dish-typical macro ranges.";
  } catch (err) {
    els.runStatus.textContent = `Macro suggest failed: ${err.message}`;
    els.runStatus.classList.add("error");
  } finally {
    els.macroSuggest.disabled = !state.selected?.canonical_id;
  }
});

els.macroClear?.addEventListener("click", () => {
  setMacrosEnabled(false);
  els.runStatus.textContent = "Macro targets cleared — optimizing for dish proportions.";
  els.runStatus.classList.remove("error");
});

for (const key of MACRO_KEYS) {
  $(`${key}_min`).addEventListener("input", (e) => {
    if (!state.macrosEnabled) setMacrosEnabled(true, { sync: false });
    syncMacroVisuals(e.target.id);
  });
  $(`${key}_max`).addEventListener("input", (e) => {
    if (!state.macrosEnabled) setMacrosEnabled(true, { sync: false });
    syncMacroVisuals(e.target.id);
  });
}

function setLiveVisible(on) {
  els.stageIdle.classList.toggle("hidden", on);
  els.stageLive.classList.toggle("hidden", !on);
}

function resetStage() {
  state.eventCount = 0;
  state.iteration = 0;
  state.maxIterations = MAX_LOOP_ROUNDS;
  state.lastNode = null;
  state.firstRatio = null;
  state.firstNutrient = null;
  state.lastFinal = null;
  state.liveIngredients = [];
  state.originalGrams = null;
  state.baselineSnapshot = null;
  state.unitMode = {};
  state.editingAmountIdx = null;
  state.openMenu = null;
  state.macroFocus = null;
  state.activeMacroTip = null;
  state.browseCandidates = [];
  state.browseIndex = 0;
  state.calorieScaleTarget = null;
  if (els.macroTip) {
    els.macroTip.classList.add("hidden");
    els.macroTip.innerHTML = "";
  }
  if (els.revertAmounts) els.revertAmounts.classList.add("hidden");
  if (els.candidatePager) els.candidatePager.classList.add("hidden");
  if (els.candidateScoreStrip) {
    els.candidateScoreStrip.classList.add("hidden");
    els.candidateScoreStrip.textContent = "";
  }
  if (els.judgeRationale) {
    els.judgeRationale.classList.add("hidden");
    if (els.judgeRationaleBody) els.judgeRationaleBody.textContent = "";
  }
  els.page.classList.remove("done", "error");
  els.runStatus.classList.remove("error");
  els.feed.innerHTML = "";
  els.feedCount.textContent = "0 steps";
  if (els.progressRemaining) els.progressRemaining.textContent = "";
  els.resultBlock.classList.add("hidden");
  els.resultMacros.innerHTML = "";
  els.resultVerdict.textContent = "";
  els.resultVerdict.className = "result-verdict";
  els.resultIngs.innerHTML = "";
  els.phaseKicker.textContent = "Working";
  els.phaseTitle.textContent = "Starting…";
  els.phaseDetail.textContent = "Connecting to the agent.";
}

function estimateStepsLeft(node, iteration) {
  const maxIt = Math.max(1, Number(state.maxIterations) || MAX_LOOP_ROUNDS);
  const it = Math.max(0, Math.min(maxIt, Number(iteration) || 0));
  const wrap = ["build_finalists", "pareto_and_rank", "judge_final", "finalize"];
  const wrapIdx = wrap.indexOf(node);
  if (wrapIdx >= 0) return Math.max(0, wrap.length - wrapIdx - 1);

  const roundsLeftIncludingCurrent = Math.max(0, maxIt - it);
  const positionInRound = {
    diagnose: 3,
    save_candidate: 3,
    save_moderate: 3,
    propose: 2,
    decide: 1,
    apply: 0,
  };
  let left = roundsLeftIncludingCurrent * STEPS_PER_LOOP_ROUND + WRAP_UP_STEPS;
  if (node in positionInRound && roundsLeftIncludingCurrent > 0) {
    left =
      (roundsLeftIncludingCurrent - 1) * STEPS_PER_LOOP_ROUND +
      positionInRound[node] +
      WRAP_UP_STEPS;
  } else if (["load", "init", "deduce_tags", "llm_draft", "ground_recipe"].includes(node)) {
    // Preamble still ahead of the first diagnose loop.
    left = maxIt * STEPS_PER_LOOP_ROUND + WRAP_UP_STEPS + 2;
  }
  return Math.max(0, left);
}

function updateProgressMeta(node = state.lastNode) {
  const maxIt = Math.max(1, Number(state.maxIterations) || MAX_LOOP_ROUNDS);
  const it = Math.max(0, Number(state.iteration) || 0);
  const displayRound = Math.min(maxIt, Math.max(1, it || 1));
  const left = estimateStepsLeft(node, it);
  const roundLabel =
    node && ["build_finalists", "pareto_and_rank", "judge_final", "finalize"].includes(node)
      ? "Finishing up"
      : `Pass ${displayRound} of ${maxIt}`;
  if (els.progressRemaining) {
    if (els.page.classList.contains("done")) {
      els.progressRemaining.textContent = "All steps complete";
    } else if (els.page.classList.contains("error")) {
      els.progressRemaining.textContent = "";
    } else {
      const leftLabel =
        left <= 0 ? "Wrapping up…" : `About ${left} step${left === 1 ? "" : "s"} left`;
      els.progressRemaining.textContent = `${roundLabel} · ${leftLabel}`;
    }
  }
  const done = state.eventCount;
  els.feedCount.textContent =
    left > 0 && state.running
      ? `${done} done · ~${left} left`
      : `${done} step${done === 1 ? "" : "s"}`;
  if (state.running && !["finalize", "judge_final"].includes(node)) {
    const base = (els.phaseKicker.textContent || "Working").split("·")[0].trim() || "Working";
    if (!["Error", "Done", "Finishing"].includes(base)) {
      els.phaseKicker.textContent = `Working · pass ${displayRound}/${maxIt}`;
    }
  }
}

function setPhase(title, detail, kicker = "Working") {
  const maxIt = Math.max(1, Number(state.maxIterations) || MAX_LOOP_ROUNDS);
  const it = Math.max(0, Number(state.iteration) || 0);
  const displayRound = Math.min(maxIt, Math.max(1, it || 1));
  let kick = kicker;
  if (kicker === "Working" && state.running) {
    kick = `Working · pass ${displayRound}/${maxIt}`;
  }
  els.phaseKicker.textContent = kick;
  els.phaseTitle.textContent = title;
  els.phaseDetail.textContent = detail || "";
  updateProgressMeta(state.lastNode);
}

function markFeedItemDone(li) {
  if (!li) return;
  li.classList.remove("active", "pending");
  li.classList.add("done");
  const spin = li.querySelector(".act-spinner");
  if (spin) spin.remove();
  if (!li.querySelector(".act-check")) {
    const check = document.createElement("span");
    check.className = "act-check";
    check.setAttribute("aria-hidden", "true");
    check.textContent = "✓";
    li.appendChild(check);
  }
}

function completePendingAndActive() {
  els.feed.querySelectorAll("li.active, li.pending").forEach((li) => markFeedItemDone(li));
}

function pushCompletedStep(title, detail = "") {
  completePendingAndActive();
  state.eventCount += 1;
  const li = document.createElement("li");
  li.className = "done";
  li.innerHTML = `
    <div class="act-copy">
      <span class="act-title">${escapeHtml(title)}</span>
      ${detail ? `<span class="act-detail">${escapeHtml(detail)}</span>` : ""}
    </div>
    <span class="act-check" aria-hidden="true">✓</span>`;
  els.feed.prepend(li);
  while (els.feed.children.length > 24) els.feed.lastChild.remove();
  updateProgressMeta(state.lastNode);
}

function setPendingStep(title, detail = "") {
  els.feed.querySelectorAll("li.pending").forEach((li) => li.remove());
  const li = document.createElement("li");
  li.className = "active pending";
  li.innerHTML = `
    <div class="act-copy">
      <span class="act-title">${escapeHtml(title)}</span>
      ${detail ? `<span class="act-detail">${escapeHtml(detail)}</span>` : ""}
    </div>
    <span class="act-spinner" aria-hidden="true"></span>`;
  els.feed.prepend(li);
  while (els.feed.children.length > 24) els.feed.lastChild.remove();
  updateProgressMeta(state.lastNode);
}

function advanceStep(nodeKey, title, detail) {
  state.lastNode = nodeKey;
  pushCompletedStep(title, detail);
  const hint = NEXT_HINT[nodeKey];
  if (hint) {
    setPhase(hint.title, hint.detail);
    setPendingStep(hint.title, hint.detail);
  } else {
    setPhase(title, detail, "Done");
  }
  updateProgressMeta(nodeKey);
}

function ingredientLabel(ing) {
  return (
    ing?.label ||
    ing?.name ||
    ing?.food_name ||
    ing?.fdc_description ||
    ing?.source_text ||
    null
  );
}

function ingredientAmount(ing) {
  if (ing?.amount_display) return String(ing.amount_display);
  if (ing?.amount_value != null && ing?.amount_unit) {
    const v = Number(ing.amount_value);
    const unit = String(ing.amount_unit);
    if (unit === "g" || unit === "gram" || unit === "grams") {
      return `${Math.round(v)} g`;
    }
    const rounded = Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(1);
    return `${rounded} ${unit}`;
  }
  if (ing?.grams != null && Number.isFinite(Number(ing.grams))) {
    return `${Math.round(Number(ing.grams))} g`;
  }
  return "—";
}

function kitchenAmountParts(ing) {
  if (ing?.amount_value != null && Number.isFinite(Number(ing.amount_value))) {
    const v = Number(ing.amount_value);
    const unit = String(ing.amount_unit || "").trim();
    const qtyText =
      Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : String(Number(v.toFixed(3)));
    return { qty: v, qtyText, unit };
  }
  const display = ingredientAmount(ing);
  const m = String(display || "").trim().match(/^([+-]?\d*\.?\d+)\s*(.*)$/);
  if (!m) return { qty: null, qtyText: "", unit: "" };
  const qty = Number(m[1]);
  return {
    qty: Number.isFinite(qty) ? qty : null,
    qtyText: m[1],
    unit: String(m[2] || "").trim(),
  };
}

function kitchenQtyToGrams(ing, newQty, idx = null) {
  const qty = Number(newQty);
  if (!Number.isFinite(qty) || qty < 0) return null;
  const gw = Number(ing?.portion_gram_weight);
  let grams = null;
  if (Number.isFinite(gw) && gw > 0) grams = qty * gw;
  else {
    const curQty = Number(ing?.amount_value);
    const curGrams = Number(ing?.grams);
    if (Number.isFinite(curQty) && curQty > 1e-9 && Number.isFinite(curGrams)) {
      grams = (curGrams / curQty) * qty;
    } else {
      grams = qty;
    }
  }
  if (idx != null && state.originalGrams?.[idx] != null && gramsNear(grams, state.originalGrams[idx])) {
    return Number(state.originalGrams[idx]);
  }
  return grams;
}

function amountEditValue(ing, mode) {
  const g = Number(ing?.grams);
  if (mode === "oz" && Number.isFinite(g)) return (g / G_PER_OZ).toFixed(2);
  if (mode === "lb" && Number.isFinite(g)) return (g / G_PER_LB).toFixed(3);
  if (mode === "g" && Number.isFinite(g)) return String(Math.round(g));
  const parts = kitchenAmountParts(ing);
  return parts.qtyText || (Number.isFinite(g) ? String(Math.round(g)) : "");
}

function amountEditUnitLabel(ing, mode) {
  if (mode === "oz") return "oz";
  if (mode === "lb") return "lb";
  if (mode === "g") return "g";
  return kitchenAmountParts(ing).unit || "";
}

function ingredientRows(final) {
  const scores = final?.display_scores || {};
  let ings = scores.ingredients || [];
  if (!ings.length) {
    const chosen = final?.chosen || final?.chosen_recipe || {};
    ings =
      chosen.ingredients ||
      chosen.entry?.ingredients ||
      (final?.problem?.chosen_recipe || {}).ingredients ||
      [];
  }
  return Array.isArray(ings) ? ings : [];
}

function shortFoodName(text) {
  let raw = String(text || "").trim();
  if (!raw) return "ingredient";
  if (raw.includes(",")) raw = raw.split(",")[0].trim();
  const words = raw.split(/\s+/);
  return (words.length > 3 ? words.slice(0, 3).join(" ") : raw).toLowerCase();
}

function editPhraseFromPayload(edit) {
  if (!edit || typeof edit !== "object") return null;
  const action = String(edit.action || "").toLowerCase();
  const label = shortFoodName(edit.label);
  const replaced = shortFoodName(edit.replace_label || edit.swap_out_label);
  if (action === "add") return `added ${label}`;
  if (action === "remove") return `removed ${label}`;
  if (action === "swap") {
    if (edit.replace_label || edit.swap_out_label) return `swapped ${replaced}`;
    return `swapped in ${label}`;
  }
  return null;
}

function collectEditPhrases(payload) {
  const edits = [];
  const push = (e) => {
    const phrase = editPhraseFromPayload(e);
    if (phrase) edits.push(phrase);
  };
  const decision = payload?.decision || {};
  for (const e of decision.edits || []) push(e);

  const lac = payload?.last_applied_candidate;
  if (lac) {
    if (Array.isArray(lac.edits) && lac.edits.length) {
      for (const e of lac.edits) push(e);
    } else if (lac.action) {
      push(lac);
    }
  }

  for (const tool of payload?.tools_used || []) {
    const out = tool.output || tool.output_summary || {};
    for (const e of out.edits || []) push(e);
    if (out.action && out.label && !(out.edits || []).length) push(out);
  }

  const bid = decision.chosen_bundle_id;
  if (bid != null) {
    for (const b of payload?.bundles || []) {
      if (String(b?.bundle_id) !== String(bid)) continue;
      for (const e of b.edits || []) push(e);
      break;
    }
  }

  const outcomes = payload?.decision_outcomes || [];
  if (outcomes.length) {
    const od = outcomes[outcomes.length - 1]?.decision || {};
    for (const e of od.edits || []) push(e);
  }

  return [...new Set(edits)];
}

function lossImprovementPct(first, current) {
  if (first == null || current == null) return null;
  const a = Number(first);
  const b = Number(current);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 1e-12) return null;
  if (b >= a - 1e-12) return null;
  const pct = Math.round(((a - b) / a) * 100);
  return pct >= 1 ? pct : null;
}

function liveLossValue(live, key) {
  const raw = live?.[key];
  if (raw == null) return null;
  if (typeof raw === "object") return raw.value ?? null;
  return raw;
}

function stepDetailForNode(node, payload, baseDetail) {
  if (payload?.progress_detail) return String(payload.progress_detail);

  const unique = collectEditPhrases(payload);
  if ((node === "apply" || node === "apply_or_expand" || node === "decide") && unique.length) {
    return unique.join("; ");
  }

  if (node === "diagnose") {
    const live = payload?.live_scores || {};
    const ratio = liveLossValue(live, "ratio_loss");
    const nutrient = liveLossValue(live, "nutrient_loss");
    const parts = [];

    // Prefer latest outcome edits on re-diagnose after an apply.
    const outcomes = payload?.decision_outcomes || [];
    if (outcomes.length) {
      const od = outcomes[outcomes.length - 1]?.decision || {};
      const recent = [];
      for (const e of od.edits || []) {
        const phrase = editPhraseFromPayload(e);
        if (phrase) recent.push(phrase);
      }
      if (recent.length) parts.push([...new Set(recent)].join("; "));
    }

    if (ratio != null && Number.isFinite(Number(ratio))) {
      const r = Number(ratio);
      if (state.firstRatio == null) state.firstRatio = r;
      else {
        const pct = lossImprovementPct(state.firstRatio, r);
        if (pct != null) parts.push(`Improved cookability by ${pct}%`);
      }
    }
    if (nutrient != null && Number.isFinite(Number(nutrient))) {
      const n = Number(nutrient);
      if (state.firstNutrient == null) state.firstNutrient = n;
      else {
        const pct = lossImprovementPct(state.firstNutrient, n);
        if (pct != null) parts.push(`Improved nutrient fit by ${pct}%`);
      }
    }
    if (parts.length) return parts.join(" · ");
  }

  if (node === "propose" && (payload?.bundles || []).length) {
    const n = payload.bundles.length;
    return `Scored ${n} edit bundle${n === 1 ? "" : "s"}`;
  }

  return baseDetail;
}

function hasDistributionData(ing) {
  const iqr = ing?.share_iqr;
  if (!iqr) return false;
  const q1 = Number(iqr.q1);
  const q3 = Number(iqr.q3);
  if (![q1, q3].every(Number.isFinite)) return false;
  const n = Number(iqr.n);
  if (Number.isFinite(n) && n < 5) return false;
  const lo = Math.min(q1, q3);
  const hi = Math.max(q1, q3);
  const width = hi - lo;
  if (width <= 1e-12) return false;
  const scale = Math.max(Math.abs(lo), Math.abs(hi), 1e-9);
  if (width / scale <= 1e-6) return false;
  return true;
}

function isZeroPortion(ing) {
  const grams = Number(ing?.grams);
  if (Number.isFinite(grams)) return Math.abs(grams) <= 1e-9;
  const amount = Number(ing?.amount_value);
  if (Number.isFinite(amount)) return Math.abs(amount) <= 1e-9;
  const cal = Number(ing?.calories);
  if (Number.isFinite(cal) && cal > 0) return false;
  return true;
}

function shouldShowIngredient(ing) {
  const name = ingredientLabel(ing);
  if (!name) return false;
  // Zero amount + no neighborhood distribution → omit (phantom / empty lines).
  if (isZeroPortion(ing) && !hasDistributionData(ing)) return false;
  return true;
}

function shareToneFromIqr(recipeShare, iqr) {
  if (recipeShare == null || !iqr) return null;
  const share = Number(recipeShare);
  const q1 = Number(iqr.q1);
  const q3 = Number(iqr.q3);
  const n = Number(iqr.n);
  if (![share, q1, q3].every(Number.isFinite)) return null;
  // Grey = too few neighbor recipes to trust the band (not related to amount).
  if (Number.isFinite(n) && n < 5) return "tone-unknown";
  const lo = Math.min(q1, q3);
  const hi = Math.max(q1, q3);
  const width = hi - lo;
  // Degenerate / zero-width bands: do not penalize (treat as unknown).
  if (width <= 1e-12) return "tone-unknown";
  const scale = Math.max(Math.abs(lo), Math.abs(hi), 1e-9);
  if (width / scale <= 1e-6) return "tone-unknown";
  const edgeEps = 0.01;
  const fenceLo = q1 - 1.5 * width;
  const fenceHi = q3 + 1.5 * width;
  // Soften IQR edges so shares sitting on q1/q3 stay green.
  if (share >= q1 - edgeEps && share <= q3 + edgeEps) return "tone-good";
  if (share >= fenceLo && share <= fenceHi) return "tone-warn";
  return "tone-bad";
}

function lossTone(band, lossContribution, ing) {
  const fromIqr = shareToneFromIqr(ing?.recipe_share, ing?.share_iqr);
  if (fromIqr) return fromIqr;
  if (band === "good") return "tone-good";
  if (band === "warn") return "tone-warn";
  if (band === "bad") return "tone-bad";
  if (lossContribution == null) return "tone-unknown";
  const v = Number(lossContribution);
  if (!Number.isFinite(v)) return "tone-unknown";
  if (v <= 0.05) return "tone-good";
  if (v <= 0.15) return "tone-warn";
  return "tone-bad";
}

function shareBoxplot(iqr, recipeShare, toneClass, { editable = false, idx = null } = {}) {
  if (!iqr || iqr.min == null || iqr.max == null) {
    return `<span class="spark-empty">—</span>`;
  }
  const min = Number(iqr.min);
  const max = Number(iqr.max);
  const q1 = Number(iqr.q1);
  const med = Number(iqr.median);
  const q3 = Number(iqr.q3);
  const w = 120;
  const h = 28;
  const pad = 6;
  const span = max - min || 1e-9;
  const x = (v) => pad + ((Number(v) - min) / span) * (w - 2 * pad);
  const y = h / 2;
  const clamped =
    recipeShare == null || Number.isNaN(Number(recipeShare))
      ? null
      : Math.min(max, Math.max(min, Number(recipeShare)));
  const marker =
    clamped == null
      ? ""
      : `<circle class="bw-marker ${toneClass || ""}" cx="${x(clamped)}" cy="${y}" r="4" />`;
  const editableAttrs = editable
    ? ` class="share-boxplot editable" data-edit-idx="${idx}" data-min="${min}" data-max="${max}" role="slider" aria-valuemin="${min}" aria-valuemax="${max}" aria-valuenow="${
        clamped ?? med
      }" aria-label="Drag to edit typical mass share" tabindex="0"`
    : ` class="share-boxplot" aria-hidden="true"`;
  return `<svg${editableAttrs} viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">
    <line class="bw-whisker" x1="${x(min)}" y1="${y}" x2="${x(max)}" y2="${y}" />
    <rect class="bw-box" x="${x(q1)}" y="${y - 6}" width="${Math.max(1, x(q3) - x(q1))}" height="12" rx="2" />
    <line class="bw-median" x1="${x(med)}" y1="${y - 7}" x2="${x(med)}" y2="${y + 7}" />
    ${marker}
  </svg>`;
}

function gramsToDisplay(grams, mode, ing) {
  const g = Number(grams);
  if (!Number.isFinite(g)) return "—";
  if (mode === "oz") return `${(g / G_PER_OZ).toFixed(Math.abs(g / G_PER_OZ) >= 10 ? 0 : 1)} oz`;
  if (mode === "lb") return `${(g / G_PER_LB).toFixed(2)} lb`;
  if (mode === "g") return `${Math.round(g)} g`;
  return ingredientAmount(ing);
}

function displayToGrams(value, mode) {
  const v = Number(value);
  if (!Number.isFinite(v)) return null;
  if (mode === "oz") return v * G_PER_OZ;
  if (mode === "lb") return v * G_PER_LB;
  return v;
}

function macroTargetsFromFinal(final) {
  const box = final?.macro_targets || final?.display_scores?.macro_targets || {};
  const out = {};
  for (const key of MACRO_KEYS) {
    const lo = box[`${key}_min`];
    const hi = box[`${key}_max`];
    if (lo != null && hi != null) {
      out[`${key}_min`] = Number(lo);
      out[`${key}_max`] = Number(hi);
      continue;
    }
    out[`${key}_min`] = frac(`${key}_min`);
    out[`${key}_max`] = frac(`${key}_max`);
  }
  return out;
}

function macroAxisStatus(key, valuePct, targets) {
  if (valuePct == null || !Number.isFinite(Number(valuePct))) return null;
  const lo = Number(targets[`${key}_min`]) * 100;
  const hi = Number(targets[`${key}_max`]) * 100;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  const v = Number(valuePct);
  if (v < lo - 0.05) {
    return { key, status: "low", value: v, lo, hi, delta: lo - v };
  }
  if (v > hi + 0.05) {
    return { key, status: "high", value: v, lo, hi, delta: v - hi };
  }
  return { key, status: "ok", value: v, lo, hi, delta: 0 };
}

function macroLabel(key) {
  if (key === "carb") return "Carbs";
  if (key === "protein") return "Protein";
  if (key === "fat") return "Fat";
  return key;
}

function macroTipCopy(status) {
  if (!status || status.status === "ok") return "";
  const name = macroLabel(status.key);
  const range = `${Math.round(status.lo)}–${Math.round(status.hi)}%`;
  if (status.status === "low") {
    return `${name} is ${Math.round(status.value)}%, below your ${range} target. Raise foods rich in ${name.toLowerCase()}, or trim foods that dilute it.`;
  }
  return `${name} is ${Math.round(status.value)}%, above your ${range} target. Trim foods driving ${name.toLowerCase()}, or balance with the other macros.`;
}

function ingredientFocusIdxs(ings, problem, status) {
  if (!status || status.status === "ok") return [];
  const M = problem?.M;
  const rowIdx = { protein: 0, carb: 1, fat: 2 }[status.key];
  if (rowIdx == null || !Array.isArray(M) || !M[rowIdx]) return [];
  const row = M[rowIdx];
  const scored = ings
    .map((ing, idx) => {
      const dens = Number(row[idx]);
      const grams = Number(ing.grams) || 0;
      if (!Number.isFinite(dens)) return null;
      return { idx, dens, contribution: dens * grams };
    })
    .filter(Boolean);
  if (!scored.length) return [];
  const ranked =
    status.status === "high"
      ? [...scored].sort((a, b) => b.contribution - a.contribution)
      : [...scored].sort((a, b) => b.dens - a.dens || b.contribution - a.contribution);
  return ranked.slice(0, Math.min(4, ranked.length)).map((r) => r.idx);
}

function amountsAreDirty() {
  if (!state.originalGrams || !state.liveIngredients?.length) return false;
  if (state.originalGrams.length !== state.liveIngredients.length) return true;
  return state.liveIngredients.some((row, i) => !gramsNear(row.grams, state.originalGrams[i]));
}

function syncRevertButton() {
  if (!els.revertAmounts) return;
  els.revertAmounts.classList.toggle("hidden", !amountsAreDirty());
}

function hideMacroTip() {
  state.activeMacroTip = null;
  if (!els.macroTip) return;
  els.macroTip.classList.add("hidden");
  els.macroTip.innerHTML = "";
}

function showMacroTip(status) {
  if (!els.macroTip || !status || status.status === "ok") {
    hideMacroTip();
    return;
  }
  state.activeMacroTip = status.key;
  const idxs = ingredientFocusIdxs(
    state.liveIngredients,
    state.lastFinal?.problem || {},
    status
  );
  els.macroTip.classList.remove("hidden");
  els.macroTip.innerHTML = `
    <div>${escapeHtml(macroTipCopy(status))}</div>
    <div class="tip-actions">
      <button type="button" class="primary" data-macro-highlight="${escapeHtml(status.key)}">
        Highlight ingredients to edit
      </button>
      <button type="button" data-macro-clear>Clear highlight</button>
      <button type="button" data-macro-dismiss>Dismiss</button>
    </div>`;
  els.macroTip.querySelector("[data-macro-highlight]")?.addEventListener("click", () => {
    state.macroFocus = { key: status.key, idxs };
    renderIngredientTable(state.liveIngredients);
  });
  els.macroTip.querySelector("[data-macro-clear]")?.addEventListener("click", () => {
    state.macroFocus = null;
    renderIngredientTable(state.liveIngredients);
  });
  els.macroTip.querySelector("[data-macro-dismiss]")?.addEventListener("click", hideMacroTip);
}

function renderMacros(macros, pfc) {
  const protein =
    macros?.protein != null
      ? macros.protein
      : pfc?.protein != null
        ? Math.round(Number(pfc.protein) * 100)
        : null;
  const carb =
    macros?.carb != null
      ? macros.carb
      : pfc?.carb != null || pfc?.carbs != null
        ? Math.round(Number(pfc.carb ?? pfc.carbs) * 100)
        : null;
  const fat =
    macros?.fat != null
      ? macros.fat
      : pfc?.fat != null
        ? Math.round(Number(pfc.fat) * 100)
        : null;
  const calories = macros?.calories;
  const calInput =
    state.calorieScaleTarget != null && Number.isFinite(Number(state.calorieScaleTarget))
      ? state.calorieScaleTarget
      : calories;
  const targets = macroTargetsFromFinal(state.lastFinal);
  const statuses = {
    protein: macroAxisStatus("protein", protein, targets),
    carb: macroAxisStatus("carb", carb, targets),
    fat: macroAxisStatus("fat", fat, targets),
  };
  const chip = (key, label, value, suffix = "%") => {
    const st = statuses[key];
    const out = st && st.status !== "ok";
    const dir = out ? ` direction-${st.status === "high" ? "high" : "low"}` : "";
    const bang = out
      ? `<button type="button" class="macro-bang" data-macro-tip="${key}" title="${escapeHtml(
          macroTipCopy(st)
        )}" aria-label="${escapeHtml(macroTipCopy(st))}">!</button>`
      : "";
    return `<div class="macro-chip${out ? " out" : ""}${dir}" data-macro="${key}">
      <span class="label">${label}</span>
      <strong>${value != null ? `${value}${suffix}` : "—"}</strong>
      ${bang}
    </div>`;
  };
  els.resultMacros.innerHTML = `
    ${chip("protein", "Protein", protein)}
    ${chip("carb", "Carbs", carb)}
    ${chip("fat", "Fat", fat)}
    <div class="macro-chip macro-chip-calories" data-macro="calories">
      <span class="label">Calories</span>
      <div class="calorie-edit-row">
        <input type="number" class="calorie-input" id="result-calorie-input" min="100" max="8000" step="1"
          value="${calInput != null ? escapeHtml(calInput) : ""}"
          aria-label="Target calories" ${calories == null ? "disabled" : ""} />
        <button type="button" class="text-btn calorie-scale-btn" id="scale-to-target-cals"
          title="Scale all ingredients so the recipe hits this calorie total"
          ${calories == null ? "disabled" : ""}>Scale to target cals</button>
      </div>
    </div>`;

  const calEl = els.resultMacros.querySelector("#result-calorie-input");
  const scaleBtn = els.resultMacros.querySelector("#scale-to-target-cals");
  calEl?.addEventListener("input", () => {
    const v = Number(calEl.value);
    state.calorieScaleTarget = Number.isFinite(v) ? v : calEl.value;
  });
  calEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      scaleBtn?.click();
    }
  });
  scaleBtn?.addEventListener("click", async (e) => {
    e.stopPropagation();
    try {
      await scaleRecipeToTargetCalories(Number(calEl?.value));
    } catch (err) {
      els.runStatus.textContent = `Could not scale calories: ${err.message || err}`;
      els.runStatus.classList.add("error");
    }
  });

  els.resultMacros.querySelectorAll("[data-macro-tip]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const key = btn.dataset.macroTip;
      const st = statuses[key];
      if (state.activeMacroTip === key) hideMacroTip();
      else showMacroTip(st);
    });
  });
  if (state.activeMacroTip && statuses[state.activeMacroTip]?.status !== "ok") {
    showMacroTip(statuses[state.activeMacroTip]);
  } else if (state.activeMacroTip) {
    hideMacroTip();
  }
}

async function scaleRecipeToTargetCalories(targetRaw) {
  const target = Number(targetRaw);
  if (!Number.isFinite(target) || target < 100 || target > 8000) {
    throw new Error("Enter a calorie target between 100 and 8000.");
  }
  const current = Number(state.lastFinal?.display_scores?.macros?.calories);
  if (!Number.isFinite(current) || current <= 1e-6) {
    throw new Error("Current recipe calories are unknown, so amounts cannot be scaled.");
  }
  const ings = state.liveIngredients || [];
  if (!ings.length) throw new Error("No ingredients to scale.");

  if (Math.abs(current - target) / target <= 0.005) {
    state.calorieScaleTarget = null;
    const macros = {
      ...(state.lastFinal?.display_scores?.macros || {}),
      calories: Math.round(target),
    };
    if (state.lastFinal?.display_scores) state.lastFinal.display_scores.macros = macros;
    renderMacros(macros, state.lastFinal?.display_scores?.pfc_after);
    els.runStatus.textContent = `Already at ${Math.round(target)} calories.`;
    els.runStatus.classList.remove("error");
    return;
  }

  const scale = target / current;
  const next = ings.map((row) => Math.max(0.5, (Number(row.grams) || 0) * scale));
  state.calorieScaleTarget = null;
  // Don't snap individuals back to the pre-scale baseline mid-scale.
  await recomputeFromGrams(next, { allowBaselineRestore: false });
  // Prefer the user target on the chip when recompute lands within ~2%.
  const landed = Number(state.lastFinal?.display_scores?.macros?.calories);
  if (Number.isFinite(landed) && Math.abs(landed - target) / target <= 0.02) {
    const macros = {
      ...(state.lastFinal.display_scores.macros || {}),
      calories: Math.round(target),
    };
    state.lastFinal.display_scores.macros = macros;
    const active = state.browseCandidates[state.browseIndex];
    if (active?.display_scores) {
      active.display_scores.macros = macros;
      if (active.score_summary) active.score_summary.macros = macros;
    }
    renderMacros(macros, state.lastFinal.display_scores.pfc_after);
  }
  els.runStatus.textContent = `Scaled recipe to ${Math.round(target)} calories.`;
  els.runStatus.classList.remove("error");
}

function renderVerdict(scores) {
  const ratio = scores?.ratio_loss || {};
  const nutrient = scores?.nutrient_loss || {};
  const cook = scores?.cookability || {};
  const parts = [];
  if (cook.summary) {
    parts.push(`<span class="verdict-bit">${escapeHtml(cook.summary)}</span>`);
  }
  if (ratio.band_summary) {
    const css = ratio.proportion_css || proportionCssFromBand(ratio.band);
    parts.push(
      `<span class="verdict-proportion proportion-${escapeHtml(css)}">${escapeHtml(
        ratio.band_summary
      )}</span>`
    );
  }
  if (nutrient.band_summary) {
    parts.push(
      `<span class="verdict-bit band-${escapeHtml(nutrient.band || "unknown")}">${escapeHtml(
        nutrient.band_summary
      )}</span>`
    );
  }
  els.resultVerdict.innerHTML = parts.join(" ");
  els.resultVerdict.className = "result-verdict";
}

function proportionCssFromBand(band) {
  if (band === "good") return "very-typical";
  if (band === "warn") return "somewhat-different";
  if (band === "bad") return "substantially-off";
  return "unknown";
}

function closeMenus() {
  state.openMenu = null;
  els.resultIngs.querySelectorAll(".ing-menu.open").forEach((el) => el.classList.remove("open"));
}

function captureBaselineSnapshot() {
  const scores = state.lastFinal?.display_scores || {};
  state.baselineSnapshot = {
    ingredients: (state.liveIngredients || []).map((r) => ({ ...r })),
    display_scores: JSON.parse(JSON.stringify(scores)),
    problem: state.lastFinal?.problem
      ? JSON.parse(JSON.stringify(state.lastFinal.problem))
      : null,
    originalGrams: Array.isArray(state.originalGrams) ? [...state.originalGrams] : null,
  };
}

function cloneBaselineSnapshot(snap) {
  if (!snap) return null;
  return {
    ingredients: (snap.ingredients || []).map((r) => ({ ...r })),
    display_scores: snap.display_scores
      ? JSON.parse(JSON.stringify(snap.display_scores))
      : null,
    problem: snap.problem ? JSON.parse(JSON.stringify(snap.problem)) : null,
    originalGrams: Array.isArray(snap.originalGrams) ? [...snap.originalGrams] : null,
  };
}

function gramsNear(a, b, atol = 0.05, rtol = 0.001) {
  const x = Number(a);
  const y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  return Math.abs(x - y) <= Math.max(atol, rtol * Math.max(Math.abs(x), Math.abs(y)));
}

function shareNear(a, b, atol = 0.005) {
  const x = Number(a);
  const y = Number(b);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  return Math.abs(x - y) <= atol;
}

function snapGramsToBaseline(grams) {
  const base = state.originalGrams;
  if (!Array.isArray(base) || !Array.isArray(grams) || base.length !== grams.length) {
    return grams.map((g) => Number(g));
  }
  return grams.map((g, i) => (gramsNear(g, base[i]) ? Number(base[i]) : Number(g)));
}

function gramsMatchBaseline(grams) {
  const base = state.originalGrams;
  if (!Array.isArray(base) || !Array.isArray(grams) || base.length !== grams.length) {
    return false;
  }
  return grams.every((g, i) => gramsNear(g, base[i]));
}

function restoreBaselineSnapshot() {
  const snap = state.baselineSnapshot;
  if (!snap?.ingredients?.length || !snap.display_scores) return false;
  state.liveIngredients = snap.ingredients.map((r) => ({ ...r }));
  state.originalGrams = Array.isArray(snap.originalGrams)
    ? [...snap.originalGrams]
    : state.liveIngredients.map((r) => Number(r.grams) || 0);
  const final = state.lastFinal || {};
  const nextScores = {
    ...snap.display_scores,
    ingredients: state.liveIngredients,
  };
  state.lastFinal = {
    ...final,
    display_scores: nextScores,
    problem: snap.problem || final.problem,
  };
  const active = state.browseCandidates[state.browseIndex];
  if (active) {
    active.display_scores = nextScores;
    active.problem = state.lastFinal.problem;
    active.liveIngredients = state.liveIngredients.map((r) => ({ ...r }));
    active.originalGrams = [...state.originalGrams];
    active.score_summary = {
      ...(active.score_summary || {}),
      macros: nextScores.macros,
      ratio_loss: nextScores.ratio_loss?.value,
      ratio_band: nextScores.ratio_loss?.band,
      nutrient_loss: nextScores.nutrient_loss?.value,
      nutrient_band: nextScores.nutrient_loss?.band,
      cookability: nextScores.cookability?.summary,
    };
  }
  renderMacros(nextScores.macros, nextScores.pfc_after);
  renderVerdict(nextScores);
  renderIngredientTable(state.liveIngredients);
  renderCandidateChrome();
  syncRevertButton();
  return true;
}

async function recomputeFromGrams(grams, { allowBaselineRestore = true } = {}) {
  const final = state.lastFinal;
  if (!final) return;
  const snapped = allowBaselineRestore
    ? snapGramsToBaseline(grams)
    : (grams || []).map((g) => Number(g));
  if (allowBaselineRestore && gramsMatchBaseline(snapped) && restoreBaselineSnapshot()) {
    return;
  }
  const scores = final.display_scores || {};
  const ings = state.liveIngredients.length
    ? state.liveIngredients
    : scores.ingredients || [];
  const res = await fetch("/api/recipe/recompute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      problem: final.problem || {},
      ingredients: ings,
      grams: snapped,
      macro_targets: final.macro_targets || macroTargetsFromFinal(final),
      score_history: scores.score_history || final.score_history || [],
      baseline_ratio: state.firstRatio ?? scores.ratio_loss?.value ?? null,
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || res.statusText);
  // Keep a stable 1:1 row mapping with the pre-edit list (do not drop fields on shorter payloads).
  const returned = Array.isArray(data.ingredients) ? data.ingredients : [];
  state.liveIngredients = ings.map((prev, i) => {
    const row = returned[i] || {};
    return {
      ...prev,
      ...row,
      grams: snapped[i] != null ? Number(snapped[i]) : Number(row.grams ?? prev.grams) || 0,
      edit_note: row.edit_note || prev.edit_note,
      edit_action: row.edit_action || prev.edit_action,
      added_during_process: row.added_during_process ?? prev.added_during_process ?? false,
      added_during_process_title:
        row.added_during_process_title || prev.added_during_process_title,
      portion_gram_weight: row.portion_gram_weight || prev.portion_gram_weight,
      quantity: prev.quantity,
      unit: prev.unit,
      original_grams: prev.original_grams,
      // Preserve kitchen unit identity for round-trip typing.
      amount_unit: row.amount_unit || prev.amount_unit,
      amount_source: row.amount_source || prev.amount_source,
    };
  });
  const nextScores = {
    ...scores,
    macros: data.macros,
    pfc_after: data.pfc_after,
    ratio_loss: { ...(scores.ratio_loss || {}), ...(data.ratio_loss || {}) },
    nutrient_loss: { ...(scores.nutrient_loss || {}), ...(data.nutrient_loss || {}) },
    cookability: data.cookability || scores.cookability,
    ingredients: state.liveIngredients,
  };
  state.lastFinal = { ...final, display_scores: nextScores };
  // Keep the active browse candidate in sync when the user edits amounts.
  const active = state.browseCandidates[state.browseIndex];
  if (active) {
    active.display_scores = nextScores;
    active.problem = state.lastFinal.problem;
    active.liveIngredients = state.liveIngredients.map((r) => ({ ...r }));
    active.score_summary = {
      ...(active.score_summary || {}),
      macros: nextScores.macros,
      ratio_loss: nextScores.ratio_loss?.value,
      ratio_band: nextScores.ratio_loss?.band,
      nutrient_loss: nextScores.nutrient_loss?.value,
      nutrient_band: nextScores.nutrient_loss?.band,
      cookability: nextScores.cookability?.summary,
    };
  }
  renderMacros(nextScores.macros, nextScores.pfc_after);
  renderVerdict(nextScores);
  renderIngredientTable(state.liveIngredients);
  renderCandidateChrome();
  syncRevertButton();
}

function shareToGrams(targetShare, idx) {
  const s = Number(targetShare);
  if (!Number.isFinite(s) || s < 0) return null;
  const ings = state.liveIngredients || [];
  if (!ings.length || idx < 0 || idx >= ings.length) return null;

  // Hold other ingredients fixed: share = g / (g + others) ⇒ g = share/(1-share)*others.
  // This round-trips: setting the original share restores the original grams.
  let others = 0;
  for (let i = 0; i < ings.length; i++) {
    if (i === idx) continue;
    others += Number(ings[i].grams) || 0;
  }
  const base = state.originalGrams?.[idx];
  const baseOthers = (state.originalGrams || []).reduce(
    (acc, g, i) => (i === idx ? acc : acc + (Number(g) || 0)),
    0
  );
  if (others <= 1e-9) {
    const fallbackTotal =
      (state.originalGrams || []).reduce((a, b) => a + (Number(b) || 0), 0) ||
      ings.reduce((a, r) => a + (Number(r.grams) || 0), 0) ||
      100;
    const grams = Math.max(0.5, Math.min(0.999, s) * fallbackTotal);
    if (base != null && gramsNear(grams, base)) return Number(base);
    return grams;
  }
  const share = Math.min(0.999, s);
  // If others are still at baseline and the marker is near the original share,
  // snap exactly so macros/verdict match the start (pixel drag is imprecise).
  if (base != null && Number.isFinite(baseOthers) && baseOthers > 1e-9 && gramsNear(others, baseOthers)) {
    const baseShare = Number(base) / (Number(base) + baseOthers);
    if (shareNear(share, baseShare)) return Number(base);
  }
  const grams = (share / Math.max(1e-12, 1 - share)) * others;
  if (base != null && gramsNear(grams, base)) return Number(base);
  return Math.max(0.5, grams);
}

function bindEditableBoxplot(svg, applyGramsAt) {
  const idx = Number(svg.dataset.editIdx);
  const min = Number(svg.dataset.min);
  const max = Number(svg.dataset.max);
  if (!Number.isFinite(idx) || !Number.isFinite(min) || !Number.isFinite(max)) return;
  const pad = 6;
  const w = 120;
  const span = max - min || 1e-9;

  const shareFromClientX = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const xSvg = ((clientX - rect.left) / Math.max(rect.width, 1)) * w;
    const t = (xSvg - pad) / (w - 2 * pad);
    return min + Math.min(1, Math.max(0, t)) * span;
  };

  const updateMarker = (share) => {
    const marker = svg.querySelector(".bw-marker");
    if (!marker) return;
    const cx = pad + ((share - min) / span) * (w - 2 * pad);
    marker.setAttribute("cx", String(cx));
    svg.setAttribute("aria-valuenow", String(share));
  };

  let dragging = false;
  const onMove = (clientX) => {
    const share = shareFromClientX(clientX);
    updateMarker(share);
    return share;
  };
  const finish = async (clientX) => {
    const share = onMove(clientX);
    const grams = shareToGrams(share, idx);
    if (grams != null) await applyGramsAt(idx, grams);
  };

  svg.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragging = true;
    svg.setPointerCapture?.(e.pointerId);
    onMove(e.clientX);
  });
  svg.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    onMove(e.clientX);
  });
  svg.addEventListener("pointerup", async (e) => {
    if (!dragging) return;
    dragging = false;
    try {
      await finish(e.clientX);
    } catch (err) {
      els.runStatus.textContent = `Could not update amounts: ${err.message || err}`;
      els.runStatus.classList.add("error");
    }
  });
  svg.addEventListener("keydown", async (e) => {
    const step = (max - min) / 40;
    let share = Number(svg.getAttribute("aria-valuenow"));
    if (!Number.isFinite(share)) share = (min + max) / 2;
    if (e.key === "ArrowLeft") share = Math.max(min, share - step);
    else if (e.key === "ArrowRight") share = Math.min(max, share + step);
    else return;
    e.preventDefault();
    updateMarker(share);
    const grams = shareToGrams(share, idx);
    if (grams != null) await applyGramsAt(idx, grams);
  });
}

function renderIngredientTable(ings) {
  if (!ings?.length) {
    els.resultIngs.innerHTML = `<p class="hint">No ingredients were returned for this run.</p>`;
    return;
  }
  const focusSet = new Set(state.macroFocus?.idxs || []);
  const rows = ings
    .map((ing, idx) => {
      if (!shouldShowIngredient(ing)) return "";
      const name = ingredientLabel(ing);
      const tone = lossTone(ing.loss_band, ing.loss_contribution, ing);
      const mode = state.unitMode[idx] || "kitchen";
      const amount = gramsToDisplay(ing.grams, mode, ing);
      const note = ing.edit_note ? `<span class="edit-note">${escapeHtml(ing.edit_note)}</span>` : "";
      const novel =
        ing.added_during_process
          ? `<span class="novel-star" title="${escapeHtml(
              ing.added_during_process_title ||
                "Added during the process (not in the reference recipe)"
            )}" aria-label="${escapeHtml(
              ing.added_during_process_title ||
                "Added during the process (not in the reference recipe)"
            )}">*</span>`
          : "";
      const modeAmount = Number(ing.grams) || 0;
      const focusClass = focusSet.has(idx) ? " macro-focus" : "";
      const editing = state.editingAmountIdx === idx;
      const editVal = amountEditValue(ing, mode);
      const editUnit = amountEditUnitLabel(ing, mode);
      const amountHtml = editing
        ? `<span class="amount-editor">
            <input type="number" class="amount-inline-input" data-idx="${idx}" step="any" min="0" value="${escapeHtml(
              editVal
            )}" aria-label="Edit amount" />
            ${editUnit ? `<span class="amount-inline-unit">${escapeHtml(editUnit)}</span>` : ""}
          </span>`
        : `<button type="button" class="amount-text-btn ${tone}" data-edit-amount="${idx}" title="Click to edit amount">${escapeHtml(
            amount
          )}</button>`;
      return `<tr class="ing-row ${tone}${focusClass}" data-idx="${idx}">
        <td class="ing-name">
          <div class="ing-name-line">
            <strong class="${tone}">${escapeHtml(name)}${novel}</strong>
            ${note}
          </div>
          <div class="ing-boxplot">${shareBoxplot(ing.share_iqr, ing.recipe_share, tone, {
            editable: true,
            idx,
          })}</div>
        </td>
        <td class="num amount-cell ${tone}">${amountHtml}</td>
        <td class="ing-actions">
          <button type="button" class="dots-btn" data-menu="${idx}" aria-label="Ingredient options">⋯</button>
          <div class="ing-menu" data-menu-panel="${idx}">
            <label class="menu-label">Amount
              <input type="number" class="mass-input" data-idx="${idx}" step="any" value="${
                mode === "kitchen"
                  ? ""
                  : mode === "oz"
                    ? (modeAmount / G_PER_OZ).toFixed(2)
                    : mode === "lb"
                      ? (modeAmount / G_PER_LB).toFixed(3)
                      : Math.round(modeAmount)
              }" ${mode === "kitchen" ? 'disabled placeholder="Click amount or switch unit"' : ""} />
            </label>
            <div class="unit-toggles" data-idx="${idx}">
              <button type="button" data-unit="kitchen" class="${mode === "kitchen" ? "active" : ""}">Recipe unit</button>
              <button type="button" data-unit="g" class="${mode === "g" ? "active" : ""}">g</button>
              <button type="button" data-unit="oz" class="${mode === "oz" ? "active" : ""}">oz</button>
              <button type="button" data-unit="lb" class="${mode === "lb" ? "active" : ""}">lb</button>
            </div>
          </div>
        </td>
      </tr>`;
    })
    .filter(Boolean)
    .join("");
  els.resultIngs.innerHTML = `
    <table class="ing-table">
      <thead><tr><th>Ingredient <span class="col-slide-hint">slide to edit</span></th><th>Amount</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hint">Boxplot: whiskers = neighborhood range, box = P15–P85 central band, line = median, colored dot = this recipe’s mass share (drag to edit). Green = within band, orange = outside band, red = outlier (beyond 1.5× band width). Grey = fewer than 5 neighbor recipes or a degenerate (zero-width) band — excluded from proportion fit.</p>`;

  const applyGramsAt = async (idx, grams) => {
    const next = state.liveIngredients.map((row, i) =>
      i === idx ? Number(grams) : Number(row.grams) || 0
    );
    try {
      await recomputeFromGrams(next);
      if (state.openMenu != null) {
        const panel = els.resultIngs.querySelector(`[data-menu-panel="${state.openMenu}"]`);
        panel?.classList.add("open");
      }
    } catch (err) {
      els.runStatus.textContent = `Could not update amounts: ${err.message || err}`;
      els.runStatus.classList.add("error");
    }
  };

  const commitAmountEdit = async (idx, rawValue) => {
    const ing = state.liveIngredients[idx];
    if (!ing) {
      state.editingAmountIdx = null;
      renderIngredientTable(state.liveIngredients);
      return;
    }
    const mode = state.unitMode[idx] || "kitchen";
    const grams =
      mode === "kitchen"
        ? kitchenQtyToGrams(ing, rawValue, idx)
        : displayToGrams(rawValue, mode);
    if (mode !== "kitchen" && grams != null && state.originalGrams?.[idx] != null) {
      const snapped = gramsNear(grams, state.originalGrams[idx])
        ? Number(state.originalGrams[idx])
        : grams;
      state.editingAmountIdx = null;
      await applyGramsAt(idx, snapped);
      return;
    }
    state.editingAmountIdx = null;
    if (grams == null || !Number.isFinite(Number(grams))) {
      renderIngredientTable(state.liveIngredients);
      return;
    }
    await applyGramsAt(idx, grams);
  };

  els.resultIngs.querySelectorAll("[data-edit-amount]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      closeMenus();
      state.editingAmountIdx = Number(btn.dataset.editAmount);
      renderIngredientTable(state.liveIngredients);
      const input = els.resultIngs.querySelector(
        `.amount-inline-input[data-idx="${state.editingAmountIdx}"]`
      );
      if (input) {
        input.focus();
        input.select();
      }
    });
  });

  els.resultIngs.querySelectorAll(".amount-inline-input").forEach((input) => {
    const idx = Number(input.dataset.idx);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commitAmountEdit(idx, input.value);
      } else if (e.key === "Escape") {
        e.preventDefault();
        state.editingAmountIdx = null;
        renderIngredientTable(state.liveIngredients);
      }
    });
    input.addEventListener("blur", () => {
      if (state.editingAmountIdx !== idx) return;
      commitAmountEdit(idx, input.value);
    });
    input.addEventListener("click", (e) => e.stopPropagation());
  });

  els.resultIngs.querySelectorAll(".dots-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = Number(btn.dataset.menu);
      const panel = els.resultIngs.querySelector(`[data-menu-panel="${idx}"]`);
      const opening = !panel.classList.contains("open");
      closeMenus();
      if (opening) {
        state.openMenu = idx;
        renderIngredientTable(state.liveIngredients);
        const reopened = els.resultIngs.querySelector(`[data-menu-panel="${idx}"]`);
        reopened?.classList.add("open");
        state.openMenu = idx;
      }
    });
  });

  els.resultIngs.querySelectorAll(".unit-toggles button").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const wrap = btn.parentElement;
      const idx = Number(wrap.dataset.idx);
      state.unitMode[idx] = btn.dataset.unit;
      renderIngredientTable(state.liveIngredients);
      const panel = els.resultIngs.querySelector(`[data-menu-panel="${idx}"]`);
      panel?.classList.add("open");
      state.openMenu = idx;
    });
  });

  els.resultIngs.querySelectorAll(".mass-input").forEach((input) => {
    input.addEventListener("change", () => {
      const idx = Number(input.dataset.idx);
      const mode = state.unitMode[idx] || "kitchen";
      if (mode === "kitchen") return;
      const g = displayToGrams(input.value, mode);
      if (g != null) applyGramsAt(idx, g);
    });
  });

  els.resultIngs.querySelectorAll("svg.share-boxplot.editable").forEach((svg) => {
    bindEditableBoxplot(svg, applyGramsAt);
  });
}

function renderJudgeRationale(final) {
  if (!els.judgeRationale || !els.judgeRationaleBody) return;
  const weirdIds = final?.weird_candidate_ids || [];
  const text =
    (Array.isArray(weirdIds) && weirdIds.length
      ? final?.judge_rationale || final?.final_judgment?.rationale || ""
      : "") || "";
  const clean = String(text || "").trim();
  if (!clean) {
    els.judgeRationale.classList.add("hidden");
    els.judgeRationaleBody.textContent = "";
    return;
  }
  els.judgeRationale.classList.remove("hidden");
  els.judgeRationaleBody.textContent = clean;
}

function renderCandidateChrome() {
  const cards = state.browseCandidates || [];
  const idx = state.browseIndex || 0;
  const card = cards[idx];
  if (els.candidatePager) {
    if (cards.length > 1) {
      els.candidatePager.classList.remove("hidden");
      if (els.candidatePagerTitle) {
        els.candidatePagerTitle.textContent = card?.title || (idx === 0 ? "Recommended" : `Option ${idx + 1}`);
      }
      if (els.candidatePagerCount) {
        els.candidatePagerCount.textContent = `${idx + 1} of ${cards.length}`;
      }
    } else {
      els.candidatePager.classList.add("hidden");
    }
  }
}

function persistBrowseCandidate() {
  const cards = state.browseCandidates || [];
  const card = cards[state.browseIndex];
  if (!card) return;
  card.liveIngredients = (state.liveIngredients || []).map((r) => ({ ...r }));
  // Keep the pristine baseline; do not overwrite with edited live state.
  if (!card.baselineSnapshot && state.baselineSnapshot) {
    card.baselineSnapshot = cloneBaselineSnapshot(state.baselineSnapshot);
  }
  card.originalGrams = Array.isArray(card.baselineSnapshot?.originalGrams)
    ? [...card.baselineSnapshot.originalGrams]
    : Array.isArray(state.originalGrams)
      ? [...state.originalGrams]
      : null;
  card.display_scores = state.lastFinal?.display_scores || card.display_scores;
  card.problem = state.lastFinal?.problem || card.problem;
  card.macro_targets = state.lastFinal?.macro_targets || card.macro_targets;
  card.score_summary = {
    ...(card.score_summary || {}),
    macros: card.display_scores?.macros,
    ratio_loss: card.display_scores?.ratio_loss?.value,
    ratio_band: card.display_scores?.ratio_loss?.band,
    nutrient_loss: card.display_scores?.nutrient_loss?.value,
    nutrient_band: card.display_scores?.nutrient_loss?.band,
    cookability: card.display_scores?.cookability?.summary,
    holistic_0_10: card.display_scores?.holistic_0_10?.value,
  };
}

function loadBrowseCandidate(idx) {
  const cards = state.browseCandidates || [];
  if (!cards.length || idx < 0 || idx >= cards.length) return;
  persistBrowseCandidate();
  state.browseIndex = idx;
  const card = cards[idx];
  const scores = card.display_scores || {};
  const ings = (card.liveIngredients || scores.ingredients || []).map((r) => ({ ...r }));
  state.liveIngredients = ings;
  if (card.baselineSnapshot?.originalGrams) {
    state.originalGrams = [...card.baselineSnapshot.originalGrams];
    state.baselineSnapshot = cloneBaselineSnapshot(card.baselineSnapshot);
  } else {
    state.originalGrams = Array.isArray(card.originalGrams)
      ? [...card.originalGrams]
      : ings.map((r) => Number(r.grams) || 0);
    state.lastFinal = {
      ...(state.lastFinal || {}),
      display_scores: scores,
      problem: card.problem || state.lastFinal?.problem || {},
      macro_targets: card.macro_targets || state.lastFinal?.macro_targets || {},
    };
    captureBaselineSnapshot();
    card.baselineSnapshot = cloneBaselineSnapshot(state.baselineSnapshot);
  }
  state.unitMode = {};
  state.editingAmountIdx = null;
  state.openMenu = null;
  state.macroFocus = null;
  state.activeMacroTip = null;
  hideMacroTip();
  state.lastFinal = {
    ...(state.lastFinal || {}),
    display_scores: scores,
    problem: card.problem || state.lastFinal?.problem || {},
    macro_targets: card.macro_targets || state.lastFinal?.macro_targets || {},
  };
  renderMacros(scores.macros, scores.pfc_after);
  renderVerdict(scores);
  renderIngredientTable(state.liveIngredients);
  renderCandidateChrome();
  syncRevertButton();
}

function showResult(final) {
  state.running = false;
  state.lastNode = "finalize";
  els.page.classList.add("done");
  els.phaseKicker.textContent = "Done";
  els.phaseTitle.textContent = "Recipe ready";
  els.phaseDetail.textContent = "Here is the optimized recipe.";
  els.feed.querySelectorAll("li.pending").forEach((li) => markFeedItemDone(li));
  pushCompletedStep("Finished", "Recipe ready to review.");
  updateProgressMeta("finalize");

  state.lastFinal = final;
  state.macroFocus = null;
  state.activeMacroTip = null;
  hideMacroTip();
  const scores = final?.display_scores || {};
  if (scores.cookability?.summary) {
    const already = [...els.feed.querySelectorAll(".act-detail")].some((el) =>
      (el.textContent || "").includes("Improved cookability")
    );
    if (!already) pushCompletedStep("Balancing the recipe", scores.cookability.summary);
  }
  for (const edit of scores.applied_edits || []) {
    if (edit.phrase) {
      const already = [...els.feed.querySelectorAll(".act-detail")].some(
        (el) => el.textContent === edit.phrase
      );
      if (!already) pushCompletedStep("Applied an edit", edit.phrase);
    }
  }

  let browse = Array.isArray(final?.browse_candidates) ? final.browse_candidates.slice(0, 4) : [];
  if (!browse.length) {
    browse.push({
      candidate_id: "recommended",
      title: "Recommended",
      is_recommended: true,
      display_scores: scores,
      problem: final?.problem || {},
      macro_targets: final?.macro_targets || {},
      score_summary: {
        macros: scores.macros,
        ratio_loss: scores.ratio_loss?.value,
        ratio_band: scores.ratio_loss?.band,
        nutrient_loss: scores.nutrient_loss?.value,
        nutrient_band: scores.nutrient_loss?.band,
        cookability: scores.cookability?.summary,
        holistic_0_10: scores.holistic_0_10?.value,
      },
    });
  } else {
    // Prefer the server recommended card's scores, but keep the enriched main display when ids match.
    const rec = browse.find((c) => c.is_recommended) || browse[0];
    if (rec && scores?.ingredients?.length) {
      rec.display_scores = scores;
      rec.problem = final?.problem || rec.problem;
      rec.macro_targets = final?.macro_targets || rec.macro_targets;
      rec.score_summary = {
        ...(rec.score_summary || {}),
        macros: scores.macros,
        ratio_loss: scores.ratio_loss?.value,
        ratio_band: scores.ratio_loss?.band,
        nutrient_loss: scores.nutrient_loss?.value,
        nutrient_band: scores.nutrient_loss?.band,
        cookability: scores.cookability?.summary,
        holistic_0_10: scores.holistic_0_10?.value,
      };
    }
  }
  state.browseCandidates = browse.map((c) => {
    const liveIngredients = (c.display_scores?.ingredients || []).map((r) => ({ ...r }));
    const originalGrams = liveIngredients.map((r) => Number(r.grams) || 0);
    const display_scores = c.display_scores || {};
    return {
      ...c,
      liveIngredients,
      originalGrams,
      baselineSnapshot: {
        ingredients: liveIngredients.map((r) => ({ ...r })),
        display_scores: JSON.parse(JSON.stringify(display_scores)),
        problem: c.problem ? JSON.parse(JSON.stringify(c.problem)) : null,
        originalGrams: [...originalGrams],
      },
    };
  });
  state.browseIndex = Math.max(
    0,
    state.browseCandidates.findIndex((c) => c.is_recommended)
  );
  if (state.browseIndex < 0) state.browseIndex = 0;

  const active = state.browseCandidates[state.browseIndex];
  state.liveIngredients = (active?.liveIngredients || ingredientRows(final)).map((r) => ({ ...r }));
  state.originalGrams = Array.isArray(active?.originalGrams)
    ? [...active.originalGrams]
    : state.liveIngredients.map((r) => Number(r.grams) || 0);
  state.lastFinal = {
    ...final,
    display_scores: active?.display_scores || scores,
    problem: active?.problem || final?.problem || {},
    macro_targets: active?.macro_targets || final?.macro_targets || {},
  };
  state.baselineSnapshot = cloneBaselineSnapshot(active?.baselineSnapshot);
  if (!state.baselineSnapshot) captureBaselineSnapshot();

  els.resultBlock.classList.remove("hidden");
  els.resultStatus.textContent = final?.status || "ready";
  renderJudgeRationale(final);
  renderMacros(
    state.lastFinal.display_scores?.macros,
    state.lastFinal.display_scores?.pfc_after || (final?.opt || {}).pfc_after
  );
  renderVerdict(state.lastFinal.display_scores || scores);
  renderIngredientTable(state.liveIngredients);
  renderCandidateChrome();
  syncRevertButton();
}

document.addEventListener("click", (e) => {
  if (e.target.closest(".ing-actions")) return;
  if (e.target.closest(".share-boxplot.editable")) return;
  if (e.target.closest(".amount-cell")) return;
  if (e.target.closest(".macro-tip")) return;
  if (e.target.closest(".macro-bang")) return;
  closeMenus();
  if (state.editingAmountIdx != null) {
    state.editingAmountIdx = null;
    if (state.liveIngredients?.length) renderIngredientTable(state.liveIngredients);
  }
});

els.revertAmounts?.addEventListener("click", async () => {
  if (!state.originalGrams) return;
  try {
    await recomputeFromGrams([...state.originalGrams]);
    els.runStatus.textContent = "Reverted to original amounts.";
    els.runStatus.classList.remove("error");
  } catch (err) {
    els.runStatus.textContent = `Could not revert: ${err.message || err}`;
    els.runStatus.classList.add("error");
  }
});

els.candidatePrev?.addEventListener("click", (e) => {
  e.stopPropagation();
  const n = state.browseCandidates.length;
  if (n < 2) return;
  loadBrowseCandidate((state.browseIndex - 1 + n) % n);
});

els.candidateNext?.addEventListener("click", (e) => {
  e.stopPropagation();
  const n = state.browseCandidates.length;
  if (n < 2) return;
  loadBrowseCandidate((state.browseIndex + 1) % n);
});

function parseSseChunk(buffer) {
  const events = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() || "";
  for (const part of parts) {
    let eventType = "message";
    let data = "";
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) continue;
    try {
      events.push({ eventType, data: JSON.parse(data) });
    } catch {
      /* ignore */
    }
  }
  return { events, rest };
}

function handleEvent(eventType, data) {
  const type = eventType || data.type;
  if (type === "load") {
    const copy = STEP_COPY.load;
    advanceStep("load", copy.title, copy.detail);
    return;
  }
  if (type === "step") {
    const node = data.node || "step";
    if (node === "shadow_gpt_candidate") return;
    const payload = data.payload || {};
    const iterRaw = data.iteration ?? payload.iteration;
    if (iterRaw != null && Number.isFinite(Number(iterRaw))) {
      state.iteration = Math.max(0, Math.min(state.maxIterations, Number(iterRaw)));
    }
    state.lastNode = node;
    const copy = STEP_COPY[node] || { title: "Working", detail: "Continuing the optimization." };
    const detail = stepDetailForNode(node, payload, copy.detail);
    if (node === "finalize") {
      pushCompletedStep(copy.title, detail);
      setPhase(copy.title, detail, "Finishing");
      updateProgressMeta(node);
      return;
    }
    advanceStep(node, copy.title, detail);
    return;
  }
  if (type === "transcript" || type === "graph_meta") return;
  if (type === "done" || type === "result") {
    showResult(data.final || data);
    return;
  }
  if (type === "error") {
    state.running = false;
    els.page.classList.add("error");
    setPhase("Something failed", "The run stopped before a recipe was ready.", "Error");
    els.runStatus.textContent = data.error || "Error";
    els.runStatus.classList.add("error");
    els.feed.querySelectorAll("li.pending").forEach((li) => markFeedItemDone(li));
    pushCompletedStep("Error", "The run stopped early.");
    if (els.progressRemaining) els.progressRemaining.textContent = "";
  }
}

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.selected?.canonical_id) {
    els.runStatus.textContent = "Select a recipe family to optimize.";
    els.runStatus.classList.add("error");
    els.menuSearch?.focus();
    return;
  }
  const errors = validateMacros();
  if (errors.length) {
    els.runStatus.textContent = `Fix macros: ${errors.join("; ")}`;
    els.runStatus.classList.add("error");
    return;
  }
  const ask = els.semantic.value.trim();
  const kcalRaw = els.kcalTarget?.value?.trim() || "";
  let kcalTarget = null;
  if (kcalRaw) {
    kcalTarget = Number(kcalRaw);
    if (!Number.isFinite(kcalTarget) || kcalTarget < 100 || kcalTarget > 8000) {
      els.runStatus.textContent = "Calories must be between 100 and 8000.";
      els.runStatus.classList.add("error");
      els.kcalTarget?.focus();
      return;
    }
  }

  resetStage();
  setLiveVisible(true);
  state.running = true;
  state.maxIterations = MAX_LOOP_ROUNDS;
  els.runBtn.disabled = true;
  els.runStatus.classList.remove("error");
  els.runStatus.textContent = "Starting…";
  setPhase("Starting…", "Connecting to the agent.");
  updateProgressMeta("load");

  const title = state.selected.title || "";
  const body = {
    mode: "neighborhood",
    user_request: ask,
    taste_text: ask || title,
    title,
    canonical_id: state.selected.canonical_id,
    start_metric: "l1_pfc",
    use_macro_targets: state.macrosEnabled,
    kcal_target: kcalTarget,
    F_accept: 1.0,
    F_max: 1.5,
    // Product UI hard-caps the diagnose→propose→apply loop at 3 passes.
    max_iterations: MAX_LOOP_ROUNDS,
  };
  if (state.macrosEnabled) {
    body.protein_min = frac("protein_min");
    body.protein_max = frac("protein_max");
    body.carb_min = frac("carb_min");
    body.carb_max = frac("carb_max");
    body.fat_min = frac("fat_min");
    body.fat_max = frac("fat_max");
  }

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(typeof err.detail === "string" ? err.detail : res.statusText);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parsed = parseSseChunk(buf);
      buf = parsed.rest;
      for (const { eventType, data } of parsed.events) {
        handleEvent(eventType, data);
      }
    }
    els.runStatus.textContent = "Finished.";
  } catch (err) {
    els.page.classList.add("error");
    setLiveVisible(true);
    setPhase("Something failed", "The run stopped before a recipe was ready.", "Error");
    els.runStatus.textContent = String(err.message || err);
    els.runStatus.classList.add("error");
    pushCompletedStep("Error", "The run stopped early.");
  } finally {
    state.running = false;
    els.runBtn.disabled = false;
  }
});

syncMacroVisuals();
setMacrosEnabled(false);
loadDishes();

// Expose helpers for automated checks
window.__macroiq = {
  ingredientLabel,
  ingredientAmount,
  ingredientRows,
  showResult,
  syncMacroVisuals,
  validateMacros,
  setMacrosEnabled,
  scaledDisplay: ingredientAmount,
};
