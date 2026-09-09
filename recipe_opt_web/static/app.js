/* Recipe Opt Agent playground — SSE + flow graph + transcript */

const FLOW_VIEW_W = 980;
const FLOW_VIEW_H = 500;

// Neighborhood: finalize sits under decide (not under propose) so diagnose→finalize
// is a diagonal — stacking diagnose/propose/finalize on one x looked like propose→finalize.
const NODE_LAYOUT = {
  init: { x: 40, y: 36 },
  deduce_tags: { x: 40, y: 130 },
  llm_draft: { x: 40, y: 224 },
  ground_recipe: { x: 40, y: 318 },
  shadow_gpt_candidate: { x: 140, y: 36 },
  diagnose: { x: 250, y: 36 },
  save_candidate: { x: 500, y: 36 },
  save_moderate: { x: 500, y: 36 },
  propose: { x: 250, y: 170 },
  decide: { x: 500, y: 170 },
  apply: { x: 750, y: 170 },
  build_finalists: { x: 750, y: 36 },
  pareto_and_rank: { x: 750, y: 300 },
  judge_final: { x: 500, y: 300 },
  finalize: { x: 500, y: 410 },
};

const NODE_LABELS = {
  init: "init",
  deduce_tags: "deduce_tags",
  llm_draft: "llm_draft",
  ground_recipe: "ground",
  shadow_gpt_candidate: "extra draft",
  diagnose: "diagnose (+opt)",
  save_candidate: "save_candidate",
  save_moderate: "save_candidate",
  propose: "propose",
  decide: "decide",
  apply: "apply",
  build_finalists: "finalists",
  pareto_and_rank: "pareto+rank",
  judge_final: "judge",
  finalize: "finalize",
};

const FALLBACK_DOCS = {
  init: {
    title: "Initialize",
    summary: "Bind the dropdown-selected canonical recipe as semantic input; load its FoodOn neighborhood + modification candidates.",
    detail:
      "No embedding search. taste_text = dish title. Starting NLG recipe is closest to the macro box among top FoodOn-hit neighbors.",
    compute: "deterministic",
  },
  diagnose: {
    title: "Diagnose (+ optimizer)",
    summary: "This is where the optimizer runs, together with hull geometry and fidelity bands.",
    detail:
      "Calls region_intersects_hull, then optimize_weighted_empirical_obj (the LP/convex solver), then diagnose_optimizer_result for IQR bands and retry triggers.",
    compute: "deterministic",
  },
  save_moderate: {
    title: "Save moderate candidate",
    summary: "Park a feasible-but-not-great solution in candidate_pool.",
    detail: "Only when fidelity_band=moderate.",
    compute: "deterministic",
  },
  save_candidate: {
    title: "Save candidate",
    summary: "Park feasible snapshots in candidate_pool.",
    detail: "Moderate / feasible must_retry / creative accept.",
    compute: "deterministic",
  },
  propose: {
    title: "Propose modifications",
    summary: "Shortlist add/swap/remove candidates (identity-filtered).",
    detail: "LLM may only pick from this list.",
    compute: "deterministic",
  },
  decide: {
    title: "Decide action (LLM)",
    summary: "Choose accept / add / swap / remove / expand with rationale.",
    detail: "gpt-4o-mini or deterministic heuristic.",
    compute: "llm_controller",
  },
  apply: {
    title: "Apply or expand",
    summary: "Mutate the recipe or widen neighbors, then usually re-diagnose (re-optimize).",
    detail: "Loops back to diagnose unless accepting.",
    compute: "deterministic",
  },
  finalize: {
    title: "Finalize",
    summary: "Emit accepted, pool-best, or best-effort result.",
    detail: "End of the graph.",
    compute: "deterministic",
  },
  deduce_tags: {
    title: "Deduce requirement tags",
    summary: "Extract hard dietary/macro tags from the user request.",
    detail: "LLM (or lexical heuristic).",
    compute: "llm_content",
  },
  llm_draft: {
    title: "LLM draft recipe",
    summary: "Draft a structured recipe from the request.",
    detail: "Creative warm-start.",
    compute: "llm_content",
  },
  ground_recipe: {
    title: "Ground draft",
    summary: "Resolve draft lines to FDC foods.",
    detail: "Deterministic grounding.",
    compute: "deterministic",
  },
  shadow_gpt_candidate: {
    title: "Extra draft candidate",
    summary: "Quietly parks one additional optimized draft for comparison.",
    detail: "Does not replace the working recipe.",
    compute: "llm_content",
  },
  build_finalists: {
    title: "Build finalists",
    summary: "Collect pool into a finalist set.",
    detail: "Creative mode.",
    compute: "deterministic",
  },
  pareto_and_rank: {
    title: "Pareto + rank",
    summary: "Score and filter finalists.",
    detail: "Deterministic multi-axis ranking.",
    compute: "deterministic",
  },
  judge_final: {
    title: "Judge finalists (LLM)",
    summary: "LLM picks the winner among Pareto survivors.",
    detail: "Creative mode.",
    compute: "llm_content",
  },
};

const FALLBACK_COMPUTE_KINDS = {
  deterministic: {
    label: "Deterministic tools",
    blurb: "Pure Python / LP — no LLM call",
  },
  llm_content: {
    label: "LLM content",
    blurb: "LLM generates structured content (tags, draft, judgment)",
  },
  llm_controller: {
    label: "LLM chooses tools",
    blurb: "LLM selects the next action / tool to invoke",
  },
};

const NODE_W = 168;
const NODE_H = 52;

// Exact agent fractions (0–1). UI shows Math.round(100 * x) as percent defaults.
const DEFAULT_MACRO_FRACTIONS = {
  protein_min: 0.19,
  protein_max: 0.23,
  carb_min: 0.345,
  carb_max: 0.545,
  fat_min: 0.245,
  fat_max: 0.445,
};

/** Percent field → 0–1 fraction; keep exact defaults when the rounded % is unchanged. */
function fractionFromPercentInput(id) {
  const pct = Number(document.getElementById(id).value);
  const exact = DEFAULT_MACRO_FRACTIONS[id];
  if (exact != null && Math.round(exact * 100) === Math.round(pct)) {
    return exact;
  }
  return pct / 100;
}

function setMacroPercentInput(id, fraction) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = String(Math.round(Number(fraction) * 100));
}

function applyMacroBox(box) {
  if (!box) return;
  setMacroPercentInput("protein_min", box.protein_min);
  setMacroPercentInput("protein_max", box.protein_max);
  setMacroPercentInput("carb_min", box.carb_min);
  setMacroPercentInput("carb_max", box.carb_max);
  setMacroPercentInput("fat_min", box.fat_min);
  setMacroPercentInput("fat_max", box.fat_max);
}

/** Errors if the macro box can't contain a point with P+C+F = 100%. */
function validateMacroBox(b) {
  const errors = [];
  for (const [name, lo, hi] of [
    ["protein", b.protein_min, b.protein_max],
    ["carb", b.carb_min, b.carb_max],
    ["fat", b.fat_min, b.fat_max],
  ]) {
    if (lo > hi) {
      errors.push(`${name} min (${Math.round(lo * 100)}%) exceeds ${name} max (${Math.round(hi * 100)}%)`);
    }
  }
  const sumMin = b.protein_min + b.carb_min + b.fat_min;
  const sumMax = b.protein_max + b.carb_max + b.fat_max;
  if (sumMax < 1 - 1e-9) {
    errors.push(
      `macro maxes sum to ${Math.round(sumMax * 100)}% — must be ≥ 100% so protein+carbs+fat can reach 100%`
    );
  }
  if (sumMin > 1 + 1e-9) {
    errors.push(
      `macro mins sum to ${Math.round(sumMin * 100)}% — must be ≤ 100% so protein+carbs+fat can total 100%`
    );
  }
  return errors;
}

let flowEdges = [];
let flowDocs = { ...FALLBACK_DOCS };
let flowComputeKinds = { ...FALLBACK_COMPUTE_KINDS };
let visitedNodes = new Set();
let lastNode = null;
/** Flight recorder for the latest finished run (flow summary + debugging). */
let lastRunBundle = null;
let runChatMessages = [];
let runChatBusy = false;

const appShell = document.getElementById("app-shell");
const form = document.getElementById("run-form");
const modeNeighborhood = document.getElementById("mode_neighborhood");
const modeCreative = document.getElementById("mode_creative");
const creativePanel = document.getElementById("creative-panel");
const userRequestEl = document.getElementById("user_request");
const canonicalSearchInput = document.getElementById("canonical_search_input");
const canonicalIdInput = document.getElementById("canonical_id");
const canonicalResults = document.getElementById("canonical_results");
const canonicalSearchWrap = document.getElementById("canonical_search_wrap");
const canonicalMeta = document.getElementById("canonical-meta");
const timeline = document.getElementById("timeline");
const flowGraph = document.getElementById("flow-graph");
const flowInspector = document.getElementById("flow-inspector");
const flowStatus = document.getElementById("flow-status");
const optimizerNote = document.getElementById("optimizer-note");
const runBtn = document.getElementById("run-btn");
const runStatus = document.getElementById("run-status");
const resultPanel = document.getElementById("result-panel");
const resultJson = document.getElementById("result-json");
const finalistCards = document.getElementById("finalist-cards");
const displayScoresEl = document.getElementById("display-scores");
const finalIngredientsEl = document.getElementById("final-ingredients");
const pathFinalsEl = document.getElementById("path-finals");
const flowSummaryBtn = document.getElementById("flow-summary-btn");
const flowSummaryStatus = document.getElementById("flow-summary-status");
const flowSummaryEl = document.getElementById("flow-summary");
const runChatBox = document.getElementById("run-chat-box");
const runChatMessagesEl = document.getElementById("run-chat-messages");
const runChatForm = document.getElementById("run-chat-form");
const runChatInput = document.getElementById("run-chat-input");
const runChatSendBtn = document.getElementById("run-chat-send");
const runChatClearBtn = document.getElementById("run-chat-clear");
const runChatStatus = document.getElementById("run-chat-status");
const healthHint = document.getElementById("health-hint");
const showTranscript = document.getElementById("show_transcript");
const transcriptPanel = document.getElementById("transcript-panel");
const transcriptEl = document.getElementById("transcript");
const closeTranscript = document.getElementById("close-transcript");
const liveStepsPanel = document.getElementById("live-steps-panel");
const agentFlowPanel = document.getElementById("agent-flow-panel");

function syncWorkspaceHeight() {
  if (!appShell) return;
  if (window.matchMedia("(max-width: 900px)").matches) {
    appShell.style.removeProperty("--workspace-h");
    return;
  }
  const header = document.querySelector(".app-header");
  const headerH = header ? header.getBoundingClientRect().height : 0;
  const formH = form ? form.getBoundingClientRect().height : 0;
  // Cap how much of the viewport the form/header can steal so the workspace stays usable.
  const reserved = Math.min(headerH + formH + 8, window.innerHeight * 0.42);
  const h = Math.max(480, Math.round(window.innerHeight - reserved));
  appShell.style.setProperty("--workspace-h", `${h}px`);
}

window.addEventListener("resize", syncWorkspaceHeight);

let canonicalDishes = [];
let selectedCanonical = null;
let canonicalSearchTimer = null;
let canonicalTotalCount = null;

function getSelectedCanonicalId() {
  const id = Number(canonicalIdInput?.value);
  return Number.isFinite(id) && id > 0 ? id : null;
}

function currentMode() {
  return modeCreative?.checked ? "creative" : "neighborhood";
}

function currentStartMetric() {
  const loss = document.getElementById("start_metric_loss");
  return loss?.checked ? "loss_projection" : "l1_pfc";
}

function setRunStatus(msg, { error = false } = {}) {
  if (!runStatus) return;
  runStatus.textContent = msg || "";
  runStatus.classList.toggle("run-status-error", Boolean(error && msg));
}

function clearCanonicalSelection() {
  selectedCanonical = null;
  if (canonicalIdInput) canonicalIdInput.value = "";
  if (canonicalSearchInput) canonicalSearchInput.value = "";
  hideCanonicalResults();
  updateCanonicalMeta();
}

function updateModeUI() {
  const creative = currentMode() === "creative";
  if (creativePanel) creativePanel.hidden = !creative;
  const startMetricRow = document.getElementById("start-metric-row");
  const startMetricHint = document.getElementById("start-metric-hint");
  if (startMetricRow) startMetricRow.hidden = creative;
  if (startMetricHint) startMetricHint.hidden = creative;
  if (canonicalSearchInput) {
    canonicalSearchInput.required = !creative;
    if (creative) {
      // Creative treats neighborhood as optional — clear auto-selected dish so runs
      // don't silently block on a heavy DB neighborhood load.
      clearCanonicalSelection();
      canonicalSearchInput.placeholder = "Optional: search neighborhood catalog…";
    } else {
      canonicalSearchInput.placeholder = "Search all canonical recipes…";
      if (!getSelectedCanonicalId()) {
        // Restore a default pick when returning to neighborhood mode.
        runCanonicalSearch("").then((data) => {
          const dishes = data?.dishes || [];
          const carbonara = dishes.find((d) => /carbonara/i.test(d.title));
          if (carbonara) selectCanonicalDish(carbonara);
          else if (dishes.length) selectCanonicalDish(dishes[0]);
        }).catch(() => {});
      }
    }
  }
  const hint = document.getElementById("canonical-hint");
  if (hint) {
    hint.textContent = creative
      ? "Optional in Creative mode. Leave blank to draft/ground offline; pick a dish only if you want its FDC neighborhood catalog."
      : "Search the full canonical catalog (not limited to top 50). Required for neighborhood mode.";
  }
  loadFlowForMode(currentMode());
  requestAnimationFrame(syncWorkspaceHeight);
}

modeNeighborhood?.addEventListener("change", updateModeUI);
modeCreative?.addEventListener("change", updateModeUI);

async function loadFlowForMode(mode) {
  try {
    const flow = await fetch(`/api/flow?mode=${encodeURIComponent(mode)}`).then((r) => r.json());
    flowEdges = (flow.edges || []).map((e) => [e.from, e.to]);
    if (flow.docs) flowDocs = { ...FALLBACK_DOCS, ...flow.docs };
    if (flow.compute_kinds) flowComputeKinds = { ...FALLBACK_COMPUTE_KINDS, ...flow.compute_kinds };
    renderFlowLegend();
    renderFlow(flow.nodes || Object.keys(NODE_LAYOUT));
  } catch (err) {
    console.warn("flow load failed", err);
  }
}

function selectedDish() {
  if (selectedCanonical) return selectedCanonical;
  const id = getSelectedCanonicalId();
  return canonicalDishes.find((d) => d.canonical_id === id) || null;
}

function updateCanonicalMeta() {
  const d = selectedDish();
  if (!d) {
    const total = canonicalTotalCount != null ? `${canonicalTotalCount} recipes in catalog · ` : "";
    canonicalMeta.textContent = `${total}Search and pick a canonical dish.`;
    return;
  }
  canonicalMeta.textContent = `${d.title} · ${d.n_matches} neighborhood matches · id=${d.canonical_id}`;
}

function hideCanonicalResults() {
  canonicalResults?.classList.add("hidden");
}

function showCanonicalResults() {
  canonicalResults?.classList.remove("hidden");
}

function renderCanonicalResults(dishes, { q = "", total = null } = {}) {
  if (!canonicalResults) return;
  canonicalResults.innerHTML = "";
  if (!dishes.length) {
    const li = document.createElement("li");
    li.className = "empty-hint";
    li.textContent = q.trim()
      ? `No matches for “${q.trim()}”${total != null ? ` (${total} total in catalog)` : ""}.`
      : "No canonical recipes found.";
    canonicalResults.appendChild(li);
    showCanonicalResults();
    return;
  }
  for (const d of dishes) {
    const li = document.createElement("li");
    li.role = "option";
    li.dataset.id = String(d.canonical_id);
    li.innerHTML = `${escapeHtml(d.title)}<span class="matches">${d.n_matches} matches · id ${d.canonical_id}</span>`;
    li.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      selectCanonicalDish(d);
    });
    canonicalResults.appendChild(li);
  }
  if (total != null && total > dishes.length) {
    const li = document.createElement("li");
    li.className = "empty-hint";
    li.textContent = `Showing ${dishes.length} of ${total} — refine search to narrow.`;
    canonicalResults.appendChild(li);
  }
  showCanonicalResults();
}

let macroPresetCache = null;
let macroPresetCacheId = null;
let macroPresetBusy = false;

const macroPresetStatus = document.getElementById("macro-preset-status");
const macroPresetBtns = {
  neighborhood_coverage: document.getElementById("macro-preset-neighborhood-coverage"),
  neighborhood_mean: document.getElementById("macro-preset-neighborhood-mean"),
};

function setMacroPresetStatus(msg, { error = false } = {}) {
  if (!macroPresetStatus) return;
  macroPresetStatus.textContent = msg || "";
  macroPresetStatus.style.color = error ? "var(--danger, #b42318)" : "";
}

function fmtPct(x) {
  return `${Math.round(Number(x) * 100)}%`;
}

function describePreset(preset) {
  if (!preset) return "";
  const mid = preset.midpoint || {};
  const pad = preset.pad_pct != null ? `±${preset.pad_pct}%` : "";
  const n = preset.n_recipes != null ? `${preset.n_recipes} recipes` : "";
  return (
    `${fmtPct(mid.protein)} P · ${fmtPct(mid.carbs)} C · ${fmtPct(mid.fat)} F` +
    (pad ? ` · ${pad}` : "") +
    (n ? ` · ${n}` : "")
  );
}

function selectCanonicalDish(dish) {
  selectedCanonical = dish;
  if (canonicalIdInput) canonicalIdInput.value = String(dish.canonical_id);
  if (canonicalSearchInput) canonicalSearchInput.value = dish.title;
  hideCanonicalResults();
  updateCanonicalMeta();
  // Invalidate cached hull presets when the dish changes.
  macroPresetCache = null;
  macroPresetCacheId = null;
  setMacroPresetStatus("");
  document.querySelectorAll(".macro-preset-btns button").forEach((b) => b.classList.remove("active"));
}

async function ensureMacroPresets() {
  const id = getSelectedCanonicalId();
  if (!id) {
    throw new Error("Select a canonical dish first.");
  }
  if (macroPresetCache && macroPresetCacheId === id) {
    return macroPresetCache;
  }
  const res = await fetch("/api/macro_targets/suggest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ canonical_id: id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  const data = await res.json();
  macroPresetCache = data;
  macroPresetCacheId = id;
  return data;
}

async function applyMacroPreset(kind) {
  if (macroPresetBusy) return;
  const btn = macroPresetBtns[kind];
  macroPresetBusy = true;
  Object.values(macroPresetBtns).forEach((b) => {
    if (b) b.disabled = true;
  });
  setMacroPresetStatus("Probing representative hull…");
  try {
    const data = await ensureMacroPresets();
    const preset = (data.presets || {})[kind];
    if (!preset?.box) {
      throw new Error(`No ${kind} preset available for this neighborhood.`);
    }
    applyMacroBox(preset.box);
    document.querySelectorAll(".macro-preset-btns button").forEach((b) => b.classList.remove("active"));
    btn?.classList.add("active");
    setMacroPresetStatus(describePreset(preset));
  } catch (err) {
    setMacroPresetStatus(String(err.message || err), { error: true });
  } finally {
    macroPresetBusy = false;
    Object.values(macroPresetBtns).forEach((b) => {
      if (b) b.disabled = false;
    });
  }
}

Object.entries(macroPresetBtns).forEach(([kind, btn]) => {
  btn?.addEventListener("click", () => applyMacroPreset(kind));
});

async function fetchCanonicalSearch(q = "") {
  const params = new URLSearchParams({ min_neighborhood: "5", limit: "40" });
  if (q.trim()) params.set("q", q.trim());
  const res = await fetch(`/api/canonicals/search?${params}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function runCanonicalSearch(q = "") {
  try {
    const data = await fetchCanonicalSearch(q);
    canonicalDishes = data.dishes || [];
    if (!q.trim() && canonicalTotalCount == null) {
      canonicalTotalCount = data.total;
    }
    renderCanonicalResults(canonicalDishes, { q, total: data.total });
    return data;
  } catch (err) {
    hideCanonicalResults();
    canonicalMeta.textContent = `Search failed: ${err.message || err}`;
    throw err;
  }
}

function scheduleCanonicalSearch(q) {
  clearTimeout(canonicalSearchTimer);
  canonicalSearchTimer = setTimeout(() => {
    runCanonicalSearch(q).catch(() => {});
  }, 220);
}

canonicalSearchInput?.addEventListener("input", () => {
  const q = canonicalSearchInput.value;
  if (selectedCanonical && q !== selectedCanonical.title) {
    selectedCanonical = null;
    if (canonicalIdInput) canonicalIdInput.value = "";
  }
  scheduleCanonicalSearch(q);
});

canonicalSearchInput?.addEventListener("focus", () => {
  runCanonicalSearch(canonicalSearchInput.value).catch(() => {});
});

canonicalSearchInput?.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") hideCanonicalResults();
});

document.addEventListener("click", (ev) => {
  if (canonicalSearchWrap && !canonicalSearchWrap.contains(ev.target)) {
    hideCanonicalResults();
  }
});

async function loadCanonicals() {
  try {
    const countRes = await fetch("/api/canonicals?count_only=1&min_neighborhood=5");
    if (countRes.ok) {
      const countData = await countRes.json();
      canonicalTotalCount = countData.count;
    }
    const data = await runCanonicalSearch("");
    const carbonara = (data.dishes || []).find((d) => /carbonara/i.test(d.title));
    if (carbonara) {
      selectCanonicalDish(carbonara);
    } else if ((data.dishes || []).length) {
      selectCanonicalDish(data.dishes[0]);
    } else {
      updateCanonicalMeta();
    }
  } catch (err) {
    if (canonicalSearchInput) canonicalSearchInput.placeholder = "Unavailable — DB error";
    canonicalMeta.textContent = `Could not load dishes: ${err.message || err}`;
  }
}

function setTranscriptVisible(on) {
  showTranscript.checked = on;
  appShell.classList.toggle("with-transcript", on);
  transcriptPanel.classList.toggle("hidden", !on);
  syncWorkspaceHeight();
}

showTranscript.addEventListener("change", () => setTranscriptVisible(showTranscript.checked));
closeTranscript.addEventListener("click", () => setTranscriptVisible(false));

async function initHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    healthHint.textContent = data.has_openai_key
      ? "OPENAI_API_KEY detected — decide_action uses gpt-4o-mini."
      : "No OPENAI_API_KEY — decide_action uses the deterministic heuristic.";
    const flow = await fetch(`/api/flow?mode=${encodeURIComponent(currentMode())}`).then((r) => r.json());
    flowEdges = (flow.edges || []).map((e) => [e.from, e.to]);
    if (flow.docs) flowDocs = { ...FALLBACK_DOCS, ...flow.docs };
    if (flow.compute_kinds) flowComputeKinds = { ...FALLBACK_COMPUTE_KINDS, ...flow.compute_kinds };
    if (flow.optimizer_note && optimizerNote) {
      optimizerNote.textContent = flow.optimizer_note;
    }
    renderFlowLegend();
    renderFlow(flow.nodes || Object.keys(NODE_LAYOUT));
  } catch (err) {
    healthHint.textContent = `Health check failed: ${err}`;
    renderFlowLegend();
    renderFlow(Object.keys(NODE_LAYOUT));
  }
  await loadCanonicals();
}

/* ---------- Compute-kind color coding ---------- */

function computeKindForNode(name) {
  const doc = flowDocs[name] || FALLBACK_DOCS[name] || {};
  const kind = doc.compute || "deterministic";
  return FALLBACK_COMPUTE_KINDS[kind] ? kind : "deterministic";
}

function renderFlowLegend() {
  const el = document.getElementById("flow-legend");
  if (!el) return;
  const order = ["deterministic", "llm_content", "llm_controller"];
  el.innerHTML = order
    .map((kind) => {
      const meta = flowComputeKinds[kind] || FALLBACK_COMPUTE_KINDS[kind];
      return `<span class="flow-legend-item compute-${kind}" title="${escapeHtml(meta.blurb || "")}">
        <span class="flow-legend-swatch" aria-hidden="true"></span>
        <span class="flow-legend-label">${escapeHtml(meta.label || kind)}</span>
      </span>`;
    })
    .join("");
}

/* ---------- Side inspector (purpose / tools / prompts / edges) ---------- */

let selectedFlowNode = null;
/** Runtime prompts captured from the current/last agent run, keyed by node. */
const runPromptsByNode = {};

function clearRunPrompts() {
  for (const k of Object.keys(runPromptsByNode)) delete runPromptsByNode[k];
}

function recordRunPrompt(node, entry) {
  if (!node) return;
  if (!runPromptsByNode[node]) runPromptsByNode[node] = [];
  runPromptsByNode[node].push(entry);
}

function nodeEdgeLists(name) {
  const doc = flowDocs[name] || {};
  const incoming = doc.incoming && doc.incoming.length
    ? doc.incoming
    : flowEdges.filter(([, b]) => b === name).map(([a]) => a);
  const outgoing = doc.outgoing && doc.outgoing.length
    ? doc.outgoing
    : flowEdges.filter(([a]) => a === name).map(([, b]) => b);
  return { incoming: [...new Set(incoming)], outgoing: [...new Set(outgoing)] };
}

function renderEmptyInspector() {
  if (!flowInspector) return;
  flowInspector.innerHTML = `
    <div class="flow-inspector-empty">
      <h3>Node inspector</h3>
      <p class="hint">Select a node in the graph to see its purpose, tools, and prompts.</p>
    </div>`;
  flowInspector.classList.remove("has-selection");
  selectedFlowNode = null;
  flowGraph?.querySelectorAll(".flow-node.selected").forEach((n) => n.classList.remove("selected"));
}

function selectFlowNode(name) {
  if (!flowInspector) return;
  if (selectedFlowNode === name) {
    renderEmptyInspector();
    return;
  }
  selectedFlowNode = name;
  flowGraph?.querySelectorAll(".flow-node.selected").forEach((n) => n.classList.remove("selected"));
  const nodeEl = flowGraph?.querySelector(`.flow-node[data-node="${name}"]`);
  if (nodeEl) nodeEl.classList.add("selected");
  renderFlowInspector(name);
}

function renderFlowInspector(name) {
  if (!flowInspector) return;
  const doc = flowDocs[name] || FALLBACK_DOCS[name] || { title: name, summary: "", detail: "" };
  const { incoming, outgoing } = nodeEdgeLists(name);
  const tools = doc.tools || [];
  const compute = computeKindForNode(name);
  const computeMeta = flowComputeKinds[compute] || FALLBACK_COMPUTE_KINDS[compute];
  const runPrompts = runPromptsByNode[name] || [];

  const edgeChip = (n, dir) =>
    `<button type="button" class="edge-chip edge-${dir}" data-jump="${escapeHtml(n)}">${dir === "in" ? "←" : "→"} ${escapeHtml(NODE_LABELS[n] || n)}</button>`;

  const toolsHtml = tools.length
    ? `<div class="inspector-section">
        <h5>Tools</h5>
        ${tools
          .map((t, i) => {
            const prompts = t.prompts || [];
            const promptsHtml = prompts.length
              ? `<details class="inspector-dropdown prompt-dropdown">
                  <summary>Prompts (${prompts.length})</summary>
                  <div class="inspector-dropdown-body">
                    ${prompts
                      .map(
                        (p) => `<details class="inspector-dropdown nested">
                          <summary><span class="role-pill role-${escapeHtml(p.role || "user")}">${escapeHtml(p.role || "user")}</span> ${escapeHtml(p.name || "prompt")}</summary>
                          <div class="inspector-dropdown-body">
                            ${p.summary ? `<p class="hint">${escapeHtml(p.summary)}</p>` : ""}
                            <pre class="prompt-block">${escapeHtml(p.content || "")}</pre>
                          </div>
                        </details>`
                      )
                      .join("")}
                  </div>
                </details>`
              : `<p class="hint tool-no-prompt">No LLM prompts — this tool is deterministic.</p>`;
            return `<details class="inspector-dropdown tool-dropdown" ${i === 0 ? "open" : ""}>
              <summary><code>${escapeHtml(t.name || "")}</code> — ${escapeHtml(t.purpose || "")}</summary>
              <div class="inspector-dropdown-body">
                <p>${escapeHtml(t.detail || t.purpose || "")}</p>
                ${promptsHtml}
              </div>
            </details>`;
          })
          .join("")}
      </div>`
    : `<div class="inspector-section"><h5>Tools</h5><p class="hint">No tools on this node.</p></div>`;

  const runPromptsHtml = runPrompts.length
    ? `<div class="inspector-section">
        <h5>This run — prompts used</h5>
        ${runPrompts
          .map(
            (p) => `<details class="inspector-dropdown prompt-dropdown">
              <summary><span class="role-pill role-${escapeHtml(p.role || "user")}">${escapeHtml(p.role || "user")}</span> ${escapeHtml(p.tool || p.name || "prompt")}${p.model ? ` · ${escapeHtml(p.model)}` : ""}</summary>
              <div class="inspector-dropdown-body">
                ${p.mode ? `<p class="hint">mode: ${escapeHtml(p.mode)}</p>` : ""}
                <pre class="prompt-block">${escapeHtml(p.content || "")}</pre>
              </div>
            </details>`
          )
          .join("")}
      </div>`
    : "";

  flowInspector.classList.add("has-selection");
  flowInspector.innerHTML = `
    <div class="flow-inspector-head">
      <h3><code>${escapeHtml(NODE_LABELS[name] || name)}</code></h3>
      <button type="button" class="ghost inspector-clear" aria-label="Clear selection">Clear</button>
    </div>
    <p class="inspector-title">${escapeHtml(doc.title || name)}</p>
    <p class="compute-badge compute-${compute}" title="${escapeHtml(computeMeta.blurb || "")}">${escapeHtml(computeMeta.label || compute)}</p>
    <p class="inspector-summary">${escapeHtml(doc.summary || "")}</p>
    ${doc.detail ? `<p class="inspector-detail">${escapeHtml(doc.detail).replace(/\n/g, "<br/>")}</p>` : ""}
    <div class="inspector-section">
      <h5>Edges</h5>
      <div class="inspector-edges">
        ${incoming.map((n) => edgeChip(n, "in")).join("")}
        ${outgoing.map((n) => edgeChip(n, "out")).join("")}
        ${!incoming.length && !outgoing.length ? '<span class="hint">none</span>' : ""}
      </div>
    </div>
    ${toolsHtml}
    ${runPromptsHtml}`;

  flowInspector.querySelector(".inspector-clear")?.addEventListener("click", renderEmptyInspector);
  flowInspector.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.addEventListener("click", () => selectFlowNode(btn.getAttribute("data-jump")));
  });
}

document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && selectedFlowNode) renderEmptyInspector();
});

/* ---------- Directed flow graph ---------- */

/** Point where the ray from the rect center toward (tx, ty) exits the rect border. */
function rectBorderPoint(pos, tx, ty) {
  const cx = pos.x + NODE_W / 2;
  const cy = pos.y + NODE_H / 2;
  const dx = tx - cx;
  const dy = ty - cy;
  if (dx === 0 && dy === 0) return { x: cx, y: cy };
  const sx = dx !== 0 ? NODE_W / 2 / Math.abs(dx) : Infinity;
  const sy = dy !== 0 ? NODE_H / 2 / Math.abs(dy) : Infinity;
  const s = Math.min(sx, sy);
  return { x: cx + dx * s, y: cy + dy * s };
}

function nodeRect(pos) {
  return { x: pos.x, y: pos.y, w: NODE_W, h: NODE_H };
}

function segmentHitsRect(x1, y1, x2, y2, rect, pad = 6) {
  const left = rect.x - pad;
  const right = rect.x + rect.w + pad;
  const top = rect.y - pad;
  const bottom = rect.y + rect.h + pad;
  for (let i = 1; i <= 8; i++) {
    const t = i / 9;
    const x = x1 + (x2 - x1) * t;
    const y = y1 + (y2 - y1) * t;
    if (x >= left && x <= right && y >= top && y <= bottom) return true;
  }
  return false;
}

function edgeBlockedByNode(fromName, toName, pa, pb, nodeNames) {
  const caX = pa.x + NODE_W / 2;
  const caY = pa.y + NODE_H / 2;
  const cbX = pb.x + NODE_W / 2;
  const cbY = pb.y + NODE_H / 2;
  for (const name of nodeNames) {
    if (name === fromName || name === toName) continue;
    const pos = NODE_LAYOUT[name];
    if (!pos) continue;
    if (segmentHitsRect(caX, caY, cbX, cbY, nodeRect(pos))) return true;
  }
  return false;
}

/** Route edges so they match the logical graph (no phantom propose→finalize). */
function pathForEdge(fromName, toName, pa, pb, nodeNames) {
  const caX = pa.x + NODE_W / 2;
  const caY = pa.y + NODE_H / 2;
  const cbX = pb.x + NODE_W / 2;
  const cbY = pb.y + NODE_H / 2;
  const backward = cbX < caX - 1;
  if (backward && Math.abs(cbY - caY) < NODE_H) {
    const midX = (caX + cbX) / 2;
    const bowY = Math.min(caY, cbY) - NODE_H * 1.25;
    const p1 = rectBorderPoint(pa, midX, bowY);
    const p2 = rectBorderPoint(pb, midX, bowY);
    return `M ${p1.x} ${p1.y} Q ${midX} ${bowY} ${p2.x} ${p2.y}`;
  }
  if (edgeBlockedByNode(fromName, toName, pa, pb, nodeNames)) {
    const midX = (caX + cbX) / 2;
    const midY = (caY + cbY) / 2;
    const dx = cbX - caX;
    const dy = cbY - caY;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const side = Math.abs(dy) > Math.abs(dx) * 0.4 ? -1 : 1;
    const offset = 110;
    const ctrlX = midX + side * nx * offset;
    const ctrlY = midY + side * ny * offset;
    const p1 = rectBorderPoint(pa, ctrlX, ctrlY);
    const p2 = rectBorderPoint(pb, ctrlX, ctrlY);
    return `M ${p1.x} ${p1.y} Q ${ctrlX} ${ctrlY} ${p2.x} ${p2.y}`;
  }
  const p1 = rectBorderPoint(pa, cbX, cbY);
  const p2 = rectBorderPoint(pb, caX, caY);
  return `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y}`;
}

function renderFlow(nodes) {
  const keepSelection = selectedFlowNode && nodes.includes(selectedFlowNode);
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${FLOW_VIEW_W} ${FLOW_VIEW_H}`);
  svg.setAttribute("class", "flow-svg");

  // Arrowhead markers (default / active / visited).
  const defs = document.createElementNS(svgNS, "defs");
  for (const variant of ["", "-active", "-visited"]) {
    const marker = document.createElementNS(svgNS, "marker");
    marker.setAttribute("id", `flow-arrow${variant}`);
    marker.setAttribute("viewBox", "0 0 10 10");
    marker.setAttribute("refX", "9");
    marker.setAttribute("refY", "5");
    marker.setAttribute("markerWidth", "7");
    marker.setAttribute("markerHeight", "7");
    marker.setAttribute("orient", "auto-start-reverse");
    const tip = document.createElementNS(svgNS, "path");
    tip.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    tip.setAttribute("class", `flow-arrowhead${variant ? ` arrowhead${variant}` : ""}`);
    marker.appendChild(tip);
    defs.appendChild(marker);
  }
  svg.appendChild(defs);

  const edgeLayer = document.createElementNS(svgNS, "g");
  edgeLayer.setAttribute("id", "edge-layer");
  svg.appendChild(edgeLayer);

  const nodeSet = new Set(nodes);
  for (const [a, b] of flowEdges.length ? flowEdges : defaultEdges()) {
    const pa = NODE_LAYOUT[a];
    const pb = NODE_LAYOUT[b];
    if (!pa || !pb || !nodeSet.has(a) || !nodeSet.has(b)) continue;
    const line = document.createElementNS(svgNS, "path");
    line.setAttribute("d", pathForEdge(a, b, pa, pb, nodes));
    line.setAttribute("class", "flow-edge");
    line.setAttribute("marker-end", "url(#flow-arrow)");
    line.dataset.from = a;
    line.dataset.to = b;
    edgeLayer.appendChild(line);
  }

  for (const name of nodes) {
    const pos = NODE_LAYOUT[name] || { x: 40, y: 40 };
    const compute = computeKindForNode(name);
    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", `flow-node compute-${compute}`);
    g.dataset.node = name;
    g.dataset.compute = compute;
    g.style.cursor = "pointer";
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", pos.x);
    rect.setAttribute("y", pos.y);
    rect.setAttribute("width", NODE_W);
    rect.setAttribute("height", NODE_H);
    rect.setAttribute("rx", 8);
    // Presentation attributes as a fallback if a stale CSS cache misses the
    // compute-kind rules (CSS still overrides these when present).
    const fills = {
      deterministic: { fill: "#d7ebe2", stroke: "#1f6f54", stripe: "#1f6f54" },
      llm_content: { fill: "#d6e4f7", stroke: "#1d4e89", stripe: "#1d4e89" },
      llm_controller: { fill: "#f6e0a8", stroke: "#8a5a00", stripe: "#8a5a00" },
    };
    const palette = fills[compute] || fills.deterministic;
    rect.setAttribute("fill", palette.fill);
    rect.setAttribute("stroke", palette.stroke);
    rect.setAttribute("stroke-width", "2");
    const stripe = document.createElementNS(svgNS, "rect");
    stripe.setAttribute("class", "compute-stripe");
    stripe.setAttribute("x", pos.x);
    stripe.setAttribute("y", pos.y);
    stripe.setAttribute("width", 7);
    stripe.setAttribute("height", NODE_H);
    stripe.setAttribute("rx", 3);
    stripe.setAttribute("fill", palette.stripe);
    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", pos.x + NODE_W / 2);
    text.setAttribute("y", pos.y + NODE_H / 2 + 5);
    text.setAttribute("text-anchor", "middle");
    text.textContent = NODE_LABELS[name] || name;
    const hint = document.createElementNS(svgNS, "text");
    hint.setAttribute("x", pos.x + NODE_W - 12);
    hint.setAttribute("y", pos.y + 16);
    hint.setAttribute("text-anchor", "middle");
    hint.setAttribute("class", "flow-node-info");
    hint.textContent = "ⓘ";
    g.appendChild(rect);
    g.appendChild(stripe);
    g.appendChild(text);
    g.appendChild(hint);
    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectFlowNode(name);
    });
    svg.appendChild(g);
  }

  flowGraph.innerHTML = "";
  flowGraph.appendChild(svg);
  if (keepSelection) {
    const el = flowGraph.querySelector(`.flow-node[data-node="${selectedFlowNode}"]`);
    if (el) el.classList.add("selected");
    renderFlowInspector(selectedFlowNode);
  } else if (!selectedFlowNode) {
    renderEmptyInspector();
  }
}

function defaultEdges() {
  return [
    ["init", "diagnose"],
    ["diagnose", "save_candidate"],
    ["diagnose", "propose"],
    ["diagnose", "finalize"],
    ["save_candidate", "propose"],
    ["propose", "decide"],
    ["decide", "apply"],
    ["decide", "finalize"],
    ["apply", "diagnose"],
    ["apply", "finalize"],
  ];
}

function fmtScore(v, digits = 3) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(digits);
}

function emptyScores() {
  return {
    ready: false,
    ratio_loss: {
      value: null,
      band: "unknown",
      label: "Ratio loss",
      explanation: "Neighborhood pasta∶egg ratio surrogate (lower better).",
      source: null,
    },
    nutrient_loss: {
      value: null,
      band: "unknown",
      label: "Nutrient loss",
      explanation: "PFC box slack (0 = inside the box).",
      source: null,
    },
    holistic_0_10: {
      value: null,
      band: "unknown",
      label: "Holistic",
      explanation: "LLM judge score 0–10 when available.",
      source: null,
    },
    ingredients: [],
    score_history: [],
  };
}

let scoreHistory = [];

function clientDisplayScores(final) {
  /** Fallback if backend omitted display_scores (older payloads). */
  const tel = final?.run_telemetry || {};
  const judge = final?.judge_result || final?.chosen?.judge || {};
  let holistic = judge.holistic_score_0_10 ?? judge.holistic_0_10;
  if (holistic == null && tel.final_holistic != null) holistic = Number(tel.final_holistic) * 10;
  if (holistic != null) holistic = Math.max(0, Math.min(10, Number(holistic)));
  // Only trust telemetry ratio when source is an explicit ratio key (not share-sum junk).
  const ratioSrc = tel.final_ratio_source;
  const ratio =
    ratioSrc === "ratio_surrogate" || ratioSrc === "ratio_loss" || ratioSrc === "ratio"
      ? tel.final_ratio_term
      : null;
  const nutrient = tel.final_nutrient_slack;
  const bandLoss = (v, goodMax, warnMax) => {
    if (v == null || Number.isNaN(Number(v))) return "unknown";
    const n = Number(v);
    if (n <= goodMax) return "good";
    if (n <= warnMax) return "warn";
    return "bad";
  };
  const bandHol = (v) => {
    if (v == null || Number.isNaN(Number(v))) return "unknown";
    if (v >= 8) return "good";
    if (v >= 5) return "warn";
    return "bad";
  };
  return {
    ready: true,
    ratio_loss: {
      value: ratio,
      band: bandLoss(ratio, 0.015, 0.04),
      label: "Ratio loss",
      explanation: "Neighborhood pasta∶egg ratio surrogate (lower better).",
      source: ratioSrc || null,
    },
    nutrient_loss: {
      value: nutrient,
      band: bandLoss(nutrient, 0.0005, 0.025),
      label: "Nutrient loss",
      explanation: "PFC box slack (0 = inside).",
      source: nutrient != null ? "telemetry" : null,
    },
    holistic_0_10: {
      value: holistic,
      band: bandHol(holistic),
      label: "Holistic",
      source: judge.holistic_score_0_10 != null ? "llm_judge" : "intent_overlap",
      explanation: "0–10 dish fit (judge preferred).",
    },
    ingredients: [],
    score_history: final?.display_scores?.score_history || final?.score_history || [],
  };
}

function renderDisplayScores(scores, { titleHint } = {}) {
  if (!displayScoresEl) return;
  const s = scores || emptyScores();
  const cards = [s.ratio_loss, s.nutrient_loss, s.holistic_0_10].filter(Boolean);
  displayScoresEl.innerHTML = cards
    .map((card) => {
      const isHolistic = (card.label || "").toLowerCase().includes("holistic");
      const digits = isHolistic ? 1 : 3;
      const unit = isHolistic && card.value != null ? " / 10" : "";
      const hint =
        card.band === "good"
          ? "Looking good"
          : card.band === "warn"
            ? "Worth watching"
            : card.band === "bad"
              ? "Needs attention"
              : "No signal";
      return `<article class="score-card band-${escapeHtml(card.band || "unknown")}" title="${escapeHtml(
        card.explanation || ""
      )}">
        <p class="score-label">${escapeHtml(card.label || "score")}</p>
        <p class="score-value">${fmtScore(card.value, digits)}${unit}</p>
        <p class="score-hint">${escapeHtml(hint)}${
          card.source ? ` · ${escapeHtml(card.source)}` : ""
        }</p>
      </article>`;
    })
    .join("");
  const hintEl = document.getElementById("scores-hint");
  if (hintEl) {
    if (titleHint) hintEl.textContent = titleHint;
    else if (!s.ready && s.ratio_loss?.value == null)
      hintEl.textContent = "Blank until the agent diagnoses; then shows the best sample each iteration.";
    else if (s.iteration != null)
      hintEl.textContent = `Best sample after iteration ${s.iteration}.`;
    else hintEl.textContent = "Current best sample.";
  }
  if (Array.isArray(s.score_history) && s.score_history.length) {
    scoreHistory = s.score_history;
  }
  renderLossChart();
}

function shareSparkline(iqr, recipeShare) {
  if (!iqr || iqr.min == null || iqr.max == null) {
    return `<span class="spark-empty">—</span>`;
  }
  const min = Number(iqr.min);
  const max = Number(iqr.max);
  const q1 = Number(iqr.q1);
  const med = Number(iqr.median);
  const q3 = Number(iqr.q3);
  const w = 88;
  const h = 18;
  const pad = 3;
  const span = max - min || 1e-9;
  const x = (v) => pad + ((Number(v) - min) / span) * (w - 2 * pad);
  const y = h / 2;
  const marker =
    recipeShare == null || Number.isNaN(Number(recipeShare))
      ? ""
      : (() => {
          const cx = x(Math.min(max, Math.max(min, Number(recipeShare))));
          return `<circle cx="${cx}" cy="${y}" r="3.2" class="spark-recipe" />`;
        })();
  return `<svg class="share-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
    <line x1="${x(min)}" y1="${y}" x2="${x(max)}" y2="${y}" class="spark-range" />
    <rect x="${x(q1)}" y="${y - 4}" width="${Math.max(1, x(q3) - x(q1))}" height="8" class="spark-iqr" rx="1" />
    <line x1="${x(med)}" y1="${y - 5}" x2="${x(med)}" y2="${y + 5}" class="spark-median" />
    ${marker}
  </svg>`;
}

function buildIngredientsTableHtml(ings, { title = "Final ingredients" } = {}) {
  if (!Array.isArray(ings) || !ings.length) return "";
  const rich = ings.some((r) => r.foodon_leaf_label != null || r.basis_node_label != null || r.share_iqr);
  const rows = ings
    .map((r) => {
      const label = r.label || r.name || r.fdc_description || "?";
      const grams = r.grams != null ? Number(r.grams).toFixed(1) : "—";
      if (!rich) {
        return `<tr><td>${escapeHtml(String(label))}</td><td class="grams">${escapeHtml(grams)} g</td></tr>`;
      }
      const foodon = r.foodon_leaf_label || r.foodon_leaf_id || "—";
      const basis = r.basis_node_label || r.basis_node_id || "—";
      const levels = r.aggregation_levels != null ? String(r.aggregation_levels) : "—";
      const loss =
        r.loss_contribution != null && !Number.isNaN(Number(r.loss_contribution))
          ? Number(r.loss_contribution).toFixed(3)
          : r.basis_n_hits != null && Number(r.basis_n_hits) < 5
            ? "n/a"
            : "—";
      const kcal =
        r.calories != null && !Number.isNaN(Number(r.calories))
          ? Number(r.calories).toFixed(0)
          : "—";
      const band = r.loss_band || "unknown";
      return `<tr>
        <td>${escapeHtml(String(label))}</td>
        <td class="grams">${escapeHtml(grams)} g</td>
        <td title="${escapeHtml(String(r.foodon_leaf_id || ""))}">${escapeHtml(String(foodon))}</td>
        <td title="${escapeHtml(String(r.basis_node_id || ""))}">${escapeHtml(String(basis))}</td>
        <td class="num">${escapeHtml(levels)}</td>
        <td class="num loss-band-${escapeHtml(band)}">${escapeHtml(loss)}</td>
        <td class="num">${escapeHtml(kcal)}</td>
        <td class="spark-cell">${shareSparkline(r.share_iqr, r.recipe_share)}</td>
      </tr>`;
    })
    .join("");
  if (!rich) {
    return `<h3>${escapeHtml(title)}</h3>
      <table>
        <thead><tr><th>Ingredient</th><th>Quantity</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }
  return `<h3>${escapeHtml(title)}</h3>
    <div class="ing-table-wrap">
      <table class="ing-rich">
        <thead>
          <tr>
            <th>Ingredient</th>
            <th>Qty</th>
            <th>FoodOn</th>
            <th>Basis</th>
            <th>↑</th>
            <th title="Mass-share ratio loss vs neighborhood samples">Ratio loss</th>
            <th>kcal</th>
            <th title="Neighborhood share: range, IQR box, median, recipe marker">Share</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="hint spark-legend">Share sparkline: bar = min–max, box = IQR, tick = median, dot = this recipe.</p>`;
}

function renderPathScoreCards(scores) {
  const s = scores || emptyScores();
  const cards = [s.ratio_loss, s.nutrient_loss, s.holistic_0_10].filter(Boolean);
  return cards
    .map((card) => {
      const isHolistic = (card.label || "").toLowerCase().includes("holistic");
      const digits = isHolistic ? 1 : 3;
      const unit = isHolistic && card.value != null ? " / 10" : "";
      return `<article class="score-card band-${escapeHtml(card.band || "unknown")} compact">
        <p class="score-label">${escapeHtml(card.label || "score")}</p>
        <p class="score-value">${fmtScore(card.value, digits)}${unit}</p>
      </article>`;
    })
    .join("");
}

function renderPathFinalCard(pathKey, pathFinal) {
  const label = pathFinal?.path_label || (pathKey === "ood" ? "OOD" : "In-distribution");
  const branch = pathFinal?.chosen?.branch || pathKey;
  const edits = pathFinal?.chosen?.edits || pathFinal?.chosen?.entry?.edits || [];
  const editSummary =
    edits.length > 0
      ? edits.map((e) => `${e.action || "?"} ${e.label || ""}`.trim()).join("; ")
      : "No edits (baseline or quantity-only)";
  const delta = pathFinal?.chosen?.delta_L_star ?? pathFinal?.chosen?.entry?.delta_L_star;
  const deltaTxt =
    delta != null && !Number.isNaN(Number(delta)) ? `ΔL* ${Number(delta).toFixed(3)}` : "";
  const scores = pathFinal?.display_scores || clientDisplayScores(pathFinal);
  const ings = scores.ingredients?.length
    ? scores.ingredients
    : pathFinal?.chosen?.ingredients || pathFinal?.chosen?.entry?.ingredients || [];
  const color = BRANCH_COLORS[branch] || BRANCH_COLORS[pathKey] || "#445";
  return `<article class="path-final-card path-${escapeHtml(pathKey)}" style="--path-accent:${color}">
    <header class="path-final-head">
      <h3>${escapeHtml(label)}</h3>
      <p class="path-meta">${escapeHtml(branch)}${deltaTxt ? ` · ${escapeHtml(deltaTxt)}` : ""}</p>
      <p class="path-edits hint">${escapeHtml(editSummary)}</p>
    </header>
    <div class="path-scores display-scores">${renderPathScoreCards(scores)}</div>
    <div class="path-ingredients">${buildIngredientsTableHtml(ings, { title: "Ingredients" })}</div>
  </article>`;
}

function renderPathFinals(final) {
  if (!pathFinalsEl) return false;
  const paths = final?.path_finals || {};
  const idPath = paths.in_distribution;
  const oodPath = paths.ood;
  if (!idPath && !oodPath) {
    pathFinalsEl.classList.add("hidden");
    pathFinalsEl.innerHTML = "";
    return false;
  }
  pathFinalsEl.classList.remove("hidden");
  const cards = [
    idPath
      ? renderPathFinalCard("in_distribution", idPath)
      : `<article class="path-final-card path-empty"><h3>In-distribution</h3><p class="hint">No in-distribution candidate was explored.</p></article>`,
    oodPath
      ? renderPathFinalCard("ood", oodPath)
      : `<article class="path-final-card path-empty"><h3>OOD</h3><p class="hint">No OOD candidate was explored.</p></article>`,
  ];
  pathFinalsEl.innerHTML = cards.join("");
  return true;
}

function renderFinalIngredients(final) {
  if (!finalIngredientsEl) return;
  const fromScores = final?.display_scores?.ingredients;
  let ings = Array.isArray(fromScores) && fromScores.length ? fromScores : null;
  if (!ings) {
    const chosen = final?.chosen || {};
    ings =
      chosen.ingredients ||
      chosen.entry?.ingredients ||
      chosen.entry?.metrics?.raw?.entry?.ingredients ||
      [];
  }
  if (!Array.isArray(ings) || !ings.length) {
    finalIngredientsEl.classList.add("hidden");
    finalIngredientsEl.innerHTML = "";
    return;
  }
  finalIngredientsEl.classList.remove("hidden");
  finalIngredientsEl.innerHTML = buildIngredientsTableHtml(ings);
}

const BRANCH_COLORS = {
  in_distribution: "#1d4e89",
  ood: "#9b2c2c",
  ood_protein: "#9b2c2c",
  hybrid: "#8a5a00",
  current: "#1f6b42",
};

function renderLossChart() {
  const el = document.getElementById("loss-chart");
  const metricSel = document.getElementById("loss-chart-metric");
  if (!el) return;
  const metric = metricSel?.value || "ratio_loss";
  const hist = Array.isArray(scoreHistory) ? scoreHistory : [];
  const points = hist.filter((p) => p && p[metric] != null && !Number.isNaN(Number(p[metric])));
  if (!points.length) {
    el.innerHTML = `<p class="hint">No ${escapeHtml(metric)} history yet.</p>`;
    return;
  }
  const byBranch = {};
  for (const p of points) {
    const b = p.branch || "current";
    if (!byBranch[b]) byBranch[b] = [];
    byBranch[b].push(p);
  }
  const iters = [...new Set(points.map((p) => Number(p.iteration) || 0))].sort((a, b) => a - b);
  const vals = points.map((p) => Number(p[metric]));
  const ymin = Math.min(0, ...vals);
  const ymax = Math.max(...vals, ymin + 1e-6);
  const w = 420;
  const h = 160;
  const padL = 36;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const xSpan = Math.max(1, iters[iters.length - 1] - iters[0]);
  const xOf = (it) => padL + ((it - iters[0]) / xSpan) * (w - padL - padR);
  const yOf = (v) => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * (h - padT - padB);
  const lines = Object.entries(byBranch)
    .map(([branch, rows]) => {
      const sorted = [...rows].sort((a, b) => (a.iteration || 0) - (b.iteration || 0));
      // Keep best (lowest loss) per iteration for this branch
      const bestByIt = {};
      for (const r of sorted) {
        const it = Number(r.iteration) || 0;
        const v = Number(r[metric]);
        if (bestByIt[it] == null || v < bestByIt[it]) bestByIt[it] = v;
      }
      const its = Object.keys(bestByIt)
        .map(Number)
        .sort((a, b) => a - b);
      if (!its.length) return "";
      const d = its.map((it, i) => `${i ? "L" : "M"}${xOf(it).toFixed(1)},${yOf(bestByIt[it]).toFixed(1)}`).join(" ");
      const color = BRANCH_COLORS[branch] || "#445";
      const dots = its
        .map(
          (it) =>
            `<circle cx="${xOf(it).toFixed(1)}" cy="${yOf(bestByIt[it]).toFixed(1)}" r="3" fill="${color}" />`
        )
        .join("");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" />${dots}`;
    })
    .join("");
  const legend = Object.keys(byBranch)
    .map((b) => {
      const color = BRANCH_COLORS[b] || "#445";
      return `<span class="chart-legend-item"><span class="swatch" style="background:${color}"></span>${escapeHtml(
        b
      )}</span>`;
    })
    .join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" role="img">
      <line x1="${padL}" y1="${yOf(0)}" x2="${w - padR}" y2="${yOf(0)}" stroke="#c5d0c9" stroke-dasharray="3 3" />
      <text x="4" y="${yOf(ymax) + 4}" class="chart-axis">${ymax.toFixed(3)}</text>
      <text x="4" y="${yOf(ymin) + 4}" class="chart-axis">${ymin.toFixed(3)}</text>
      ${lines}
      <text x="${padL}" y="${h - 6}" class="chart-axis">iter ${iters[0]}</text>
      <text x="${w - padR - 40}" y="${h - 6}" class="chart-axis">iter ${iters[iters.length - 1]}</text>
    </svg>
    <div class="chart-legend">${legend}</div>`;
}

function renderFinalistCards(final) {
  if (!finalistCards) return;
  const list = final?.scored_finalists || [];
  if (!list.length) {
    finalistCards.classList.add("hidden");
    finalistCards.innerHTML = "";
    return;
  }
  finalistCards.classList.remove("hidden");
  const winnerId =
    final?.chosen?.entry?.candidate_id ||
    final?.judge_result?.winner_id ||
    (final?.chosen?.entry || {}).candidate_id;
  finalistCards.innerHTML = list
    .map((s) => {
      const good = s.good || {};
      const branch = s.metrics?.raw?.branch || s.metrics?.raw?.entry?.branch || s.branch || "";
      const isWinner = s.candidate_id === winnerId;
      return `<article class="finalist-card${isWinner ? " winner" : ""}${s.dominated ? " dominated" : ""}">
        <h3>${escapeHtml(s.candidate_id || "?")}${isWinner ? " ★ winner" : ""}${
        s.dominated ? " (dominated)" : ""
      }${branch ? ` · ${escapeHtml(branch)}` : ""}</h3>
        <p class="composite">composite <strong>${(s.composite ?? 0).toFixed(3)}</strong></p>
        <ul class="metric-list">
          <li>nutrient good ${(good.nutrient_dist ?? 0).toFixed(3)}</li>
          <li>ratio good ${(good.ratio_badness ?? 0).toFixed(3)}</li>
          <li>intent good ${(good.intent_gap ?? 0).toFixed(3)}</li>
          <li>churn good ${(good.churn ?? 0).toFixed(3)}</li>
        </ul>
      </article>`;
    })
    .join("");
}

function showFinalResult(final) {
  resultPanel.classList.remove("hidden");
  const hasPathFinals = renderPathFinals(final);
  const scores = final?.display_scores || clientDisplayScores(final);
  if (Array.isArray(scores.score_history)) scoreHistory = scores.score_history;
  if (hasPathFinals) {
    if (finalIngredientsEl) {
      finalIngredientsEl.classList.add("hidden");
      finalIngredientsEl.innerHTML = "";
    }
    renderDisplayScores(scores, {
      titleHint: "Agent's applied path (scores panel). Side-by-side ID vs OOD champions below.",
    });
  } else {
    renderDisplayScores(scores, { titleHint: "Final chosen recipe scores." });
    renderFinalIngredients(final);
  }
  renderFinalistCards(final);
  resultJson.textContent = JSON.stringify(final, null, 2);
  if (lastRunBundle) {
    lastRunBundle.final = final;
    lastRunBundle.history = final?.history || lastRunBundle.history || [];
    lastRunBundle.decision_outcomes = final?.decision_outcomes || [];
    lastRunBundle.run_telemetry = final?.run_telemetry || {};
    lastRunBundle.score_history = scoreHistory;
  }
  if (flowSummaryBtn) {
    flowSummaryBtn.disabled = !lastRunBundle;
  }
  if (flowSummaryStatus) {
    flowSummaryStatus.textContent = lastRunBundle
      ? "Ready — asks gpt-4o for a holistic review of this run only."
      : "Available after a finished run.";
  }
  setRunChatEnabled(Boolean(lastRunBundle?.final));
}

function simpleMarkdownToHtml(md) {
  const lines = String(md || "").split("\n");
  const out = [];
  let inList = false;
  const flushList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^###\s+/.test(line)) {
      flushList();
      out.push(`<h3>${inline(line.replace(/^###\s+/, ""))}</h3>`);
    } else if (/^##\s+/.test(line)) {
      flushList();
      out.push(`<h3>${inline(line.replace(/^##\s+/, ""))}</h3>`);
    } else if (/^#\s+/.test(line)) {
      flushList();
      out.push(`<h3>${inline(line.replace(/^#\s+/, ""))}</h3>`);
    } else if (/^[-*]\s+/.test(line)) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${inline(line.replace(/^[-*]\s+/, ""))}</li>`);
    } else if (!line.trim()) {
      flushList();
    } else {
      flushList();
      out.push(`<p>${inline(line)}</p>`);
    }
  }
  flushList();
  return out.join("\n");
}

async function requestFlowSummary() {
  if (!lastRunBundle) {
    if (flowSummaryStatus) flowSummaryStatus.textContent = "Run the agent first.";
    return;
  }
  if (flowSummaryBtn) flowSummaryBtn.disabled = true;
  if (flowSummaryStatus) flowSummaryStatus.textContent = "Asking gpt-4o…";
  if (flowSummaryEl) {
    flowSummaryEl.classList.remove("hidden");
    flowSummaryEl.textContent = "Generating holistic review…";
  }
  try {
    const res = await fetch("/api/flow_summary", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: lastRunBundle.mode,
        user_request: lastRunBundle.user_request,
        title: lastRunBundle.title,
        final: lastRunBundle.final,
        steps: lastRunBundle.steps,
        history: lastRunBundle.history,
        decision_outcomes: lastRunBundle.decision_outcomes,
        llm_calls: lastRunBundle.llm_calls,
        run_telemetry: lastRunBundle.run_telemetry,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    if (flowSummaryEl) {
      flowSummaryEl.innerHTML = simpleMarkdownToHtml(data.summary_markdown || "");
    }
    const toks = data.usage?.total_tokens;
    if (flowSummaryStatus) {
      flowSummaryStatus.textContent = toks
        ? `Done (gpt-4o · ${toks} tokens).`
        : "Done (gpt-4o).";
    }
  } catch (err) {
    if (flowSummaryEl) flowSummaryEl.textContent = String(err.message || err);
    if (flowSummaryStatus) flowSummaryStatus.textContent = `Error: ${err.message || err}`;
  } finally {
    if (flowSummaryBtn) flowSummaryBtn.disabled = !lastRunBundle;
  }
}

flowSummaryBtn?.addEventListener("click", () => {
  requestFlowSummary();
});

function setRunChatEnabled(enabled) {
  const on = Boolean(enabled && lastRunBundle?.final);
  if (runChatInput) runChatInput.disabled = !on || runChatBusy;
  if (runChatSendBtn) runChatSendBtn.disabled = !on || runChatBusy;
  if (runChatClearBtn) runChatClearBtn.disabled = !on || runChatBusy || runChatMessages.length === 0;
  if (runChatStatus) {
    if (!on) {
      runChatStatus.textContent = "Available after a finished run.";
    } else if (runChatBusy) {
      runChatStatus.textContent = "Asking gpt-4o…";
    } else {
      runChatStatus.textContent = "Ask about steps, losses, ingredients, or ID vs OOD paths.";
    }
  }
}

function renderRunChatMessages() {
  if (!runChatMessagesEl) return;
  if (!runChatMessages.length) {
    runChatMessagesEl.innerHTML = `<p class="hint run-chat-empty">No messages yet. Try a specific question about this run.</p>`;
    return;
  }
  runChatMessagesEl.innerHTML = runChatMessages
    .map((msg) => {
      const role = msg.role === "assistant" ? "assistant" : "user";
      const body =
        role === "assistant"
          ? simpleMarkdownToHtml(msg.content || "")
          : `<p>${escapeHtml(msg.content || "")}</p>`;
      return `<article class="run-chat-msg run-chat-msg-${role}">
        <p class="run-chat-role">${role === "assistant" ? "gpt-4o" : "You"}</p>
        <div class="run-chat-body">${body}</div>
      </article>`;
    })
    .join("");
  runChatMessagesEl.scrollTop = runChatMessagesEl.scrollHeight;
}

function resetRunChat() {
  runChatMessages = [];
  runChatBusy = false;
  if (runChatInput) runChatInput.value = "";
  renderRunChatMessages();
  setRunChatEnabled(false);
}

function runChatPayload() {
  if (!lastRunBundle) return null;
  return {
    mode: lastRunBundle.mode,
    user_request: lastRunBundle.user_request,
    title: lastRunBundle.title,
    final: lastRunBundle.final,
    steps: lastRunBundle.steps,
    history: lastRunBundle.history,
    decision_outcomes: lastRunBundle.decision_outcomes,
    llm_calls: lastRunBundle.llm_calls,
    run_telemetry: lastRunBundle.run_telemetry,
    messages: runChatMessages.map((m) => ({ role: m.role, content: m.content })),
  };
}

async function sendRunChatMessage(text) {
  const question = String(text || "").trim();
  if (!question || !lastRunBundle?.final || runChatBusy) return;

  runChatMessages.push({ role: "user", content: question });
  renderRunChatMessages();
  runChatBusy = true;
  setRunChatEnabled(true);
  if (runChatInput) runChatInput.value = "";

  try {
    const res = await fetch("/api/run_chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runChatPayload()),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    runChatMessages.push({ role: "assistant", content: data.reply_markdown || "" });
    renderRunChatMessages();
    const toks = data.usage?.total_tokens;
    if (runChatStatus) {
      runChatStatus.textContent = toks
        ? `Reply received (gpt-4o · ${toks} tokens).`
        : "Reply received.";
    }
  } catch (err) {
    runChatMessages.push({
      role: "assistant",
      content: `**Error:** ${err.message || err}`,
    });
    renderRunChatMessages();
    if (runChatStatus) runChatStatus.textContent = `Error: ${err.message || err}`;
  } finally {
    runChatBusy = false;
    setRunChatEnabled(true);
    runChatInput?.focus();
  }
}

runChatForm?.addEventListener("submit", (ev) => {
  ev.preventDefault();
  sendRunChatMessage(runChatInput?.value || "");
});

runChatInput?.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendRunChatMessage(runChatInput.value || "");
  }
});

runChatClearBtn?.addEventListener("click", () => {
  runChatMessages = [];
  renderRunChatMessages();
  setRunChatEnabled(true);
});

function highlightNode(node, durationMs = null) {
  if (!node) return;
  const svg = flowGraph.querySelector("svg");
  if (!svg) return;

  if (lastNode) {
    visitedNodes.add(lastNode);
    const prev = svg.querySelector(`.flow-node[data-node="${lastNode}"]`);
    if (prev) {
      prev.classList.remove("active");
      prev.classList.add("visited");
    }
    svg.querySelectorAll(".flow-edge").forEach((edge) => {
      if (edge.dataset.from === lastNode && edge.dataset.to === node) {
        edge.classList.add("active");
        setTimeout(() => {
          edge.classList.remove("active");
          edge.classList.add("visited");
        }, 400);
      }
    });
  }

  const el = svg.querySelector(`.flow-node[data-node="${node}"]`);
  if (el) {
    el.classList.add("active");
    el.classList.remove("visited");
  }
  lastNode = node;
  const elapsed =
    durationMs != null && Number.isFinite(Number(durationMs))
      ? ` · ${(Number(durationMs) / 1000).toFixed(2)}s`
      : "";
  flowStatus.textContent = `Last completed: ${NODE_LABELS[node] || node}${elapsed} · next node running…`;
}

function resetRunUI() {
  visitedNodes = new Set();
  lastNode = null;
  lastRunBundle = {
    mode: currentMode(),
    user_request: (userRequestEl?.value || "").trim() || selectedCanonical?.title || "",
    title: selectedCanonical?.title || "",
    steps: [],
    llm_calls: [],
    history: [],
    decision_outcomes: [],
    run_telemetry: {},
    final: null,
  };
  clearRunPrompts();
  if (selectedFlowNode) renderFlowInspector(selectedFlowNode);
  timeline.innerHTML = "";
  transcriptEl.innerHTML = "";
  resultPanel.classList.add("hidden");
  resultJson.textContent = "";
  if (finalistCards) {
    finalistCards.classList.add("hidden");
    finalistCards.innerHTML = "";
  }
  if (displayScoresEl) {
    scoreHistory = [];
    renderDisplayScores(emptyScores());
  }
  if (finalIngredientsEl) {
    finalIngredientsEl.classList.add("hidden");
    finalIngredientsEl.innerHTML = "";
  }
  if (flowSummaryEl) {
    flowSummaryEl.classList.add("hidden");
    flowSummaryEl.innerHTML = "";
  }
  if (flowSummaryBtn) flowSummaryBtn.disabled = true;
  if (flowSummaryStatus) flowSummaryStatus.textContent = "Available after a finished run.";
  resetRunChat();
  flowStatus.textContent = "Running…";
  setRunStatus("Running…");
  const svg = flowGraph.querySelector("svg");
  if (svg) {
    svg.querySelectorAll(".flow-node").forEach((n) => n.classList.remove("active", "visited"));
    svg.querySelectorAll(".flow-edge").forEach((e) => e.classList.remove("active", "visited"));
  }
}

function bandClass(band) {
  if (!band) return "";
  return `band-pill band-${band}`;
}

function fmt(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return typeof v === "number" ? (Math.abs(v) >= 0.001 && Math.abs(v) < 1000 ? v.toFixed(4) : String(v)) : String(v);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function pretty(obj) {
  return escapeHtml(JSON.stringify(obj, null, 2));
}

function detailBlock(title, contentHtml) {
  return `<div class="detail-block"><h4>${escapeHtml(title)}</h4>${contentHtml}</div>`;
}

function renderChosenRecipe(chosen) {
  if (!chosen) return "";
  const ings = (chosen.ingredients || [])
    .slice(0, 12)
    .map((i) => `${i.label || "?"} (${fmt(i.grams)}g)`)
    .join("; ");
  const more = (chosen.ingredients || []).length > 12 ? "…" : "";
  const sel = chosen.selection || {};
  const cacheHit = sel.neighborhood_from_cache === true;
  const cacheMiss = sel.neighborhood_from_cache === false;
  const cacheLabel = cacheHit
    ? "Neighborhood: Jaccard cache"
    : cacheMiss
      ? "Neighborhood: live fast basis (cache miss)"
      : "";
  return `<div class="recipe-box">
    <strong>Chosen recipe:</strong> ${escapeHtml(chosen.title || "?")}
    ${chosen.recipe_nlg_id ? ` · NLG <code>${escapeHtml(chosen.recipe_nlg_id)}</code>` : ""}
    ${chosen.canonical_id != null ? ` · canonical ${chosen.canonical_id}` : ""}
    ${cacheLabel ? `<div class="hint">${escapeHtml(cacheLabel)}</div>` : ""}
    <div class="hint">${escapeHtml(chosen.selection_note || "")}</div>
    ${sel.distance_to_target_box != null ? `<div>PFC distance to target box: ${fmt(sel.distance_to_target_box)}${sel.switched_from_default ? " (switched from FoodOn-default start)" : ""}</div>` : ""}
    <div><strong>Ingredients:</strong> ${escapeHtml(ings || "(none)")}${more}</div>
  </div>`;
}

function renderTriggers(triggers) {
  if (!triggers?.length) return "";
  return triggers
    .map((t) => {
      const primary = t.primary ? " (primary)" : "";
      return `<div class="trigger-box">
        <strong>${escapeHtml(t.metric)}${primary}</strong>
        <div>${escapeHtml(t.reason || "")}</div>
        <div class="step-detail">current=${escapeHtml(JSON.stringify(t.current_value))}
threshold_to_clear=${escapeHtml(JSON.stringify(t.threshold_to_clear))}
${escapeHtml(t.clearance || "")}</div>
      </div>`;
    })
    .join("");
}

function renderHullDistance(hull) {
  const dist = hull?.distance || hull?.residual?.distance;
  if (!dist && !hull) return "";
  const interp = dist?.interpretation || "";
  const score = dist?.outside_score;
  const boxDist = dist?.min_distance_box_to_hull;
  const midDist = dist?.min_distance_midpoint_to_hull_sample;
  const gaps = dist?.axis_gaps;
  return `<div class="hull-box">
    <strong>Hull distance</strong>
    <div>intersects=${hull?.intersects} · geometric=${hull?.geometric_intersects} · lp=${hull?.lp_feasible}</div>
    <div>box→hull=${fmt(boxDist)} · mid→nearest sample=${fmt(midDist)} · outside_score=${fmt(score)}</div>
    ${gaps ? `<div>axis gaps: P=${fmt(gaps.protein)} C=${fmt(gaps.carbs)} F=${fmt(gaps.fat)}</div>` : ""}
    ${interp ? `<div>${escapeHtml(interp)}</div>` : ""}
  </div>`;
}

function renderCandidates(cands, dropped) {
  if (!cands?.length && !dropped?.length) return "<p class=\"hint\">No candidates at this step.</p>";
  const items = (cands || [])
    .map(
      (c) =>
        `<li><code>${escapeHtml(c.candidate_id || "?")}</code> · ${escapeHtml(c.action || "")} · ${escapeHtml(
          c.label || ""
        )} · L*=${fmt(c.L_star)} · cooc=${fmt(c.cooccurrence)} · geom=${fmt(c.geom_score)}</li>`
    )
    .join("");
  const drop =
    dropped?.length
      ? `<p class="hint">Dropped (${dropped.length}): ${dropped
          .map((d) => `${d.candidate?.label || d.candidate?.candidate_id} (${d.reason})`)
          .join(", ")}</p>`
      : "";
  return `${cands?.length ? `<ul class="cand-list">${items}</ul>` : ""}${drop}`;
}

function renderTools(tools) {
  if (!tools?.length) return "";
  return tools
    .map(
      (t) =>
        detailBlock(
          `Tool: ${t.name || "?"}${t.mode ? ` [${t.mode}]` : ""}`,
          `<p class="hint">${escapeHtml(t.purpose || "")}</p><pre>${pretty(t.output_summary || t)}</pre>`
        )
    )
    .join("");
}

function appendStep(ev) {
  const payload = ev.payload || {};
  const nodeName = ev.node || payload.node || "";
  for (const tool of payload.tools_used || []) {
    const trace = tool.llm_trace;
    if (!trace || !nodeName) continue;
    for (const msg of trace.messages || []) {
      recordRunPrompt(nodeName, {
        role: msg.role,
        content: msg.content,
        mode: trace.mode,
        model: trace.model,
        tool: tool.name,
        name: tool.name,
      });
    }
  }
  if (selectedFlowNode === nodeName) renderFlowInspector(nodeName);

  const card = document.createElement("div");
  card.className = "step-card active-step";
  timeline.querySelectorAll(".step-card").forEach((c) => c.classList.remove("active-step"));

  const band = ev.fidelity_band || payload.fidelity_band;
  const head = document.createElement("div");
  head.className = "step-head";
  head.innerHTML = `
    <span class="step-node">${ev.node || payload.node || "?"}</span>
    <span class="step-meta">#${ev.seq ?? "—"} · iter ${ev.iteration ?? payload.iteration ?? "—"}${
      ev.duration_ms != null ? ` · ${(Number(ev.duration_ms) / 1000).toFixed(2)}s` : ""
    }</span>
    ${band ? `<span class="${bandClass(band)}">${band}</span>` : ""}
  `;
  card.appendChild(head);

  if (payload.chosen_recipe) {
    const wrap = document.createElement("div");
    wrap.innerHTML = renderChosenRecipe(payload.chosen_recipe);
    card.appendChild(wrap);
  }

  if (payload.neighborhood_n != null || payload.neighborhood_recipes?.length) {
    const n = payload.neighborhood_n ?? payload.neighborhood_recipes.length;
    const c = document.createElement("div");
    c.className = "step-detail";
    c.textContent = `neighborhood recipes: ${n} (FoodOn matches for this canonical dish)`;
    card.appendChild(c);
  }

  const decision = payload.decision;
  if (decision) {
    const rat = document.createElement("p");
    rat.className = "rationale";
    const identity = decision.identity || {};
    rat.innerHTML = `<strong>${escapeHtml(decision.action || "decision")}</strong> — ${escapeHtml(
      decision.rationale || "(no rationale)"
    )}
      ${identity.preserves_dish === false ? " · ⚠ may break dish identity" : ""}
      ${identity.acceptable_variant ? " · variant OK" : ""}`;
    card.appendChild(rat);
  }

  const diag = payload.diagnosis;
  if (diag) {
    const d = document.createElement("div");
    d.className = "step-detail";
    d.textContent = [
      `diagnosis=${diag.diagnosis} · ${diag.meaning || ""}`,
      `L_max_norm=${fmt(diag.L_max_norm)}  L_total=${fmt(diag.L_total)}  n_red=${diag.n_red}`,
      diag.binding_macros?.length ? `binding: ${diag.binding_macros.join(", ")}` : "",
      diag.band_thresholds
        ? `thresholds: F_accept=${diag.band_thresholds.F_accept} F_max=${diag.band_thresholds.F_max}`
        : "",
    ]
      .filter(Boolean)
      .join("\n");
    card.appendChild(d);

    if (band === "must_retry" || band === "moderate") {
      const wrap = document.createElement("div");
      wrap.innerHTML = renderTriggers(diag.retry_triggers || []);
      card.appendChild(wrap);
    }
  }

  if (payload.hull) {
    const wrap = document.createElement("div");
    wrap.innerHTML = renderHullDistance(payload.hull);
    card.appendChild(wrap);
  }

  if (payload.opt) {
    const o = document.createElement("div");
    o.className = "step-detail";
    o.textContent = `opt status=${payload.opt.status} feasible=${payload.opt.feasible} obj=${fmt(payload.opt.objective)}`;
    card.appendChild(o);
  }

  if (payload.candidates?.length) {
    const c = document.createElement("div");
    c.className = "step-detail";
    c.textContent = `candidates (${payload.candidates.length}): ${payload.candidates
      .slice(0, 5)
      .map((x) => x.label || x.candidate_id || x.action)
      .join(", ")}${payload.candidates.length > 5 ? "…" : ""}`;
    card.appendChild(c);
  }

  if (payload.tools_used?.length) {
    const t = document.createElement("div");
    t.className = "step-detail";
    t.textContent = `tools: ${payload.tools_used.map((x) => x.name).join(", ")}`;
    card.appendChild(t);
  }

  // Expandable deep detail
  const more = document.createElement("details");
  more.className = "step-more";
  const summary = document.createElement("summary");
  summary.textContent = "More context";
  more.appendChild(summary);

  let html = "";
  if (payload.chosen_recipe) {
    html += detailBlock("Chosen recipe (full)", `<pre>${pretty(payload.chosen_recipe)}</pre>`);
  }
  if (payload.neighborhood_recipes?.length) {
    html += detailBlock(
      `Neighborhood recipes (${payload.neighborhood_recipes.length})`,
      `<pre>${pretty(payload.neighborhood_recipes.slice(0, 40))}</pre>`
    );
  }
  if (diag?.retry_triggers?.length) {
    html += detailBlock("Retry triggers (full)", `<pre>${pretty(diag.retry_triggers)}</pre>`);
  }
  if (diag?.terms?.length) {
    html += detailBlock("Fidelity terms / zones", `<pre>${pretty(diag.terms)}</pre>`);
  }
  if (payload.hull) {
    html += detailBlock("Hull (full)", `<pre>${pretty(payload.hull)}</pre>`);
  }
  if (payload.opt) {
    html += detailBlock("Optimizer", `<pre>${pretty(payload.opt)}</pre>`);
  }
  if (payload.candidates || payload.candidates_dropped) {
    html += detailBlock(
      "Candidates at this stage",
      renderCandidates(payload.candidates, payload.candidates_dropped)
    );
    html += detailBlock("Candidates JSON", `<pre>${pretty({ kept: payload.candidates, dropped: payload.candidates_dropped })}</pre>`);
  }
  if (payload.tools_used?.length) {
    html += renderTools(payload.tools_used);
  }
  if (payload.detail?.tools_used) {
    html += detailBlock("Tool outputs (full)", `<pre>${pretty(payload.detail.tools_used)}</pre>`);
  }
  if (payload.llm_trace) {
    const trace = payload.llm_trace;
    html += detailBlock(
      `LLM call (${trace.mode || "?"}${trace.model ? ` / ${trace.model}` : ""})`,
      `<p class="hint">Reasoning: ${escapeHtml(trace.rationale || "")}</p>` +
        (trace.messages || [])
          .map(
            (m) =>
              detailBlock(
                `Prompt · ${m.role}`,
                `<pre>${escapeHtml(m.content || "")}</pre>`
              )
          )
          .join("") +
        detailBlock("Raw LLM response", `<pre>${escapeHtml(trace.raw_response || "")}</pre>`)
    );
  }
  if (payload.decision_context) {
    html += detailBlock("DecisionContext sent to LLM", `<pre>${pretty(payload.decision_context)}</pre>`);
  }
  if (payload.candidate_pool?.length) {
    html += detailBlock("Candidate pool (saved moderate solutions)", `<pre>${pretty(payload.candidate_pool)}</pre>`);
  }
  if (payload.last_applied_candidate) {
    html += detailBlock("Applied candidate", `<pre>${pretty(payload.last_applied_candidate)}</pre>`);
  }
  if (payload.detail) {
    html += detailBlock("Raw step detail", `<pre>${pretty(payload.detail)}</pre>`);
  }

  const body = document.createElement("div");
  body.innerHTML = html || "<p class=\"hint\">No extra detail for this node.</p>";
  more.appendChild(body);
  card.appendChild(more);

  timeline.prepend(card);
}

function appendTranscriptEntry(entry, meta = {}) {
  const div = document.createElement("div");
  div.className = "transcript-entry";
  const kind = entry.kind || "note";
  const nodeName = meta.node || entry.node || "";
  if (kind === "prompt" && nodeName) {
    recordRunPrompt(nodeName, {
      role: entry.role,
      content: entry.content,
      mode: entry.mode,
      model: entry.model,
      tool: entry.name || entry.tool,
      name: entry.name,
    });
    if (selectedFlowNode === nodeName) renderFlowInspector(nodeName);
  }
  const title =
    kind === "prompt"
      ? `PROMPT · ${entry.role || ""}`
      : kind === "tool"
        ? `TOOL · ${entry.name || ""}`
        : kind === "llm_response"
          ? `LLM RESPONSE · ${entry.mode || ""}`
          : kind === "reasoning"
            ? "LLM REASONING"
            : kind === "retry_trigger"
              ? `RETRY · ${entry.metric || ""}`
              : kind === "candidates"
                ? "CANDIDATES"
                : kind.toUpperCase();

  let body = "";
  if (kind === "prompt" || kind === "llm_response" || kind === "reasoning") {
    body = `<pre>${escapeHtml(entry.content || "")}</pre>`;
  } else if (kind === "tool") {
    body = `<div class="meta">${escapeHtml(entry.purpose || "")}</div><pre>${pretty(entry.output ?? entry.summary)}</pre>`;
  } else if (kind === "retry_trigger") {
    body = `<div>${escapeHtml(entry.reason || "")}</div>
      <div class="meta">clear when: ${escapeHtml(JSON.stringify(entry.threshold_to_clear))}</div>
      <div>${escapeHtml(entry.clearance || "")}</div>`;
  } else if (kind === "candidates") {
    body = renderCandidates(entry.candidates, entry.dropped);
  } else {
    body = `<pre>${pretty(entry)}</pre>`;
  }

  div.innerHTML = `
    <span class="kind kind-${kind}">${escapeHtml(kind)}</span>
    <strong>${escapeHtml(title)}</strong>
    <div class="meta">node=${escapeHtml(meta.node || entry.node || "")} · seq=${meta.seq ?? "—"} · iter=${meta.iteration ?? "—"}</div>
    ${body}
  `;
  transcriptEl.appendChild(div);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

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
      /* ignore malformed */
    }
  }
  return { events, rest };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetRunUI();
  runBtn.disabled = true;
  if (showTranscript.checked) setTranscriptVisible(true);

  const dish = selectedDish();
  const mode = currentMode();
  const body = {
    mode,
    user_request: userRequestEl?.value?.trim() || "",
    taste_text: dish?.title || "",
    title: dish?.title || "",
    canonical_id: getSelectedCanonicalId(),
    start_metric: currentStartMetric(),
    protein_min: fractionFromPercentInput("protein_min"),
    protein_max: fractionFromPercentInput("protein_max"),
    carb_min: fractionFromPercentInput("carb_min"),
    carb_max: fractionFromPercentInput("carb_max"),
    fat_min: fractionFromPercentInput("fat_min"),
    fat_max: fractionFromPercentInput("fat_max"),
    F_accept: Number(document.getElementById("F_accept").value),
    F_max: Number(document.getElementById("F_max").value),
    max_iterations: Number(document.getElementById("max_iterations").value),
  };
  if (lastRunBundle) {
    lastRunBundle.mode = mode;
    lastRunBundle.user_request = body.user_request || body.taste_text || "";
    lastRunBundle.title = body.title || "";
  }
  const boxErrors = validateMacroBox(body);
  if (boxErrors.length) {
    const msg = `Infeasible macro targets: ${boxErrors.join("; ")}`;
    flowStatus.textContent = msg;
    setRunStatus(msg, { error: true });
    runBtn.disabled = false;
    return;
  }
  if (mode === "neighborhood" && !body.canonical_id) {
    const msg = "Search and select a canonical recipe.";
    flowStatus.textContent = msg;
    setRunStatus(msg, { error: true });
    runBtn.disabled = false;
    return;
  }
  if (mode === "creative" && !body.user_request) {
    const msg = "Enter a creative user request above.";
    flowStatus.textContent = msg;
    setRunStatus(msg, { error: true });
    userRequestEl?.focus();
    runBtn.disabled = false;
    return;
  }
  setRunStatus(
    mode === "creative"
      ? "Starting creative agent…"
      : "Building neighborhood and starting agent…"
  );

  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
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
    flowStatus.textContent = lastNode ? `Finished (last: ${lastNode})` : "Finished";
    setRunStatus(lastNode ? `Finished (last: ${lastNode})` : "Finished");
  } catch (err) {
    const msg = `Error: ${err.message || err}`;
    flowStatus.textContent = msg;
    setRunStatus(msg, { error: true });
    const card = document.createElement("div");
    card.className = "step-card";
    card.textContent = String(err.message || err);
    timeline.prepend(card);
  } finally {
    runBtn.disabled = false;
  }
});

function appendLoadEvent(data) {
  setRunStatus(data.message || "Loading…");
  const card = document.createElement("div");
  card.className = "step-card active-step";
  timeline.querySelectorAll(".step-card").forEach((c) => c.classList.remove("active-step"));
  const head = document.createElement("div");
  head.className = "step-head";
  head.innerHTML = `<span class="step-node">load</span><span class="step-meta">${escapeHtml(data.phase || "")}</span>`;
  card.appendChild(head);
  const msg = document.createElement("p");
  msg.className = "rationale";
  msg.textContent = data.message || "";
  card.appendChild(msg);
  if (data.chosen_recipe) {
    const wrap = document.createElement("div");
    wrap.innerHTML = renderChosenRecipe(data.chosen_recipe);
    card.appendChild(wrap);
  }
  timeline.prepend(card);
  flowStatus.textContent = data.message || "Loading…";
  appendTranscriptEntry(
    { kind: "tool", name: `load:${data.phase || "phase"}`, purpose: data.message, output: data },
    { node: "load", seq: "—", iteration: "—" }
  );
}

function handleEvent(eventType, data) {
  if (eventType === "graph_meta" || data.type === "graph_meta") {
    flowEdges = (data.edges || []).map((e) => (Array.isArray(e) ? e : [e.from, e.to]));
    renderFlow(data.nodes || Object.keys(NODE_LAYOUT));
    return;
  }
  if (eventType === "load" || data.type === "load") {
    appendLoadEvent(data);
    return;
  }
  if (eventType === "transcript" || data.type === "transcript") {
    appendTranscriptEntry(data.entry || data, {
      seq: data.seq,
      node: data.node,
      iteration: data.iteration,
    });
    return;
  }
  if (eventType === "step" || data.type === "step") {
    highlightNode(data.node, data.duration_ms);
    appendStep(data);
    const payload = data.payload || {};
    if (payload.live_scores) {
      if (Array.isArray(payload.score_history)) scoreHistory = payload.score_history;
      else if (Array.isArray(payload.live_scores.score_history))
        scoreHistory = payload.live_scores.score_history;
      renderDisplayScores(
        { ...payload.live_scores, score_history: scoreHistory },
        {
          titleHint:
            data.iteration != null
              ? `Best sample after iteration ${data.iteration} (${data.node}).`
              : `Updated after ${data.node}.`,
        }
      );
    } else if (Array.isArray(payload.score_history) && payload.score_history.length) {
      scoreHistory = payload.score_history;
      renderLossChart();
    }
    if (lastRunBundle) {
      lastRunBundle.steps.push({
        seq: data.seq,
        node: data.node,
        iteration: data.iteration,
        fidelity_band: data.fidelity_band,
        decision: payload.decision,
        tools: (payload.tools_used || []).map((t) => ({
          name: t.name,
          purpose: t.purpose,
          mode: t.mode,
          model: t.model,
          output_summary: t.output_summary,
        })),
      });
      for (const tool of payload.tools_used || []) {
        if (tool.llm_trace) {
          lastRunBundle.llm_calls.push({
            seq: data.seq,
            node: data.node,
            tool: tool.name,
            mode: tool.llm_trace.mode,
            model: tool.llm_trace.model,
            messages: tool.llm_trace.messages,
            raw_response: tool.llm_trace.raw_response,
            parsed: tool.llm_trace.parsed,
            rationale: tool.llm_trace.rationale,
            usage: tool.llm_trace.usage,
          });
        }
      }
      if (payload.llm_trace && !(payload.tools_used || []).some((t) => t.llm_trace)) {
        lastRunBundle.llm_calls.push({
          seq: data.seq,
          node: data.node,
          mode: payload.llm_trace.mode,
          model: payload.llm_trace.model,
          messages: payload.llm_trace.messages,
          raw_response: payload.llm_trace.raw_response,
          usage: payload.llm_trace.usage,
        });
      }
    }
    return;
  }
  if (eventType === "done" || data.type === "done") {
    if (data.final) showFinalResult(data.final);
    return;
  }
  if (eventType === "result" || data.type === "result") {
    showFinalResult(data.final || data);
    return;
  }
  if (eventType === "error" || data.type === "error") {
    const msg = `Error: ${data.error}`;
    flowStatus.textContent = msg;
    setRunStatus(msg, { error: true });
    const card = document.createElement("div");
    card.className = "step-card";
    card.textContent = data.error;
    timeline.prepend(card);
  }
}

initHealth();
updateModeUI();
syncWorkspaceHeight();
renderDisplayScores(emptyScores());
document.getElementById("loss-chart-metric")?.addEventListener("change", renderLossChart);
// Form height can settle after fonts/canonical meta load.
window.addEventListener("load", syncWorkspaceHeight);
