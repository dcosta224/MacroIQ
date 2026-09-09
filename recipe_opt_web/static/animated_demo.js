(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  // Headless recorders often spoof prefers-reduced-motion; ?record=1 keeps real timing.
  const forceFullMotion = params.has("record") || params.get("fullmotion") === "1";
  const reduceMotion =
    !forceFullMotion && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.__motionFlags = { forceFullMotion, reduceMotion, search: window.location.search };

  const NODE_ORDER = [
    "query",
    "neighborhood",
    "draft",
    "opt1",
    "propose",
    "decide",
    "apply",
    "opt2",
    "compare",
    "result",
  ];

  const CONTENT = {
    query: "High-Protein Carbonara",
    neighborhood: [
      "Carbonara Tradizionali",
      "Penne Carbonara",
      "Chorizo Carbonara",
      "Light Asparagus Carbonara",
      "…",
    ],
    draftTitle: "Higher-Protein Carbonara",
    draftIngs: ["120 g pasta", "36 g egg yolk", "40 g parmesan", "50 g bacon", "…"],
    proposeEdits: [
      "SWAP whole wheat spaghetti",
      "INSERT seared chicken breast",
      "SWAP lean pancetta",
    ],
    proposeSearch: [
      "CHICKEN BREAST · Protein 80% · Co-occurrence 0.12",
      "LEAN PANCETTA · Protein 22% · Co-occurrence 0.21",
      "WHOLE WHEAT PASTA · Protein 18% · Co-occurrence 0.16",
    ],
    decideChoice: "CHICKEN BREAST",
    decideReason: "Chicken breast can boost protein without overwhelming pasta and pork.",
    expand: [
      "chicken penne pasta bake",
      "chicken breast fettuccine alfredo",
      "creamy pasta with chicken",
    ],
    applyIngs: [
      "120 g pasta",
      "36 g egg yolk",
      "40 g parmesan",
      "50 g bacon",
      "100 g CHICKEN BREAST",
    ],
    // Final mass shares (~338 g total). All markers sit inside their IQR,
    // slightly off median: spaghetti below; chicken above; others barely above.
    resultIngs: [
      {
        label: "118 g spaghetti",
        share: 0.349,
        iqr: { min: 0.16, q1: 0.27, median: 0.37, q3: 0.46, max: 0.58 },
      },
      {
        label: "2 egg yolks",
        share: 0.101,
        iqr: { min: 0.035, q1: 0.072, median: 0.095, q3: 0.125, max: 0.19 },
      },
      {
        label: "38 g parmesan",
        share: 0.112,
        iqr: { min: 0.04, q1: 0.08, median: 0.105, q3: 0.145, max: 0.24 },
      },
      {
        label: "3 slices bacon",
        share: 0.142,
        iqr: { min: 0.055, q1: 0.095, median: 0.132, q3: 0.185, max: 0.3 },
      },
      {
        label: "100 g chicken breast",
        share: 0.296,
        iqr: { min: 0.1, q1: 0.2, median: 0.265, q3: 0.34, max: 0.5 },
      },
    ],
    resultExplain:
      "Lean chicken raises protein into the nutrient target range while pasta, egg yolks, and hard cheese keep classic carbonara identity. Ingredient amounts stay typical for related pasta dishes.",
  };

  const OPT1 = {
    iters: 20,
    animMs: 1750,
    holdMs: 650,
    startLoss: 0.41,
    endLoss: 0.17,
    startMacros: { p: 29.5, c: 18.0, f: 52.4 },
    endMacros: { p: 27.2, c: 31.5, f: 41.3 },
    endStatus: "OUTSIDE_TARGET",
  };

  const OPT2 = {
    iters: 15,
    animMs: 1300,
    holdMs: 400,
    startLoss: 0.17,
    endLoss: 0.04,
    startMacros: { p: 27.2, c: 31.5, f: 41.3 },
    endMacros: { p: 28.1, c: 39.8, f: 32.1 },
    endStatus: "IN_BOX",
  };

  let runToken = 0;
  let panX = 0;
  let panY = 0;
  /** Precomputed vertical camera lock per node (final expanded layout). */
  let lockedPanY = {};
  /** Per-node content [minY, maxY] in viewport coords at lock time (debug / QA). */
  let lockedContentBounds = {};
  let currentNodeId = "query";
  /** Keep at least this fraction of the stage as top/bottom whitespace when centering. */
  const VERTICAL_MARGIN_FRAC = 0.05;
  const VERTICAL_MARGIN_MIN_PX = 20;
  /** Stages that show a database satellite + connector edge. */
  const DB_FOR_NODE = {
    neighborhood: "recipe",
    propose: "ingredient",
    decide: "recipe",
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function wait(ms) {
    window.__waitCalls = (window.__waitCalls || 0) + 1;
    window.__waitMs = (window.__waitMs || 0) + (Number(ms) || 0);
    if (reduceMotion) return Promise.resolve();
    return new Promise((r) => setTimeout(r, ms));
  }

  function setCaption(text) {
    const el = $("#stage_caption");
    if (el) el.textContent = text;
  }

  function clearList(el) {
    if (el) el.innerHTML = "";
  }

  function clearText(el) {
    if (el) el.textContent = "";
  }

  function resetReadout(root) {
    if (!root) return;
    root.classList.remove("is-converged");
    $$("[data-field]", root).forEach((el) => {
      el.textContent = "—";
      el.classList.remove("is-outside-target");
    });
    $$(".check", root).forEach((el) => {
      el.hidden = true;
    });
  }

  function layoutSliders() {
    const minVisualSpan = 100 / 6; // at least 1/6 of the range bar
    $$(".slider-row").forEach((row) => {
      const lo = Number(row.dataset.lo);
      const hi = Number(row.dataset.hi);
      const dual = $(".range-dual", row);
      if (!dual) return;
      const mid = (lo + hi) / 2;
      const span = Math.max(hi - lo, minVisualSpan);
      let visLo = mid - span / 2;
      let visHi = mid + span / 2;
      if (visLo < 0) {
        visHi = Math.min(100, visHi - visLo);
        visLo = 0;
      }
      if (visHi > 100) {
        visLo = Math.max(0, visLo - (visHi - 100));
        visHi = 100;
      }
      dual.style.setProperty("--lo-pct", `${visLo}%`);
      dual.style.setProperty("--hi-pct", `${visHi}%`);
      dual.style.setProperty("--fill-left", `${visLo}%`);
      dual.style.setProperty("--fill-width", `${visHi - visLo}%`);
    });
  }

  function animateSlidersIn() {
    $$(".slider-row").forEach((row) => {
      const dual = $(".range-dual", row);
      if (!dual) return;
      dual.style.setProperty("--lo-pct", "45%");
      dual.style.setProperty("--hi-pct", "55%");
      dual.style.setProperty("--fill-left", "45%");
      dual.style.setProperty("--fill-width", "10%");
    });
    requestAnimationFrame(() => {
      requestAnimationFrame(layoutSliders);
    });
  }

  function applyPan() {
    const rail = $("#graph_rail");
    if (!rail) return;
    rail.style.setProperty("--pan-x", `${panX}px`);
    rail.style.setProperty("--pan-y", `${panY}px`);
  }

  function dbElForNode(nodeId) {
    const kind = DB_FOR_NODE[nodeId];
    if (!kind) return null;
    return kind === "ingredient" ? $("#db_ingredient") : $("#db_recipe");
  }

  /** Frame used for camera math (stage-cam if present, else viewport). */
  function cameraFrame() {
    return $("#stage_cam") || $("#stage_viewport");
  }

  function getStageZoom() {
    const cam = $("#stage_cam");
    if (!cam) return 1;
    const z = parseFloat(getComputedStyle(cam).getPropertyValue("--stage-zoom"));
    return Number.isFinite(z) && z > 0 ? z : 1;
  }

  /**
   * Active step content band in viewport coords.
   * When includeDb, expands to keep the DB card (+ padding) inside the band.
   */
  function contentYBounds(step = $(".graph-step.is-active"), { includeDb = false } = {}) {
    const circle = step && $(".node-circle", step);
    const body = step && $(".node-body", step);
    if (!circle || !body) return null;
    const cr = circle.getBoundingClientRect();
    const br = body.getBoundingClientRect();
    let minY = cr.top;
    let maxY = br.bottom;
    if (includeDb) {
      const db = dbElForNode(step.dataset.node);
      if (db) {
        const dr = db.getBoundingClientRect();
        minY = Math.min(minY, dr.top - 8);
        maxY = Math.max(maxY, dr.bottom + 8);
      }
    }
    return {
      minY,
      maxY,
      height: maxY - minY,
    };
  }

  /** Top inset needed so a DB card (fixed in the cam) stays fully visible. */
  function dbTopReserve(frameEl) {
    const db = $("#db_recipe") || $("#db_ingredient");
    if (!db || !frameEl) return 0;
    const fr = frameEl.getBoundingClientRect();
    const dr = db.getBoundingClientRect();
    // Distance from frame top through the DB card, plus a small gap under it.
    return Math.max(0, dr.bottom - fr.top + 10);
  }

  /**
   * Vertical pan so circle+card is centered in the available frame band.
   * DB stages keep extra top inset so the satellite + edge stay on-screen.
   */
  function measureVerticalPan(step = $(".graph-step.is-active")) {
    const frame = cameraFrame();
    const bounds = contentYBounds(step, { includeDb: false });
    if (!frame || !bounds) return panY;
    const fr = frame.getBoundingClientRect();
    const viewH = fr.height;
    const margin = Math.max(VERTICAL_MARGIN_MIN_PX, viewH * VERTICAL_MARGIN_FRAC);
    const needsDb = Boolean(DB_FOR_NODE[step?.dataset?.node]);
    const topMargin = needsDb ? Math.max(margin, dbTopReserve(frame)) : margin;
    const botMargin = margin;
    const { minY, height } = bounds;

    const availTop = fr.top + topMargin;
    const availBot = fr.bottom - botMargin;
    const availH = availBot - availTop;

    let targetMinY;
    if (height >= availH) {
      // Tall card: prefer keeping the bottom (LLM explain on result) in frame.
      if (step?.dataset?.node === "result") {
        targetMinY = availBot - height;
      } else {
        // Other tall cards: pin under the top reserve (keeps DB clear when present).
        targetMinY = availTop;
      }
    } else {
      targetMinY = availTop + (availH - height) / 2;
    }

    // getBoundingClientRect is in screen px; --pan-y is in pre-zoom cam CSS px.
    const zoom = getStageZoom();
    return panY + (targetMinY - minY) / zoom;
  }

  function fillList(el, lines, { markLastAdded = false } = {}) {
    if (!el) return;
    el.innerHTML = "";
    lines.forEach((line, i) => {
      const li = document.createElement("li");
      li.textContent = typeof line === "string" ? line : line.label || "";
      if (markLastAdded && i === lines.length - 1) {
        const text = typeof line === "string" ? line : line.label || "";
        if (!text.startsWith("…")) li.classList.add("added");
      }
      li.classList.add("is-in");
      el.appendChild(li);
    });
  }

  /** MacroIQ-style share boxplot (whiskers / IQR / median / recipe marker). */
  function shareBoxplotSvg(iqr, recipeShare, toneClass = "tone-good", { compact = false } = {}) {
    if (!iqr || iqr.min == null || iqr.max == null) return "";
    const min = Number(iqr.min);
    const max = Number(iqr.max);
    const q1 = Number(iqr.q1);
    const med = Number(iqr.median);
    const q3 = Number(iqr.q3);
    const w = compact ? 108 : 120;
    const h = compact ? 22 : 28;
    const pad = compact ? 5 : 6;
    const span = max - min || 1e-9;
    const x = (v) => pad + ((Number(v) - min) / span) * (w - 2 * pad);
    const y = h / 2;
    const boxH = compact ? 10 : 12;
    const medHalf = compact ? 6 : 7;
    const markerR = compact ? 3.2 : 4;
    const clamped =
      recipeShare == null || Number.isNaN(Number(recipeShare))
        ? null
        : Math.min(max, Math.max(min, Number(recipeShare)));
    const marker =
      clamped == null
        ? ""
        : `<circle class="bw-marker ${toneClass}" cx="${x(clamped)}" cy="${y}" r="${markerR}" />`;
    return `<svg class="share-boxplot" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
      <line class="bw-whisker" x1="${x(min)}" y1="${y}" x2="${x(max)}" y2="${y}" />
      <rect class="bw-box" x="${x(q1)}" y="${y - boxH / 2}" width="${Math.max(1, x(q3) - x(q1))}" height="${boxH}" rx="2" />
      <line class="bw-median" x1="${x(med)}" y1="${y - medHalf}" x2="${x(med)}" y2="${y + medHalf}" />
      ${marker}
    </svg>`;
  }

  function resultIngLi(ing) {
    const li = document.createElement("li");
    li.className = "ing-with-boxplot";
    if (ing.added) li.classList.add("added");
    li.innerHTML = `<div class="ing-name-line"><strong class="tone-good">${ing.label}</strong></div>
      <div class="ing-boxplot">${shareBoxplotSvg(ing.iqr, ing.share, "tone-good", { compact: true })}</div>`;
    return li;
  }

  function fillResultIngs(el, ings) {
    if (!el) return;
    el.innerHTML = "";
    el.classList.add("has-boxplots");
    ings.forEach((ing) => {
      const li = resultIngLi(ing);
      li.classList.add("is-in");
      el.appendChild(li);
    });
  }

  async function streamResultIngs(el, ings, durationMs, token) {
    if (!el) return;
    el.innerHTML = "";
    el.classList.add("has-boxplots");
    if (reduceMotion) {
      fillResultIngs(el, ings);
      return;
    }
    const per = durationMs / Math.max(ings.length, 1);
    for (let i = 0; i < ings.length; i++) {
      if (token !== runToken) return;
      const li = resultIngLi(ings[i]);
      el.appendChild(li);
      void li.offsetWidth;
      li.classList.add("is-in");
      await wait(per);
    }
  }

  function setStatusText(statusEl, text) {
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.classList.toggle("is-outside-target", text === "OUTSIDE_TARGET");
  }

  function fillOptReadout(root, cfg) {
    if (!root || !cfg) return;
    const lossEl = $('[data-field="loss"]', root);
    const macrosEl = $('[data-field="macros"]', root);
    const statusEl = $('[data-field="status"]', root);
    if (lossEl) lossEl.textContent = cfg.endLoss.toFixed(2);
    if (macrosEl) {
      macrosEl.textContent = `P ${cfg.endMacros.p.toFixed(1)}% · C ${cfg.endMacros.c.toFixed(1)}% · F ${cfg.endMacros.f.toFixed(1)}%`;
    }
    setStatusText(statusEl, cfg.endStatus);
    root.classList.add("is-converged");
    $$(".check", root).forEach((c) => {
      c.hidden = false;
    });
  }

  function populateFinalContent() {
    const qt = $("#query_text");
    if (qt) qt.textContent = CONTENT.query;
    fillList($("#neighborhood_list"), CONTENT.neighborhood);
    const dt = $("#draft_title");
    if (dt) dt.textContent = CONTENT.draftTitle;
    fillList($("#draft_ings"), CONTENT.draftIngs);
    fillList($("#propose_edits"), CONTENT.proposeEdits);
    fillList($("#propose_search"), CONTENT.proposeSearch);
    const dc = $("#decide_choice");
    if (dc) dc.textContent = CONTENT.decideChoice;
    const dr = $("#decide_reason");
    if (dr) dr.textContent = CONTENT.decideReason;
    fillList($("#expand_list"), CONTENT.expand);
    fillList($("#apply_ings"), CONTENT.applyIngs, { markLastAdded: true });
    fillResultIngs($("#result_ings"), CONTENT.resultIngs);
    const re = $("#result_explain");
    if (re) re.textContent = CONTENT.resultExplain;
    fillOptReadout($("#opt1_readout"), OPT1);
    fillOptReadout($("#opt2_readout"), OPT2);
    layoutSliders();
  }

  function clearStreamingContent() {
    clearText($("#query_text"));
    clearList($("#neighborhood_list"));
    clearText($("#draft_title"));
    clearList($("#draft_ings"));
    clearList($("#propose_edits"));
    clearList($("#propose_search"));
    clearText($("#decide_choice"));
    clearText($("#decide_reason"));
    clearList($("#expand_list"));
    clearList($("#apply_ings"));
    clearList($("#result_ings"));
    clearText($("#result_explain"));
    resetReadout($("#opt1_readout"));
    resetReadout($("#opt2_readout"));
  }

  function lockBodyHeights() {
    $$(".graph-step .node-body").forEach((body) => {
      body.style.minHeight = `${body.offsetHeight}px`;
    });
  }

  function clearBodyHeightLocks() {
    $$(".graph-step .node-body").forEach((body) => {
      body.style.minHeight = "";
      body.style.maxHeight = "";
    });
  }

  /**
   * Fill every node to its final expanded size, lock card heights + camera Y,
   * draw edges against that stable geometry, then clear streaming text.
   */
  function computeLockedCamera() {
    const rail = $("#graph_rail");
    if (!rail) return;
    rail.classList.add("is-measuring");
    clearBodyHeightLocks();
    // Measure true full content height (ignore stage max-height clipping).
    $$(".graph-step .node-body").forEach((body) => {
      body.style.maxHeight = "none";
    });
    populateFinalContent();

    const prevX = panX;
    const prevId = currentNodeId;
    lockedPanY = {};
    panY = 0;
    applyPan();

    lockedContentBounds = {};
    NODE_ORDER.forEach((id) => {
      setActiveClasses(id);
      panX = getPanForNode(id).x;
      panY = 0;
      applyPan();
      const step = $(`.graph-step[data-node="${id}"]`);
      void (step && step.offsetHeight);
      // Show the DB for stages that use it so top-reserve / bounds are accurate.
      const dbKind = DB_FOR_NODE[id];
      if (dbKind) showDb(dbKind);
      else hideDbs();
      const before = contentYBounds(step, { includeDb: Boolean(dbKind) });
      lockedPanY[id] = measureVerticalPan(step);
      panY = lockedPanY[id];
      applyPan();
      const after = contentYBounds(step, { includeDb: Boolean(dbKind) });
      const frame = cameraFrame()?.getBoundingClientRect();
      const db = dbElForNode(id);
      const dbRect = db?.getBoundingClientRect();
      lockedContentBounds[id] = {
        height: before?.height ?? 0,
        minY: after?.minY ?? 0,
        maxY: after?.maxY ?? 0,
        panY: lockedPanY[id],
        dbInFrame: !dbRect || !frame
          ? true
          : dbRect.top >= frame.top - 1 &&
            dbRect.bottom <= frame.bottom + 1 &&
            dbRect.left >= frame.left - 1 &&
            dbRect.right <= frame.right + 1,
      };
      panY = 0;
      applyPan();
    });
    hideDbs();
    window.__lockedContentBounds = lockedContentBounds;

    lockBodyHeights();
    drawRailEdges();
    clearStreamingContent();
    // Restore default max-height except result (CSS keeps that unclipped).
    $$(".graph-step:not(.node-result) .node-body").forEach((body) => {
      body.style.maxHeight = "";
    });

    currentNodeId = prevId;
    setActiveClasses(prevId);
    panX = prevX;
    panY = lockedPanY[prevId] ?? 0;
    applyPan();
    rail.classList.remove("is-measuring");
  }

  function lockedYFor(nodeId) {
    return lockedPanY[nodeId] ?? 0;
  }

  function circleLayoutInRail(step) {
    const circle = $(".node-circle", step);
    if (!circle) return null;
    const r = circle.offsetWidth / 2;
    return {
      cx: step.offsetLeft + circle.offsetLeft + r,
      cy: step.offsetTop + circle.offsetTop + r,
      r,
    };
  }

  /** Horizontal pan so this node's circle is centered in the viewport. */
  function getPanForNode(nodeId) {
    const steps = $$(".graph-step");
    const first = steps[0];
    const target = steps.find((n) => n.dataset.node === nodeId) || first;
    if (!first || !target) return { x: 0 };
    const a = circleLayoutInRail(first);
    const b = circleLayoutInRail(target);
    if (!a || !b) {
      return { x: -(target.offsetLeft - first.offsetLeft) };
    }
    return { x: -(b.cx - a.cx) };
  }

  function setActiveClasses(nodeId) {
    const idx = NODE_ORDER.indexOf(nodeId);
    $$(".graph-step").forEach((n) => {
      const id = n.dataset.node;
      n.classList.toggle("is-active", id === nodeId);
      n.classList.toggle("is-peek", id === NODE_ORDER[idx + 1]);
    });
    const prev = NODE_ORDER[idx - 1];
    highlightRailEdge(prev, nodeId, "active");
  }

  function drawRailEdges() {
    const rail = $("#graph_rail");
    const svg = $("#rail_edges");
    if (!rail || !svg) return;

    const steps = $$(".graph-step");
    const width = Math.max(rail.scrollWidth, rail.offsetWidth);
    const height = Math.max(rail.scrollHeight, rail.offsetHeight);
    svg.setAttribute("width", String(width));
    svg.setAttribute("height", String(height));
    svg.style.width = `${width}px`;
    svg.style.height = `${height}px`;
    svg.innerHTML = "";

    for (let i = 0; i < steps.length - 1; i++) {
      const a = circleLayoutInRail(steps[i]);
      const b = circleLayoutInRail(steps[i + 1]);
      if (!a || !b) continue;
      const dx = b.cx - a.cx;
      const dy = b.cy - a.cy;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len;
      const uy = dy / len;
      const x1 = a.cx + ux * a.r;
      const y1 = a.cy + uy * a.r;
      const x2 = b.cx - ux * b.r;
      const y2 = b.cy - uy * b.r;
      // slight bow so the path reads as a graph edge, not a ruler
      const mx = (x1 + x2) / 2 - uy * 10;
      const my = (y1 + y2) / 2 + ux * 10;

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${x1} ${y1} Q ${mx} ${my} ${x2} ${y2}`);
      path.classList.add("rail-link");
      path.dataset.from = steps[i].dataset.node;
      path.dataset.to = steps[i + 1].dataset.node;
      svg.appendChild(path);

      // arrowhead at end
      const angle = Math.atan2(y2 - my, x2 - mx);
      const ah = 9;
      const p1x = x2 - Math.cos(angle) * ah + Math.cos(angle + Math.PI / 2) * 4.5;
      const p1y = y2 - Math.sin(angle) * ah + Math.sin(angle + Math.PI / 2) * 4.5;
      const p2x = x2 - Math.cos(angle) * ah + Math.cos(angle - Math.PI / 2) * 4.5;
      const p2y = y2 - Math.sin(angle) * ah + Math.sin(angle - Math.PI / 2) * 4.5;
      const arrow = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      arrow.setAttribute("points", `${x2},${y2} ${p1x},${p1y} ${p2x},${p2y}`);
      arrow.classList.add("rail-arrow");
      arrow.dataset.from = steps[i].dataset.node;
      arrow.dataset.to = steps[i + 1].dataset.node;
      svg.appendChild(arrow);
    }
  }

  function highlightRailEdge(fromId, toId, mode) {
    const svg = $("#rail_edges");
    if (!svg) return;
    $$(".rail-link, .rail-arrow", svg).forEach((el) => {
      el.classList.remove("is-active-edge", "is-traversing");
      if (fromId && toId && el.dataset.from === fromId && el.dataset.to === toId) {
        el.classList.add(mode === "traverse" ? "is-traversing" : "is-active-edge");
      }
    });
  }

  function panTo(nodeId) {
    const target = getPanForNode(nodeId);
    panX = target.x;
    panY = lockedYFor(nodeId);
    setActiveClasses(nodeId);
    currentNodeId = nodeId;
    highlightRailEdge(NODE_ORDER[NODE_ORDER.indexOf(nodeId) - 1], nodeId, "active");
    applyPan();
  }

  /** Animate camera along the solid edge from the current node to nodeId. */
  async function followEdgeTo(nodeId, durationMs, token) {
    const idx = NODE_ORDER.indexOf(nodeId);
    if (idx < 0) return;
    const fromId = currentNodeId;
    const startX = panX;
    const startY = panY;
    const endX = getPanForNode(nodeId).x;
    const endY = lockedYFor(nodeId);

    $$(".graph-step").forEach((n) => {
      const id = n.dataset.node;
      n.classList.toggle("is-peek", id === nodeId);
      n.classList.toggle("is-active", id === fromId || id === nodeId);
    });
    highlightRailEdge(fromId, nodeId, "traverse");

    if (reduceMotion) {
      panX = endX;
      panY = endY;
      applyPan();
      currentNodeId = nodeId;
      setActiveClasses(nodeId);
      return;
    }

    const frames = Math.max(8, Math.round(durationMs / 16));
    for (let i = 1; i <= frames; i++) {
      if (token !== runToken) return;
      const t = easeInOutCubic(i / frames);
      panX = lerp(startX, endX, t);
      panY = lerp(startY, endY, t);
      applyPan();
      await wait(durationMs / frames);
    }

    panX = endX;
    panY = endY;
    applyPan();
    currentNodeId = nodeId;
    setActiveClasses(nodeId);
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function hideDbs() {
    ["#db_recipe", "#db_ingredient"].forEach((sel) => {
      const el = $(sel);
      if (!el) return;
      el.classList.remove("is-visible", "is-pulse");
    });
    clearDbEdges();
  }

  function showDb(which) {
    hideDbs();
    const el = which === "ingredient" ? $("#db_ingredient") : $("#db_recipe");
    if (el) el.classList.add("is-visible");
  }

  function pulseDb(which, on) {
    const el = which === "ingredient" ? $("#db_ingredient") : $("#db_recipe");
    if (!el) return;
    el.classList.toggle("is-pulse", on);
  }

  function clearDbEdges() {
    const svg = $("#edge_layer");
    if (!svg) return;
    $$("path.is-on", svg).forEach((p) => p.remove());
  }

  function drawEdgeToDb(dbWhich) {
    clearDbEdges();
    const frame = cameraFrame();
    const active = $(".graph-step.is-active");
    const circle = active && $(".node-circle", active);
    const db = dbWhich === "ingredient" ? $("#db_ingredient") : $("#db_recipe");
    const svg = $("#edge_layer");
    if (!frame || !circle || !db || !svg) return;

    const fr = frame.getBoundingClientRect();
    const cr = circle.getBoundingClientRect();
    const dr = db.getBoundingClientRect();
    const zoom = getStageZoom();
    // Convert screen deltas into stage-cam local SVG coordinates.
    const x1 = (cr.left + cr.width / 2 - fr.left) / zoom;
    const y1 = (cr.top + cr.height / 2 - fr.top) / zoom;
    const x2 = (dr.left - fr.left) / zoom + 8;
    const y2 = (dr.top + dr.height / 2 - fr.top) / zoom;
    const cx = (x1 + x2) / 2;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${cx} ${y1}, ${cx} ${y2}, ${x2} ${y2}`);
    path.classList.add("is-on");
    svg.appendChild(path);
  }

  async function typeText(el, text, durationMs, token) {
    if (!el) return;
    if (reduceMotion) {
      el.textContent = text;
      return;
    }
    el.textContent = "";
    const n = Math.max(text.length, 1);
    const step = Math.max(12, durationMs / n);
    for (let i = 1; i <= n; i++) {
      if (token !== runToken) return;
      el.textContent = text.slice(0, i);
      await wait(step);
    }
  }

  async function streamLines(el, lines, durationMs, token, { markLastAdded = false } = {}) {
    if (!el) return;
    el.innerHTML = "";
    if (reduceMotion) {
      fillList(el, lines, { markLastAdded });
      return;
    }
    const per = durationMs / Math.max(lines.length, 1);
    for (let i = 0; i < lines.length; i++) {
      if (token !== runToken) return;
      const li = document.createElement("li");
      li.textContent = lines[i];
      if (markLastAdded && i === lines.length - 1 && !lines[i].startsWith("…")) {
        li.classList.add("added");
      }
      el.appendChild(li);
      // force reflow for transition
      void li.offsetWidth;
      li.classList.add("is-in");
      await wait(per);
    }
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  /** Front-loaded iteration schedule: early iters fast, later slow. */
  function iterationDelays(n, totalMs) {
    const weights = [];
    for (let i = 0; i < n; i++) {
      weights.push(0.35 + easeOutCubic(i / Math.max(n - 1, 1)) * 1.65);
    }
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map((w) => (w / sum) * totalMs);
  }

  /**
   * Steep convex loss bowl L(x) — minimization field with a stronger quadratic slope.
   * Screen Y grows downward: high loss on the upper rim, minimum at the bottom.
   * pointT: 0 at high loss on the rim → 1 at the minimum.
   */
  function drawBowl(ctx, w, h, pointT) {
    ctx.clearRect(0, 0, w, h);
    const cx = w * 0.5;
    const cy = h * 0.82; // min sits low on the canvas
    const rx = w * 0.36;
    const ry = h * 0.22;
    const slope = 2.85;

    function surfaceY(u, z) {
      // Larger |u| → higher on screen (smaller y); u=0 → bottom (min)
      return cy - slope * u * u * ry + z * 14 + Math.abs(u) * z * 6;
    }

    // Contour ellipses (level sets) — higher loss rings sit above the min
    for (let i = 7; i >= 0; i--) {
      const k = i / 7;
      const lift = k * 36; // up the screen
      ctx.beginPath();
      ctx.ellipse(cx, cy - lift - 6, rx * (0.4 + k * 0.6), ry * (0.28 + k * 0.55), 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(196, 92, 38, ${0.1 + k * 0.1})`;
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }

    // Steep parabolic mesh ridges
    ctx.strokeStyle = "rgba(196, 92, 38, 0.28)";
    ctx.lineWidth = 1.35;
    for (let a = -4; a <= 4; a++) {
      ctx.beginPath();
      const z = (a / 4) * 0.85;
      for (let u = -1; u <= 1; u += 0.04) {
        const x = cx + u * rx;
        const y = surfaceY(u, z);
        if (u <= -1 + 1e-6) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Axis hint: minimize L (arrow toward the bottom / min)
    ctx.fillStyle = "rgba(92, 107, 100, 0.85)";
    ctx.font = "600 11px 'IBM Plex Sans', sans-serif";
    ctx.fillText("minimize L(x)", 12, 16);
    ctx.strokeStyle = "rgba(92, 107, 100, 0.45)";
    ctx.beginPath();
    ctx.moveTo(w - 28, 18);
    ctx.lineTo(w - 28, h - 18);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(w - 28, h - 18);
    ctx.lineTo(w - 33, h - 28);
    ctx.lineTo(w - 23, h - 28);
    ctx.closePath();
    ctx.fillStyle = "rgba(92, 107, 100, 0.55)";
    ctx.fill();

    // Gradient-descent point rides the steep surface down to the min.
    // Start near the top of the parabolic ridges (u ≈ ±1), then descend to u=0.
    const uStart = -1.0;
    const u = lerp(uStart, 0.0, pointT);
    const px = cx + u * rx * 0.95;
    const py = surfaceY(u, 0);

    ctx.beginPath();
    for (let s = 0; s <= pointT; s += 0.03) {
      const uu = lerp(uStart, 0.0, s);
      const x = cx + uu * rx * 0.95;
      const y = surfaceY(uu, 0);
      if (s === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "rgba(196, 92, 38, 0.55)";
    ctx.lineWidth = 2.5;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#c45c26";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();

    if (pointT > 0.92) {
      ctx.fillStyle = "rgba(58, 125, 92, 0.95)";
      ctx.font = "700 12px 'IBM Plex Sans', sans-serif";
      ctx.fillText("min", px + 10, py + 4);
    }
  }

  async function runOptimizer(cfg, canvas, readout, token) {
    if (!canvas || !readout) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const delays = iterationDelays(cfg.iters, cfg.animMs);
    const lossEl = $('[data-field="loss"]', readout);
    const macrosEl = $('[data-field="macros"]', readout);
    const statusEl = $('[data-field="status"]', readout);
    const checks = $$(".check", readout);

    readout.classList.remove("is-converged");
    checks.forEach((c) => {
      c.hidden = true;
    });
    setStatusText(statusEl, "ITERATING…");

    if (reduceMotion) {
      drawBowl(ctx, w, h, 1);
      if (lossEl) lossEl.textContent = cfg.endLoss.toFixed(2);
      if (macrosEl) {
        macrosEl.textContent = `P ${cfg.endMacros.p.toFixed(1)}% · C ${cfg.endMacros.c.toFixed(1)}% · F ${cfg.endMacros.f.toFixed(1)}%`;
      }
      setStatusText(statusEl, cfg.endStatus);
      readout.classList.add("is-converged");
      checks.forEach((c) => {
        c.hidden = false;
      });
      await wait(cfg.holdMs);
      return;
    }

    // Show the dot on the upper rim before the first descent step.
    drawBowl(ctx, w, h, 0);
    if (lossEl) lossEl.textContent = cfg.startLoss.toFixed(2);
    if (macrosEl) {
      macrosEl.textContent = `P ${cfg.startMacros.p.toFixed(1)}% · C ${cfg.startMacros.c.toFixed(1)}% · F ${cfg.startMacros.f.toFixed(1)}%`;
    }
    await wait(Math.max(80, Math.floor(delays[0] * 0.5)));

    for (let i = 0; i < cfg.iters; i++) {
      if (token !== runToken) return;
      const t = (i + 1) / cfg.iters;
      const te = easeOutCubic(t);
      drawBowl(ctx, w, h, te);
      const loss = lerp(cfg.startLoss, cfg.endLoss, te);
      const p = lerp(cfg.startMacros.p, cfg.endMacros.p, te);
      const c = lerp(cfg.startMacros.c, cfg.endMacros.c, te);
      const f = lerp(cfg.startMacros.f, cfg.endMacros.f, te);
      if (lossEl) lossEl.textContent = loss.toFixed(2);
      if (macrosEl) macrosEl.textContent = `P ${p.toFixed(1)}% · C ${c.toFixed(1)}% · F ${f.toFixed(1)}%`;
      await wait(delays[i]);
    }

    if (token !== runToken) return;
    drawBowl(ctx, w, h, 1);
    if (lossEl) lossEl.textContent = cfg.endLoss.toFixed(2);
    if (macrosEl) {
      macrosEl.textContent = `P ${cfg.endMacros.p.toFixed(1)}% · C ${cfg.endMacros.c.toFixed(1)}% · F ${cfg.endMacros.f.toFixed(1)}%`;
    }
    setStatusText(statusEl, cfg.endStatus);
    readout.classList.add("is-converged");
    checks.forEach((c) => {
      c.hidden = false;
    });
    await wait(cfg.holdMs);
  }

  function resetAll() {
    clearStreamingContent();
    ["#opt1_canvas", "#opt2_canvas"].forEach((sel) => {
      const c = $(sel);
      if (c) {
        const ctx = c.getContext("2d");
        ctx.clearRect(0, 0, c.width, c.height);
      }
    });
    hideDbs();
    layoutSliders();
    if (!Object.keys(lockedPanY).length) {
      computeLockedCamera();
    } else {
      drawRailEdges();
    }
    currentNodeId = "query";
    panTo("query");
    setCaption("Ready");
  }

  async function runDemo() {
    const token = ++runToken;
    window.__demoDone = false;
    resetAll();
    setCaption("User query");

    // Timed for ~22s total: faster streams, shorter post-content holds.
    // Scene 0 — User query
    panTo("query");
    animateSlidersIn();
    await typeText($("#query_text"), CONTENT.query, reduceMotion ? 0 : 720, token);
    if (token !== runToken) return;
    await wait(780);

    // Scene 1 — Neighborhood (DB early)
    if (token !== runToken) return;
    setCaption("Query neighborhood · retrieving from Recipe Embeddings");
    showDb("recipe");
    pulseDb("recipe", true);
    await followEdgeTo("neighborhood", 400, token);
    if (token !== runToken) return;
    drawEdgeToDb("recipe");
    await streamLines($("#neighborhood_list"), CONTENT.neighborhood, 1200, token);
    if (token !== runToken) return;
    pulseDb("recipe", false);
    await wait(160);
    hideDbs();

    // Scene 2 — Draft
    if (token !== runToken) return;
    setCaption("LLM draft · building a candidate ingredient set");
    await followEdgeTo("draft", 380, token);
    if (token !== runToken) return;
    await typeText($("#draft_title"), CONTENT.draftTitle, 360, token);
    if (token !== runToken) return;
    await streamLines($("#draft_ings"), CONTENT.draftIngs, 950, token);
    if (token !== runToken) return;
    await wait(120);

    // Scene 3 — Opt1
    if (token !== runToken) return;
    setCaption("Diagnose · minimizing nutrient loss on a convex field");
    await followEdgeTo("opt1", 380, token);
    if (token !== runToken) return;
    await runOptimizer(OPT1, $("#opt1_canvas"), $("#opt1_readout"), token);
    if (token !== runToken) return;

    // Scene 4 — Propose (DB early, before edit streams)
    setCaption("LLM propose · edit ideas + ingredient search");
    showDb("ingredient");
    pulseDb("ingredient", true);
    await followEdgeTo("propose", 380, token);
    if (token !== runToken) return;
    drawEdgeToDb("ingredient");
    await streamLines($("#propose_edits"), CONTENT.proposeEdits, 580, token);
    if (token !== runToken) return;
    await streamLines($("#propose_search"), CONTENT.proposeSearch, 720, token);
    if (token !== runToken) return;
    pulseDb("ingredient", false);
    await wait(120);
    hideDbs();

    // Scene 5 — Decide + expand (DB early)
    if (token !== runToken) return;
    setCaption("LLM decide · then expand the recipe neighborhood");
    showDb("recipe");
    pulseDb("recipe", true);
    await followEdgeTo("decide", 380, token);
    if (token !== runToken) return;
    drawEdgeToDb("recipe");
    await typeText($("#decide_choice"), CONTENT.decideChoice, 300, token);
    if (token !== runToken) return;
    await typeText($("#decide_reason"), CONTENT.decideReason, 450, token);
    if (token !== runToken) return;
    await streamLines($("#expand_list"), CONTENT.expand, 580, token);
    if (token !== runToken) return;
    pulseDb("recipe", false);
    await wait(120);
    hideDbs();

    // Scene 6 — Apply
    if (token !== runToken) return;
    setCaption("Apply edits · new candidate with chicken breast");
    await followEdgeTo("apply", 360, token);
    if (token !== runToken) return;
    await streamLines($("#apply_ings"), CONTENT.applyIngs, 820, token, { markLastAdded: true });
    if (token !== runToken) return;
    await wait(120);

    // Scene 7 — Opt2
    if (token !== runToken) return;
    setCaption("Optimizer round 2 · continue descent from prior loss");
    await followEdgeTo("opt2", 360, token);
    if (token !== runToken) return;
    await runOptimizer(OPT2, $("#opt2_canvas"), $("#opt2_readout"), token);
    if (token !== runToken) return;

    // Scene 8 — Compare
    setCaption("LLM compare · scoring draft vs edited candidate");
    await followEdgeTo("compare", 360, token);
    if (token !== runToken) return;
    await wait(1750);

    // Scene 9 — Result
    if (token !== runToken) return;
    setCaption("Result · in-box macros with carbonara identity intact");
    await followEdgeTo("result", 360, token);
    if (token !== runToken) return;
    await streamResultIngs($("#result_ings"), CONTENT.resultIngs, 900, token);
    if (token !== runToken) return;
    await typeText($("#result_explain"), CONTENT.resultExplain, reduceMotion ? 0 : 1100, token);
    if (token !== runToken) return;
    setCaption("Done · press Replay to watch again");
    // Hold the final result frame for presentation / recording.
    await wait(5000);
    if (token !== runToken) return;
    window.__demoDone = true;
  }

  function onResize() {
    computeLockedCamera();
    const active = $(".graph-step.is-active");
    if (active) panTo(active.dataset.node);
  }

  function stripChromeForRecord() {
    if (!params.has("record")) return;
    document.documentElement.classList.add("record-mode");
    document.querySelectorAll(".demo-header, header.demo-header, .nav-hint").forEach((el) => {
      el.remove();
    });
    const shell = document.querySelector(".demo-shell");
    if (shell) {
      [...shell.children].forEach((c) => {
        if (!c.classList.contains("stage-wrap")) c.remove();
      });
    }
  }

  function boot() {
    stripChromeForRecord();
    layoutSliders();
    $("#replay_btn")?.addEventListener("click", () => {
      runDemo();
    });
    window.addEventListener("resize", onResize);
    // Used by headless recording scripts (stage-only capture).
    window.__runAnimatedDemo = runDemo;
    /**
     * Test / QA helper: lock camera with full final content, pan to result,
     * and report whether #result_explain is fully inside the stage frame.
     */
    window.__measureResultExplainFit = function measureResultExplainFit() {
      computeLockedCamera();
      populateFinalContent();
      const nodeId = "result";
      currentNodeId = nodeId;
      setActiveClasses(nodeId);
      const p = getPanForNode(nodeId);
      panX = p.x;
      panY = lockedYFor(nodeId);
      applyPan();
      const explain = $("#result_explain");
      const frameEl = cameraFrame();
      if (!explain || !frameEl) {
        return { ok: false, error: "missing explain or frame" };
      }
      const er = explain.getBoundingClientRect();
      const fr = frameEl.getBoundingClientRect();
      const slack = 1.5;
      const overflowTop = Math.max(0, fr.top - er.top);
      const overflowBottom = Math.max(0, er.bottom - fr.bottom);
      const ok = overflowTop <= slack && overflowBottom <= slack;
      return {
        ok,
        overflowTop,
        overflowBottom,
        explain: { top: er.top, bottom: er.bottom, height: er.height },
        frame: { top: fr.top, bottom: fr.bottom, height: fr.height },
        locked: lockedContentBounds.result || null,
        zoom: getStageZoom(),
        text: String(explain.textContent || ""),
      };
    };
    window.__recordReady = false;
    window.__demoDone = false;
    const skipAutoplay = params.has("record") || params.get("autoplay") === "0";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        stripChromeForRecord();
        computeLockedCamera();
        resetAll();
        window.__demoDone = false;
        stripChromeForRecord();
        window.__recordReady = true;
        if (!skipAutoplay) runDemo();
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
