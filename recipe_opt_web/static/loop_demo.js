(() => {
  const boxLine = document.getElementById("box_line");
  const rows = [...document.querySelectorAll(".slider-row[data-macro]")];

  function clamp(n, lo, hi) {
    return Math.min(hi, Math.max(lo, n));
  }

  function read(row) {
    return {
      min: Number(row.dataset.min),
      max: Number(row.dataset.max),
      lo: Number(row.dataset.lo),
      hi: Number(row.dataset.hi),
    };
  }

  function paint(row) {
    const { min, max, lo, hi } = read(row);
    const span = max - min || 1;
    const left = ((lo - min) / span) * 100;
    const right = ((hi - min) / span) * 100;
    row.style.setProperty("--fill-left", `${left}%`);
    row.style.setProperty("--fill-width", `${Math.max(0, right - left)}%`);
    row.style.setProperty("--lo-pct", `${left}%`);
    row.style.setProperty("--hi-pct", `${right}%`);
    const val = row.querySelector("[data-val]");
    if (val) val.textContent = `${lo}–${hi}%`;
  }

  function syncBox() {
    const parts = rows.map((row) => {
      const { lo, hi } = read(row);
      return `${row.dataset.macro} ${lo}–${hi}%`;
    });
    boxLine.textContent = `Target box: ${parts.join(" · ")}`;
  }

  function setThumb(row, which, value) {
    const { min, max, lo, hi } = read(row);
    let next = Math.round(clamp(value, min, max));
    if (which === "lo") {
      next = Math.min(next, hi);
      row.dataset.lo = String(next);
    } else {
      next = Math.max(next, lo);
      row.dataset.hi = String(next);
    }
    paint(row);
    syncBox();
  }

  function valueFromClientX(row, clientX) {
    const dual = row.querySelector(".range-dual");
    const rect = dual.getBoundingClientRect();
    const { min, max } = read(row);
    const pct = clamp((clientX - rect.left) / rect.width, 0, 1);
    return min + pct * (max - min);
  }

  function nearerThumb(row, clientX) {
    const { min, max, lo, hi } = read(row);
    const at = valueFromClientX(row, clientX);
    return Math.abs(at - lo) <= Math.abs(at - hi) ? "lo" : "hi";
  }

  rows.forEach((row) => {
    const dual = row.querySelector(".range-dual");
    const thumbLo = row.querySelector(".thumb-lo");
    const thumbHi = row.querySelector(".thumb-hi");
    let active = null;

    function startDrag(which, e) {
      active = which;
      const el = which === "lo" ? thumbLo : thumbHi;
      el.classList.add("is-dragging");
      el.setPointerCapture?.(e.pointerId);
      setThumb(row, which, valueFromClientX(row, e.clientX));
      e.preventDefault();
    }

    function moveDrag(e) {
      if (!active) return;
      setThumb(row, active, valueFromClientX(row, e.clientX));
    }

    function endDrag(e) {
      if (!active) return;
      const el = active === "lo" ? thumbLo : thumbHi;
      el.classList.remove("is-dragging");
      try {
        el.releasePointerCapture?.(e.pointerId);
      } catch (_) {
        /* ignore */
      }
      active = null;
    }

    thumbLo.addEventListener("pointerdown", (e) => startDrag("lo", e));
    thumbHi.addEventListener("pointerdown", (e) => startDrag("hi", e));

    // Click/drag on track picks the nearer thumb
    dual.addEventListener("pointerdown", (e) => {
      if (e.target.classList.contains("thumb")) return;
      startDrag(nearerThumb(row, e.clientX), e);
    });

    window.addEventListener("pointermove", moveDrag);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);

    // Keyboard: focus dual, arrows nudge nearer or both via Home/End feel
    dual.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 5 : 1;
      if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
        setThumb(row, "lo", read(row).lo - step);
        e.preventDefault();
      } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
        setThumb(row, "hi", read(row).hi + step);
        e.preventDefault();
      } else if (e.key === "[" ) {
        setThumb(row, "lo", read(row).lo - step);
        e.preventDefault();
      } else if (e.key === "]") {
        setThumb(row, "hi", read(row).hi + step);
        e.preventDefault();
      }
    });

    paint(row);
  });
  syncBox();

  document.querySelectorAll(".ghost").forEach((el) => {
    el.addEventListener("paste", (e) => {
      e.preventDefault();
      const text = (e.clipboardData || window.clipboardData).getData("text/plain");
      document.execCommand("insertText", false, text);
    });
  });
})();
