/* Results page: render the prediction stashed by the upload flow. */

(function () {
  const raw = sessionStorage.getItem("dermascan:lastResult");
  const imageSrc = sessionStorage.getItem("dermascan:lastImage");
  const empty = document.getElementById("empty");
  const content = document.getElementById("content");

  if (!raw) {
    empty.classList.remove("hidden");
    return;
  }
  content.classList.remove("hidden");

  const data = JSON.parse(raw);
  const P = data.predictions || {};

  // ── Image panels ─────────────────────────────────────────
  if (imageSrc) document.getElementById("imgOriginal").src = imageSrc;
  const heatSrc = data.gradcam_heatmap_base64
    ? `data:image/png;base64,${data.gradcam_heatmap_base64}`
    : null;
  const overlaySrc = data.gradcam_base64
    ? `data:image/png;base64,${data.gradcam_base64}`
    : null;
  if (heatSrc) document.getElementById("imgGradcam").src = heatSrc;
  if (overlaySrc) document.getElementById("imgOverlay").src = overlaySrc;

  // ── Timing ───────────────────────────────────────────────
  document.getElementById("timing").textContent =
    `processed in ${data.processing_time_ms} ms`;

  // ── Verdict cards ────────────────────────────────────────
  function fillCard(pred, ids, hideIfMissing) {
    const card = document.getElementById(ids.card);
    if (!pred) {
      if (hideIfMissing) card.classList.add("hidden");
      return;
    }
    document.getElementById(ids.badge).innerHTML = severityBadge(pred.label);
    document.getElementById(ids.conf).textContent = fmtPct(pred.confidence);
    document.getElementById(ids.raw).textContent = pred.raw_score.toFixed(3);
    const bar = document.getElementById(ids.bar);
    bar.style.width = `${pred.raw_score * 100}%`;
    bar.className = "bar-fill " + (pred.label === "high" ? "coral" : "teal");
  }

  fillCard(P.cnn, { card: "cnnCard", badge: "cnnBadge", conf: "cnnConf", raw: "cnnRaw", bar: "cnnBar" }, true);
  fillCard(P.mobilenet, { card: "mobCard", badge: "mobBadge", conf: "mobConf", raw: "mobRaw", bar: "mobBar" }, true);

  // ── Agreement indicator ──────────────────────────────────
  const agreeEl = document.getElementById("agreement");
  if (data.agreement === true) {
    agreeEl.innerHTML = `<span class="badge badge-agree">✓ Models agree</span>`;
  } else if (data.agreement === false) {
    agreeEl.innerHTML = `<span class="badge badge-disagree">✕ Models disagree — interpret with caution</span>`;
  } else {
    agreeEl.innerHTML = `<span class="badge">Single model — no agreement check</span>`;
  }

  // ── Severity breakdown (P(high) per model) ───────────────
  const breakdown = document.getElementById("breakdown");
  const rows = [];
  function pHigh(pred) {
    // index 0 == high, so P(high) = 1 - raw_score.
    return 1 - pred.raw_score;
  }
  if (P.cnn) rows.push(["Custom CNN", pHigh(P.cnn)]);
  if (P.mobilenet) rows.push(["MobileNetV2", pHigh(P.mobilenet)]);
  breakdown.innerHTML = rows
    .map(([name, v]) => `
      <div class="bar-row">
        <div class="bar-label">${name}</div>
        <div class="bar-track"><div class="bar-fill ${v >= 0.5 ? "coral" : "teal"}" style="width:${v * 100}%"></div></div>
        <div class="bar-val">${fmtPct(v)}</div>
      </div>`)
    .join("");

  // ── Raw JSON ─────────────────────────────────────────────
  document.getElementById("rawJson").innerHTML = highlightJSON(data);
})();
