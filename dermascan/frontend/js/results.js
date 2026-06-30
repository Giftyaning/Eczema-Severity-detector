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

  // Image panels
  if (imageSrc) document.getElementById("imgOriginal").src = imageSrc;
  const heatSrc = data.gradcam_heatmap_base64
    ? `data:image/png;base64,${data.gradcam_heatmap_base64}`
    : null;
  const overlaySrc = data.gradcam_base64
    ? `data:image/png;base64,${data.gradcam_base64}`
    : null;
  if (heatSrc) document.getElementById("imgGradcam").src = heatSrc;
  if (overlaySrc) document.getElementById("imgOverlay").src = overlaySrc;

  // Timing 
  document.getElementById("timing").textContent =
    `processed in ${data.processing_time_ms} ms`;

  // Verdict cards 
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

  // Agreement indicator 
  const agreeEl = document.getElementById("agreement");
  if (data.agreement === true) {
    agreeEl.innerHTML = `<span class="badge badge-agree">✓ Models agree</span>`;
  } else if (data.agreement === false) {
    agreeEl.innerHTML = `<span class="badge badge-disagree">✕ Models disagree — interpret with caution</span>`;
  } else {
    agreeEl.innerHTML = `<span class="badge">Single model — no agreement check</span>`;
  }

  // Severity breakdown (P(high) per model) 
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

  // Raw JSON 
  document.getElementById("rawJson").innerHTML = highlightJSON(data);

  // Download composed result (client-side canvas, no backend) 
  const PALETTE = {
    bg: "#0F1117", surface: "#161622", surfaceRaised: "#1A1A2E",
    border: "#2A2A3A", text: "#F0F0F0", secondary: "#888", muted: "#555",
    accent: "#7F77DD", accentLight: "#AFA9EC", teal: "#5DCAA5", coral: "#F0997B",
  };
  const FONT = "system-ui, -apple-system, sans-serif";

  const downloadBtn = document.getElementById("downloadBtn");
  const downloadErr = document.getElementById("downloadErr");

  // Resolve an image to draw: prefer the already-decoded on-page <img>; only
  // re-fetch from the source string if the element never loaded.
  function resolveImg(imgEl, src) {
    if (imgEl && imgEl.complete && imgEl.naturalWidth > 0) return Promise.resolve(imgEl);
    if (!src) return Promise.resolve(null);
    return new Promise((resolve) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => resolve(null);
      im.src = src;
    });
  }

  // Rounded-rectangle path (arcTo keeps it portable across canvas versions).
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // Draw an image into a box with object-fit: cover semantics.
  function drawCover(ctx, img, x, y, w, h) {
    const ir = img.naturalWidth / img.naturalHeight;
    const r = w / h;
    let sx, sy, sw, sh;
    if (ir > r) { sh = img.naturalHeight; sw = sh * r; sx = (img.naturalWidth - sw) / 2; sy = 0; }
    else { sw = img.naturalWidth; sh = sw / r; sx = 0; sy = (img.naturalHeight - sh) / 2; }
    ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
  }

  function panel(ctx, img, x, y, w, h, caption) {
    ctx.save();
    roundRect(ctx, x, y, w, h, 10);
    ctx.fillStyle = PALETTE.surfaceRaised;
    ctx.fill();
    if (img) {
      ctx.save();
      roundRect(ctx, x, y, w, h, 10);
      ctx.clip();
      drawCover(ctx, img, x, y, w, h);
      ctx.restore();
    } else {
      ctx.fillStyle = PALETTE.muted;
      ctx.font = `15px ${FONT}`;
      ctx.textAlign = "center";
      ctx.fillText("not available", x + w / 2, y + h / 2);
    }
    roundRect(ctx, x, y, w, h, 10);
    ctx.lineWidth = 1;
    ctx.strokeStyle = PALETTE.border;
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = PALETTE.secondary;
    ctx.font = `15px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillText(caption, x + w / 2, y + h + 24);
  }

  // Draw "Name — HIGH 79%" with the severity word coloured. Returns nothing.
  function verdictLine(ctx, x, y, name, pred) {
    ctx.textAlign = "left";
    ctx.font = `18px ${FONT}`;
    ctx.fillStyle = PALETTE.secondary;
    ctx.fillText(name, x, y);
    let cx = x + ctx.measureText(name).width;
    ctx.fillStyle = PALETTE.secondary;
    ctx.fillText("  —  ", cx, y);
    cx += ctx.measureText("  —  ").width;
    const label = pred.label.toUpperCase();
    ctx.font = `600 18px ${FONT}`;
    ctx.fillStyle = pred.label === "high" ? PALETTE.coral : PALETTE.teal;
    ctx.fillText(label, cx, y);
    cx += ctx.measureText(label).width;
    ctx.font = `18px ${FONT}`;
    ctx.fillStyle = PALETTE.text;
    ctx.fillText(`  ${Math.round(pred.confidence * 100)}%`, cx, y);
  }

  // Pill badge mirroring the on-site .badge styling. Returns its width.
  function badge(ctx, text, x, y, color) {
    ctx.font = `600 14px ${FONT}`;
    const padX = 12, h = 26;
    const w = ctx.measureText(text).width + padX * 2;
    roundRect(ctx, x, y - h / 2, w, h, 13);
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, x + padX, y + 1);
    ctx.textBaseline = "alphabetic";
    return w;
  }

  function timestamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-` +
      `${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  async function buildResultCanvas() {
    const [orig, heat, over] = await Promise.all([
      resolveImg(document.getElementById("imgOriginal"), imageSrc),
      resolveImg(document.getElementById("imgGradcam"), heatSrc),
      resolveImg(document.getElementById("imgOverlay"), overlaySrc),
    ]);

    const W = 1200, H = 680, SC = 2; // SC = supersample for a crisp shareable PNG
    const canvas = document.createElement("canvas");
    canvas.width = W * SC;
    canvas.height = H * SC;
    const ctx = canvas.getContext("2d");
    ctx.scale(SC, SC);

    // Background
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, H);

    // Title / brand watermark (top-left), with the accent dot from the nav.
    ctx.fillStyle = PALETTE.accent;
    ctx.beginPath();
    ctx.arc(38, 40, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = PALETTE.text;
    ctx.font = `600 22px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText("DermaScan", 52, 47);
    ctx.fillStyle = PALETTE.secondary;
    ctx.font = `16px ${FONT}`;
    ctx.fillText("Grad-CAM analysis", 178, 47);

    // Three panels
    const M = 32, G = 28, top = 78;
    const pw = (W - 2 * M - 2 * G) / 3, ph = pw; // square panels
    panel(ctx, orig, M, top, pw, ph, "Original");
    panel(ctx, heat, M + pw + G, top, pw, ph, "Grad-CAM heatmap");
    panel(ctx, over, M + 2 * (pw + G), top, pw, ph, "Overlay (70% heatmap)");

    // Verdicts
    let vy = top + ph + 70;
    ctx.fillStyle = PALETTE.muted;
    ctx.font = `13px ${FONT}`;
    ctx.textAlign = "left";
    ctx.fillText("MODEL VERDICTS", M, vy - 26);
    if (P.cnn) { verdictLine(ctx, M, vy, "Custom CNN", P.cnn); vy += 34; }
    if (P.mobilenet) { verdictLine(ctx, M, vy, "MobileNetV2", P.mobilenet); vy += 34; }

    // Agreement indicator (right-aligned, vertically centred on the verdict block)
    const agreeY = top + ph + 78;
    let agreeText, agreeColor;
    if (data.agreement === true) { agreeText = "✓ Models agree"; agreeColor = PALETTE.teal; }
    else if (data.agreement === false) { agreeText = "✕ Models disagree"; agreeColor = PALETTE.coral; }
    else { agreeText = "Single model"; agreeColor = PALETTE.secondary; }
    ctx.font = `600 14px ${FONT}`;
    const pillW = ctx.measureText(agreeText).width + 24; // padX (12) * 2
    badge(ctx, agreeText, W - M - pillW, agreeY, agreeColor);

    // Footer: honest framing (centre) + timestamp (right)
    ctx.fillStyle = PALETTE.border;
    ctx.fillRect(M, H - 52, W - 2 * M, 1);
    ctx.fillStyle = PALETTE.secondary;
    ctx.font = `14px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillText("DermaScan — Educational tool, not for clinical use", W / 2, H - 26);
    ctx.fillStyle = PALETTE.muted;
    ctx.font = `12px ${FONT}`;
    ctx.textAlign = "right";
    ctx.fillText(timestamp(), W - M, H - 26);

    return canvas;
  }

  downloadBtn.addEventListener("click", async () => {
    downloadErr.textContent = "";
    downloadBtn.disabled = true;
    const original = downloadBtn.textContent;
    downloadBtn.textContent = "Preparing…";
    try {
      const canvas = await buildResultCanvas();
      const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
      if (!blob) throw new Error("Could not generate the image.");
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `dermascan-result-${timestamp()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      downloadErr.textContent = e.message || "Download failed.";
    } finally {
      downloadBtn.textContent = original;
      downloadBtn.disabled = false;
    }
  });

  // Results are rendered — the button is now safe to use.
  downloadBtn.disabled = false;
})();
