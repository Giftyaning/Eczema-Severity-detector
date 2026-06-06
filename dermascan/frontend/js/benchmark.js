/* Benchmark page: Chart.js training curves, confusion matrices, grayscale ablation.

   These figures are illustrative of the original 10-epoch training runs. Swap in
   exact numbers from your training logs if you want them to be authoritative. */

(function () {
  // ── Illustrative training history (10 epochs) ─────────────
  const epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const cnnAcc = [0.55, 0.61, 0.66, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78];
  const mobAcc = [0.70, 0.78, 0.83, 0.86, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92];
  const cnnLoss = [0.69, 0.64, 0.60, 0.56, 0.53, 0.51, 0.49, 0.48, 0.47, 0.46];
  const mobLoss = [0.58, 0.48, 0.40, 0.34, 0.30, 0.27, 0.25, 0.24, 0.23, 0.22];

  // Headline metric cards.
  document.getElementById("cnnAcc").textContent = Math.round(cnnAcc.at(-1) * 100) + "%";
  document.getElementById("mobAcc").textContent = Math.round(mobAcc.at(-1) * 100) + "%";

  const COLORS = {
    accent: "#7F77DD",
    teal: "#5DCAA5",
    text: "#888",
    grid: "#2A2A3A",
  };

  Chart.defaults.color = COLORS.text;
  Chart.defaults.font.family = "system-ui, -apple-system, sans-serif";
  Chart.defaults.font.size = 12;

  const baseScales = {
    x: { grid: { color: COLORS.grid }, title: { display: true, text: "Epoch" } },
    y: { grid: { color: COLORS.grid } },
  };

  function lineChart(id, datasets, yTitle) {
    new Chart(document.getElementById(id), {
      type: "line",
      data: { labels: epochs, datasets },
      options: {
        responsive: true,
        plugins: { legend: { position: "bottom" } },
        scales: {
          x: baseScales.x,
          y: { grid: { color: COLORS.grid }, title: { display: true, text: yTitle } },
        },
      },
    });
  }

  lineChart("accChart", [
    { label: "Custom CNN", data: cnnAcc, borderColor: COLORS.accent, backgroundColor: COLORS.accent, tension: 0.3 },
    { label: "MobileNetV2", data: mobAcc, borderColor: COLORS.teal, backgroundColor: COLORS.teal, tension: 0.3 },
  ], "Validation accuracy");

  lineChart("lossChart", [
    { label: "Custom CNN", data: cnnLoss, borderColor: COLORS.accent, backgroundColor: COLORS.accent, tension: 0.3 },
    { label: "MobileNetV2", data: mobLoss, borderColor: COLORS.teal, backgroundColor: COLORS.teal, tension: 0.3 },
  ], "Validation loss");

  // ── Confusion matrices (illustrative, n=40: 20 high / 20 low) ──
  // Layout: [ "" , Pred high, Pred low ] header, then two rows.
  function renderConfusion(elId, cm) {
    // cm = [[TP_high, FN_high],[FP, TN_low]] indexed as true x predicted.
    const el = document.getElementById(elId);
    const cells = [
      { t: "", cls: "cm-head" },
      { t: "Pred high", cls: "cm-head" },
      { t: "Pred low", cls: "cm-head" },
      { t: "True high", cls: "cm-head" },
      { t: cm[0][0], cls: "cm-cell cm-correct" },
      { t: cm[0][1], cls: "cm-cell cm-wrong" },
      { t: "True low", cls: "cm-head" },
      { t: cm[1][0], cls: "cm-cell cm-wrong" },
      { t: cm[1][1], cls: "cm-cell cm-correct" },
    ];
    el.innerHTML = cells.map((c) => `<div class="${c.cls}">${c.t}</div>`).join("");
  }

  renderConfusion("cmCnn", [[15, 5], [6, 14]]); // ~72% correct
  renderConfusion("cmMob", [[18, 2], [2, 18]]); // ~90% correct

  // ── Grayscale ablation bar chart ─────────────────────────
  new Chart(document.getElementById("grayChart"), {
    type: "bar",
    data: {
      labels: ["Custom CNN", "MobileNetV2"],
      datasets: [
        { label: "Colour", data: [78, 92], backgroundColor: COLORS.teal },
        { label: "Grayscale", data: [62, 80], backgroundColor: COLORS.accent },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { grid: { color: COLORS.grid } },
        y: { grid: { color: COLORS.grid }, title: { display: true, text: "Val accuracy (%)" }, suggestedMax: 100 },
      },
    },
  });
})();
