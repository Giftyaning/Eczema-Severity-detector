/* ============================================================
   DermaScan — shared frontend JS
   Navigation highlighting + all API fetch helpers live here.
   Every page includes this file first.
   ============================================================ */

// API base resolution, in priority order:
//   1. window.DERMASCAN_API if explicitly set.
//   2. Same-origin /api/v1 when served over http(s) — this is the Docker setup,
//      where nginx proxies /api/ to the backend container.
//   3. http://54.226.38.244:8000/api/v1 fallback (e.g. opening index.html via file://).
function resolveApiBase() {
  if (window.DERMASCAN_API) return window.DERMASCAN_API;
  return "http://54.226.38.244:8000/api/v1";
}
const API_BASE = resolveApiBase().replace(/\/$/, "");

/* ── Navigation active-state highlighting ─────────────────── */
function highlightNav() {
  const here = (location.pathname.split("/").pop() || "index.html").toLowerCase();
  document.querySelectorAll(".nav-links a").forEach((a) => {
    const target = (a.getAttribute("href") || "").toLowerCase();
    if (target === here || (here === "" && target === "index.html")) {
      a.classList.add("active");
    }
  });
}

/* ── Fetch helpers ────────────────────────────────────────── */

// Shared single-image prediction call used by every page that predicts.
async function fetchPredict(file, { model = "both", gradcam = true } = {}) {
  const form = new FormData();
  form.append("image", file);
  form.append("model", model);
  form.append("gradcam", String(gradcam));

  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

// Two-image comparison call.
async function fetchCompare(fileA, fileB, { model = "both" } = {}) {
  const form = new FormData();
  form.append("image_a", fileA);
  form.append("image_b", fileB);
  form.append("model", model);

  const res = await fetch(`${API_BASE}/predict/compare`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

/* ── Utilities ────────────────────────────────────────────── */

// Minimal JSON syntax highlighter -> HTML using the .json-* classes.
function highlightJSON(obj) {
  let json = typeof obj === "string" ? obj : JSON.stringify(obj, truncateBase64, 2);
  json = json.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return json.replace(
    /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|\bnull\b|-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)/g,
    (match) => {
      let cls = "json-number";
      if (/^"/.test(match)) {
        cls = /:$/.test(match) ? "json-key" : "json-string";
      } else if (/true|false/.test(match)) {
        cls = "json-bool";
      } else if (/null/.test(match)) {
        cls = "json-null";
      }
      return `<span class="${cls}">${match}</span>`;
    }
  );
}

// Keep base64 blobs short in pretty-printed JSON so the panel stays readable.
function truncateBase64(key, value) {
  if (key === "gradcam_base64" && typeof value === "string" && value.length > 48) {
    return value.slice(0, 32) + "…(" + value.length + " chars)";
  }
  return value;
}

function severityBadge(label) {
  const cls = label === "high" ? "badge-high" : "badge-low";
  return `<span class="badge ${cls}">${label.toUpperCase()}</span>`;
}

function fmtPct(x) {
  return (x * 100).toFixed(1) + "%";
}

document.addEventListener("DOMContentLoaded", highlightNav);
