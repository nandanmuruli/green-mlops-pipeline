"""Live, auto-refreshing Green MLOps dashboard.

Unlike build_dashboard.py (which bakes a static snapshot into
reports/dashboard.html), this runs a tiny local Flask server that
re-reads results/*.csv and logs/emissions.csv on every request and
serves them as JSON. The page itself polls that endpoint every few
seconds and redraws hand-rolled, dependency-free SVG charts with hover
tooltips -- so re-running an experiment and leaving this page open
shows the new numbers without re-running build_dashboard.py or
reloading the page.

No external JS libraries are loaded (no CDN dependency), so this works
fully offline -- important for a reproducible pipeline artifact.

Run:  python -m src.dashboard.live_server
Then open http://127.0.0.1:5000
"""
import csv
import os
import time

from flask import Flask, jsonify, Response

from src.utils.config import load_config

app = Flask(__name__)


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@app.route("/api/metrics")
def api_metrics():
    cfg = load_config()
    rq2 = _read_csv(cfg["compression"]["results_file"])
    rq3_capping = _read_csv(os.path.join("results", "rq3_capping_results.csv"))
    rq3_sched = _read_csv(os.path.join("results", "rq3_scheduling_results.csv"))
    emissions = _read_csv(os.path.join(cfg["telemetry"]["log_dir"], cfg["telemetry"]["emissions_file"]))
    return jsonify({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sla_latency_ms": cfg["rq3"]["sla_latency_ms"],
        "rq2": rq2,
        "rq3_capping": rq3_capping,
        "rq3_scheduling": rq3_sched,
        "emissions_recent": emissions[-15:],
        "emissions_total_runs": len(emissions),
    })


@app.route("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Green MLOps -- Live Dashboard</title>
<style>
:root { --forest:#1B4332; --moss:#74A57F; --amber:#E8A33D; --ink:#1F2937; --muted:#6B7280; --card:#F3F7F4; --border:#E5E7EB; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background: #FFFFFF; color: var(--ink); }
header { background: var(--forest); color: #fff; padding: 1.1rem 2rem; display: flex; align-items: center; justify-content: space-between; }
header h1 { font-size: 1.25rem; margin: 0; }
.live-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--amber); margin-right: 6px; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
.meta { font-size: .8rem; color: #CFE3D6; }
main { max-width: 1180px; margin: 0 auto; padding: 1.6rem 2rem 3rem; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .6rem; border-bottom: 1px solid var(--border); padding-bottom: .35rem; }
.kpis { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.kpi { background: var(--card); border-radius: 10px; padding: .9rem 1.2rem; min-width: 170px; }
.kpi .v { font-size: 1.35rem; font-weight: 700; color: var(--forest); }
.kpi .l { font-size: .78rem; color: var(--muted); }
.charts-row { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.chart-card { background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: .9rem 1rem 0.5rem; flex: 1 1 340px; min-width: 300px; }
.chart-card h3 { font-size: .88rem; margin: 0 0 .5rem; color: var(--forest); }
svg { width: 100%; height: 260px; overflow: visible; }
table { border-collapse: collapse; width: 100%; margin-top: .4rem; font-size: .82rem; }
th, td { border: 1px solid var(--border); padding: .35rem .55rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #FAFBFA; }
.note { font-size: .78rem; color: var(--muted); margin: .3rem 0 0; }
.legend { display: flex; gap: 1rem; font-size: .78rem; color: var(--muted); margin-top: .3rem; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
#tooltip { position: absolute; display: none; background: #111827; color: #fff; font-size: .78rem; padding: .35rem .55rem; border-radius: 6px; pointer-events: none; z-index: 10; white-space: nowrap; }
.empty { color: var(--muted); font-size: .85rem; font-style: italic; padding: 1rem 0; }
</style></head>
<body>
<div id="tooltip"></div>
<header>
  <h1>Green MLOps -- Live Dashboard</h1>
  <div class="meta"><span class="live-dot"></span>Live -- auto-refreshes every 3s -- last updated <span id="ts">--</span></div>
</header>
<main>
  <div class="kpis" id="kpis"></div>

  <h2>RQ2 -- Compression trade-off</h2>
  <div class="charts-row" id="rq2-charts"></div>
  <table id="rq2-table"></table>

  <h2>RQ3a -- Resource capping vs. latency SLA</h2>
  <div class="charts-row">
    <div class="chart-card" style="flex: 2 1 500px;">
      <h3>Latency vs. CPU thread cap</h3>
      <svg id="rq3a-chart"></svg>
      <div class="legend">
        <span style="--c: var(--forest)"><span style="background:var(--forest)"></span>Avg latency</span>
        <span><span style="background:var(--amber)"></span>p95 latency</span>
        <span><span style="background:#B91C1C"></span>SLA limit</span>
      </div>
    </div>
  </div>
  <table id="rq3a-table"></table>

  <h2>RQ3b -- Scheduling: grid search vs. early stopping</h2>
  <div class="charts-row">
    <div class="chart-card" style="flex: 1 1 340px;">
      <h3>Total energy by strategy (mg CO2e)</h3>
      <svg id="rq3b-chart"></svg>
    </div>
  </div>
  <table id="rq3b-table"></table>

  <p class="note">Reads results/rq2_compression_results.csv, results/rq3_capping_results.csv, results/rq3_scheduling_results.csv and logs/emissions.csv fresh on every poll -- no need to re-run build_dashboard.py or reload this page after a new experiment run.</p>
</main>

<script>
const FOREST = "#1B4332", MOSS = "#74A57F", AMBER = "#E8A33D";
const tooltip = document.getElementById("tooltip");

function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  tooltip.style.left = (evt.pageX + 14) + "px";
  tooltip.style.top = (evt.pageY + 10) + "px";
}
function hideTip() { tooltip.style.display = "none"; }
function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }

function drawBarChart(svg, labels, values, colors, fmt) {
  clear(svg);
  const W = svg.clientWidth || 380, H = 260;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const padL = 44, padR = 12, padT = 24, padB = 40;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const maxV = Math.max(...values, 0.0001) * 1.2;
  const gap = plotW / values.length, bw = gap * 0.5;
  svg.appendChild(svgEl("line", { x1: padL, y1: padT, x2: padL, y2: H - padB, stroke: "#E5E7EB" }));
  svg.appendChild(svgEl("line", { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: "#E5E7EB" }));
  values.forEach((v, i) => {
    const x = padL + gap * i + (gap - bw) / 2;
    const h = maxV > 0 ? (v / maxV) * plotH : 0;
    const y = H - padB - h;
    const rect = svgEl("rect", { x, y, width: bw, height: Math.max(h, 1), fill: colors[i % colors.length], rx: 3 });
    rect.addEventListener("mousemove", (e) => showTip(e, `<b>${labels[i]}</b><br>${fmt(v)}`));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);
    const lbl = svgEl("text", { x: x + bw / 2, y: H - padB + 16, "text-anchor": "middle", "font-size": 10.5, fill: "#374151" });
    lbl.textContent = labels[i];
    svg.appendChild(lbl);
    const vlbl = svgEl("text", { x: x + bw / 2, y: y - 6, "text-anchor": "middle", "font-size": 11, "font-weight": "bold", fill: "#111827" });
    vlbl.textContent = fmt(v);
    svg.appendChild(vlbl);
  });
}

function drawLineChart(svg, xLabels, seriesList, slaValue) {
  clear(svg);
  const W = svg.clientWidth || 500, H = 260;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const padL = 46, padR = 16, padT = 20, padB = 34;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const allVals = seriesList.flatMap(s => s.values).concat([slaValue || 0]);
  const maxV = Math.max(...allVals, 1) * 1.1;
  const stepX = xLabels.length > 1 ? plotW / (xLabels.length - 1) : plotW;

  svg.appendChild(svgEl("line", { x1: padL, y1: padT, x2: padL, y2: H - padB, stroke: "#E5E7EB" }));
  svg.appendChild(svgEl("line", { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, stroke: "#E5E7EB" }));

  if (slaValue) {
    const y = H - padB - (slaValue / maxV) * plotH;
    svg.appendChild(svgEl("line", { x1: padL, y1: y, x2: W - padR, y2: y, stroke: "#B91C1C", "stroke-width": 1.5, "stroke-dasharray": "5,4" }));
  }

  xLabels.forEach((lab, i) => {
    const x = padL + stepX * i;
    const t = svgEl("text", { x, y: H - padB + 16, "text-anchor": "middle", "font-size": 10.5, fill: "#374151" });
    t.textContent = lab;
    svg.appendChild(t);
  });

  seriesList.forEach(s => {
    let pts = "";
    s.values.forEach((v, i) => {
      const x = padL + stepX * i;
      const y = H - padB - (v / maxV) * plotH;
      pts += `${x},${y} `;
    });
    svg.appendChild(svgEl("polyline", { points: pts.trim(), fill: "none", stroke: s.color, "stroke-width": 2.5 }));
    s.values.forEach((v, i) => {
      const x = padL + stepX * i;
      const y = H - padB - (v / maxV) * plotH;
      const c = svgEl("circle", { cx: x, cy: y, r: 5, fill: s.color });
      c.addEventListener("mousemove", (e) => showTip(e, `<b>${s.name}</b><br>${xLabels[i]}: ${v.toFixed(1)} ms`));
      c.addEventListener("mouseleave", hideTip);
      svg.appendChild(c);
    });
  });
}

function renderTable(el, rows) {
  clear(el);
  if (!rows || rows.length === 0) {
    el.parentElement.insertAdjacentHTML("beforeend", "");
    el.innerHTML = '<tr><td class="empty">No data yet -- run the corresponding pipeline stage.</td></tr>';
    return;
  }
  const cols = Object.keys(rows[0]);
  const thead = "<thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead>";
  const tbody = "<tbody>" + rows.map(r => "<tr>" + cols.map(c => `<td>${r[c]}</td>`).join("") + "</tr>").join("") + "</tbody>";
  el.innerHTML = thead + tbody;
}

function kpiCard(value, label) {
  return `<div class="kpi"><div class="v">${value}</div><div class="l">${label}</div></div>`;
}

async function refresh() {
  let data;
  try {
    const res = await fetch("/api/metrics");
    data = await res.json();
  } catch (e) {
    return; // server briefly restarting or unreachable -- keep last good view
  }
  document.getElementById("ts").textContent = data.generated_at;

  const rq2 = data.rq2, byVariant = {};
  rq2.forEach(r => byVariant[r.variant] = r);
  const kpis = [];
  if (byVariant.baseline && byVariant.quantized) {
    const b = byVariant.baseline, q = byVariant.quantized;
    const sizeRed = (1 - q.model_size_mb / b.model_size_mb) * 100;
    const latRed = (1 - q.avg_inference_latency_ms / b.avg_inference_latency_ms) * 100;
    const accDelta = (q.accuracy - b.accuracy) * 100;
    kpis.push(kpiCard(sizeRed.toFixed(0) + "%", "model size reduction (quantized)"));
    kpis.push(kpiCard(latRed.toFixed(0) + "%", "latency reduction (quantized)"));
    kpis.push(kpiCard((accDelta >= 0 ? "+" : "") + accDelta.toFixed(1) + " pp", "accuracy change (quantized)"));
  }
  const sched = data.rq3_scheduling;
  const gridTotal = sched.filter(r => r.strategy === "grid_search").reduce((a, r) => a + parseFloat(r.emissions_kg_co2e || 0), 0);
  const esTotal = sched.filter(r => r.strategy === "early_stopping").reduce((a, r) => a + parseFloat(r.emissions_kg_co2e || 0), 0);
  if (gridTotal > 0 && esTotal > 0) {
    kpis.push(kpiCard(((1 - esTotal / gridTotal) * 100).toFixed(1) + "%", "energy saved: early-stopping vs. grid search"));
  }
  kpis.push(kpiCard(data.emissions_total_runs, "total tracked runs (logs/emissions.csv)"));
  document.getElementById("kpis").innerHTML = kpis.join("");

  // RQ2 charts
  const rq2Container = document.getElementById("rq2-charts");
  if (rq2.length) {
    rq2Container.innerHTML = `
      <div class="chart-card"><h3>Accuracy</h3><svg id="rq2-acc"></svg></div>
      <div class="chart-card"><h3>Avg. inference latency (ms)</h3><svg id="rq2-lat"></svg></div>
      <div class="chart-card"><h3>Weights size (MB)</h3><svg id="rq2-size"></svg></div>`;
    const labels = rq2.map(r => r.variant);
    const colors = [FOREST, AMBER, MOSS];
    drawBarChart(document.getElementById("rq2-acc"), labels, rq2.map(r => +r.accuracy), colors, v => v.toFixed(2));
    drawBarChart(document.getElementById("rq2-lat"), labels, rq2.map(r => +r.avg_inference_latency_ms), colors, v => v.toFixed(0) + " ms");
    drawBarChart(document.getElementById("rq2-size"), labels, rq2.map(r => +r.model_size_mb), colors, v => v.toFixed(0) + " MB");
  } else {
    rq2Container.innerHTML = '<div class="empty">No RQ2 results yet -- run: python -m src.pipeline.compress</div>';
  }
  renderTable(document.getElementById("rq2-table"), rq2);

  // RQ3a chart
  const capping = data.rq3_capping;
  if (capping.length) {
    drawLineChart(
      document.getElementById("rq3a-chart"),
      capping.map(r => "cap=" + r.thread_cap),
      [
        { name: "Avg latency", color: FOREST, values: capping.map(r => +r.avg_inference_latency_ms) },
        { name: "p95 latency", color: AMBER, values: capping.map(r => +r.p95_inference_latency_ms) },
      ],
      data.sla_latency_ms
    );
  }
  renderTable(document.getElementById("rq3a-table"), capping);

  // RQ3b chart
  if (sched.length) {
    drawBarChart(
      document.getElementById("rq3b-chart"),
      ["Grid search", "Early stopping"],
      [gridTotal * 1e6, esTotal * 1e6],
      [FOREST, AMBER],
      v => v.toFixed(2) + " mg"
    );
  }
  renderTable(document.getElementById("rq3b-table"), sched);
}

refresh();
setInterval(refresh, 3000);
window.addEventListener("resize", refresh);
</script>
</body></html>
"""


if __name__ == "__main__":
    print("Green MLOps live dashboard -- open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
