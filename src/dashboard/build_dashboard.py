"""Builds a self-contained static HTML dashboard summarizing operational
vs. environmental metrics from all three research questions, reading the
CSVs produced by compress.py / rq3_experiments.py / the CodeCarbon log.

This supplements (not replaces) the live MLflow UI --
`mlflow ui --backend-store-uri sqlite:///mlflow.db` -- per the documented
scope adaptation (no Grafana/Kubernetes deployment on this hardware).
"""
import base64
import csv
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _chart_rq2(rows):
    variants = [r["variant"] for r in rows]
    accuracy = [float(r["accuracy"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]
    latency = [float(r["avg_inference_latency_ms"]) for r in rows]
    size = [float(r["model_size_mb"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = range(len(variants))

    axes[0].bar([i - 0.15 for i in x], accuracy, width=0.3, label="Accuracy")
    axes[0].bar([i + 0.15 for i in x], f1, width=0.3, label="F1")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(variants)
    axes[0].set_title("RQ2: Accuracy / F1 by variant"); axes[0].legend(); axes[0].set_ylim(0, 1)

    axes[1].bar(x, latency, color="#d97706")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(variants)
    axes[1].set_title("Avg inference latency (ms)")

    axes[2].bar(x, size, color="#0369a1")
    axes[2].set_xticks(list(x)); axes[2].set_xticklabels(variants)
    axes[2].set_title("On-disk weights size (MB)")

    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_rq3_capping(rows, sla_ms):
    threads = [int(r["thread_cap"]) for r in rows]
    avg_lat = [float(r["avg_inference_latency_ms"]) for r in rows]
    p95_lat = [float(r["p95_inference_latency_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(threads, avg_lat, marker="o", label="avg latency")
    ax.plot(threads, p95_lat, marker="o", label="p95 latency")
    ax.axhline(sla_ms, color="red", linestyle="--", label=f"SLA ({sla_ms} ms)")
    ax.set_xlabel("CPU thread cap"); ax.set_ylabel("Latency (ms)")
    ax.set_title("RQ3: Latency vs. execution-level resource cap")
    ax.set_xticks(threads)
    ax.legend()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_rq3_scheduling(rows):
    grid_rows = [r for r in rows if r["strategy"] == "grid_search"]
    es_rows = [r for r in rows if r["strategy"] == "early_stopping"]
    grid_total = sum(float(r["emissions_kg_co2e"]) for r in grid_rows if r["emissions_kg_co2e"])
    es_total = sum(float(r["emissions_kg_co2e"]) for r in es_rows if r["emissions_kg_co2e"])

    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(["Brute-force\ngrid search", "Early\nstopping"],
                   [grid_total * 1e6, es_total * 1e6], color=["#b91c1c", "#15803d"])
    ax.set_ylabel("Emissions (mg CO2e)")
    ax.set_title("RQ3: Search-strategy energy cost")
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom")
    fig.tight_layout()
    reduction = None
    if grid_total:
        reduction = round((1 - es_total / grid_total) * 100, 1)
    return _fig_to_base64(fig), reduction


def build():
    cfg = load_config()
    os.makedirs("reports", exist_ok=True)

    rq2_rows = _read_csv(cfg["compression"]["results_file"])
    rq3_capping_rows = _read_csv(os.path.join("results", "rq3_capping_results.csv"))
    rq3_sched_rows = _read_csv(os.path.join("results", "rq3_scheduling_results.csv"))

    rq2_img = _chart_rq2(rq2_rows)
    rq3_cap_img = _chart_rq3_capping(rq3_capping_rows, cfg["rq3"]["sla_latency_ms"])
    rq3_sched_img, reduction_pct = _chart_rq3_scheduling(rq3_sched_rows)

    baseline = next(r for r in rq2_rows if r["variant"] == "baseline")
    quantized = next(r for r in rq2_rows if r["variant"] == "quantized")
    size_reduction_pct = round((1 - float(quantized["model_size_mb"]) / float(baseline["model_size_mb"])) * 100, 1)
    latency_reduction_pct = round((1 - float(quantized["avg_inference_latency_ms"]) / float(baseline["avg_inference_latency_ms"])) * 100, 1)
    acc_delta_pp = round((float(quantized["accuracy"]) - float(baseline["accuracy"])) * 100, 1)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Green MLOps Dashboard</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 1100px; color: #1f2937; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid #e5e7eb; padding-bottom: .3rem;}}
.kpis {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0 2rem; }}
.kpi {{ background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: .9rem 1.2rem; min-width: 180px; }}
.kpi .v {{ font-size: 1.4rem; font-weight: 700; }} .kpi .l {{ font-size: .8rem; color: #64748b; }}
table {{ border-collapse: collapse; width: 100%; margin-top: .5rem; }}
th, td {{ border: 1px solid #e5e7eb; padding: .4rem .6rem; text-align: right; font-size: .85rem; }}
th:first-child, td:first-child {{ text-align: left; }}
img {{ max-width: 100%; }}
.note {{ font-size: .82rem; color: #64748b; }}
</style></head>
<body>
<h1>Green MLOps: Operational vs. Environmental Metrics</h1>
<p class="note">Generated from real experiment runs (results/*.csv, logs/emissions.csv). Live per-run traces: <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code>.</p>

<div class="kpis">
  <div class="kpi"><div class="v">{size_reduction_pct}%</div><div class="l">model size reduction (quantization)</div></div>
  <div class="kpi"><div class="v">{latency_reduction_pct}%</div><div class="l">inference latency reduction (quantization)</div></div>
  <div class="kpi"><div class="v">{acc_delta_pp:+.1f} pp</div><div class="l">accuracy change (quantization)</div></div>
  <div class="kpi"><div class="v">{reduction_pct}%</div><div class="l">energy saved: early-stopping vs. grid search</div></div>
</div>

<h2>RQ2 -- Compression trade-off (accuracy/F1 vs. energy &amp; latency)</h2>
<img src="data:image/png;base64,{rq2_img}">

<h2>RQ3a -- Execution-level resource capping vs. latency SLA</h2>
<img src="data:image/png;base64,{rq3_cap_img}">

<h2>RQ3b -- Scheduling strategy: grid search vs. early stopping</h2>
<img src="data:image/png;base64,{rq3_sched_img}">

<h2>Raw results</h2>
<p class="note">RQ2 compression benchmark</p>
{_html_table(rq2_rows)}
<p class="note">RQ3 capping benchmark (SLA = {cfg['rq3']['sla_latency_ms']} ms)</p>
{_html_table(rq3_capping_rows)}
<p class="note">RQ3 scheduling benchmark</p>
{_html_table(rq3_sched_rows)}

</body></html>
"""
    out_path = os.path.join("reports", "dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {out_path}")


def _html_table(rows):
    if not rows:
        return "<p>no data</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


if __name__ == "__main__":
    build()
