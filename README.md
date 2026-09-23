# Green MLOps: Framework for Energy-Efficient ML Operations Pipelines

Research project (SRH University Heidelberg) implementing and evaluating a
"Green MLOps" pipeline that treats energy efficiency as a first-class metric
alongside model accuracy, per the accompanying Research Project Expose.

Author: Nandan Muruli | Supervisor: Prof. Dr. Kamellia Reshadi

## Research questions

- **RQ1** – How can power-consumption tracking be non-intrusively integrated
  into automated ML training/deployment loops?
- **RQ2** – What is the quantitative trade-off between model performance
  (F1) and energy consumption under model compression (quantization,
  pruning)?
- **RQ3** – Can data-driven scheduling and execution-level resource capping
  reduce carbon emissions without violating operational latency SLAs?

## Architecture

| Layer | Tool | Status |
|---|---|---|
| Pipeline & versioning | DVC (`dvc.yaml`) | implemented |
| Experiment tracking | MLflow (`sqlite:///mlflow.db`) | implemented |
| Energy telemetry | CodeCarbon (`src/telemetry/energy_tracker.py`) | implemented |
| Containerization | Docker (`Dockerfile`) | implemented |
| CI/CD | GitHub Actions (`.github/workflows/ci.yml`) | implemented (lint + smoke test) |
| Dashboard | MLflow UI + static HTML summary (`reports/dashboard.html`) + live auto-refreshing Flask dashboard (`src/dashboard/live_server.py`) | implemented |

### Documented scope adaptation

The expose specifies **Scaphandre** (Intel RAPL-based bare-metal/Kubernetes
power telemetry) and **GPU power capping** for RQ3. The development and
execution hardware for this project (Apple Silicon Mac, and a CPU-only
Linux execution environment) has **no Intel RAPL interface and no NVIDIA
GPU**, so neither tool is applicable. This is a hardware constraint, not a
design choice, and is treated as a stated limitation of the study:

- Energy telemetry uses **CodeCarbon only** (process/CPU/RAM-level
  estimation), not Scaphandre.
- "Hardware-level power capping" is implemented as a **CPU-thread-count
  cap** (`torch.set_num_threads`) as a software-controllable proxy for
  execution-level resource restriction, benchmarked against the SLA
  latency threshold in `configs/config.yaml`.
- The dashboard uses **MLflow's built-in UI** plus a static HTML summary
  instead of a full Grafana/Kubernetes deployment.

## Repository layout

```
configs/config.yaml         # single source of truth for all run parameters
src/pipeline/ingest_data.py # DVC stage 1: download IMDb dataset
src/pipeline/train_model.py # DVC stage 2: fine-tune DistilBERT baseline
src/pipeline/evaluate.py    # accuracy / F1 / inference latency evaluation
src/pipeline/compress.py    # RQ2: quantization + pruning benchmark
src/pipeline/rq3_experiments.py # RQ3: thread-capping + scheduling benchmark
src/telemetry/energy_tracker.py # CodeCarbon + MLflow energy logging wrapper
src/dashboard/build_dashboard.py# static HTML metrics dashboard
src/dashboard/live_server.py     # live/dynamic dashboard (Flask + polling)
results/                    # CSV outputs consumed by the report/dashboard
logs/emissions.csv          # raw CodeCarbon telemetry log
reports/                    # case study report + dashboard.html
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Stage 1: data ingestion (DVC-tracked)
python -m src.pipeline.ingest_data

# Stage 2: baseline fine-tuning (already produced models/baseline/checkpoint-126)
python -m src.pipeline.train_model

# RQ2: compression trade-off benchmark (quantization + pruning)
python -m src.pipeline.compress

# RQ3: thread-capping + scheduling benchmark
python -m src.pipeline.rq3_experiments

# Build the static dashboard summary from all logged results
python -m src.dashboard.build_dashboard

# Or drive all DVC-tracked stages at once
dvc repro
```

Live experiment tracking: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Live dashboard

In addition to the static `reports/dashboard.html`, `src/dashboard/live_server.py`
serves the same metrics as a small Flask app that re-reads the results CSVs on
every request instead of baking them into a fixed HTML snapshot:

```bash
python -m src.dashboard.live_server
# then open http://127.0.0.1:5000
```

This gives it three properties the static version doesn't have:

- **Interactive charts** – the RQ2/RQ3 bar and line charts are hand-rolled SVG
  (no external JS library, so it still works fully offline) with hover
  tooltips showing exact values per bar/point.
- **Auto-loads latest results** – every page load (or refresh) reflects
  whatever is currently in `results/*.csv` and `logs/emissions.csv` — rerun
  any pipeline stage and reload the page to see the new numbers, no rebuild
  step needed.
- **Live-updating without refresh** – the page polls `/api/metrics` every 3
  seconds and redraws in place, so if a pipeline stage is still writing
  results (e.g. mid-training), the dashboard updates on its own.

## Results

See `results/rq2_compression_results.csv`, `results/rq3_results.csv`,
`reports/dashboard.html` and `reports/case_study_report.docx` for the full
write-up of quantitative findings against RQ1–RQ3.

## Reproducibility notes

During development, `results/rq2_compression_results.csv` was observed to
change between reruns of `python -m src.pipeline.compress` (baseline
accuracy moved between roughly 0.80 and 0.84 across otherwise-identical
runs, with the model checkpoint, tokenizer, and eval-set selection all
confirmed bit-identical). The cause was CPU floating-point non-determinism:
multi-threaded matmul on CPU does not guarantee the same summation order
across runs/machines, and with only 100 eval examples a single flipped
prediction moves accuracy by a full percentage point.

Two fixes were applied in `configs/config.yaml` / `src/pipeline/evaluate.py`:
pinning `compression.eval_num_threads` (so thread count — and therefore
floating-point rounding — is fixed rather than host-dependent) and enabling
`torch.use_deterministic_algorithms` for the RQ2 evaluation pass, plus
tripling `compression.eval_subset_size` (100 → 300) so the metric is less
sensitive to any remaining borderline predictions. Repeated reruns after
this fix produced bit-identical accuracy/F1 (only wall-clock latency, which
is expected to vary with system load, differed).

Separately, `data/raw/imdb`'s DVC-tracked hash kept drifting because HF
`datasets` writes `cache-*.arrow` byproduct files into the dataset directory
on every `.shuffle()/.select()/.map()` call. `datasets.disable_caching()` is
now called at import time in `src/utils/config.py` and `src/pipeline/evaluate.py`
to stop this at the source rather than needing a manual DVC resync after
every run.
