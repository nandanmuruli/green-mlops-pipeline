"""RQ3 -- can data-driven scheduling and execution-level resource capping
reduce carbon emissions without violating operational latency SLAs?

Two independent benchmarks, both logged to MLflow + CodeCarbon:

1. Capping benchmark: run baseline-model inference under different
   torch.set_num_threads() caps (a CPU-level, software-controllable proxy
   for the GPU power-capping described in the expose, since this hardware
   has no NVIDIA GPU to cap). Checks whether latency stays within the
   configured SLA at each cap level.

2. Scheduling benchmark: brute-force grid search over (learning_rate,
   batch_size) vs. a single early-stopped run, comparing total energy
   spent searching for good hyperparameters.
"""
import json
import os
import shutil
import time

import mlflow
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.pipeline.evaluate import evaluate_model
from src.telemetry.energy_tracker import GreenTracker
from src.utils.config import load_config

SCRATCH_DIR = "./models/rq3_scratch"


def _tokenized_datasets(cfg, tokenizer, train_size, eval_size):
    dataset = load_from_disk(cfg["dataset"]["raw_dir"])
    train = dataset["train"].shuffle(seed=cfg["dataset"]["seed"]).select(range(train_size))
    test = dataset["test"].shuffle(seed=cfg["dataset"]["seed"]).select(range(eval_size))

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True,
                          max_length=cfg["dataset"]["max_seq_length"])

    return train.map(tok, batched=True), test.map(tok, batched=True)


def _train_once(cfg, tokenizer, train_ds, eval_ds, lr, batch_size, max_epochs, early_stopping):
    run_tag = f"lr{lr}_bs{batch_size}_{'earlystop' if early_stopping else 'grid'}"
    out_dir = os.path.join(SCRATCH_DIR, run_tag)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["base_model"], num_labels=cfg["model"]["num_labels"]
    )
    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch" if early_stopping else "no",
        load_best_model_at_end=early_stopping,
        metric_for_best_model="eval_loss",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        report_to=[],
        logging_steps=50,
        disable_tqdm=True,
    )
    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["rq3"]["early_stopping"]["patience"]))

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, callbacks=callbacks)

    t0 = time.time()
    with GreenTracker(project_name=f"rq3_scheduling_{run_tag}") as tracker:
        train_result = trainer.train()
    duration_s = time.time() - t0
    eval_metrics = trainer.evaluate()
    epochs_run = trainer.state.epoch

    shutil.rmtree(out_dir, ignore_errors=True)  # scratch only, not a deliverable artifact
    return {
        "run_tag": run_tag,
        "strategy": "early_stopping" if early_stopping else "grid_search",
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs_run": round(float(epochs_run), 2),
        "train_duration_s": round(duration_s, 3),
        "eval_loss": round(float(eval_metrics["eval_loss"]), 4),
    }


PARTIAL_PATH = os.path.join("results", "rq3_scheduling_partial.jsonl")


def _load_partial():
    if not os.path.exists(PARTIAL_PATH):
        return {}
    done = {}
    with open(PARTIAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["run_tag"]] = row
    return done


def _append_partial(row):
    os.makedirs("results", exist_ok=True)
    with open(PARTIAL_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_scheduling_benchmark(cfg, max_new_runs=None):
    """Resumable: already-completed run_tags (from a prior, timed-out call)
    are loaded from results/rq3_scheduling_partial.jsonl and skipped, so
    this can safely be re-invoked across several shorter calls until all
    grid combinations + the early-stopping run are done. `max_new_runs`
    caps how many NEW training runs this invocation performs, so a single
    call can be kept under a wall-clock budget.
    """
    rq3cfg = cfg["rq3"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    train_size = 200
    eval_size = 50
    train_ds, eval_ds = _tokenized_datasets(cfg, tokenizer, train_size, eval_size)

    done = _load_partial()
    new_runs = 0

    print("\n=== Brute-force grid search ===")
    for lr in rq3cfg["grid_search"]["learning_rates"]:
        for bs in rq3cfg["grid_search"]["batch_sizes"]:
            tag = f"lr{lr}_bs{bs}_grid"
            if tag in done:
                print(f"[skip] {tag} already completed")
                continue
            if max_new_runs is not None and new_runs >= max_new_runs:
                print(f"[stop] reached max_new_runs={max_new_runs} for this invocation")
                return _finalize_if_complete(cfg, done)
            with GreenTracker(project_name=f"rq3_grid_lr{lr}_bs{bs}") as tracker:
                row = _train_once(cfg, tokenizer, train_ds, eval_ds, lr, bs, max_epochs=1, early_stopping=False)
            row["emissions_kg_co2e"] = tracker.final_emissions if hasattr(tracker, "final_emissions") else None
            _append_partial(row)
            done[tag] = row
            new_runs += 1
            print(row)

    es_tag_lr = rq3cfg["grid_search"]["learning_rates"][0]
    es_tag_bs = rq3cfg["grid_search"]["batch_sizes"][0]
    es_tag = f"lr{es_tag_lr}_bs{es_tag_bs}_earlystop"
    if es_tag not in done:
        if max_new_runs is not None and new_runs >= max_new_runs:
            print(f"[stop] reached max_new_runs={max_new_runs} for this invocation")
            return _finalize_if_complete(cfg, done)
        print("\n=== Early-stopped single run ===")
        with GreenTracker(project_name="rq3_early_stopping") as tracker:
            es_row = _train_once(cfg, tokenizer, train_ds, eval_ds, es_tag_lr, es_tag_bs, max_epochs=4, early_stopping=True)
        es_row["emissions_kg_co2e"] = tracker.final_emissions if hasattr(tracker, "final_emissions") else None
        _append_partial(es_row)
        done[es_tag] = es_row
        print(es_row)
    else:
        print(f"[skip] {es_tag} already completed")

    return _finalize_if_complete(cfg, done)


def _finalize_if_complete(cfg, done):
    rq3cfg = cfg["rq3"]
    expected_tags = [
        f"lr{lr}_bs{bs}_grid"
        for lr in rq3cfg["grid_search"]["learning_rates"]
        for bs in rq3cfg["grid_search"]["batch_sizes"]
    ]
    es_tag = f"lr{rq3cfg['grid_search']['learning_rates'][0]}_bs{rq3cfg['grid_search']['batch_sizes'][0]}_earlystop"
    expected_tags.append(es_tag)

    if not all(t in done for t in expected_tags):
        missing = [t for t in expected_tags if t not in done]
        print(f"Scheduling benchmark incomplete -- still missing: {missing}. Re-run to continue.")
        return None, None

    rows = [done[t] for t in expected_tags]
    grid_energy_kg = sum(done[t]["emissions_kg_co2e"] for t in expected_tags[:-1] if done[t].get("emissions_kg_co2e"))
    es_emissions = done[es_tag].get("emissions_kg_co2e")
    summary = {
        "grid_search_total_runs": len(expected_tags) - 1,
        "grid_search_total_emissions_kg_co2e": grid_energy_kg,
        "early_stopping_total_emissions_kg_co2e": es_emissions,
        "estimated_emissions_reduction_pct": (
            round((1 - (es_emissions / grid_energy_kg)) * 100, 1)
            if grid_energy_kg and es_emissions else None
        ),
    }
    print("\nScheduling summary:", summary)
    return rows, summary


def run_capping_benchmark(cfg):
    print("\n=== Execution-level resource capping (CPU thread cap proxy) ===")
    from transformers import AutoModelForSequenceClassification as ACSeq
    model = ACSeq.from_pretrained(cfg["compression"]["baseline_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    sla_ms = cfg["rq3"]["sla_latency_ms"]

    rows = []
    for threads in cfg["rq3"]["thread_caps"]:
        with GreenTracker(project_name=f"rq3_capping_{threads}threads") as tracker:
            metrics = evaluate_model(model, tokenizer=tokenizer, cfg=cfg, num_threads=threads)
        emissions = tracker.final_emissions if hasattr(tracker, "final_emissions") else None
        row = {
            "thread_cap": threads,
            "avg_inference_latency_ms": metrics["avg_inference_latency_ms"],
            "p95_inference_latency_ms": metrics["p95_inference_latency_ms"],
            "sla_latency_ms": sla_ms,
            "sla_violated": metrics["p95_inference_latency_ms"] > sla_ms,
            "emissions_kg_co2e": emissions,
        }
        rows.append(row)
        print(row)
    return rows


def run(max_new_runs=None):
    cfg = load_config()
    os.makedirs("results", exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    if not os.path.exists(os.path.join("results", "rq3_capping_results.csv")):
        capping_rows = run_capping_benchmark(cfg)
        import csv
        with open(os.path.join("results", "rq3_capping_results.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(capping_rows[0].keys()))
            writer.writeheader()
            writer.writerows(capping_rows)
    else:
        print("[skip] capping benchmark already completed")

    scheduling_csv_path = os.path.join("results", "rq3_scheduling_results.csv")
    if os.path.exists(scheduling_csv_path) and not os.path.exists(PARTIAL_PATH):
        print("[skip] scheduling benchmark already completed")
        return True

    scheduling_rows, scheduling_summary = run_scheduling_benchmark(cfg, max_new_runs=max_new_runs)
    if scheduling_rows is None:
        print("Scheduling benchmark not yet complete -- re-run this stage to continue.")
        return False

    import csv
    with open(scheduling_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scheduling_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scheduling_rows)

    with mlflow.start_run(run_name="rq3_summary"):
        for k, v in scheduling_summary.items():
            if isinstance(v, (int, float)) and v is not None:
                mlflow.log_metric(k, v)

    # Combined results file referenced by dvc.yaml / the dashboard
    with open(cfg["rq3"]["results_file"], "w", newline="") as f:
        f.write("== capping ==\n")
        with open(os.path.join("results", "rq3_capping_results.csv")) as cf:
            f.write(cf.read())
        f.write("\n== scheduling ==\n")
        with open(os.path.join("results", "rq3_scheduling_results.csv")) as sf:
            f.write(sf.read())
        f.write(f"\n== summary ==\n{scheduling_summary}\n")

    partial_path = os.path.join("results", "rq3_scheduling_partial.jsonl")
    try:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    except OSError:
        pass  # best-effort cleanup only; the sandbox filesystem may not permit deletes
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    print(f"\nRQ3 results written to results/rq3_capping_results.csv, results/rq3_scheduling_results.csv, {cfg['rq3']['results_file']}")
    return True


if __name__ == "__main__":
    import sys
    _max = None
    for arg in sys.argv[1:]:
        if arg.startswith("--max-new-runs="):
            _max = int(arg.split("=", 1)[1])
    run(max_new_runs=_max)
