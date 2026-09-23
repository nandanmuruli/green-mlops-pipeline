"""RQ2 -- quantitative trade-off between model performance (F1) and energy
consumption under model compression.

Takes the already fine-tuned baseline checkpoint and produces two
post-training compressed variants:
  * dynamic INT8 quantization (torch.quantization.quantize_dynamic)
  * L1 unstructured magnitude pruning (torch.nn.utils.prune)

Each variant (plus the baseline itself, as the control) is evaluated with
the shared evaluate_model() routine, wrapped in GreenTracker so CodeCarbon
records the energy/emissions of *evaluation* for that variant specifically.
Results (accuracy, F1, energy, latency, on-disk size) are written to
results/rq2_compression_results.csv and logged to MLflow.
"""
import copy
import os

import mlflow
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.pipeline.evaluate import evaluate_model
from src.telemetry.energy_tracker import GreenTracker
from src.utils.config import load_config


def _dir_size_mb(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def apply_dynamic_quantization(model):
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized


def apply_magnitude_pruning(model, amount):
    pruned = copy.deepcopy(model)
    for module in pruned.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # bake the mask into the weights
    return pruned


def run():
    cfg = load_config()
    ccfg = cfg["compression"]
    os.makedirs("results", exist_ok=True)
    os.makedirs(ccfg["quantized_output_dir"], exist_ok=True)
    os.makedirs(ccfg["pruned_output_dir"], exist_ok=True)

    os.environ.setdefault("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    print(f"Loading baseline checkpoint from {ccfg['baseline_checkpoint']}...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        ccfg["baseline_checkpoint"]
    )
    # Compare like-for-like "deployed inference weights" size only -- the
    # baseline checkpoint dir also contains optimizer/scheduler/rng state
    # (needed to resume training, irrelevant to a deployment footprint),
    # which would otherwise make the comparison unfair.
    baseline_weights_file = os.path.join(ccfg["baseline_checkpoint"], "model.safetensors")
    baseline_size_mb = _dir_size_mb(baseline_weights_file)

    variants = {}
    variants["baseline"] = (baseline_model, baseline_size_mb)

    print("Applying dynamic INT8 quantization...")
    quantized_model = apply_dynamic_quantization(copy.deepcopy(baseline_model))
    quant_path = os.path.join(ccfg["quantized_output_dir"], "model.pt")
    torch.save(quantized_model.state_dict(), quant_path)
    variants["quantized"] = (quantized_model, _dir_size_mb(ccfg["quantized_output_dir"]))

    print(f"Applying L1 unstructured pruning (amount={ccfg['pruning_amount']})...")
    pruned_model = apply_magnitude_pruning(baseline_model, ccfg["pruning_amount"])
    pruned_path = os.path.join(ccfg["pruned_output_dir"], "model.pt")
    torch.save(pruned_model.state_dict(), pruned_path)
    variants["pruned"] = (pruned_model, _dir_size_mb(ccfg["pruned_output_dir"]))

    rows = []
    for variant_name, (model, size_mb) in variants.items():
        print(f"\n--- Evaluating variant: {variant_name} ---")
        with mlflow.start_run(run_name=f"rq2_{variant_name}_eval"):
            with GreenTracker(project_name=f"rq2_{variant_name}_eval"):
                metrics = evaluate_model(
                    model, tokenizer=tokenizer, cfg=cfg,
                    num_threads=ccfg.get("eval_num_threads"),
                    eval_subset_size=ccfg.get("eval_subset_size"),
                    deterministic=True,
                )
            metrics["variant"] = variant_name
            metrics["model_size_mb"] = round(size_mb, 3)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_param("variant", variant_name)
            rows.append(metrics)
            print(metrics)

    import csv
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(ccfg["results_file"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRQ2 results written to {ccfg['results_file']}")


if __name__ == "__main__":
    run()
