"""Shared evaluation utility: accuracy, F1 and inference latency for a
sequence-classification checkpoint on the IMDb test subset.

Used by both the RQ2 compression benchmark and the RQ3 capping/scheduling
benchmark so every variant is measured the same way.
"""
import time

import numpy as np
import torch
from datasets import load_from_disk
from datasets import disable_caching as _disable_hf_datasets_caching
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from src.utils.config import load_config

# By default, HF `datasets` writes a cache-*.arrow file next to the source
# dataset for every .shuffle()/.select()/.map() call, so it can skip
# recomputing that transform next time. That's convenient, but our source
# dataset (data/raw/imdb) is DVC-tracked, so those byproduct files kept
# changing its tracked hash every time evaluate_model() ran -- with no
# actual change to the data itself. Disabling the on-disk cache here (once,
# for the whole process) stops that drift; transforms are recomputed
# in-memory each run instead, which is cheap at our subset sizes (<=500
# examples).
_disable_hf_datasets_caching()


def _load_eval_set(cfg, subset_size=None):
    subset_size = subset_size or cfg["dataset"]["eval_subset_size"]
    dataset = load_from_disk(cfg["dataset"]["raw_dir"])
    eval_split = dataset["test"].shuffle(seed=cfg["dataset"]["seed"]).select(
        range(subset_size)
    )
    return eval_split


def evaluate_model(model, tokenizer=None, cfg=None, num_threads=None, batch_size=None,
                    eval_subset_size=None, deterministic=False):
    """Run inference over the eval subset and return a metrics dict.

    Parameters
    ----------
    model : a torch.nn.Module / HF model already loaded in eval mode
    tokenizer : HF tokenizer (defaults to the base model tokenizer if None)
    cfg : parsed configs/config.yaml (loaded if None)
    num_threads : if set, calls torch.set_num_threads(num_threads) before
        timing inference -- used by the RQ3 capping benchmark. Also pins the
        thread count so multi-threaded CPU matmul reduction order (and thus
        floating-point rounding) is identical across machines/runs, which
        avoids RQ2 accuracy drifting a few points from run to run.
    batch_size : override the eval batch size from config.
    eval_subset_size : override the number of eval examples from config
        (RQ2 uses a larger subset than RQ3 to be less sensitive to the
        occasional prediction flip caused by CPU floating-point rounding).
    deterministic : if True, requests bitwise-deterministic CPU ops via
        torch.use_deterministic_algorithms (best-effort; some ops may not
        support it, in which case we fall back to non-deterministic mode
        rather than crashing the whole evaluation).
    """
    cfg = cfg or load_config()
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # pragma: no cover -- best-effort only
            print(f"[evaluate_model] could not enable fully deterministic "
                  f"algorithms ({exc}); continuing without it.")

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    batch_size = batch_size or cfg["training"]["per_device_eval_batch_size"]
    max_length = cfg["dataset"]["max_seq_length"]

    eval_set = _load_eval_set(cfg, subset_size=eval_subset_size)
    texts = eval_set["text"]
    labels = np.array(eval_set["label"])

    model.eval()
    all_preds = []
    batch_latencies_ms = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            t0 = time.perf_counter()
            outputs = model(**inputs)
            t1 = time.perf_counter()
            batch_latencies_ms.append((t1 - t0) * 1000.0)

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    all_preds = np.array(all_preds)
    return {
        "accuracy": float(accuracy_score(labels, all_preds)),
        "f1": float(f1_score(labels, all_preds, average="weighted")),
        "avg_inference_latency_ms": float(np.mean(batch_latencies_ms)),
        "p95_inference_latency_ms": float(np.percentile(batch_latencies_ms, 95)),
        "n_eval_examples": int(len(texts)),
        "eval_batch_size": int(batch_size),
    }
