"""Shared evaluation utility: accuracy, F1 and inference latency for a
sequence-classification checkpoint on the IMDb test subset.

Used by both the RQ2 compression benchmark and the RQ3 capping/scheduling
benchmark so every variant is measured the same way.
"""
import time

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from src.utils.config import load_config


def _load_eval_set(cfg):
    dataset = load_from_disk(cfg["dataset"]["raw_dir"])
    eval_split = dataset["test"].shuffle(seed=cfg["dataset"]["seed"]).select(
        range(cfg["dataset"]["eval_subset_size"])
    )
    return eval_split


def evaluate_model(model, tokenizer=None, cfg=None, num_threads=None, batch_size=None):
    """Run inference over the eval subset and return a metrics dict.

    Parameters
    ----------
    model : a torch.nn.Module / HF model already loaded in eval mode
    tokenizer : HF tokenizer (defaults to the base model tokenizer if None)
    cfg : parsed configs/config.yaml (loaded if None)
    num_threads : if set, calls torch.set_num_threads(num_threads) before
        timing inference -- used by the RQ3 capping benchmark.
    batch_size : override the eval batch size from config.
    """
    cfg = cfg or load_config()
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg["model"]["base_model"])
    batch_size = batch_size or cfg["training"]["per_device_eval_batch_size"]
    max_length = cfg["dataset"]["max_seq_length"]

    eval_set = _load_eval_set(cfg)
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
