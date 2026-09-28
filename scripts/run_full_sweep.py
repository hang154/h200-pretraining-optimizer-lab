#!/usr/bin/env python3
"""Run every requested AdamW stage, selecting candidates from measured val loss."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).with_name("train_1b_trial.py")
RUNS = ROOT / "artifacts" / "runs"
CONFIGS = ROOT / "artifacts" / "resolved_configs"


def manifest(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "manifest.json").read_text())


def execute(config: dict, gpu: int) -> dict:
    run_id = config["run_id"]
    target = RUNS / run_id / "manifest.json"
    if target.exists():
        return manifest(run_id)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS / f"{run_id}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    subprocess.run([sys.executable, str(TRAIN), "--config-json", str(config_path)], env=env, check=True)
    return manifest(run_id)


def parallel(configs: list[dict], gpus: list[int]) -> list[dict]:
    slots = {gpu: [] for gpu in gpus}
    for index, config in enumerate(configs):
        slots[gpus[index % len(gpus)]].append(config)

    def worker(gpu: int, queue: list[dict]) -> list[dict]:
        return [execute(config, gpu) for config in queue]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        batches = list(pool.map(lambda pair: worker(*pair), slots.items()))
    return [item for batch in batches for item in batch]


def cfg(run_id: str, **kwargs) -> dict:
    base = {
        "run_id": run_id,
        "output_dir": str(RUNS),
        "seed": 42,
        "lr": 4e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "warmup_ratio": 0.01,
        "scheduler": "cosine",
        "sequence_length": 1024,
        "gradient_accumulation": 2,
        "steps": 8,
        "validation_batches": 4,
        "phase": "scratch",
        "save_checkpoint": False,
    }
    base.update(kwargs)
    return base


def best(rows: list[dict], count: int) -> list[dict]:
    return sorted(rows, key=lambda row: row["validation_loss"])[:count]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="2,3")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    RUNS.mkdir(parents=True, exist_ok=True)

    stage1 = parallel(
        [cfg(f"s1-lr-{lr:g}", lr=lr) for lr in (1e-4, 2e-4, 4e-4, 6e-4, 8e-4)], gpus
    )
    top_lrs = [row["lr"] for row in best(stage1, 2)]
    stage2_configs = []
    for lr in top_lrs:
        for beta2 in (0.95, 0.98, 0.99):
            for wd in (0.05, 0.1):
                stage2_configs.append(cfg(f"s2-lr{lr:g}-b{beta2:g}-wd{wd:g}", lr=lr, beta2=beta2, weight_decay=wd, steps=10))
    stage2 = parallel(stage2_configs, gpus)
    anchor = best(stage2, 1)[0]

    warmup = parallel([
        cfg(f"warmup-{value:g}", lr=anchor["lr"], beta2=anchor["beta2"], weight_decay=anchor["weight_decay"], warmup_ratio=value, steps=12)
        for value in (0.005, 0.01, 0.02)
    ], gpus)
    batch = parallel([
        cfg(f"batch-proxy-{accum}", lr=anchor["lr"], beta2=anchor["beta2"], weight_decay=anchor["weight_decay"], gradient_accumulation=accum, steps=8)
        for accum in (1, 2, 4)
    ], gpus)
    anchor_warmup = best(warmup, 1)[0]["warmup_ratio"]
    anchor_accum = best(batch, 1)[0]["gradient_accumulation"]

    candidates = best(stage2, 3)
    stage3 = parallel([
        cfg(f"s3-long-{index}", lr=row["lr"], beta2=row["beta2"], weight_decay=row["weight_decay"], warmup_ratio=anchor_warmup, gradient_accumulation=anchor_accum, steps=20)
        for index, row in enumerate(candidates)
    ], gpus)
    finalists = best(stage3, 2)
    stage4_configs = []
    for index, row in enumerate(finalists):
        for seed in (17, 42, 31415):
            stage4_configs.append(cfg(f"s4-final{index}-seed{seed}", seed=seed, lr=row["lr"], beta2=row["beta2"], weight_decay=row["weight_decay"], warmup_ratio=anchor_warmup, gradient_accumulation=anchor_accum, steps=24, save_checkpoint=index == 0 and seed == 42))
    stage4 = parallel(stage4_configs, gpus)
    checkpoint = RUNS / "s4-final0-seed42" / "model.pt"

    cpt = parallel([
        cfg(f"cpt-lr-{lr:g}", phase="cpt", dataset_revision="structured-high-quality-v1", init_checkpoint=str(checkpoint), lr=lr, beta2=finalists[0]["beta2"], weight_decay=finalists[0]["weight_decay"], warmup_ratio=0.01, steps=16)
        for lr in (finalists[0]["lr"] / 4, finalists[0]["lr"] / 2, finalists[0]["lr"])
    ], gpus)

    all_rows = stage1 + stage2 + warmup + batch + stage3 + stage4 + cpt
    fields = ["run_id", "phase", "seed", "lr", "beta2", "weight_decay", "warmup_ratio", "global_batch_tokens", "dataset_tokens", "start_validation_loss", "validation_loss", "tokens_per_second", "wall_time", "peak_hbm_bytes", "model_params", "manifest_sha256"]
    with (ROOT / "artifacts" / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "stage_counts": {"lr": len(stage1), "beta2_wd": len(stage2), "warmup": len(warmup), "batch": len(batch), "long": len(stage3), "multiseed": len(stage4), "cpt": len(cpt)},
        "best_scratch": best(stage4, 1)[0],
        "best_cpt": best(cpt, 1)[0],
        "all_completed": len(all_rows) == 35,
    }
    (ROOT / "artifacts" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"all_completed": summary["all_completed"], "runs": len(all_rows)}))
    return 0 if summary["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
