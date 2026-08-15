#!/usr/bin/env python3
"""Online-training GPU service characterization on CIFAR-10 images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet50_Weights, resnet50


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def elapsed(begin: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return float(begin.elapsed_time(end))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument(
        "--duty-cycle", type=float, choices=(0.5, 0.9), required=True
    )
    parser.add_argument("--microbatch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--preload-samples", type=int, default=256)
    args = parser.parse_args()
    if args.trial <= 0 or args.duration <= 0 or args.preload_samples <= 0:
        parser.error("trial, duration, and preload-samples must be positive")

    torch.cuda.set_device(0)
    torch.manual_seed(0x7A1 + args.trial)
    np.random.seed(0x7A1 + args.trial)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).cuda().half().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataset = CIFAR10(root=args.dataset_root, train=True, download=False)
    transform = weights.transforms()
    sample_count = min(args.preload_samples, len(dataset))
    selected = np.random.default_rng(0xC1FA + args.trial).choice(
        len(dataset), size=sample_count, replace=False
    )
    images = []
    labels = []
    for index in selected:
        image, label = dataset[int(index)]
        images.append(transform(image).half())
        labels.append(label)
    images_gpu = torch.stack(images).cuda()
    labels_gpu = torch.tensor(labels, dtype=torch.long, device="cuda")

    cursor = 0

    def one_unit() -> dict[str, float]:
        nonlocal cursor
        indices = (torch.arange(args.microbatch, device="cuda") + cursor) % sample_count
        cursor = (cursor + args.microbatch) % sample_count
        batch = images_gpu.index_select(0, indices)
        target = labels_gpu.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        logits = model(batch)
        loss = criterion(logits.float(), target)
        forward_end.record()
        loss.backward()
        backward_end.record()
        optimizer.step()
        end.record()
        end.synchronize()
        return {
            "forward_ms": elapsed(start, forward_end),
            "backward_ms": elapsed(forward_end, backward_end),
            "optimizer_ms": elapsed(backward_end, end),
            "gpu_ms": elapsed(start, end),
            "wall_ms": (time.perf_counter_ns() - wall_start) / 1e6,
            "loss": float(loss.detach().item()),
        }

    calibration = [one_unit()["gpu_ms"] for _ in range(args.warmup)]
    if not calibration:
        raise RuntimeError("warmup must produce a calibration sample")
    interval_ns = round(np.median(calibration) * 1e6 / args.duty_cycle)
    records = []
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + round(args.duration * 1e9)
    target_ns = start_ns
    while time.perf_counter_ns() < deadline_ns:
        wait_until(target_ns)
        actual_start_ns = time.perf_counter_ns()
        item = one_unit()
        item.update(
            {
                "arrival_ns": target_ns,
                "actual_start_ns": actual_start_ns,
                "scheduler_lateness_us": (actual_start_ns - target_ns) / 1e3,
            }
        )
        records.append(item)
        target_ns += interval_ns
    end_ns = time.perf_counter_ns()

    dataset_archive = Path(args.dataset_root) / "cifar-10-python.tar.gz"
    archive_hash = None
    if dataset_archive.is_file():
        archive_hash = sha256(dataset_archive)
    result = {
        "schema": "online-training-qualification-v1",
        "model": "torchvision.resnet50",
        "weights": str(weights),
        "dataset": "CIFAR-10 train",
        "dataset_root": args.dataset_root,
        "dataset_archive_sha256": archive_hash,
        "trial": args.trial,
        "microbatch": args.microbatch,
        "target_duty_cycle": args.duty_cycle,
        "duration_target_s": args.duration,
        "duration_actual_s": (end_ns - start_ns) / 1e9,
        "target_interval_ms": interval_ns / 1e6,
        "preloaded_real_samples": sample_count,
        "preprocessing": repr(transform),
        "calibration_gpu_ms": stats(calibration),
        "forward_ms": stats([item["forward_ms"] for item in records]),
        "backward_ms": stats([item["backward_ms"] for item in records]),
        "optimizer_ms": stats([item["optimizer_ms"] for item in records]),
        "unit_gpu_ms": stats([item["gpu_ms"] for item in records]),
        "unit_wall_ms": stats([item["wall_ms"] for item in records]),
        "samples_per_s": len(records) * args.microbatch / ((end_ns - start_ns) / 1e9),
        "records": records,
        "pass": bool(records) and all(np.isfinite(item["loss"]) for item in records),
    }
    if not result["pass"]:
        raise RuntimeError("training qualification produced invalid measurements")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(output)
    print(
        f"[TRAIN-QUAL] PASS mb={args.microbatch} duty={args.duty_cycle} "
        f"n={len(records)} gpu_p99={result['unit_gpu_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
