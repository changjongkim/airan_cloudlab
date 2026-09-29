#!/usr/bin/env python3.11
"""Summarize GPU memory of the multi-GPU and single-GPU placements (C177).

Inputs: the memory probe of job 59066149 (per-GPU samples while the
multi-GPU pipeline ran 600 periods and the single-GPU pipeline with one
receiver per cell ran 60 and 600 periods), the receiver measurement of job
59066534, the coordinator logs of the single-GPU runs that ran out of memory,
and the per-run memory samples of the C177 campaign.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_peaks(samples: Path, phases: Path) -> dict:
    spans = {}
    for line in phases.read_text().splitlines():
        parts = line.split()
        spans.setdefault(parts[0], {})[parts[1]] = dt.datetime.strptime(" ".join(parts[-2:]), "%Y/%m/%d %H:%M:%S")
    peaks = {}
    for line in samples.read_text().splitlines():
        stamp, gpu, mib = (field.strip() for field in line.split(","))
        when = dt.datetime.strptime(stamp[:19], "%Y/%m/%d %H:%M:%S")
        for name, span in spans.items():
            if span["start"] <= when <= span["end"]:
                key = f"GPU{gpu}"
                peaks.setdefault(name, {})[key] = max(peaks.get(name, {}).get(key, 0.0), int(mib) / 1024)
    return {name: {gpu: round(value, 2) for gpu, value in sorted(values.items())} for name, values in peaks.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--receiver", type=Path, required=True)
    parser.add_argument("--campaign-jobs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = args.raw / "c177memprobe_j59066149_mem_gpu.csv"
    phases = args.raw / "c177memprobe_j59066149_phases.log"
    peaks = probe_peaks(samples, phases)
    multi = peaks["multi600"]

    failures = {}
    for name in ("c177_steady_backstop_natural_j59065521_coordinator.log",
                 "c177memsingle600_steady_backstop_natural_j59066149_coordinator.log"):
        text = (args.raw / name).read_text(errors="replace")
        failures[name] = {"out_of_memory": "out of memory" in text, "sha256": sha256(args.raw / name)}

    runs = {}
    for path in sorted(p for job in args.campaign_jobs for p in args.raw.glob(f"c17[78]*_j{job}_gpu_memory.csv")):
        values = [float(line.split(",")[1]) for line in path.read_text().splitlines() if line.strip()]
        runs[path.name.replace("_gpu_memory.csv", "")] = round(max(values) / 1024, 2)

    receiver = json.loads(args.receiver.read_text())
    result = {
        "schema": "softwall-c177-memory-v1",
        "gpu_capacity_gib": 80.0,
        "multi_gpu_600_periods_peak_gib": multi,
        "multi_gpu_600_periods_sum_gib": round(sum(multi.values()), 2),
        "single_gpu_one_receiver_per_cell": {
            "60_periods_peak_gib": peaks["single60"].get("GPU0"),
            "600_periods": "out of memory in the coordinator while it builds one receiver per cell",
            "failed_runs": failures,
        },
        "receiver_gib": receiver["per_receiver_gib"],
        "single_gpu_shared_recovery_receiver_peak_gib": runs,
        "single_gpu_shared_recovery_receiver_max_gib": max(runs.values()) if runs else None,
        "inputs_sha256": {path.name: sha256(path) for path in (samples, phases, args.receiver)},
        "claim_scope": ("Device memory in use on A100-SXM4-80GB GPUs under MPS (nvidia-smi samples every 0.5-1 s). "
                        "The multi-GPU placement holds three owners on GPU0, one owner on GPU1, the four-receiver "
                        "recovery lane and Qwen on GPU2, and NeuralRx on GPU3."),
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("multi_gpu_600_periods_peak_gib", "multi_gpu_600_periods_sum_gib",
                                            "single_gpu_one_receiver_per_cell", "receiver_gib",
                                            "single_gpu_shared_recovery_receiver_max_gib")}, indent=1))


if __name__ == "__main__":
    main()
