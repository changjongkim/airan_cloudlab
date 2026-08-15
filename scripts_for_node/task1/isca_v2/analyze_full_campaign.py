#!/usr/bin/env python3
"""Strict validator and summary builder for the regular full-day core campaign."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
    }


def validate_nrx(root: Path) -> list[dict]:
    paths = sorted((root / "02_nrx_stack").glob("trial_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected 5 NRx trials, found {len(paths)}")
    rows = []
    for path in paths:
        value = json.loads(path.read_text())
        if not value.get("pass") or value["iterations"] != 10000:
            raise RuntimeError(f"invalid NRx trial: {path}")
        for variant, item in value["variants"].items():
            metric = item["metrics"]["gpu_ms"]
            if metric["n"] != 10000:
                raise RuntimeError(f"invalid sample count: {path} {variant}")
            if "correctness" in item and not all(
                output["allclose_rtol_1e-3_atol_1e-3"]
                for output in item["correctness"].values()
            ):
                raise RuntimeError(f"correctness failure: {path} {variant}")
            rows.append({
                "trial": value["trial"], "variant": variant,
                **{f"gpu_ms_{key}": metric[key] for key in (
                    "mean", "p50", "p95", "p99", "p99_9", "min", "max"
                )},
            })
    return rows


def validate_workloads(root: Path) -> list[dict]:
    directory = root / "03_workload_qualification"
    paths = sorted(
        path for path in directory.glob("*.json")
        if path.name != "DATA_MANIFEST.json"
    )
    if len(paths) != 48:
        raise RuntimeError(f"expected 48 workload trials, found {len(paths)}")
    rows = []
    for path in paths:
        value = json.loads(path.read_text())
        if not value.get("pass") or value["duration_actual_s"] < 89.1:
            raise RuntimeError(f"invalid workload trial: {path}")
        if path.name.startswith("qwen_"):
            metric = value["service_gpu_ms"]
            kind = "qwen"
            detail = value["mode"]
            topology = path.name.split("_")[1]
            work_rate = value["units_per_s"]
            provenance = value["provenance"]["sha256"]
        elif path.name.startswith("training_"):
            metric = value["unit_gpu_ms"]
            kind = "training"
            detail = f"microbatch_{value['microbatch']}"
            topology = path.name.split("_")[1]
            work_rate = value["samples_per_s"]
            provenance = value["dataset_archive_sha256"]
        else:
            raise RuntimeError(f"unknown workload artifact: {path}")
        rows.append({
            "file": path.name, "kind": kind, "detail": detail,
            "topology": topology, "trial": value["trial"],
            "duty_cycle": value["target_duty_cycle"],
            "service_mean_ms": metric["mean"],
            "service_p99_ms": metric["p99"],
            "service_p99_9_ms": metric["p99_9"],
            "work_rate": work_rate, "data_sha256": provenance,
        })
    return rows


def validate_capacity(root: Path) -> list[dict]:
    paths = sorted((root / "04_nrx_capacity").glob("capacity_*.json"))
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 capacity files, found {len(paths)}")
    rows = []
    for path in paths:
        value = json.loads(path.read_text())
        if not value.get("pass") or len(value["configurations"]) != 4:
            raise RuntimeError(f"invalid capacity result: {path}")
        for configuration in value["configurations"]:
            saturation = configuration["saturation"]
            if saturation["duration_actual_s"] < 59.4:
                raise RuntimeError(f"short saturation window: {path}")
            rows.append({
                "topology": value["topology"], "trial": value["trial"],
                "replicas": configuration["replicas"], "kind": "saturation",
                "load_fraction": None,
                "rate_slots_s": saturation["throughput_slots_per_s"],
                "throughput_slots_s": saturation["throughput_slots_per_s"],
                "latency_mean_ms": saturation["service_ms"]["mean"],
                "latency_p99_ms": saturation["service_ms"]["p99"],
                "miss_5ms": None, "backlog": None,
            })
            for item in configuration["open_loop"]:
                if item["duration_s"] < 59.4:
                    raise RuntimeError(f"short open-loop window: {path}")
                rows.append({
                    "topology": value["topology"], "trial": value["trial"],
                    "replicas": configuration["replicas"], "kind": "open_loop",
                    "load_fraction": item["load_fraction_of_measured_capacity"],
                    "rate_slots_s": item["arrival_rate_slots_per_s"],
                    "throughput_slots_s": None,
                    "latency_mean_ms": item["latency_ms"]["mean"],
                    "latency_p99_ms": item["latency_ms"]["p99"],
                    "miss_5ms": item["deadline_miss_ratio"]["5ms"],
                    "backlog": item["backlog_at_window_end"],
                })
    return rows


def validate_fiveway(root: Path) -> list[dict]:
    directory = root / "05_fiveway_compute"
    rows = []
    for approach in ("mps", "mig_local", "mig_mps", "p2p", "gdr"):
        paths = sorted(directory.glob(f"{approach}/load_*/trial_*/*.json"))
        if len(paths) != 20:
            raise RuntimeError(f"expected 20 {approach} files, found {len(paths)}")
        for path in paths:
            value = json.loads(path.read_text())
            if not value.get("pass", True):
                raise RuntimeError(f"failed result: {path}")
            load = float(path.parts[-3].split("_", 1)[1])
            if approach == "gdr":
                e2e_raw = value["raw_ms"]
                sojourn_raw = value["raw_sojourn_ms"]
                l1 = None
                transport = "GPUDirect RDMA"
            else:
                e2e_raw = [x["e2e_ms"] for x in value["raw"]]
                sojourn_raw = [x["sojourn_ms"] for x in value["raw"]]
                l1 = value["metrics"]["l1_active_ms"]["mean"]
                transport = (
                    "CUDA P2P" if approach == "p2p" else
                    "CUDA IPC" if approach in ("mps", "mig_mps") else
                    "in-process binding"
                )
            e2e = stats(e2e_raw); sojourn = stats(sojourn_raw)
            rows.append({
                "approach": approach, "load_fraction": load,
                "trial": value["trial"],
                "arrival_rate_slots_s": value["arrival_rate_slots_per_s"],
                "samples": value["iterations"], "transport": transport,
                "l1_active_mean_ms": l1,
                **{f"e2e_ms_{key}": e2e[key] for key in e2e},
                **{f"sojourn_ms_{key}": sojourn[key] for key in sojourn},
                "miss_5ms": float(np.mean(np.asarray(sojourn_raw) > 5.0)),
            })
    return rows


def median_rows(rows: list[dict], group: tuple[str, ...], values: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[item] for item in group)
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in sorted(groups.items()):
        value = dict(zip(group, key))
        for field in values:
            numbers = [float(item[field]) for item in items if item[field] is not None]
            value[f"median_{field}"] = statistics.median(numbers) if numbers else None
        value["trials"] = len(items)
        output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    nrx = validate_nrx(root)
    workloads = validate_workloads(root)
    capacity = validate_capacity(root)
    fiveway = validate_fiveway(root)
    write_csv(output / "NRX_STACK.csv", nrx)
    write_csv(output / "WORKLOAD_QUALIFICATION.csv", workloads)
    write_csv(output / "NRX_CAPACITY.csv", capacity)
    write_csv(output / "FIVEWAY_COMPUTE.csv", fiveway)
    fiveway_median = median_rows(
        fiveway, ("approach", "load_fraction"),
        ("e2e_ms_mean", "e2e_ms_p99", "sojourn_ms_p99", "miss_5ms"),
    )
    write_csv(output / "FIVEWAY_MEDIAN.csv", fiveway_median)
    report = [
        "# Full-day core campaign validation",
        "",
        "This report excludes the earlier quick campaign.",
        "",
        f"- NRx regular trial rows: {len(nrx)} (5 trials × 5 variants)",
        f"- Qualified workload trials: {len(workloads)}",
        f"- Capacity rows: {len(capacity)}",
        f"- Five-way regular trials: {len(fiveway)}",
        "",
        "## Five-way median results",
        "",
        "| approach | load | E2E mean ms | E2E p99 ms | sojourn p99 ms | miss >5ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in fiveway_median:
        report.append(
            f"| {row['approach']} | {row['load_fraction']:.2f} | "
            f"{row['median_e2e_ms_mean']:.3f} | {row['median_e2e_ms_p99']:.3f} | "
            f"{row['median_sojourn_ms_p99']:.3f} | {row['median_miss_5ms']:.6f} |"
        )
    report.extend([
        "", "## Scope", "",
        "This is the regular NRx/workload/capacity/five-way core. It does not by itself "
        "mark the integrated DART reservation, epoch commit, fallback, and lease campaign complete.",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "validated": True, "quick_results_used": False,
        "nrx_rows": len(nrx), "workload_rows": len(workloads),
        "capacity_rows": len(capacity), "fiveway_trials": len(fiveway),
    }
    (output / "VALIDATION.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[FULL-ANALYSIS] PASS output={output}", flush=True)


if __name__ == "__main__":
    main()
