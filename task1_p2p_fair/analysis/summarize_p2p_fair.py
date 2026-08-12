#!/usr/bin/env python3
"""Validate, summarize, and plot the fair same-MIG versus cross-MIG P2P trials."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
TRIAL_CSV = ROOT / "P2P_FAIR_TRIALS.csv"
AGGREGATE_CSV = ROOT / "P2P_FAIR_AGGREGATE.csv"

CONFIGS = {
    "l1_only_4g": ("topology_a", "standalone", 1),
    "same_overlap_4g": ("topology_a", "same", 1),
    "l1_only_2g": ("topology_b", "standalone", 1),
    "cross_p2p_2g2g": ("topology_b", "p2p", 2),
}
BASELINE = {
    "l1_only_4g": "l1_only_4g",
    "same_overlap_4g": "l1_only_4g",
    "l1_only_2g": "l1_only_2g",
    "cross_p2p_2g2g": "l1_only_2g",
}
QWEN_RE = re.compile(r"\[Qwen\] progress: (\d+) iters, ([0-9.]+) it/s")


def find_trials() -> list[tuple[str, Path]]:
    trials = [("trial1", ROOT)]
    trials.extend(
        (path.name, path)
        for path in sorted(ROOT.glob("trial[0-9]*"))
        if path.is_dir()
    )
    for name, path in trials:
        if not (path / "COMPLETE").is_file():
            raise RuntimeError(f"{name}: missing COMPLETE marker")
    if len(trials) != 3:
        raise RuntimeError(f"expected exactly 3 complete trials, found {len(trials)}")
    return trials


def validate_p2p_gate() -> None:
    path = ROOT / "p2p_gate_2g2g" / "p2p_copy_gate.json"
    data = json.loads(path.read_text())
    if data.get("visible_cuda_devices") != 2:
        raise RuntimeError(f"{path}: expected two visible devices")
    if data.get("can_access_peer") != {"0_to_1": 1, "1_to_0": 1}:
        raise RuntimeError(f"{path}: peer access is not bidirectional")
    directions = data.get("directions", [])
    expected = [(0, 1, 1_415_232), (1, 0, 1_257_984)]
    if len(directions) != len(expected):
        raise RuntimeError(f"{path}: expected two directions")
    for direction, (source, destination, size) in zip(directions, expected):
        observed = (
            direction.get("source_device"),
            direction.get("destination_device"),
            direction.get("payload_bytes"),
        )
        if observed != (source, destination, size):
            raise RuntimeError(f"{path}: direction mismatch {observed}")
        records = direction.get("records", [])
        if direction.get("iterations") != 10 or len(records) != 10:
            raise RuntimeError(f"{path}: expected ten measured copies")
        if [record.get("sequence") for record in records] != list(range(1, 11)):
            raise RuntimeError(f"{path}: non-contiguous P2P sequences")
        if not all(record.get("verified") is True for record in records):
            raise RuntimeError(f"{path}: failed integrity record")
        latencies = np.asarray([record["latency_us"] for record in records])
        if not math.isclose(
            direction["latency_us"]["mean"],
            float(latencies.mean()),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"{path}: P2P mean does not match raw records")


def load_one(path: Path, config: str, expected_mode: str, visible_devices: int) -> dict:
    files = list(path.glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(f"{path}: expected one JSON, found {len(files)}")
    data = json.loads(files[0].read_text())
    expected = {
        "label": config,
        "mode": expected_mode,
        "iterations": 30,
        "warmup": 20,
        "ring_depth": 1 if expected_mode == "standalone" else 2,
        "visible_cuda_devices": visible_devices,
        "fwd_bytes": 1_415_232,
        "bwd_bytes": 1_257_984,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"{files[0]}: {key}={data.get(key)!r}, expected {value!r}")
    if len(data.get("raw", [])) != 30:
        raise RuntimeError(f"{files[0]}: raw sample count is not 30")
    sequences = [sample.get("sequence") for sample in data["raw"]]
    if sequences != list(range(1, 31)):
        raise RuntimeError(f"{files[0]}: non-contiguous sequence numbers")
    for name, saved in data["metrics"].items():
        raw_values = [sample[name] for sample in data["raw"] if name in sample]
        if not raw_values:
            if saved != {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}:
                raise RuntimeError(f"{files[0]}: invalid empty summary for {name}")
            continue
        array = np.asarray(raw_values, dtype=np.float64)
        recalculated = {
            "n": int(array.size),
            "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)),
        }
        for stat, value in recalculated.items():
            if stat == "n":
                matches = saved[stat] == value
            else:
                matches = math.isclose(saved[stat], value, rel_tol=0.0, abs_tol=1e-9)
            if not matches:
                raise RuntimeError(
                    f"{files[0]}: {name}.{stat}={saved[stat]}, recalculated={value}"
                )
    completed = [sample["completed_ns"] for sample in data["raw"]]
    intervals = np.diff(np.asarray(completed, dtype=np.float64)) / 1e6
    if intervals.size != 29:
        raise RuntimeError(f"{files[0]}: invalid completion interval count")
    if not math.isclose(
        data["completion_interval_ms"]["mean"],
        float(intervals.mean()),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(f"{files[0]}: completion interval does not match raw data")
    expected_throughput = 30 / float(data["wall_seconds"])
    if not math.isclose(
        data["completion_throughput_slots_s"],
        expected_throughput,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{files[0]}: throughput does not match wall time")
    return data


def last_qwen(path: Path) -> tuple[int, float]:
    matches = QWEN_RE.findall(path.read_text(errors="replace"))
    if not matches:
        raise RuntimeError(f"{path}: no Qwen progress records")
    iterations, itps = matches[-1]
    return int(iterations), float(itps)


def metric(data: dict, name: str, stat: str = "mean") -> float:
    value = data["metrics"][name][stat]
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {name}.{stat}")
    return float(value)


def collect() -> list[dict]:
    rows: list[dict] = []
    for trial_name, trial_dir in find_trials():
        data_by_config: dict[str, dict] = {}
        qwen_by_topology: dict[str, tuple[int, float]] = {}
        for topology in ("topology_a", "topology_b"):
            qwen_by_topology[topology] = last_qwen(trial_dir / topology / "qwen.log")
        for config, (topology, mode, visible_devices) in CONFIGS.items():
            data_by_config[config] = load_one(
                trial_dir / topology / config, config, mode, visible_devices
            )

        for config, (topology, mode, _) in CONFIGS.items():
            data = data_by_config[config]
            baseline_mean = metric(data_by_config[BASELINE[config]], "l1_active_ms")
            l1_mean = metric(data, "l1_active_ms")
            qwen_iterations, qwen_itps = qwen_by_topology[topology]
            has_nrx = mode != "standalone"
            rows.append(
                {
                    "trial": trial_name,
                    "config": config,
                    "mode": mode,
                    "l1_active_mean_ms": l1_mean,
                    "l1_active_p50_ms": metric(data, "l1_active_ms", "p50"),
                    "l1_active_p95_ms": metric(data, "l1_active_ms", "p95"),
                    "l1_active_p99_ms": metric(data, "l1_active_ms", "p99"),
                    "own_l1_baseline_mean_ms": baseline_mean,
                    "l1_slowdown_x": l1_mean / baseline_mean,
                    "nrx_mean_ms": metric(data, "nrx_ms") if has_nrx else "",
                    "transport_mean_us": (
                        metric(data, "transport_us") if mode == "p2p" else ""
                    ),
                    "e2e_mean_ms": metric(data, "e2e_ms"),
                    "e2e_p99_ms": metric(data, "e2e_ms", "p99"),
                    "completion_throughput_slots_s": float(
                        data["completion_throughput_slots_s"]
                    ),
                    "qwen_iterations_last": qwen_iterations,
                    "qwen_itps_last": qwen_itps,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    numeric_fields = (
        "l1_active_mean_ms",
        "l1_active_p99_ms",
        "own_l1_baseline_mean_ms",
        "l1_slowdown_x",
        "nrx_mean_ms",
        "transport_mean_us",
        "e2e_mean_ms",
        "e2e_p99_ms",
        "completion_throughput_slots_s",
        "qwen_itps_last",
    )
    output: list[dict] = []
    for config in CONFIGS:
        selected = [row for row in rows if row["config"] == config]
        for field in numeric_fields:
            values = [float(row[field]) for row in selected if row[field] != ""]
            if not values:
                continue
            output.append(
                {
                    "config": config,
                    "metric": field,
                    "n": len(values),
                    "mean": mean(values),
                    "stdev": stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )
    return output


def plot(rows: list[dict]) -> Path:
    FIGURE_DIR.mkdir(exist_ok=True)
    trial_names = [name for name, _ in find_trials()]
    by_key = {(row["trial"], row["config"]): row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), constrained_layout=True)

    overlap_configs = ["same_overlap_4g", "cross_p2p_2g2g"]
    labels = ["Same 4g\nL1 + NRx", "Cross 2g + 2g\nDirect P2P"]
    colors = ["#d95f02", "#1b9e77"]
    x = np.arange(2)
    slowdowns = np.array(
        [[by_key[(trial, cfg)]["l1_slowdown_x"] for trial in trial_names] for cfg in overlap_configs]
    )
    means = slowdowns.mean(axis=1)
    axes[0].bar(x, means, color=colors, width=0.62, zorder=2)
    for idx in range(2):
        jitter = np.linspace(-0.07, 0.07, len(trial_names))
        axes[0].scatter(
            np.full(len(trial_names), x[idx]) + jitter,
            slowdowns[idx],
            color="black",
            s=27,
            zorder=3,
            label="Individual trial" if idx == 0 else None,
        )
        axes[0].text(x[idx], means[idx] + 1.2, f"{means[idx]:.2f}×", ha="center", weight="bold")
    axes[0].axhline(1, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("L1 CUDA-elapsed slowdown vs own baseline (×)")
    axes[0].set_title("Isolation under actual L1–NRx overlap")
    axes[0].grid(axis="y", alpha=0.25, zorder=0)

    configs = ["l1_only_4g", "same_overlap_4g", "l1_only_2g", "cross_p2p_2g2g"]
    short = ["4g\nL1 only", "4g\nshared", "2g\nL1 only", "2g+2g\nP2P"]
    config_colors = ["#7570b3", "#d95f02", "#66a61e", "#1b9e77"]
    active_means = np.array(
        [[by_key[(trial, cfg)]["l1_active_mean_ms"] for trial in trial_names] for cfg in configs]
    )
    active_p99 = np.array(
        [[by_key[(trial, cfg)]["l1_active_p99_ms"] for trial in trial_names] for cfg in configs]
    )
    positions = np.arange(4)
    axes[1].bar(positions, active_means.mean(axis=1), color=config_colors, width=0.62, zorder=2)
    axes[1].scatter(
        np.repeat(positions, len(trial_names)),
        active_p99.flatten(),
        marker="D",
        s=30,
        facecolors="white",
        edgecolors="black",
        zorder=3,
        label="Per-trial p99",
    )
    for idx, value in enumerate(active_means.mean(axis=1)):
        axes[1].text(idx, value + 3.2, f"{value:.2f}", ha="center", weight="bold")
    axes[1].set_xticks(positions, short)
    axes[1].set_ylabel("L1 CUDA-stream elapsed time (ms)")
    axes[1].set_title("Absolute L1 elapsed time (bars: trial mean)")
    axes[1].grid(axis="y", alpha=0.25, zorder=0)
    axes[1].legend(loc="upper left")

    fig.suptitle(
        "Fair P2P comparison: equal aggregate 4g, Qwen isolated on 3g, N=30 × 3 trials",
        fontsize=13,
        weight="bold",
    )
    output = FIGURE_DIR / "p2p_fair_l1_isolation.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    validate_p2p_gate()
    rows = collect()
    write_csv(TRIAL_CSV, rows)
    aggregates = aggregate(rows)
    write_csv(AGGREGATE_CSV, aggregates)
    figure = plot(rows)
    print(f"validated {len(rows)} config observations across 3 complete trials")
    print(TRIAL_CSV)
    print(AGGREGATE_CSV)
    print(figure)


if __name__ == "__main__":
    main()
