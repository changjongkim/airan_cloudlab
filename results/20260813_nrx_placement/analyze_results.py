#!/usr/bin/env python3
"""Validate and summarize the direct-TRT NeuralRx placement experiments."""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def docs(pattern):
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise RuntimeError(f"no files matched {pattern}")
    return [read_json(path) for path in paths]


def mean(values):
    items = [float(value) for value in values if value is not None]
    return float(np.mean(items)) if items else float("nan")


def sample_std(values):
    items = [float(value) for value in values if value is not None]
    return float(np.std(items, ddof=1)) if len(items) > 1 else 0.0


def qwen_itps(path):
    text = Path(path).read_text(encoding="utf-8")
    matches = re.findall(
        r"\[Qwen\] (?:progress|done): \d+ iters, ([0-9.]+) it/s", text
    )
    if not matches:
        raise RuntimeError(f"Qwen progress missing: {path}")
    return float(matches[-1])


def overlap_summary(pattern):
    values = docs(pattern)
    if len(values) != 3:
        raise RuntimeError(f"expected three trials for {pattern}, got {len(values)}")
    return {
        "n_trials": len(values),
        "l1_active_ms": mean(item["metrics"]["l1_active_ms"]["mean"] for item in values),
        "l1_active_p99_ms": mean(item["metrics"]["l1_active_ms"]["p99"] for item in values),
        "e2e_ms": mean(item["metrics"]["e2e_ms"]["mean"] for item in values),
        "e2e_p99_ms": mean(item["metrics"]["e2e_ms"]["p99"] for item in values),
        "nrx_ms": mean(item["metrics"]["nrx_ms"]["mean"] for item in values),
        "transport_us": mean(item["metrics"]["transport_us"]["mean"] for item in values),
        "throughput_slots_s": mean(item["completion_throughput_slots_s"] for item in values),
        "l1_active_trial_std_ms": sample_std(
            item["metrics"]["l1_active_ms"]["mean"] for item in values
        ),
    }


def placement_row(name, family, resource, baseline, overlap, qwen, notes):
    return {
        "config": name,
        "family": family,
        "resource_layout": resource,
        "n_trials": overlap["n_trials"],
        "l1_baseline_active_ms": baseline["l1_active_ms"],
        "l1_active_ms": overlap["l1_active_ms"],
        "l1_active_p99_ms": overlap["l1_active_p99_ms"],
        "l1_slowdown": overlap["l1_active_ms"] / baseline["l1_active_ms"],
        "nrx_ms": overlap["nrx_ms"],
        "e2e_ms": overlap["e2e_ms"],
        "e2e_p99_ms": overlap["e2e_p99_ms"],
        "transport_us": overlap["transport_us"],
        "completion_throughput_slots_s": overlap["throughput_slots_s"],
        "qwen_itps": qwen,
        "notes": notes,
    }


def build_placement():
    p2p = RAW / "p2p_direct_trt"
    mig_base = overlap_summary(
        p2p / "topology_a" / "trial*" / "l1_only_4g" / "p2p_overlap_*.json"
    )
    mig_same = overlap_summary(
        p2p / "topology_a" / "trial*" / "same_overlap_4g" / "p2p_overlap_*.json"
    )
    p2p_base = overlap_summary(
        p2p / "topology_b" / "trial*" / "l1_only_2g" / "p2p_overlap_*.json"
    )
    p2p_cross = overlap_summary(
        p2p / "topology_b" / "trial*" / "cross_p2p_2g2g" / "p2p_overlap_*.json"
    )
    rows = [
        placement_row(
            "MIG same",
            "MIG",
            "4g L1+NRx | 3g Qwen",
            mig_base,
            mig_same,
            qwen_itps(p2p / "topology_a" / "qwen.log"),
            "Direct TRT, CUDA Graph, ring depth 2",
        ),
        placement_row(
            "Cross P2P",
            "P2P",
            "2g L1 | 2g NRx | 3g Qwen",
            p2p_base,
            p2p_cross,
            qwen_itps(p2p / "topology_b" / "qwen.log"),
            "Direct same-GPU MIG P2P, ring depth 2",
        ),
    ]

    mig_mps = RAW / "mig_mps_direct_qwen"
    mm_base = overlap_summary(
        mig_mps / "pct100" / "trial*" / "l1_only" / "p2p_overlap_*.json"
    )
    mm_same = overlap_summary(
        mig_mps / "pct100" / "trial*" / "same_overlap" / "p2p_overlap_*.json"
    )
    rows.append(
        placement_row(
            "MIG+MPS same",
            "MIG+MPS",
            "4g L1+NRx MPS client | 3g Qwen",
            mm_base,
            mm_same,
            qwen_itps(mig_mps / "qwen.log"),
            "One combined L1+NRx MPS client; MPS does not isolate its streams",
        )
    )

    mps = RAW / "mps_direct"
    for pct in (30, 50, 70, 100):
        base = overlap_summary(
            mps / f"pct{pct}" / "trial*" / "l1_only" / "p2p_overlap_*.json"
        )
        same = overlap_summary(
            mps / f"pct{pct}" / "trial*" / "same_overlap" / "p2p_overlap_*.json"
        )
        rows.append(
            placement_row(
                f"Full MPS Qwen {pct}%",
                "MPS",
                "Full 7g shared by L1+NRx and Qwen",
                base,
                same,
                qwen_itps(mps / f"pct{pct}" / "qwen.log"),
                "L1+NRx combined MPS client at 100%; Qwen client capped",
            )
        )

    gdr_docs = docs(RAW / "config*_gdr_direct" / "l1prod_*.json")
    if len(gdr_docs) != 2:
        raise RuntimeError(f"expected two GDR repeats, got {len(gdr_docs)}")
    for item in gdr_docs:
        if item["transport"] != "gpudirect_rdma_zero_copy_direct_trt":
            raise RuntimeError(f"unexpected GDR transport {item['transport']}")
        if item["bwd_bytes"] != 314_496:
            raise RuntimeError(f"wrong LLR payload size {item['bwd_bytes']}")
    rows.append(
        {
            "config": "Cross NIC GDR",
            "family": "NIC GDR",
            "resource_layout": "2g L1 | 2g NRx | 3g Qwen",
            "n_trials": 2,
            "l1_baseline_active_ms": float("nan"),
            "l1_active_ms": float("nan"),
            "l1_active_p99_ms": float("nan"),
            "l1_slowdown": float("nan"),
            "nrx_ms": float("nan"),
            "e2e_ms": mean(item["mean_ms"] for item in gdr_docs),
            "e2e_p99_ms": mean(item["p99_ms"] for item in gdr_docs),
            "transport_us": float("nan"),
            "completion_throughput_slots_s": mean(
                1000.0 / item["mean_ms"] for item in gdr_docs
            ),
            "qwen_itps": mean(
                qwen_itps(path)
                for path in glob.glob(str(RAW / "config*_gdr_direct" / "qwen.log"))
            ),
            "notes": "Zero-copy direct TRT; request/response depth 1, so throughput is not comparable to ring-depth-2 P2P",
        }
    )
    return rows


def build_capacity():
    rows = []
    for device in ("2g", "4g", "full"):
        data = read_json(RAW / "nrx_deep_profile" / f"nrx_replica_sweep_{device}.json")
        for item in data["configurations"]:
            rows.append(
                {
                    "device": device,
                    "replicas": item["replicas"],
                    "closed_loop_slots_s": item["closed_loop"]["throughput_slots_per_s"],
                    "service_mean_ms": item["closed_loop"]["service_ms"]["mean"],
                    "service_p99_ms": item["closed_loop"]["service_ms"]["p99"],
                }
            )
    return rows


def build_depth1_transport():
    root = RAW / "p2p_direct_trt_depth1"
    definitions = (
        ("MIG same depth 1", "4g L1+NRx | 3g Qwen", "topology_a", "same_overlap_4g"),
        ("Cross P2P depth 1", "2g L1 | 2g NRx | 3g Qwen", "topology_b", "cross_p2p_2g2g"),
    )
    rows = []
    for name, resource, topology, directory in definitions:
        item = overlap_summary(
            root / topology / "trial*" / directory / "p2p_overlap_*.json"
        )
        rows.append(
            {
                "config": name,
                "resource_layout": resource,
                "n_trials": item["n_trials"],
                "e2e_mean_ms": item["e2e_ms"],
                "e2e_p99_ms": item["e2e_p99_ms"],
                "l1_active_ms": item["l1_active_ms"],
                "nrx_ms": item["nrx_ms"],
                "transport_us": item["transport_us"],
                "completion_throughput_slots_s": item["throughput_slots_s"],
                "qwen_itps": qwen_itps(root / topology / "qwen.log"),
            }
        )

    gdr_docs = docs(RAW / "config*_gdr_direct" / "l1prod_*.json")
    rows.append(
        {
            "config": "Cross NIC GDR depth 1",
            "resource_layout": "2g L1 | 2g NRx | 3g Qwen",
            "n_trials": len(gdr_docs),
            "e2e_mean_ms": mean(item["mean_ms"] for item in gdr_docs),
            "e2e_p99_ms": mean(item["p99_ms"] for item in gdr_docs),
            "l1_active_ms": float("nan"),
            "nrx_ms": float("nan"),
            "transport_us": float("nan"),
            "completion_throughput_slots_s": mean(
                1000.0 / item["mean_ms"] for item in gdr_docs
            ),
            "qwen_itps": mean(
                qwen_itps(path)
                for path in glob.glob(str(RAW / "config*_gdr_direct" / "qwen.log"))
            ),
        }
    )
    return rows


def build_open_loop():
    rows = []
    for device in ("2g", "4g", "full"):
        data = read_json(RAW / "nrx_deep_profile" / f"nrx_replica_sweep_{device}.json")
        for configuration in data["configurations"]:
            for item in configuration["open_loop"]:
                rows.append(
                    {
                        "device": device,
                        "replicas": configuration["replicas"],
                        "arrival_rate_slots_s": item["arrival_rate_slots_per_s"],
                        "latency_mean_ms": item["latency_ms"]["mean"],
                        "latency_p99_ms": item["latency_ms"]["p99"],
                        "backlog_at_window_end": item["backlog_at_window_end"],
                        "max_outstanding": item["max_outstanding"],
                        "deadline_miss_ratio_1ms": item["deadline_miss_ratio"]["1ms"],
                        "drain_after_last_arrival_ms": item["drain_after_last_arrival_ms"],
                    }
                )
    return rows


def build_tactic_sensitivity(capacity_rows):
    reference = {
        item["device"]: item["service_mean_ms"]
        for item in capacity_rows
        if item["replicas"] == 1
    }
    native_paths = {
        "2g": RAW / "nrx_deep_profile" / "tactic_sensitivity" / "nrx_trt_direct_graph_2g_native.json",
        "full": RAW / "nrx_deep_profile" / "tactic_sensitivity" / "nrx_trt_direct_graph_full_native.json",
    }
    rows = []
    for device in ("2g", "full"):
        native = read_json(native_paths[device])["direct"]["gpu_ms"]
        reference_ms = reference[device]
        rows.append(
            {
                "device": device,
                "shared_4g_built_engine_mean_ms": reference_ms,
                "native_built_engine_mean_ms": native["mean"],
                "native_built_engine_p99_ms": native["p99"],
                "native_improvement_percent": 100.0 * (
                    reference_ms - native["mean"]
                ) / reference_ms,
            }
        )
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_wrapper():
    profile = read_json(RAW / "nrx_deep_profile" / "nrx_deep_profile_4g_bits2.json")
    direct = read_json(RAW / "nrx_deep_profile" / "nrx_trt_direct_4g.json")
    graph = read_json(RAW / "nrx_deep_profile" / "nrx_trt_direct_graph_4g.json")
    labels = ["pycuphy raw", "Direct TensorRT", "Direct + CUDA Graph"]
    values = [
        profile["metrics"]["raw_pycuphy"]["gpu_ms"]["mean"],
        direct["direct"]["gpu_ms"]["mean"],
        graph["direct"]["gpu_ms"]["mean"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=["#d95f02", "#1b9e77", "#7570b3"])
    ax.set_yscale("log")
    ax.set_ylabel("GPU latency per NRx inference (ms, log scale)")
    ax.set_title(
        "NeuralRx bottleneck is pycuphy layout conversion, not TensorRT",
        pad=18,
    )
    ax.set_ylim(top=180)
    ax.grid(axis="y", alpha=0.25, which="both")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.15, f"{value:.3f} ms", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "nrx_wrapper_decomposition.png", dpi=240)
    plt.close(fig)


def plot_capacity(rows):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"2g": "#e6ab02", "4g": "#1b9e77", "full": "#7570b3"}
    for device in ("2g", "4g", "full"):
        items = [item for item in rows if item["device"] == device]
        ax.plot(
            [item["replicas"] for item in items],
            [item["closed_loop_slots_s"] for item in items],
            marker="o",
            label=device,
            color=colors[device],
        )
    ax.axhline(1000, color="black", linestyle="--", linewidth=1, label="1 ms arrival = 1000 slots/s")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlabel("Concurrent TensorRT execution contexts / replicas")
    ax.set_ylabel("Closed-loop throughput (slots/s)")
    ax.set_title("More replicas do not create capacity inside a fixed MIG slice")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "nrx_replica_capacity.png", dpi=240)
    plt.close(fig)


def plot_open_loop(rows):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    colors = {"2g": "#e6ab02", "4g": "#1b9e77", "full": "#7570b3"}
    for device in ("2g", "4g", "full"):
        items = [
            item for item in rows
            if item["device"] == device and item["replicas"] == 1
        ]
        items.sort(key=lambda item: item["arrival_rate_slots_s"])
        ax.plot(
            [item["arrival_rate_slots_s"] for item in items],
            [item["latency_p99_ms"] for item in items],
            marker="o",
            label=f"{device}, 1 context",
            color=colors[device],
        )
    ax.axvline(1000, color="black", linestyle="--", linewidth=1, label="1 ms slot arrival")
    ax.axhline(1, color="#555555", linestyle=":", linewidth=1, label="1 ms response")
    ax.set_yscale("log")
    ax.set_xlabel("Open-loop arrival rate (slots/s)")
    ax.set_ylabel("Response latency p99 (ms, log scale)")
    ax.set_title("Queue stability depends on NRx service capacity, not transport")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "nrx_open_loop_queue.png", dpi=240)
    plt.close(fig)


def plot_placement(rows):
    selected = [
        item
        for item in rows
        if item["config"] in (
            "MIG same",
            "MIG+MPS same",
            "Cross P2P",
            "Cross NIC GDR",
            "Full MPS Qwen 30%",
            "Full MPS Qwen 50%",
            "Full MPS Qwen 70%",
            "Full MPS Qwen 100%",
        )
    ]
    order = [
        "MIG same",
        "MIG+MPS same",
        "Cross P2P",
        "Cross NIC GDR",
        "Full MPS Qwen 30%",
        "Full MPS Qwen 50%",
        "Full MPS Qwen 70%",
        "Full MPS Qwen 100%",
    ]
    selected.sort(key=lambda item: order.index(item["config"]))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    x = np.arange(len(selected))
    labels = [item["config"].replace("Full MPS ", "MPS\n") for item in selected]
    e2e = [item["e2e_ms"] for item in selected]
    colors = [
        {"MIG": "#1b9e77", "MIG+MPS": "#66a61e", "P2P": "#377eb8", "NIC GDR": "#984ea3", "MPS": "#d95f02"}[item["family"]]
        for item in selected
    ]
    axes[0].bar(x, e2e, color=colors)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Dependency-carrying slot e2e (ms)")
    axes[0].set_title("Optimized pipeline latency")
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(e2e):
        axes[0].text(index, value + 0.12, f"{value:.2f}", ha="center", fontsize=8)

    annotation_offsets = {
        "MIG same": (-78, -18),
        "MIG+MPS same": (-92, 22),
        "Cross P2P": (10, 22),
        "Cross NIC GDR": (10, -20),
    }
    for item, color in zip(selected, colors):
        axes[1].scatter(item["qwen_itps"], item["e2e_ms"], s=80, color=color)
        offset = annotation_offsets.get(item["config"], (6, 6))
        axes[1].annotate(
            item["config"],
            (item["qwen_itps"], item["e2e_ms"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.82, "ec": "none"},
        )
    axes[1].set_xlabel("Qwen throughput (it/s)")
    axes[1].set_ylabel("Slot e2e (ms)")
    axes[1].set_title("RAN latency vs. monetizable AI utility")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "placement_latency_utility.png", dpi=240)
    plt.close(fig)


def main():
    comparison = read_json(
        RAW / "nrx_deep_profile" / "nrx_trt_direct_compare_4g.json"
    )["pycuphy_comparison"]
    if not all(item["max_abs_difference"] == 0.0 for item in comparison.values()):
        raise RuntimeError(f"direct TensorRT correctness failed: {comparison}")
    placement = build_placement()
    capacity = build_capacity()
    open_loop = build_open_loop()
    depth1_transport = build_depth1_transport()
    tactic_sensitivity = build_tactic_sensitivity(capacity)
    write_csv(ROOT / "PLACEMENT_SUMMARY.csv", placement)
    write_csv(ROOT / "NRX_CAPACITY.csv", capacity)
    write_csv(ROOT / "NRX_OPEN_LOOP.csv", open_loop)
    write_csv(ROOT / "DEPTH1_TRANSPORT_COMPARISON.csv", depth1_transport)
    write_csv(ROOT / "NRX_TACTIC_SENSITIVITY.csv", tactic_sensitivity)
    plot_wrapper()
    plot_capacity(capacity)
    plot_open_loop(open_loop)
    plot_placement(placement)
    print(
        f"validated {len(placement)} placement rows, {len(capacity)} capacity rows, "
        f"{len(open_loop)} open-loop rows, and "
        f"{len(depth1_transport)} depth-1 transport rows, and "
        f"{len(tactic_sensitivity)} tactic-sensitivity rows"
    )


if __name__ == "__main__":
    main()
