#!/usr/bin/env python3
"""Generate the measurement-backed figures used by RESEARCH_WALKTHROUGH_KO.md.

Every plotted value is loaded from a preserved CSV/JSON result.  The script does
not modify raw data and intentionally keeps the scope of each experiment visible
in the figure subtitle.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "current" / "figures"

DRAIN = ROOT / "results" / "20260813_drain_free"
MIG_CAUSAL = (
    ROOT
    / "results"
    / "isca_v2"
    / "mig_causal_20260813T1138Z"
)
PLACEMENT = ROOT / "results" / "20260813_nrx_placement"
GDR_POOL = ROOT / "task1_final" / "gdr_pool_20260814T014651Z" / "analysis"
RADIO = ROOT / "task1_final" / "dart_rx_radio_pool" / "analysis"

COLORS = {
    "navy": "#16324f",
    "blue": "#2878b5",
    "cyan": "#4bb3a7",
    "green": "#4c956c",
    "orange": "#f4a261",
    "red": "#d1495b",
    "purple": "#7b6fd0",
    "gray": "#8d99ae",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dfe4ea", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(axis, bars, fmt="{:.2f}", *, scale=1.0, pad=3):
    for bar in bars:
        value = bar.get_height() * scale
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def figure_01_isolation_and_queue_cliff():
    base = DRAIN / "fixed_mig_sibling_isolation"
    alone = read_json(base / "nrx_4g_alone.json")["configurations"][0]
    sibling = read_json(base / "nrx_4g_qwen_3g.json")["configurations"][0]

    cap = [
        alone["closed_loop"]["throughput_slots_per_s"],
        sibling["closed_loop"]["throughput_slots_per_s"],
    ]
    delta = 100 * (cap[1] / cap[0] - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    bars = axes[0].bar(
        ["4g NRx alone", "4g NRx +\nQwen on sibling 3g"],
        cap,
        color=[COLORS["blue"], COLORS["green"]],
        width=0.62,
    )
    axes[0].set_ylim(0, max(cap) * 1.18)
    axes[0].set_ylabel("Closed-loop capacity (requests/s)")
    axes[0].set_title("(a) MIG sibling isolation works")
    annotate_bars(axes[0], bars, "{:.1f}")
    axes[0].text(
        0.5,
        max(cap) * 1.09,
        f"capacity delta: {delta:+.2f}%",
        ha="center",
        fontsize=10,
        color=COLORS["navy"],
        fontweight="bold",
    )
    style_axes(axes[0])

    for data, label, color, marker in [
        (alone, "4g alone", COLORS["blue"], "o"),
        (sibling, "4g + sibling Qwen", COLORS["green"], "s"),
    ]:
        rates = [x["arrival_rate_slots_per_s"] for x in data["open_loop"]]
        p99 = [x["latency_ms"]["p99"] for x in data["open_loop"]]
        axes[1].plot(rates, p99, marker=marker, linewidth=2.2, label=label, color=color)
    axes[1].axhline(5, linestyle="--", linewidth=1.2, color=COLORS["red"], label="5 ms gate")
    axes[1].axvline(cap[0], linestyle=":", linewidth=1.4, color=COLORS["gray"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Offered NRx rate (requests/s)")
    axes[1].set_ylabel("Queueing latency p99 (ms, log scale)")
    axes[1].set_title("(b) Isolation does not remove the capacity cliff")
    axes[1].legend(frameon=False, fontsize=8.5)
    style_axes(axes[1])

    fig.suptitle(
        "Measured problem: MIG protects service rate, but the fixed endpoint still overloads",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Scope: direct TensorRT NRx on one A100 4g MIG; open-loop synthetic slot arrivals.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "01_mig_isolation_queue_cliff.png")


def figure_01b_nrx_wrapper_optimization():
    base = PLACEMENT / "raw" / "nrx_deep_profile"
    wrapper = read_json(base / "nrx_deep_profile_4g.json")
    direct = read_json(base / "nrx_trt_direct_4g.json")
    graph = read_json(base / "nrx_trt_direct_graph_4g.json")
    compare = read_json(base / "nrx_trt_direct_compare_4g.json")

    labels = ["Public pycuphy\nwrapper", "Caller-owned\nTensorRT binding", "Direct binding\n+ CUDA Graph"]
    gpu_ms = [
        float(wrapper["metrics"]["raw_pycuphy"]["gpu_ms"]["mean"]),
        float(direct["direct"]["gpu_ms"]["mean"]),
        float(graph["direct"]["gpu_ms"]["mean"]),
    ]
    enqueue_us = [
        np.nan,
        float(direct["direct"]["enqueue_us"]["mean"]),
        float(graph["direct"]["enqueue_us"]["mean"]),
    ]
    equality = all(
        float(value["max_abs_difference"]) == 0.0
        for value in compare["pycuphy_comparison"].values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    bars = axes[0].bar(labels, gpu_ms, color=[COLORS["red"], COLORS["blue"], COLORS["green"]], width=0.62)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NRx GPU time (ms, log scale)")
    axes[0].set_title("(a) Wrapper cost was not neural compute")
    annotate_bars(axes[0], bars, "{:.3f}")
    style_axes(axes[0])

    x = np.arange(2)
    bars = axes[1].bar(x, enqueue_us[1:], color=[COLORS["blue"], COLORS["green"]], width=0.58)
    axes[1].set_xticks(x, labels[1:])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Host enqueue time (us, log scale)")
    axes[1].set_title("(b) Persistent CUDA Graph removes launch overhead")
    annotate_bars(axes[1], bars, "{:.1f}")
    style_axes(axes[1])
    axes[1].text(
        0.5,
        0.78,
        f"Wrapper-equivalent outputs: max abs diff = 0 ({'PASS' if equality else 'FAIL'})",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
        color=COLORS["navy"],
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )

    fig.suptitle(
        "Measured implementation correction: direct TensorRT changes the bottleneck by 74x",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Scope: one A100 4g MIG; wrapper n=30, direct/CUDA-Graph n=1,000 after warm-up.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "01b_nrx_wrapper_optimization.png")


def figure_02_fragmentation():
    rows = read_csv(
        MIG_CAUSAL / "07_multicell_workloads" / "analysis" / "MULTICELL_HARDWARE_MEDIANS.csv"
    )
    cases = [
        ("single_periodic", "1", "1000.0", "1.0", "1 cell\n1 ms, NRx 100%"),
        ("selective_bursty", "4", "1000.0", "0.1", "4 cells\n1 ms, bursty 10%"),
        ("selective_bursty", "4", "500.0", "0.1", "4 cells\n0.5 ms, bursty 10%"),
    ]
    policies = ["static_one", "predicted_finish"]
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for scenario, cells, slot_us, probability, label in cases:
        for row in rows:
            key = (row["scenario"], row["cells"], row["slot_us"], row["nrx_probability_requested"])
            if key == (scenario, cells, slot_us, probability) and row["policy"] in policies:
                selected[(label, row["policy"])] = row
    assert len(selected) == len(cases) * len(policies)

    labels = [case[-1] for case in cases]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for index, (policy, display, color) in enumerate(
        [
            ("static_one", "Static-one", COLORS["red"]),
            ("predicted_finish", "Predicted-finish", COLORS["blue"]),
        ]
    ):
        p99 = [float(selected[(label, policy)]["median_p99_ms"]) for label in labels]
        no_timely = [
            float(selected[(label, policy)]["median_no_timely_nrx_ratio"]) for label in labels
        ]
        idle = [
            float(selected[(label, policy)]["median_idle_endpoint_fraction"]) for label in labels
        ]
        offsets = x + (index - 0.5) * width
        axes[0].bar(offsets, p99, width, label=display, color=color)
        axes[1].bar(offsets, no_timely, width, label=display, color=color)
        axes[2].bar(offsets, idle, width, label=display, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("NRx response p99 (ms, log)")
    axes[0].set_title("(a) Tail latency")
    axes[1].set_ylabel("No timely NRx ratio")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("(b) Timeliness failure")
    axes[2].set_ylabel("Mean idle endpoint fraction")
    axes[2].set_ylim(0, 0.9)
    axes[2].set_title("(c) Observed idle endpoint fraction")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.5)
        style_axes(axis)
    axes[0].legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "Measured problem existence: a busy queue and idle isolated endpoints coexist",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Scope: 3 independent resident TensorRT endpoints; 3-trial medians; 5 ms experimental timeliness gate.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "02_fixed_placement_fragmentation.png")


def figure_03_placement_and_transport():
    rows = read_csv(PLACEMENT / "PLACEMENT_SUMMARY.csv")
    desired = [
        "MIG same",
        "MIG+MPS same",
        "Cross P2P",
        "Cross NIC GDR",
        "Full MPS Qwen 30%",
        "Full MPS Qwen 50%",
        "Full MPS Qwen 70%",
        "Full MPS Qwen 100%",
    ]
    by_name = {row["config"]: row for row in rows}
    assert all(name in by_name for name in desired)
    labels = [
        "MIG local",
        "MIG+MPS",
        "Cross P2P",
        "Cross GDR",
        "MPS 30%",
        "MPS 50%",
        "MPS 70%",
        "MPS 100%",
    ]
    colors = [
        COLORS["blue"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
    ]
    mean = np.array([float(by_name[name]["e2e_ms"]) for name in desired])
    p99 = np.array([float(by_name[name]["e2e_p99_ms"]) for name in desired])
    qwen = np.array([float(by_name[name]["qwen_itps"]) for name in desired])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(labels))
    bars = axes[0].bar(x, mean, color=colors, width=0.68, label="mean")
    axes[0].scatter(x, p99, marker="D", s=38, color=COLORS["navy"], label="p99", zorder=3)
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("Dependency-carrying slot E2E (ms)")
    axes[0].set_title("(a) Placement/transport latency")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars, "{:.2f}")
    style_axes(axes[0])

    family_markers = {
        "MIG": "o",
        "MIG+MPS": "P",
        "P2P": "s",
        "NIC GDR": "^",
        "MPS": "D",
    }
    label_offsets = {
        "MIG local": (-64, 0),
        "MIG+MPS": (-78, 34),
        "Cross P2P": (24, 19),
        "Cross GDR": (24, 43),
        "MPS 30%": (-10, -18),
        "MPS 50%": (12, -14),
    }
    for name, label, color in zip(desired, labels, colors):
        row = by_name[name]
        axes[1].scatter(
            float(row["qwen_itps"]),
            float(row["e2e_ms"]),
            marker=family_markers[row["family"]],
            s=72,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[1].annotate(
            label,
            (float(row["qwen_itps"]), float(row["e2e_ms"])),
            xytext=label_offsets.get(label, (6, 4)),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.7}
            if label in label_offsets
            else None,
        )
    axes[1].set_xlabel("Qwen throughput (iterations/s)")
    axes[1].set_ylabel("Slot E2E mean (ms)")
    axes[1].set_title("(b) RAN latency vs background utility")
    style_axes(axes[1])

    fig.suptitle(
        "Measured baseline space: P2P/GDR enable cross-partition placement, but NRx compute and queueing dominate",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Scope: optimized direct-TensorRT dependency chain; GDR is depth 1, while the main P2P result is depth 2.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    save(fig, "03_placement_transport_baselines.png")


def _background_metrics(workload: str, policy: str):
    base = MIG_CAUSAL / "06_background_contention" / workload / policy
    nrx = read_json(base / "nrx_timeline.json")
    background = read_json(base / "background_timeline.json")
    burst = next(item for item in nrx["phase_results"] if item["kind"] == "burst")
    activation = 0.0
    for transition in nrx.get("transitions", []):
        if transition.get("event") == "reclaim_request":
            activation = float(transition["activation_ms"])
    work_items = background.get("units", background.get("iterations", []))
    work_latency = background.get("unit_latency_ms", background.get("iteration_latency_ms"))
    assert work_items is not None and work_latency is not None
    return {
        "p99": float(burst["latency_ms"]["p99"]),
        "miss5": float(burst["deadline_miss_ratio"]["5ms"]),
        "max_outstanding": int(nrx["max_outstanding"]),
        "units": len(work_items),
        "unit_p99": float(work_latency["p99"]),
        "activation": activation,
    }


def figure_04_background_reclaim():
    workloads = [
        ("resnet50", "ResNet-50"),
        ("bert_base", "BERT-base"),
        ("whisper_base", "Whisper-base"),
        ("qwen_decode", "Qwen-7B\ndecode"),
    ]
    metrics = {
        (display, policy): _background_metrics(path, policy)
        for path, display in workloads
        for policy in ("naive_share", "adaptive_reclaim")
    }
    labels = [display for _, display in workloads]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    naive = [metrics[(label, "naive_share")]["p99"] for label in labels]
    adaptive = [metrics[(label, "adaptive_reclaim")]["p99"] for label in labels]
    axes[0].bar(x - width / 2, naive, width, color=COLORS["red"], label="Naive share")
    axes[0].bar(x + width / 2, adaptive, width, color=COLORS["green"], label="Adaptive reclaim")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Burst NRx p99 (ms, log)")
    axes[0].set_title("(a) Queue collapse")
    axes[0].legend(frameon=False, fontsize=8.5)

    naive_miss = [metrics[(label, "naive_share")]["miss5"] for label in labels]
    adaptive_miss = [metrics[(label, "adaptive_reclaim")]["miss5"] for label in labels]
    axes[1].bar(x - width / 2, naive_miss, width, color=COLORS["red"])
    axes[1].bar(x + width / 2, adaptive_miss, width, color=COLORS["green"])
    axes[1].set_ylim(0, 0.78)
    axes[1].set_ylabel("Requests over 5 ms")
    axes[1].set_title("(b) Experimental miss ratio")

    retained = [
        100 * metrics[(label, "adaptive_reclaim")]["units"] / metrics[(label, "naive_share")]["units"]
        for label in labels
    ]
    activation = [metrics[(label, "adaptive_reclaim")]["activation"] for label in labels]
    bars = axes[2].bar(x, retained, width=0.58, color=COLORS["blue"], label="background work retained")
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("Background work retained (%)")
    axes[2].set_title("(c) Utility retained / reclaim delay")
    twin = axes[2].twinx()
    twin.plot(x, activation, color=COLORS["orange"], marker="D", linewidth=2, label="reclaim delay")
    twin.set_ylabel("Reclaim activation (ms)", color=COLORS["orange"])
    twin.tick_params(axis="y", colors=COLORS["orange"])
    annotate_bars(axes[2], bars, "{:.1f}%")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.5)
        style_axes(axis)
    twin.spines["top"].set_visible(False)

    fig.suptitle(
        "Measured design opportunity: bounded background leases turn an idle spare into burst capacity",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Scope: direct-TensorRT NRx compute queue + resident background model; no cuPHY or transport in this gate.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "04_background_reclaim.png")


def figure_05_gdr_pool_policy():
    gaps = read_csv(GDR_POOL / "POLICY_GAPS.csv")
    comparisons = read_csv(GDR_POOL / "POLICY_COMPARISONS.csv")
    load_bands = ["at_or_below_1000", "1000_to_1500", "above_1500"]
    load_labels = ["<=1,000/s", "1,000-1,500/s", ">1,500/s"]
    policies = ["static_one", "static_cell", "predicted_finish", "tail_aware"]
    display = ["Static-one", "Static-cell", "Predicted-finish", "Tail-aware"]
    colors = [COLORS["red"], COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    lookup = {(row["load_band"], row["policy"]): row for row in gaps}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(load_bands))
    width = 0.19
    for i, (policy, label, color) in enumerate(zip(policies, display, colors)):
        values = [float(lookup[(band, policy)]["no_timely_ratio_median"]) for band in load_bands]
        axes[0].bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    axes[0].set_xticks(x, load_labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Median no-timely ratio")
    axes[0].set_title("(a) Full-size zero-copy GDR pool")
    axes[0].legend(frameon=False, fontsize=8.2, ncol=2)
    style_axes(axes[0])

    names = [f"{row['candidate']}\nvs {row['baseline']}" for row in comparisons]
    improvements = [100 * float(row["no_timely_improvement_median"]) for row in comparisons]
    better = [f"{row['candidate_better']}/{row['paired_traces']}" for row in comparisons]
    bars = axes[1].barh(np.arange(len(names)), improvements, color=[COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["purple"]])
    axes[1].set_yticks(np.arange(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median no-timely reduction (percentage points)")
    axes[1].set_title("(b) Paired trace dominance")
    for bar, count in zip(bars, better):
        axes[1].text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"better {count}", va="center", fontsize=8.2)
    style_axes(axes[1])

    fig.suptitle(
        "Measured scheduler result: finish prediction avoids futile remote work, but admission is still conservative",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Scope: 29 workload points x 3 trials x 4 policies = 348 runs; 5 ms gate. No-timely includes intentional conventional rejection.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "05_gdr_pool_policy.png")


def figure_06_radio_utility():
    rows = read_csv(RADIO / "SUMMARY.csv")
    rows3 = {row["mode"]: row for row in rows if row["endpoint_count"] == "3"}
    assert set(rows3) >= {"none", "all", "utility"}
    modes = ["none", "all", "utility"]
    labels = ["Conventional", "All NRx", "Utility admission"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]

    correct = [float(rows3[mode]["correct_ratio_median"]) for mode in modes]
    requests = [float(rows3[mode]["nrx_requests_median"]) for mode in modes]
    p50 = [float(rows3[mode]["decision_p50_ms_median"]) for mode in modes]
    p99 = [float(rows3[mode]["decision_p99_ms_median"]) for mode in modes]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    bars = axes[0].bar(labels, correct, color=colors, width=0.62)
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("Correct TB ratio")
    axes[0].set_title("(a) Delivered radio outcome")
    annotate_bars(axes[0], bars, "{:.2f}")

    bars = axes[1].bar(labels, requests, color=colors, width=0.62)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("NRx requests per 100 slots")
    axes[1].set_title("(b) Neural work admitted")
    annotate_bars(axes[1], bars, "{:.0f}")

    x = np.arange(len(labels))
    width = 0.34
    b1 = axes[2].bar(x - width / 2, p50, width, color=COLORS["cyan"], label="p50")
    b2 = axes[2].bar(x + width / 2, p99, width, color=COLORS["navy"], label="p99")
    axes[2].axhline(12, linestyle="--", color=COLORS["red"], linewidth=1.2, label="12 ms expiry")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 13.2)
    axes[2].set_ylabel("Decision latency (ms)")
    axes[2].set_title("(c) Actual transaction latency")
    axes[2].legend(frameon=False, fontsize=8.5)
    annotate_bars(axes[2], b1, "{:.2f}")
    annotate_bars(axes[2], b2, "{:.2f}")

    for axis in axes:
        axis.tick_params(axis="x", labelsize=8.4, rotation=8)
        style_axes(axis)

    fig.suptitle(
        "Measured vertical slice: selective NRx preserves radio gain while avoiding 25% of neural requests",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Scope: actual cuPHY CE -> GDR NRx -> LDPC/CRC, 3 endpoints, 3 trials x 100 requests; 12 ms experimental expiry.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "06_actual_radio_utility.png")


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
        }
    )
    figure_01_isolation_and_queue_cliff()
    figure_01b_nrx_wrapper_optimization()
    figure_02_fragmentation()
    figure_03_placement_and_transport()
    figure_04_background_reclaim()
    figure_05_gdr_pool_policy()
    figure_06_radio_utility()
    print(f"wrote 7 figures to {OUT}")


if __name__ == "__main__":
    main()
