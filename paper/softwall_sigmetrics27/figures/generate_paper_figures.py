#!/usr/bin/env python3
"""Generate the additional Backstop paper figures from audited result JSON.

Run from anywhere after loading the NERSC Python module:
  module load python/3.11-24.1.0
  python paper/softwall_sigmetrics27/figures/generate_paper_figures.py

The script writes vector PDF/SVG figures and a manifest beside itself.  It
asserts the headline values used by the manuscript before rendering.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SOURCES = {
    "mps_cap": ROOT / "results/softwall_same_gpu/confirm26_physical_overrun.json",
    "mps_lifecycle": ROOT / "results/softwall_same_gpu/confirm46_retirement_cpu_control.json",
    "mps_lifecycle_protocol": ROOT / "results/softwall_same_gpu/confirm46_retirement_cpu_control_protocol.json",
    "q2": ROOT / "results/softwall_multigpu/confirm159_q2_variable_two_node.json",
    "q2_protocol": ROOT / "results/softwall_multigpu/confirm159q2f_batch_dev_j58857672_protocol.json",
    "fault": ROOT / "results/softwall_multigpu/c161_full_fault_qualification.json",
    "baseline": ROOT / "results/softwall_multigpu/c173_deadline_correction_result_v1.json",
    "capacity": ROOT / "results/softwall_multigpu/c167_reclaimable_capacity_model_v2.json",
    "headroom": ROOT / "results/softwall_multigpu/c174_trace_oracle_screen_v1.json",
    "levers": ROOT / "results/softwall_multigpu/c175_lever_evaluation_v1.json",
    "lifecycle": ROOT / "results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json",
    "production": ROOT / "results/softwall_multigpu/softwall_production_exit_gate_v3.json",
    "channel": ROOT / "results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json",
    "receiver": ROOT / "results/softwall_same_gpu/confirm18_dual_receiver_transition_job58637416.json",
    "mps_light": ROOT / "results/softwall_same_gpu/confirm10_paired2_qwen_s1_job58625542_d21.json",
    "mps_light_protocol": ROOT / "results/softwall_same_gpu/confirm10_paired2_qwen_s1_protocol.json",
    "mps_heavy": ROOT / "results/softwall_same_gpu/confirm11_paired2_qwen_heavy_s1_job58625542_d21.json",
    "mps_heavy_protocol": ROOT / "results/softwall_same_gpu/confirm11_paired2_qwen_heavy_s1_protocol.json",
    "native_canary": ROOT / "results/softwall_multigpu/c168_persistent_pair_canary_seed20359400_job58871654.json",
    "kernel_analysis": ROOT / "results/softwall_multigpu/c163_raw_p2p_v5_stage_profile_analysis_job58868184.json",
    "envelope_grid": ROOT / "results/softwall_multigpu/c162_feasibility_grid_v1.json",
    "envelope_scale": ROOT / "results/softwall_multigpu/c162_scheduler_scalability_v1.json",
    "envelope_protocol": ROOT / "results/softwall_multigpu/c162a_boundary_dev_j58860486_protocol.json",
    "necessity_dev": ROOT / "results/softwall_multigpu/c172_debt_blind_job58957717_result.json",
    "necessity_holdout": ROOT / "results/softwall_multigpu/c172_debt_blind_job58957719_result.json",
    "fault_phase1": ROOT / "results/softwall_multigpu/c161_phase1_two_node.json",
    "fault_phase2": ROOT / "results/softwall_multigpu/c161_phase2_two_node.json",
    "necessity_witness": ROOT / "results/softwall_multigpu/softwall_necessity_witness_v3.json",
    "burst_campaign": ROOT / "results/softwall_multigpu/c176_burst_campaign_v1.json",
    "burst_replay": ROOT / "results/softwall_multigpu/c176_replay_slo_v1.json",
    "load_campaign": ROOT / "results/softwall_multigpu/c176_load_campaign_v1.json",
    "load_replay": ROOT / "results/softwall_multigpu/c176_replay_load_v1.json",
}

# Raw per-request records behind the two background figures. The Q2
# coordinator digests are pinned in the Q2 summary and the native-pair digest in
# its canary analysis; the derived file records every digest.
BACKGROUND_RAW = {
    "q2_dev": ROOT / "results/softwall_multigpu/raw/confirm159q2f_batch_dev_j58857672_coordinator.json",
    "q2_holdout": ROOT / "results/softwall_multigpu/raw/confirm159q2g_batch_holdout_j58858194_coordinator.json",
    "pair_python": ROOT / "results/softwall_multigpu/raw/c163_raw_p2p_job58866163_controller.json",
    "pair_native": ROOT / "results/softwall_multigpu/raw/c168_persistent_pair_canary_seed20359400_job58871654_controller.json",
    "kernel": ROOT / "results/softwall_multigpu/raw/c163_raw_p2p_v5_stage_profile_job58868184_worker.json",
}
BACKGROUND_DATA = HERE / "softwall_background_data.json"

# Per-attempt records of the two-node debt-blind diagnostic (C172); the result
# files pin the digest of each coordinator log.
EVAL_RAW = {
    "c172_dev": ROOT / "results/softwall_multigpu/raw/c172_debt_blind_job58957717_coordinator.json",
    "c172_holdout": ROOT / "results/softwall_multigpu/raw/c172_debt_blind_job58957719_coordinator.json",
}
EVAL_DATA = HERE / "softwall_eval_data.json"
RUNTIME_DATA = HERE / "softwall_runtime_data.json"


def load(name: str):
    return json.loads(SOURCES[name].read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# matplotlib tab10 palette. Backstop data is drawn in blue and reference or
# baseline data in gray; light variants are used only for box fills.
COLORS = {
    "blue": "#1F77B4",
    "light_blue": "#D2E3F1",
    "orange": "#FF7F0E",
    "light_orange": "#FFE3C9",
    "green": "#2CA02C",
    "light_green": "#D5EFD5",
    "purple": "#9467BD",
    "light_purple": "#E7DEEF",
    "red": "#D62728",
    "light_red": "#F6D3D3",
    "brown": "#8C564B",
    "cyan": "#17BECF",
    "gray": "#7F7F7F",
    "light_gray": "#E5E5E5",
    "dark": "#202020",
}

# acmsmall text width: 6.75 in paper - 2 x 46 pt margins = 5.48 in. Figures are
# drawn at this width and included at \linewidth, so font sizes are final.
FIG_WIDTH = 5.4

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "bold",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.labelweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "softwall-sigmetrics27",
    }
)

GRID = {"color": "#D8DDE3", "linewidth": 0.55}


def panel_label(ax, label: str) -> None:
    """Subfigure marker only; panel descriptions belong to the LaTeX caption."""
    ax.text(
        -0.02,
        1.03,
        f"({label})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["dark"],
    )


def save(fig, stem: str) -> None:
    # The tight bounding box of matplotlib 3.8 narrows axis labels to their
    # anchor; listing the labels keeps a long label from being cut at the edge.
    labels = fig.get_default_bbox_extra_artists() + [axis.label for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)]
    fig.savefig(
        HERE / f"{stem}.pdf",
        bbox_inches="tight",
        bbox_extra_artists=labels,
        pad_inches=0.04,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        HERE / f"{stem}.svg",
        bbox_inches="tight",
        bbox_extra_artists=labels,
        pad_inches=0.04,
        metadata={"Date": None},
    )
    plt.close(fig)




def conventional_midpoint(points: list) -> float:
    """SNR at which the conventional receiver decodes half of the TBs (linear interpolation)."""
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if y0 < 50.0 <= y1:
            return x0 + (x1 - x0) * (50.0 - y0) / (y1 - y0)
    raise ValueError("conventional waterfall does not cross 50%")


def debt_histogram_tail(counts: dict) -> list:
    """Share of epochs that retain at least k debts, k = 1..4."""
    total = sum(counts.values())
    return [sum(n for m, n in counts.items() if int(m) >= k) / total for k in range(1, 5)]


def debt_tail_model(p: float, k: int, rho: float, n: int = 4) -> float:
    """P(at least k of n debts remain) with per-TB probability p and intra-epoch correlation rho."""
    if rho == 0.0:
        return sum(math.comb(n, m) * p ** m * (1 - p) ** (n - m) for m in range(k, n + 1))
    a = p * (1 - rho) / rho
    b = (1 - p) * (1 - rho) / rho

    def log_beta(x: float, y: float) -> float:
        return math.lgamma(x) + math.lgamma(y) - math.lgamma(x + y)

    return sum(math.comb(n, m) * math.exp(log_beta(m + a, n - m + b) - log_beta(a, b))
               for m in range(k, n + 1))


def derive_background() -> None:
    """Derive the background-figure data from receiver, MPS, and runtime records."""
    if not all(path.exists() for path in BACKGROUND_RAW.values()):
        return
    pinned = load("q2")["artifact_sha256"]
    for key in ("q2_dev", "q2_holdout"):
        rel = str(BACKGROUND_RAW[key].relative_to(ROOT))
        assert pinned[rel] == sha256(BACKGROUND_RAW[key]), f"raw log differs from pinned digest: {rel}"
    assert load("native_canary")["source_sha256"]["controller_raw"] == sha256(BACKGROUND_RAW["pair_native"])

    # Receiver outcomes. Confirm18 decodes the same TBs with cuPHY and NeuralRx
    # under a synthetic block-Rayleigh channel; the Sionna holdout does the same
    # for CDL-D and CDL-E. The 10 dB control stratum is not part of a waterfall.
    receiver = load("receiver")["results"]
    channels = {"Rayleigh": [(row["snr_db"], row["trials"], row["conventional_correct"], row["neural_correct"])
                             for row in receiver]}
    for model in ("D", "E"):
        strata = load("channel")["models"][model]["strata"]
        channels[f"CDL-{model}"] = [(row["esno_db"], row["trials"], row["conventional_correct"],
                                     row["neural_correct"]) for row in strata if row["esno_db"] < 0]
    waterfalls = {}
    for name, rows in channels.items():
        midpoint = conventional_midpoint([(snr, 100.0 * conv / n) for snr, n, conv, _ in rows])
        waterfalls[name] = {
            "conventional_50pct_snr_db": midpoint,
            "rows": [{"snr_db": snr, "trials": n, "conventional_correct": conv, "neural_correct": nrx}
                     for snr, n, conv, nrx in rows],
        }

    # Receiver trials are independent channel draws; four consecutive trials
    # form one epoch with four TBs at that SNR.
    grouped_trials = []
    for row in receiver:
        records = sorted(row["records"], key=lambda record: record["trial"])
        failures = sum(not record["neural_correct"] for record in records)
        assert failures == row["trials"] - row["neural_correct"]
        if 0 < failures < len(records):
            counts = {str(m): 0 for m in range(5)}
            for start in range(0, len(records) - len(records) % 4, 4):
                counts[str(sum(not record["neural_correct"] for record in records[start:start + 4]))] += 1
            grouped_trials.append({"snr_db": row["snr_db"], "p": failures / len(records),
                                   "epochs": sum(counts.values()), "retained_debt_epochs": counts})

    # Deployment epochs (Q2). The transition-SNR TBs of consecutive epochs of
    # one run form a four-TB epoch; the first certificate build of each epoch
    # is the host control sample.
    transition = [tuple(key) for key in load("q2_protocol")["low_snr_keys"]]
    joined = {str(m): 0 for m in range(5)}
    failed = outcomes = 0
    build_ms = []
    for key in ("q2_dev", "q2_holdout"):
        rounds = sorted(json.loads(BACKGROUND_RAW[key].read_text())["rounds"], key=lambda row: row["sequence"])
        assert len(rounds) % 2 == 0
        per_epoch = []
        for row in rounds:
            unresolved = {tuple(item) for item in row["outcome_transition"]["unresolved_obligations"]}
            assert unresolved <= set(transition), unresolved
            per_epoch.append(len(unresolved))
            failed += len(unresolved)
            outcomes += len(transition)
            build_ms.append(round(row["launch_revalidation_attempts"][0]["elapsed_ms"], 4))
        for first, second in zip(per_epoch[0::2], per_epoch[1::2]):
            joined[str(first + second)] += 1
    deployment = {"p": failed / outcomes, "epochs": sum(joined.values()), "retained_debt_epochs": joined}

    # MPS share sweeps: one two-cell PUSCH pipeline co-running a light or a
    # heavy Qwen prefill class at each active-thread share.
    sweeps = {}
    for name in ("light", "heavy"):
        protocol = load(f"mps_{name}_protocol")
        by_share = {}
        for row in load(f"mps_{name}"):
            share = 0 if row["condition"] == "alone" else int(row["condition"].split("_")[0][len("cap"):])
            entry = by_share.setdefault(str(share), {"p99_ms": [], "max_ms": [], "misses": 0, "samples": 0,
                                                     "ai_units_per_s": []})
            entry["p99_ms"].append(row["ran_p99_ms"])
            entry["max_ms"].append(row["ran_max_ms"])
            entry["misses"] += row["deadline_misses"]
            entry["samples"] += row["samples"]
            entry["ai_units_per_s"].append(row["ai_units_per_s"])
        sweeps[name] = {"ai": protocol["ai"], "deadline_ms": protocol["ran"]["deadline_ms"], "by_share": by_share}

    # End-to-end paths against one GPU kernel. The receiver pairs meet a 4.5 ms
    # indication deadline; the kernel is the NeuralRx TensorRT graph.
    pair_python = json.loads(BACKGROUND_RAW["pair_python"].read_text())
    pair_native = json.loads(BACKGROUND_RAW["pair_native"].read_text())
    kernel_warmup = load("kernel_analysis")["warmup_excluded"]
    kernel = [record["stage_profile"]["stages_gpu_ms"]["tensorrt_graph"]
              for record in json.loads(BACKGROUND_RAW["kernel"].read_text())["records"]
              if record["sequence"] > kernel_warmup]
    latency = {
        "certificate_build": build_ms,
        "pair_python": [round(record["parallel_pair_wall_ms"], 4) for record in pair_python["records"]],
        "pair_native": [round(record["parallel_pair_wall_ms"], 4) for record in pair_native["records"]],
        "kernel": [round(value, 5) for value in kernel],
    }

    data = {
        "schema": "softwall-background-data-v1",
        "scope": ("Receiver waterfalls from Confirm18 (synthetic block Rayleigh, cuPHY and NeuralRx, 500 TBs per "
                  "SNR) and the Sionna CDL-D/E holdout (50 TBs per stratum); grouped receiver trials and Q2 "
                  "deployment epochs joined into four-TB epochs; MPS share sweeps of Confirm10/11 (two-cell "
                  "273-PRB PUSCH, 21 ms deadline); host certificate build times of Q2; cross-GPU receiver pairs "
                  "of C163 and C168 at the 4.5 ms scale; and the TensorRT graph stage of C163."),
        "raw_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in BACKGROUND_RAW.values()},
        "waterfalls": waterfalls,
        "grouped_receiver_trials": grouped_trials,
        "deployment_joined": deployment,
        "mps_sweeps": sweeps,
        "latency_ms": latency,
    }
    BACKGROUND_DATA.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def top_legend(ax, ncol: int, handles=None, labels=None, align: str = "center", title=None) -> None:
    """Legend above the axes and above the panel marker; a right-aligned legend ends at the right spine."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    anchor = {"center": ("lower center", 0.5), "right": ("lower right", 1.0)}[align]
    legend = ax.legend(handles, labels, loc=anchor[0], bbox_to_anchor=(anchor[1], 1.13), ncol=ncol,
                       frameon=False, handlelength=1.3, handletextpad=0.35, columnspacing=0.7, labelspacing=0.2,
                       borderaxespad=0.0, title=title, title_fontproperties={"weight": "bold", "size": 7.5})
    if title:
        legend._legend_box.align = "left"


def make_execution() -> dict:
    """GPU sharing and end-to-end latency tails."""
    data = json.loads(BACKGROUND_DATA.read_text())
    sweeps = data["mps_sweeps"]
    shares = [20, 40, 60, 80, 100]
    deadline = sweeps["heavy"]["deadline_ms"]
    assert deadline == sweeps["light"]["deadline_ms"] == 21
    assert sweeps["heavy"]["by_share"]["100"]["misses"] == 93
    assert all(sweeps[name]["by_share"][str(s)]["misses"] == 0
               for name in ("light", "heavy") for s in [0] + shares if (name, s) != ("heavy", 100))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3), gridspec_kw={"width_ratios": [0.85, 1.15]})
    fig.subplots_adjust(wspace=0.45)

    # (a) Radio p99 against the MPS share of a co-running Qwen prefill class.
    # Lines connect the mean over runs; bars span the runs.
    ax = axes[0]
    alone = [value for sweep in sweeps.values() for value in sweep["by_share"]["0"]["p99_ms"]]
    ax.axhspan(min(alone), max(alone), color=COLORS["light_gray"], zorder=1, label="radio alone")
    ax.axhline(deadline, color=COLORS["red"], linestyle="--", linewidth=1.2, zorder=2, label="radio deadline")
    summary = {}
    for name, color, marker, label in (("light", COLORS["green"], "o", "light AI"),
                                       ("heavy", COLORS["purple"], "s", "heavy AI")):
        runs = [sweeps[name]["by_share"][str(s)]["p99_ms"] for s in shares]
        ax.vlines(shares, [min(r) for r in runs], [max(r) for r in runs], color=color, linewidth=1.3, zorder=3)
        ax.plot(shares, [statistics.mean(r) for r in runs], color=color, marker=marker, markersize=3.8,
                linewidth=1.5, zorder=4, label=label)
        summary[name] = {str(s): [min(r), max(r)] for s, r in zip(shares, runs)}
    ax.set_xticks(shares)
    ax.set_xlim(10, 110)
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 10, 20])
    ax.set_xlabel("AI GPU share by MPS (%)")
    ax.set_ylabel("radio p99\nlatency (ms)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("light AI", "radio alone", "heavy AI", "radio deadline")]
    top_legend(ax, ncol=2, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
    panel_label(ax, "a")

    # (b) Complementary CDF of latency normalized by its own median.
    ax = axes[1]
    series = (("certificate_build", "schedule build", COLORS["orange"]),
              ("pair_python", "receivers (Python)", COLORS["brown"]),
              ("pair_native", "receivers (C++)", COLORS["cyan"]),
              ("kernel", "NeuralRx kernel", COLORS["gray"]))
    tails = {}
    for key, label, color in series:
        values = sorted(data["latency_ms"][key])
        median = statistics.median(values)
        n = len(values)
        ax.step([value / median for value in values], [(n - i) / n for i in range(n)], where="pre", color=color,
                linewidth=1.5, label=label)
        tails[key] = {"n": n, "median_ms": median, "max_ms": values[-1], "max_over_median": values[-1] / median}
    assert len(data["latency_ms"]["certificate_build"]) == 1200
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.8, 40)
    ax.set_ylim(5e-4, 1.5)
    ax.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    ax.set_xlabel("latency / median latency")
    ax.set_ylabel("fraction of runs\nabove")
    ax.grid(which="major", **GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("schedule build", "receivers (Python)", "receivers (C++)",
                                             "NeuralRx kernel")]
    # No curve enters the upper-right area, which holds the legend.
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", frameon=False,
              fontsize=6.6, handlelength=1.4, handletextpad=0.35, labelspacing=0.15, borderaxespad=0.2)
    panel_label(ax, "b")

    save(fig, "softwall_execution")
    return {"deadline_ms": deadline, "p99_range_ms": summary,
            "alone_p99_range_ms": [min(alone), max(alone)], "tails": tails}


def make_debt() -> dict:
    """Receiver outcome uncertainty and the retained debt of a four-TB epoch."""
    data = json.loads(BACKGROUND_DATA.read_text())
    deployment = data["deployment_joined"]
    assert deployment["epochs"] == 600

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3), gridspec_kw={"width_ratios": [0.85, 1.15]})
    fig.subplots_adjust(wspace=0.6)

    # (a) Waterfalls of both receivers, shifted so that the conventional
    # receiver decodes half of the TBs at 0 dB in every channel model.
    ax = axes[0]
    channel_colors = {"Rayleigh": COLORS["orange"], "CDL-D": COLORS["green"], "CDL-E": COLORS["purple"]}
    for name, color in channel_colors.items():
        waterfall = data["waterfalls"][name]
        offset = [row["snr_db"] - waterfall["conventional_50pct_snr_db"] for row in waterfall["rows"]]
        neural = [100.0 * row["neural_correct"] / row["trials"] for row in waterfall["rows"]]
        conventional = [100.0 * row["conventional_correct"] / row["trials"] for row in waterfall["rows"]]
        ax.plot(offset, neural, color=color, linestyle="-", marker="o", markersize=3.0, linewidth=1.5, zorder=3)
        ax.plot(offset, conventional, color=color, linestyle="--", marker="o", markersize=3.0,
                markerfacecolor="white", linewidth=1.2, zorder=3)
    ax.set_xlim(-1.7, 0.6)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("relative SNR (dB)")
    ax.set_ylabel("TBs decoded (%)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=color, linewidth=2.2) for color in channel_colors.values()]
    handles += [Line2D([], [], color=COLORS["dark"], linestyle="-", linewidth=1.5),
                Line2D([], [], color=COLORS["dark"], linestyle="--", linewidth=1.2)]
    labels = list(channel_colors) + ["NeuralRx", "conventional"]
    top_legend(ax, ncol=2, handles=handles, labels=labels)
    panel_label(ax, "a")

    # (b) Probability that an epoch with four admitted TBs retains at least k
    # debts. Curves are independent failures; the band raises the correlation
    # of the all-four event to 0.3. Markers are measured epochs.
    ax = axes[1]
    ps = [10 ** (-2 + 2 * i / 240) for i in range(240)]
    shades = ["#BDBDBD", "#969696", "#636363", "#252525"]
    for k in range(1, 5):
        ax.plot(ps, [debt_tail_model(p, k, 0.0) for p in ps], color=shades[k - 1], linewidth=1.5, zorder=3)
    ax.fill_between(ps, [debt_tail_model(p, 4, 0.0) for p in ps], [debt_tail_model(p, 4, 0.3) for p in ps],
                    color=COLORS["light_red"], linewidth=0, zorder=2)
    ax.plot(ps, [debt_tail_model(p, 4, 0.3) for p in ps], color=COLORS["red"], linestyle="--", linewidth=1.0,
            zorder=2)
    # Curve labels sit where no other curve or the band passes.
    for k, (p, y, va) in zip(range(1, 5), ((0.013, 0.09, "bottom"), (0.04, 0.0165, "bottom"),
                                           (0.05, 0.0008, "bottom"), (0.36, 0.0012, "top"))):
        ax.text(p, y, f"$k$={k}", ha="center", va=va, fontsize=6.8,
                color=shades[k - 1] if k > 1 else COLORS["dark"])
    measured = [(deployment["p"], debt_histogram_tail(deployment["retained_debt_epochs"]), "o")]
    measured += [(group["p"], debt_histogram_tail(group["retained_debt_epochs"]), "s")
                 for group in data["grouped_receiver_trials"]]
    for p, tail, marker in measured:
        for k, y in zip(range(1, 5), tail):
            if y > 0:
                ax.scatter(p, y, marker=marker, s=16, color=shades[k - 1], edgecolor=COLORS["dark"],
                           linewidth=0.6, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1.0)
    ax.set_ylim(1e-4, 1.6)
    ax.set_xticks([0.01, 0.1, 1.0], ["0.01", "0.1", "1"])
    ax.set_xlabel("per-TB failure probability")
    ax.set_ylabel("P(at least $k$ of\n4 TBs fail)")
    ax.grid(which="major", **GRID)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=shades[3], linewidth=1.5),
               Rectangle((0, 0), 1, 1, facecolor=COLORS["light_red"], edgecolor=COLORS["red"], linestyle="--"),
               Line2D([], [], color=shades[1], marker="o", linestyle="none", markeredgecolor=COLORS["dark"],
                      markersize=4.5),
               Line2D([], [], color=shades[1], marker="s", linestyle="none", markeredgecolor=COLORS["dark"],
                      markersize=4.2)]
    top_legend(ax, ncol=2, handles=[handles[0], handles[2], handles[1], handles[3]],
               labels=["independent", "measured periods", "correlated", "grouped trials"], align="right")
    panel_label(ax, "b")

    save(fig, "softwall_debt")
    return {
        "waterfall_midpoints_db": {name: data["waterfalls"][name]["conventional_50pct_snr_db"]
                                   for name in channel_colors},
        "deployment_p": deployment["p"],
        "deployment_tail": debt_histogram_tail(deployment["retained_debt_epochs"]),
        "grouped_trials": [{"snr_db": g["snr_db"], "p": g["p"], "tail": debt_histogram_tail(g["retained_debt_epochs"])}
                           for g in data["grouped_receiver_trials"]],
        "model_at_p_0_1": {"at_least_one": debt_tail_model(0.1, 1, 0.0), "all_four": debt_tail_model(0.1, 4, 0.0),
                           "all_four_rho_0_1": debt_tail_model(0.1, 4, 0.1),
                           "all_four_rho_0_3": debt_tail_model(0.1, 4, 0.3)},
    }


def make_envelope() -> dict:
    """Certified AI-admission frontier and scheduler cost (C162)."""
    grid = load("envelope_grid")
    by_count = load("envelope_scale")["large_grid"]["by_debt"]
    cases = load("envelope_protocol")["cases"]
    contexts = [64, 128, 256, 512]
    shades = {64: "#9ECAE1", 128: "#6BAED6", 256: "#3182BD", 512: "#08519C"}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3), gridspec_kw={"width_ratios": [1.1, 1.05]})
    fig.subplots_adjust(wspace=0.5)

    # (a) Latest decision time at which one AI unit of each class still admits
    # an all-fail schedule, for four admitted TBs and 0-4 pending recoveries.
    ax = axes[0]
    frontier = {}
    for context in contexts:
        points = []
        for pending in range(5):
            safe = [row["decision_time_ms"] for row in grid["qualified_rows"]
                    if row["accepted_debts"] == 4 and row["unresolved_debts"] == pending
                    and row["context_length"] == context and row["state"] == "QSU"]
            if safe:
                points.append((max(safe), pending))
        frontier[str(context)] = points
        ax.plot([x for x, _ in points], [y for _, y in points], color=shades[context], marker="o",
                markersize=3.2, linewidth=1.5, label=f"{context} tokens")
    physical = []
    # Prompt-length label of each physical test: offset from the marker and
    # alignment, chosen so that no label touches a line or another label.
    label_at = {"E3": (-1.2, 0.33, "right"), "E4": (1.4, 0.33, "left"), "E5": (1.4, -0.33, "left"),
                "E6a": (-1.4, -0.33, "right"), "E6b": (1.4, 0.33, "left")}
    for case in cases:
        if case["context_length"] is None:
            continue
        pending = 4 - case["success_count"]
        name = case["case_id"].split("_")[0]
        x = case["conservative_prediction_time_ms"]
        physical.append({"case": name, "decision_ms": x, "pending": pending, "admit": case["expected_lease"]})
        dx, dy, align = label_at[name]
        ax.text(x + dx, pending + dy, str(case["context_length"]), ha=align, va="center", fontsize=6.2,
                color=COLORS["dark"])
        if case["expected_lease"]:
            ax.scatter(x, pending, marker="o", s=22, facecolor="white", edgecolor=COLORS["green"], linewidth=1.4,
                       zorder=5)
        else:
            ax.scatter(x, pending, marker="x", s=24, color=COLORS["red"], linewidth=1.5, zorder=5)
    # A decision left of a line admits that AI unit; a later decision rejects it.
    ax.text(56, 3.75, "\u2190 admit", ha="right", va="center", fontsize=6.4, color=COLORS["gray"])
    ax.text(104, 3.75, "reject \u2192", ha="left", va="center", fontsize=6.4, color=COLORS["gray"])
    ax.set_xlim(35, 120)
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(range(5))
    ax.set_xlabel("decision time after release (ms)")
    ax.set_ylabel("pending recoveries")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                       markeredgecolor=COLORS["green"], markeredgewidth=1.4, markersize=4.5),
                Line2D([], [], marker="x", linestyle="none", color=COLORS["red"], markeredgewidth=1.5, markersize=4.5)]
    labels += ["tested: admitted", "tested: rejected"]
    top_legend(ax, ncol=3, handles=handles, labels=labels, title="Backstop admission limit by prompt length")
    panel_label(ax, "a")

    # (b) Candidate construction and verification latency against the number
    # of pending recoveries.
    ax = axes[1]
    counts = [1, 2, 4, 8, 16, 32, 64]
    decision_p99 = [by_count[str(n)]["decision_us"]["p99"] / 1000 for n in counts]
    decision_max = [by_count[str(n)]["decision_us"]["max"] / 1000 for n in counts]
    verify_p99 = [by_count[str(n)]["verify_us"]["p99"] / 1000 for n in counts]
    assert round(decision_p99[-1], 3) == 1.494
    ax.plot(counts, decision_p99, color=COLORS["blue"], marker="o", markersize=3.2, linewidth=1.5,
            label="Backstop decision p99")
    ax.plot(counts, decision_max, color=COLORS["blue"], marker="s", markersize=3.0, linewidth=1.2,
            linestyle="--", label="Backstop decision max")
    ax.plot(counts, verify_p99, color=COLORS["green"], marker="D", markersize=3.0, linewidth=1.5,
            label="Backstop verifier p99")
    ax.axhline(5.0, color=COLORS["red"], linestyle="--", linewidth=1.2)
    ax.text(70, 5.1, "5 ms control budget", ha="right", va="bottom", fontsize=6.4, color=COLORS["red"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(n) for n in counts])
    ax.set_xlim(0.8, 80)
    ax.set_xlabel("pending recoveries")
    ax.set_ylabel("latency (ms)")
    ax.set_ylim(0, 6.0)
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), frameon=False, fontsize=6.6, handlelength=1.6,
              handletextpad=0.35, labelspacing=0.15, borderaxespad=0.2)
    panel_label(ax, "b")

    save(fig, "softwall_envelope")
    return {"frontier": frontier, "physical_cases": physical, "decision_p99_ms": decision_p99,
            "decision_max_ms": decision_max, "verify_p99_ms": verify_p99}


def figure_legend(fig, handles, labels, ncol: int, compact: bool = False) -> None:
    """One legend above all panels of a figure; compact spacing keeps long names within the width."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
               handlelength=1.3 if compact else 1.8, handletextpad=0.35 if compact else 0.4,
               columnspacing=0.8 if compact else 1.2, borderaxespad=0.0)


def derive_eval() -> None:
    """Per-attempt boundary records of the two-node debt-blind diagnostic (C172)."""
    if not all(path.exists() for path in EVAL_RAW.values()):
        return
    records = []
    for key, result in (("c172_dev", "necessity_dev"), ("c172_holdout", "necessity_holdout")):
        pinned = load(result)["artifact_sha256"]["coordinator"]
        assert sha256(EVAL_RAW[key]) == pinned, f"raw log differs from pinned digest: {key}"
        for row in json.loads(EVAL_RAW[key].read_text())["rounds"]:
            records.append({"run": key, "scenario": row["scenario_id"], "policy": row["policy"],
                            "lease_accepted": row["lease_accepted"],
                            "decision_ms": round(row["observed_decision_ms"], 4),
                            "last_recovery_complete_ms": round(row["last_recovery_complete_ms"], 4)})
    data = {
        "schema": "softwall-eval-data-v1",
        "scope": ("Every physical attempt of the frozen C172 development and holdout runs. The last recovery "
                  "completion is measured from release; the guard is 153 ms and the expiry 155 ms."),
        "raw_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in EVAL_RAW.values()},
        "boundary_attempts": records,
    }
    EVAL_DATA.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def make_eval_capacity() -> dict:
    """Safe AI capacity of each policy in the exact-checked capacity model (C167)."""
    capacity = load("capacity")
    rows = [dict(zip(capacity["row_fields"], row)) for row in capacity["rows"]]
    feasible = [row for row in rows if row["mandatory_feasible"] == 1]
    assert len(feasible) == 13716 and capacity["exact_checker"]["mismatches"] == 0
    assert all(abs(row["softwall_requests"] - row["oracle_requests"]) < 1e-9 for row in feasible)
    useful = [row for row in feasible if row["oracle_requests"] > 0]
    policies = (("static_requests", "static", COLORS["gray"], ":", "o"),
                ("recovery_first_requests", "recovery-first", COLORS["gray"], "--", "s"),
                ("softwall_requests", "Backstop", COLORS["blue"], "-", "D"))

    def share(group, key):
        return 100.0 * statistics.mean(row[key] / row["oracle_requests"] for row in group)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 1.35))
    fig.subplots_adjust(wspace=0.62)
    cells = [2, 4, 8, 16]
    deadlines = [50, 100, 1000]
    summary = {"by_cells": {}, "by_deadline": {}, "idle_violation_by_cells": {}}

    # (a) Capacity as the number of cells grows.
    ax = axes[0]
    for key, label, color, style, marker in policies:
        values = [share([row for row in useful if row["cells"] == c], key) for c in cells]
        summary["by_cells"][label] = dict(zip(map(str, cells), values))
        ax.plot(range(len(cells)), values, color=color, linestyle=style, marker=marker, markersize=3.6,
                linewidth=1.5, label=label)
    ax.set_xticks(range(len(cells)), [str(c) for c in cells])
    ax.set_xlabel("number of cells")
    ax.set_ylabel("safe AI capacity\n(% of oracle)")
    ax.set_ylim(40, 105)
    ax.set_yticks([50, 75, 100])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Capacity as the AI deadline tightens.
    ax = axes[1]
    for key, label, color, style, marker in policies:
        values = [share([row for row in useful if row["ai_deadline_ms"] == d], key) for d in deadlines]
        summary["by_deadline"][label] = dict(zip(map(str, deadlines), values))
        ax.plot(range(len(deadlines)), values, color=color, linestyle=style, marker=marker, markersize=3.6,
                linewidth=1.5)
    ax.set_xticks(range(len(deadlines)), [str(d) for d in deadlines])
    ax.set_xlabel("AI deadline (ms)")
    ax.set_ylim(40, 105)
    ax.set_yticks([50, 75, 100])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # (c) Radio violation probability of idle-time admission; the certified
    # policies never violate the all-fail schedule.
    ax = axes[2]
    violation = [100.0 * statistics.mean(row["debt_blind_violation_probability"] for row in feasible
                                         if row["cells"] == c) for c in cells]
    summary["idle_violation_by_cells"] = dict(zip(map(str, cells), violation))
    ax.plot(range(len(cells)), violation, color=COLORS["red"], marker="^", markersize=3.8, linewidth=1.5,
            label="idle-time")
    ax.plot(range(len(cells)), [0.0] * len(cells), color=COLORS["blue"], marker="D", markersize=3.6,
            linewidth=1.5)
    ax.set_xticks(range(len(cells)), [str(c) for c in cells])
    ax.set_xlabel("number of cells")
    ax.set_ylabel("radio violation\nprobability (%)")
    ax.set_ylim(-2, 35)
    ax.set_yticks([0, 15, 30])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3.6, linewidth=1.5)
               for _, _, color, style, marker in policies]
    handles.append(Line2D([], [], color=COLORS["red"], marker="^", markersize=3.8, linewidth=1.5))
    figure_legend(fig, handles, ["static reservation", "recovery-first", "Backstop", "idle-time admission"],
                  ncol=4, compact=True)
    save(fig, "softwall_eval_capacity")
    summary["feasible_points"] = len(feasible)
    summary["useful_points"] = len(useful)
    return summary


def make_eval_drivers() -> dict:
    """NeuralRx success and radio expiry as drivers of safe AI capacity (C167)."""
    capacity = load("capacity")
    rows = [dict(zip(capacity["row_fields"], row)) for row in capacity["rows"]]
    feasible = [row for row in rows if row["mandatory_feasible"] == 1]
    assert len(feasible) == 13716
    useful = [row for row in feasible if row["oracle_requests"] > 0]
    policies = (("static_requests", "static", COLORS["gray"], ":", "o"),
                ("recovery_first_requests", "recovery-first", COLORS["gray"], "--", "s"),
                ("softwall_requests", "Backstop", COLORS["blue"], "-", "D"))
    successes = [0.2, 0.5, 0.8]
    expiries = [100, 155, 220]
    # Every NeuralRx success probability covers the same mandatory-feasible
    # configurations, so the success panels compare identical point sets.
    by_success = {p: [row for row in feasible if row["success_probability"] == p] for p in successes}
    assert len({len(group) for group in by_success.values()}) == 1
    summary = {"units_by_success": {}, "share_by_expiry": {}, "idle_violation_by_success": {},
               "points_per_success": len(by_success[0.2]),
               "points_by_expiry": {str(e): sum(1 for row in feasible if row["expiry_ms"] == e) for e in expiries}}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 1.35))
    fig.subplots_adjust(wspace=0.78)

    # (a) Mean AI units admitted per release period as NeuralRx succeeds more
    # often. Static reservation holds every pending recovery and cannot use
    # the capacity that a success releases.
    ax = axes[0]
    for key, label, color, style, marker in policies:
        values = [statistics.mean(row[key] for row in by_success[p]) for p in successes]
        summary["units_by_success"][label] = dict(zip(map(str, successes), values))
        ax.plot(range(len(successes)), values, color=color, linestyle=style, marker=marker, markersize=3.6,
                linewidth=1.5)
    ax.set_xticks(range(len(successes)), [str(p) for p in successes])
    ax.set_xlabel("NeuralRx success rate")
    ax.set_ylabel("admitted AI requests\nper period")
    ax.set_ylim(0.8, 1.6)
    ax.set_yticks([1.0, 1.2, 1.4])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Capacity relative to the oracle as the radio expiry tightens.
    ax = axes[1]
    for key, label, color, style, marker in policies:
        values = [100.0 * statistics.mean(row[key] / row["oracle_requests"] for row in useful
                                          if row["expiry_ms"] == e) for e in expiries]
        summary["share_by_expiry"][label] = dict(zip(map(str, expiries), values))
        ax.plot(range(len(expiries)), values, color=color, linestyle=style, marker=marker, markersize=3.6,
                linewidth=1.5)
    ax.set_xticks(range(len(expiries)), [str(e) for e in expiries])
    ax.set_xlabel("radio expiry (ms)")
    ax.set_ylabel("safe AI capacity\n(% of oracle)")
    ax.set_ylim(25, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # (c) Radio violation probability of idle-time admission as NeuralRx
    # succeeds more often; the certified policies never violate.
    ax = axes[2]
    violation = [100.0 * statistics.mean(row["debt_blind_violation_probability"] for row in by_success[p])
                 for p in successes]
    summary["idle_violation_by_success"] = dict(zip(map(str, successes), violation))
    ax.plot(range(len(successes)), violation, color=COLORS["red"], marker="^", markersize=3.8, linewidth=1.5)
    ax.plot(range(len(successes)), [0.0] * len(successes), color=COLORS["blue"], marker="D", markersize=3.6,
            linewidth=1.5)
    ax.set_xticks(range(len(successes)), [str(p) for p in successes])
    ax.set_xlabel("NeuralRx success rate")
    ax.set_ylabel("radio violation\nprobability (%)")
    ax.set_ylim(-2, 35)
    ax.set_yticks([0, 15, 30])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3.6, linewidth=1.5)
               for _, _, color, style, marker in policies]
    handles.append(Line2D([], [], color=COLORS["red"], marker="^", markersize=3.8, linewidth=1.5))
    figure_legend(fig, handles, ["static reservation", "recovery-first", "Backstop", "idle-time admission"],
                  ncol=4, compact=True)
    save(fig, "softwall_eval_drivers")
    return summary


def make_eval_trace() -> dict:
    """Timely AI value on the BurstGPT headroom points (C174/C175)."""
    rows = load("levers")["rows"]
    assert len(rows) == 362
    gpus = [1, 2, 4]
    rf = {g: [] for g in gpus}
    sw = {g: [] for g in gpus}
    ratios = []
    for row in rows:
        values = row["lever_values"]
        oracle = values["fixed_placement_oracle"]
        rf[row["gpus"]].append(100.0 * values["recovery_first"] / oracle)
        sw[row["gpus"]].append(100.0 * values["ai_first_greedy"] / oracle)
        ratios.append(values["ai_first_greedy"] / values["recovery_first"])

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3))
    fig.subplots_adjust(wspace=0.4)

    # (a) Mean timely value per GPU count, normalized to the fixed-placement oracle.
    ax = axes[0]
    xs = list(range(len(gpus)))
    width = 0.34
    rf_mean = [statistics.mean(rf[g]) for g in gpus]
    sw_mean = [statistics.mean(sw[g]) for g in gpus]
    ax.bar([x - width / 2 for x in xs], rf_mean, width=width, color=COLORS["gray"], zorder=3)
    ax.bar([x + width / 2 for x in xs], sw_mean, width=width, color=COLORS["blue"], zorder=3)
    ax.set_xticks(xs, [str(g) for g in gpus])
    ax.set_xlabel("number of GPUs")
    ax.set_ylabel("on-time AI tokens\n(% of oracle)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Distribution of the Backstop-to-recovery-first value ratio.
    ax = axes[1]
    values = sorted(ratios)
    n = len(values)
    ax.step(values, [100.0 * (i + 1) / n for i in range(n)], where="post", color=COLORS["blue"], linewidth=1.6)
    ax.axvline(1.0, color=COLORS["gray"], linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlim(0.9, 20)
    ax.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("on-time AI tokens,\nBackstop / recovery-first")
    ax.set_ylabel("headroom points\nat or below (%)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    handles = [Rectangle((0, 0), 1, 1, color=COLORS["gray"]), Rectangle((0, 0), 1, 1, color=COLORS["blue"])]
    figure_legend(fig, handles, ["recovery-first", "Backstop"], ncol=2)
    save(fig, "softwall_eval_trace")
    return {"recovery_first_pct_of_oracle": dict(zip(map(str, gpus), rf_mean)),
            "softwall_pct_of_oracle": dict(zip(map(str, gpus), sw_mean)),
            "ratio_median": statistics.median(values), "ratio_max": values[-1], "ratio_min": values[0],
            "points": n}


def make_eval_safety() -> dict:
    """Physical boundary attempts (C172) and class-wise AI admission (Q2)."""
    data = json.loads(EVAL_DATA.read_text())
    attempts = data["boundary_attempts"]
    # Boundary case, horizontal position, and policy of each attempt group.
    groups = (("E4_debt_blind_launch", -0.2, COLORS["red"], "x"),
              ("E4_softwall_reject", 0.2, COLORS["blue"], "o"),
              ("E6b_shadow_launch", 0.8, COLORS["red"], "x"),
              ("E6b_softwall_reject", 1.2, COLORS["blue"], "o"),
              ("E6a_softwall_admit", 2.0, COLORS["blue"], "o"))
    guard, expiry = 153.0, 155.0

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.45), gridspec_kw={"width_ratios": [1.4, 1.0]})
    fig.subplots_adjust(wspace=0.4)

    # (a) Completion of the last recovery in every physical attempt. A small
    # deterministic horizontal offset separates overlapping attempts.
    ax = axes[0]
    summary = {}
    for scenario, x, color, marker in groups:
        ends = [a["last_recovery_complete_ms"] for a in attempts if a["scenario"] == scenario]
        assert len(ends) == 48
        offsets = [((i % 9) - 4) * 0.03 for i in range(len(ends))]
        if marker == "x":
            ax.scatter([x + o for o in offsets], ends, s=10, marker="x", color=color, linewidths=0.9, zorder=3)
        else:
            ax.scatter([x + o for o in offsets], ends, s=10, marker="o", facecolor="none", edgecolor=color,
                       linewidths=0.9, zorder=3)
        summary[scenario] = {"attempts": len(ends), "min_ms": min(ends), "max_ms": max(ends),
                             "beyond_guard": sum(e > guard for e in ends)}
    ax.axhline(guard, color=COLORS["red"], linestyle="--", linewidth=1.1, zorder=2)
    ax.axhline(expiry, color=COLORS["dark"], linestyle=":", linewidth=1.1, zorder=2)
    ax.text(-0.55, guard - 1.5, "recovery deadline", ha="left", va="top", fontsize=6.5, color=COLORS["red"])
    ax.text(2.45, expiry + 1.5, "expiry", ha="right", va="bottom", fontsize=6.5, color=COLORS["dark"])
    ax.set_xticks([0, 1, 2], ["E4\n256 tokens\n2 recoveries", "E6b\n64 tokens\nat 89 ms",
                              "E6a\n64 tokens\nat 88 ms"])
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlim(-0.6, 2.5)
    ax.set_ylim(85, 172)
    ax.set_yticks([100, 125, 150])
    ax.set_ylabel("last recovery finish\n(ms after release)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) AI units per prompt length in the two-node campaign: run by Backstop,
    # and rejected by Backstop because they break the certified recovery bound
    # (idle-time admission would run them). Static reservation runs none.
    ax = axes[1]
    q2 = load("q2")["summary"]
    contexts = [16, 32, 64, 128, 256, 512]
    executed = [q2["qwen_executed_by_class"][str(c)] for c in contexts]
    rejected = [q2["qwen_certificate_rejected_by_class"][str(c)] for c in contexts]
    assert sum(executed) == 1077 and [e + r for e, r in zip(executed, rejected)] == [200] * 6
    xs = list(range(len(contexts)))
    ax.bar(xs, executed, width=0.62, color=COLORS["blue"], zorder=3)
    broken = [(x, e, r) for x, e, r in zip(xs, executed, rejected) if r > 0]
    ax.bar([x for x, _, _ in broken], [r for _, _, r in broken], bottom=[e for _, e, _ in broken], width=0.62,
           color=COLORS["light_red"], edgecolor=COLORS["red"], hatch="/////", linewidth=0.8, zorder=3)
    ax.set_xticks(xs, [str(c) for c in contexts])
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("AI requests\n(Qwen prefill)")
    ax.set_ylim(0, 215)
    ax.set_yticks([0, 100, 200])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # Legend rows: (a) attempt markers, then (b) bar segments.
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor=COLORS["blue"], markersize=4.5),
               Rectangle((0, 0), 1, 1, color=COLORS["blue"]),
               Line2D([], [], marker="x", linestyle="none", color=COLORS["red"], markersize=4.5),
               Rectangle((0, 0), 1, 1, facecolor=COLORS["light_red"], edgecolor=COLORS["red"], hatch="/////")]
    figure_legend(fig, handles, ["Backstop attempt", "AI run by Backstop", "AI launched anyway (baseline)",
                                 "AI rejected by Backstop"], ncol=2, compact=True)
    save(fig, "softwall_eval_safety")
    return {"boundary": summary, "qwen_executed": dict(zip(map(str, contexts), executed)),
            "qwen_rejected": dict(zip(map(str, contexts), rejected))}


FAULT_RUNS = ("c161p1a_dev_j58859044", "c161p1b_holdout_j58859145",
              "c161p2d_dev_j58859872_a0_stale_duplicate_nrx", "c161p2d_dev_j58859872_a1_post_fence_reply_delay",
              "c161p2d_dev_j58859872_a2_pre_fence_channel_loss", "c161p2d_dev_j58859872_a3_stale_duplicate_recovery",
              "c161p2e_holdout_j58859986_a0_stale_duplicate_recovery",
              "c161p2e_holdout_j58859986_a1_pre_fence_channel_loss",
              "c161p2e_holdout_j58859986_a2_post_fence_reply_delay",
              "c161p2e_holdout_j58859986_a3_stale_duplicate_nrx")
FAULT_ORDER = (("no_fault", "A0 none"), ("correlated_all_fail", "A1 all-fail"),
               ("stale_duplicate_nrx", "A2 stale NRx"), ("post_fence_reply_delay", "A3 late reply"),
               ("latest_start_nonlaunch", "A4 held lease"), ("pre_fence_channel_loss", "A5 lost channel"),
               ("stale_duplicate_recovery", "A6 stale recovery"))


def pinned_digests(*names: str) -> dict:
    """Every 64-hex digest recorded under a results path in the given summaries."""
    digests = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.startswith("results/") and len(value) == 64:
                    digests[key] = value
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for name in names:
        walk(load(name))
    return digests


def derive_runtime() -> None:
    """Per-period admission records (Q2) and fault-campaign commits and leases (C161)."""
    raw = ROOT / "results/softwall_multigpu/raw"
    coordinators = [raw / f"{run}_coordinator.json" for run in FAULT_RUNS]
    if not all(path.exists() for path in coordinators):
        return
    pinned = pinned_digests("fault_phase1", "fault_phase2")
    q2_pinned = load("q2")["artifact_sha256"]
    mode = load("q2_protocol")["mode"]

    # Q2: one record per release period of both runs. The certified end is the
    # end of the last lane action in the certificate committed at the outcome
    # batch: the lease runs to its latest-start plus its class bound, and each
    # required recovery then takes its full bound. A recovery-only certificate
    # ends at the last certified recovery finish. The measured end is the later
    # of the CUDA completion of the prefill and the completion of the last
    # recovery. Both are relative to the NeuralRx deadline.
    periods = []
    # Release-to-commit time of every TB, by commit path and by the AI lease
    # that precedes the recoveries of its period, and the physical path of
    # every recovery (CUDA start to completion, as in the Q2 analyzer).
    q2_commits = {"nrx": [], "none": [], "16-64": [], "128": [], "256": [], "512": []}
    recovery_paths = []
    for key in ("q2_dev", "q2_holdout"):
        path = BACKGROUND_RAW[key]
        assert q2_pinned[str(path.relative_to(ROOT))] == sha256(path)
        raw_run = json.loads(path.read_text())
        recoveries = {}
        for record in raw_run["physical_recoveries"]:
            recoveries.setdefault(record["sequence"], []).append(record)
            recovery_paths.append(round((record["actual_completed_ns"] - record["actual_start_ns"]) / 1e6, 4))
        by_sequence = {row["sequence"]: row for row in raw_run["rounds"]}
        owners = sorted(path.parent.glob(path.name.replace("_coordinator.json", "_home*_owner.json")))
        assert len(owners) == 4, key
        for owner in owners:
            assert q2_pinned[str(owner.relative_to(ROOT))] == sha256(owner), owner.name
            for record in json.loads(owner.read_text())["records"]:
                assert record["commit_count"] == 1 and not record["deadline_miss"]
                row = by_sequence[record["sequence"]]
                if record["commit_source"] == "actual_nrx":
                    group = "nrx"
                else:
                    assert record["commit_source"] == "shared_conventional"
                    group = ("none" if not row["lease_accepted"] else
                             "16-64" if row["context_length"] <= 64 else str(row["context_length"]))
                q2_commits[group].append(round(record["release_to_commit_ms"], 3))
        for row in sorted(raw_run["rounds"], key=lambda r: r["sequence"]):
            qwen = row["qwen"] if row["lease_accepted"] else None
            required = len(row["outcome_transition"]["unresolved_obligations"])
            lane = recoveries.get(row["sequence"], [])
            assert len(lane) == required
            assert all(r["release_to_complete_ms"] * 1e6 <= r["model_finish_ns"] for r in lane)
            ends = [r["release_to_complete_ms"] for r in lane]
            if qwen:
                latest_start = (qwen["latest_start_ns"] - row["release_wall_ns"]) / 1e6
                certified = latest_start + mode["ai_class_bounds_ms"][str(row["context_length"])] \
                    + required * mode["conventional_bound_ms"]
                ends.append((qwen["worker_completed_ns"] - row["release_wall_ns"]) / 1e6)
            else:
                certified = max(r["model_finish_ns"] for r in lane) / 1e6
            periods.append({
                "run": key, "sequence": row["sequence"], "pending": required,
                "context": row["context_length"], "admitted": row["lease_accepted"],
                "execution_ms": round(qwen["execution_ms"], 3) if qwen else None,
                "arrival_minus_latest_start_ms": (round((qwen["worker_accepted_ns"] - qwen["latest_start_ns"]) / 1e6, 4)
                                                  if qwen else None),
                "decision_ms": round(row["outcome_transition"]["applied_at_ns"] / 1e6, 4),
                "certified_end_ms": round(certified - mode["nrx_bound_ms"], 4),
                "measured_end_ms": round(max(ends) - mode["nrx_bound_ms"], 4),
            })

    # C161: commits of every TB per fault class and every lease held past its latest-start.
    commits = {name: [] for name, _ in FAULT_ORDER}
    commit_counts = {name: 0 for name, _ in FAULT_ORDER}
    misses = {name: 0 for name, _ in FAULT_ORDER}
    held = []
    for run in FAULT_RUNS:
        coordinator = raw / f"{run}_coordinator.json"
        assert pinned[str(coordinator.relative_to(ROOT))] == sha256(coordinator), run
        rounds = json.loads(coordinator.read_text())["rounds"]
        arm_by_sequence = {row["sequence"]: row["fault_arm"] for row in rounds}
        for row in rounds:
            attempt = row.get("qwen_attempt")
            if row["fault_arm"] == "latest_start_nonlaunch" and attempt:
                held.append({"arrival_minus_latest_start_ms":
                             round((attempt["worker_accepted_ns"] - attempt["latest_start_ns"]) / 1e6, 4),
                             "launched": attempt["launched"], "reason": attempt.get("reason")})
        owners = sorted(raw.glob(f"{run}_home*_owner.json"))
        assert len(owners) == 4, run
        for owner in owners:
            assert pinned[str(owner.relative_to(ROOT))] == sha256(owner), owner.name
            for record in json.loads(owner.read_text())["records"]:
                arm = arm_by_sequence[record["sequence"]]
                commits[arm].append(round(record["release_to_commit_ms"], 3))
                commit_counts[arm] += record["commit_count"]
                misses[arm] += bool(record["deadline_miss"])

    data = {
        "schema": "softwall-runtime-data-v1",
        "scope": ("Every release period of the Q2 development and holdout runs, and every TB commit and held "
                  "lease of the qualified C161 phase-1 and phase-2 runs on two nodes. Arrival is the time at "
                  "which the Qwen worker accepts a lease; a lease that arrives after its latest-start does not "
                  "launch."),
        "mode": {"expiry_ms": mode["expiry_ms"], "guard_ms": mode["guard_ms"], "nrx_bound_ms": mode["nrx_bound_ms"],
                 "conventional_bound_ms": mode["conventional_bound_ms"],
                 "launch_control_bound_ms": mode["launch_control_bound_ms"],
                 "ai_class_bounds_ms": mode["ai_class_bounds_ms"]},
        "q2_periods": periods,
        "q2_commits_ms": q2_commits,
        "q2_recovery_path_ms": recovery_paths,
        "fault_commits_ms": commits,
        "fault_commit_counts": commit_counts,
        "fault_deadline_misses": misses,
        "held_leases": held,
    }
    RUNTIME_DATA.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def make_eval_online() -> dict:
    """Certified schedules and their physical execution in the two-node campaign (Q2)."""
    data = json.loads(RUNTIME_DATA.read_text())
    mode = data["mode"]
    periods = data["q2_periods"]
    assert len(periods) == 1200 and sum(p["admitted"] for p in periods) == 1077
    window = mode["expiry_ms"] - mode["guard_ms"] - mode["nrx_bound_ms"]
    recovery = mode["conventional_bound_ms"]
    charge = mode["launch_control_bound_ms"]
    bounds = {int(k): v for k, v in mode["ai_class_bounds_ms"].items()}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.45), gridspec_kw={"width_ratios": [1.45, 1.1]})
    fig.subplots_adjust(wspace=0.36)

    # (a) The certified schedule of 36 consecutive periods: an admitted AI lease
    # first, then the required recoveries, all before the guard. A rejected AI
    # lease is drawn above the recoveries where it would cross the guard.
    ax = axes[0]
    shown = [p for p in periods if p["run"] == "q2_dev"][:36]
    for x, p in enumerate(shown):
        lease = bounds[p["context"]] + charge
        base = 0.0
        if p["admitted"]:
            ax.bar(x, lease, bottom=0, width=0.78, color=COLORS["light_blue"], edgecolor=COLORS["blue"],
                   linewidth=0.5, zorder=3)
            ax.bar(x, p["execution_ms"], bottom=charge, width=0.36, color=COLORS["blue"], zorder=4)
            base = lease
        ax.bar(x, p["pending"] * recovery, bottom=base, width=0.78, color=COLORS["light_orange"],
               edgecolor=COLORS["orange"], linewidth=0.5, zorder=3)
        if not p["admitted"]:
            ax.bar(x, lease, bottom=p["pending"] * recovery, width=0.78, color="none", edgecolor=COLORS["red"],
                   hatch="/////", linewidth=0.7, zorder=3)
    ax.axhline(window, color=COLORS["red"], linestyle="--", linewidth=1.1, zorder=5)
    ax.text(21.0, window + 3, "recovery deadline", ha="center", va="bottom", fontsize=6.5, color=COLORS["red"])
    ax.set_xlim(-0.8, 35.8)
    ax.set_ylim(0, 150)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks([0, 11, 23, 35], ["1", "12", "24", "36"])
    ax.set_xlabel("release period")
    ax.set_ylabel("time after NeuralRx\ndeadline (ms)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    lane_handles = [Rectangle((0, 0), 1, 1, facecolor=COLORS["light_blue"], edgecolor=COLORS["blue"]),
                    Rectangle((0, 0), 1, 1, facecolor=COLORS["blue"]),
                    Rectangle((0, 0), 1, 1, facecolor=COLORS["light_orange"], edgecolor=COLORS["orange"]),
                    Rectangle((0, 0), 1, 1, facecolor=COLORS["light_red"], edgecolor=COLORS["red"], hatch="/////")]
    lane_labels = ["Backstop AI slot", "measured prefill", "recovery", "rejected AI slot"]

    # (b) Measured against certified end of the last lane action, all 1,200
    # periods. Points below the diagonal finish inside their certificate.
    ax = axes[1]
    kinds = (("AI, no recovery", lambda p: p["admitted"] and p["pending"] == 0, "o", COLORS["blue"], "none"),
             ("AI, 1 recovery", lambda p: p["admitted"] and p["pending"] == 1, "^", COLORS["blue"], COLORS["blue"]),
             ("AI, 2 recoveries", lambda p: p["admitted"] and p["pending"] == 2, "s", COLORS["blue"],
              COLORS["light_blue"]),
             ("AI rejected", lambda p: not p["admitted"], "s", COLORS["orange"], COLORS["light_orange"]))
    assert sum(sum(1 for p in periods if test(p)) for _, test, *_ in kinds) == len(periods)
    handles = []
    for label, test, marker, edge, face in kinds:
        chosen = [p for p in periods if test(p)]
        ax.scatter([p["certified_end_ms"] for p in chosen], [p["measured_end_ms"] for p in chosen], s=11,
                   marker=marker, facecolor=face, edgecolor=edge, linewidths=0.6, zorder=3)
        handles.append(Line2D([], [], marker=marker, linestyle="none", markerfacecolor=face, markeredgecolor=edge,
                              markersize=4.2))
    assert all(p["measured_end_ms"] < p["certified_end_ms"] <= window for p in periods)
    ax.plot([0, 120], [0, 120], color=COLORS["gray"], linestyle=":", linewidth=1.0, zorder=2)
    ax.text(6, 112, "Backstop periods", ha="left", va="top", fontsize=6.6, color=COLORS["blue"])
    ax.text(40, 44, "measured = scheduled", ha="center", va="bottom", fontsize=6.0, color=COLORS["gray"],
            rotation=45, rotation_mode="anchor", transform_rotates_text=True)
    ax.axvline(window, color=COLORS["red"], linestyle="--", linewidth=1.1, zorder=2)
    ax.text(window - 2, 6, "recovery\ndeadline", ha="right", va="bottom", fontsize=6.5, color=COLORS["red"])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 120)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("scheduled finish (ms)")
    ax.set_ylabel("measured finish (ms)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # One legend above both panels: the first row names the lane intervals of
    # (a), the second row the schedule kinds of (b).
    order = [h for pair in zip(lane_handles, handles) for h in pair]
    names = [n for pair in zip(lane_labels, [k[0] for k in kinds]) for n in pair]
    figure_legend(fig, order, names, ncol=4, compact=True)
    save(fig, "softwall_eval_online")
    late = [p for p in periods if p["decision_ms"] > mode["nrx_bound_ms"] + 1.0]
    return {"window_ms": window,
            "measured_end_max_ms": max(p["measured_end_ms"] for p in periods),
            "certified_end_max_ms": max(p["certified_end_ms"] for p in periods),
            "min_slack_to_certificate_ms": min(p["certified_end_ms"] - p["measured_end_ms"] for p in periods),
            "late_decisions": [[p["run"], p["sequence"], p["decision_ms"], p["certified_end_ms"],
                                p["measured_end_ms"]] for p in late]}


FAULT_NAMES = {"no_fault": "no fault", "correlated_all_fail": "all TBs fail",
               "stale_duplicate_nrx": "stale NeuralRx", "post_fence_reply_delay": "late AI reply",
               "latest_start_nonlaunch": "held AI slot", "pre_fence_channel_loss": "lost AI channel",
               "stale_duplicate_recovery": "stale recovery"}


def commit_boxes(ax, groups, positions):
    ax.boxplot(groups, positions=positions, widths=0.55, whis=(0, 100), patch_artist=True,
               medianprops={"color": COLORS["dark"], "linewidth": 1.0},
               boxprops={"facecolor": COLORS["light_blue"], "edgecolor": COLORS["blue"], "linewidth": 0.8},
               whiskerprops={"color": COLORS["blue"], "linewidth": 0.8},
               capprops={"color": COLORS["blue"], "linewidth": 0.8})


def make_eval_refinement() -> dict:
    """Latest-start enforcement and radio commit time without and with faults (Q2, C161)."""
    data = json.loads(RUNTIME_DATA.read_text())
    mode = data["mode"]
    expiry = mode["expiry_ms"]
    launched = [p["arrival_minus_latest_start_ms"] for p in data["q2_periods"] if p["admitted"]]
    held = data["held_leases"]
    assert len(launched) == 1077 and len(held) == 52 and not any(h["launched"] for h in held)
    assert all(v < 0 for v in launched) and all(h["arrival_minus_latest_start_ms"] > 0 for h in held)
    assert all(v == 0 for v in data["fault_deadline_misses"].values())
    q2_commits = data["q2_commits_ms"]
    order = ["none", "16-64", "128", "256", "512"]
    assert sum(len(q2_commits[name]) for name in order + ["nrx"]) == 4800 and len(q2_commits["nrx"]) == 3488
    assert max(max(values) for values in q2_commits.values()) < expiry

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 1.45), gridspec_kw={"width_ratios": [1.15, 1.0, 1.25]})
    fig.subplots_adjust(wspace=0.62)

    # (a) Arrival of each AI lease at the Qwen worker relative to its absolute
    # latest start: leases of the two-node campaign and leases held past it.
    ax = axes[0]
    rows = ((1, launched, COLORS["blue"], "o"), (0, [h["arrival_minus_latest_start_ms"] for h in held],
                                                  COLORS["red"], "x"))
    for y, values, color, marker in rows:
        offsets = [((i % 11) - 5) * 0.035 for i in range(len(values))]
        if marker == "x":
            ax.scatter(values, [y + o for o in offsets], s=9, marker="x", color=color, linewidths=0.8, zorder=3)
        else:
            ax.scatter(values, [y + o for o in offsets], s=9, marker="o", facecolor="none", edgecolor=color,
                       linewidths=0.7, zorder=3)
    ax.axvline(0.0, color=COLORS["dark"], linestyle="--", linewidth=1.1, zorder=2)
    # Each group is labeled directly above its points.
    ax.text(statistics.median(launched), 1.42, "on time", ha="center", va="bottom", fontsize=6.3,
            color=COLORS["blue"])
    ax.text(statistics.median(h["arrival_minus_latest_start_ms"] for h in held), 0.42, "too late", ha="center",
            va="bottom", fontsize=6.3, color=COLORS["red"])
    ax.set_yticks([0, 1], ["held\nAI slots", "campaign\nAI slots"])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlim(-6, 10)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_xlabel("arrival relative to\nlatest-start (ms)")
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Commit time of every recovered TB in the two-node campaign, by the
    # AI lease that precedes its recovery; the dotted line is the median
    # commit of TBs that NeuralRx decodes.
    ax = axes[1]
    commit_boxes(ax, [q2_commits[name] for name in order], range(len(order)))
    nrx_median = statistics.median(q2_commits["nrx"])
    ax.axhline(nrx_median, color=COLORS["green"], linestyle=":", linewidth=1.2)
    ax.text(len(order) - 0.45, nrx_median - 4, "NeuralRx-decoded", ha="right", va="top", fontsize=6.2,
            color=COLORS["green"])
    ax.axhline(expiry, color=COLORS["red"], linestyle="--", linewidth=1.1)
    ax.text(len(order) - 0.45, expiry + 3, "expiry", ha="right", va="bottom", fontsize=6.5, color=COLORS["red"])
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_xlabel("AI tokens before recovery")
    ax.set_ylabel("TB commit time\n(ms after release)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # (c) Commit time of every TB in each injected fault class.
    ax = axes[2]
    commits = data["fault_commits_ms"]
    commit_boxes(ax, [commits[name] for name, _ in FAULT_ORDER], range(len(FAULT_ORDER)))
    ax.axhline(expiry, color=COLORS["red"], linestyle="--", linewidth=1.1)
    ax.text(len(FAULT_ORDER) - 0.5, expiry + 3, "expiry", ha="right", va="bottom", fontsize=6.5,
            color=COLORS["red"])
    ax.set_xticks(range(len(FAULT_ORDER)), [FAULT_NAMES[name] for name, _ in FAULT_ORDER], rotation=40,
                  ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.2)
    ax.set_xlim(-0.6, len(FAULT_ORDER) - 0.4)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_xlabel("injected fault")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=COLORS["blue"],
                      markersize=4.5),
               Line2D([], [], marker="x", linestyle="none", color=COLORS["red"], markersize=4.5),
               Line2D([], [], color=COLORS["dark"], linestyle="--", linewidth=1.1)]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=COLORS["light_blue"], edgecolor=COLORS["blue"]))
    figure_legend(fig, [handles[0], handles[2], handles[1], handles[3]],
                  ["launched by Backstop", "latest-start", "refused by Backstop", "Backstop TB commit time"],
                  ncol=2, compact=True)
    save(fig, "softwall_eval_refinement")
    return {"launched_arrival_ms": [min(launched), max(launched)],
            "held_arrival_ms": [min(h["arrival_minus_latest_start_ms"] for h in held),
                                max(h["arrival_minus_latest_start_ms"] for h in held)],
            "q2_commit_ms": {name: {"tbs": len(q2_commits[name]), "min": min(q2_commits[name]),
                                    "median": statistics.median(q2_commits[name]), "max": max(q2_commits[name])}
                             for name in order + ["nrx"]},
            "fault_commit_max_ms": {name: max(commits[name]) for name, _ in FAULT_ORDER},
            "fault_commit_counts": data["fault_commit_counts"],
            "fault_tbs": {name: len(commits[name]) for name, _ in FAULT_ORDER}}


def make_eval_sensitivity() -> dict:
    """Recovery-bound sensitivity of the boundary witnesses and measured recovery paths (C172, Q2)."""
    witness = load("necessity_witness")
    runtime = json.loads(RUNTIME_DATA.read_text())
    attempts = json.loads(EVAL_DATA.read_text())["boundary_attempts"]
    bound = witness["contract_sensitivity"]["qualified_recovery_bound_ms"]
    cases = {w["case"].split("_", 1)[0]: w for w in witness["witnesses"]}
    assert set(cases) == {"E4", "E6b"} and bound == 25

    def excess(case, recovery_bound):
        w = cases[case]
        return (w["bound_respecting_finish_ms"] + w["unresolved_debts"] * (recovery_bound - bound)
                - w["radio_guard_boundary_ms"])

    # The bound at which each witness disappears matches the recorded thresholds.
    assert excess("E4", 19) == 0 and excess("E4", 20) > 0
    assert excess("E6b", 24) == 0 and excess("E6b", 25) > 0
    paths = sorted(runtime["q2_recovery_path_ms"])
    assert len(paths) == 1312 and abs(paths[-1] - 13.4651) < 1e-3

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.42)

    # (a) Excess of the last recovery past the guard when the unit that
    # Backstop rejects runs anyway, as a function of the recovery bound, with
    # the bound-padded physical attempts at the qualified bound.
    ax = axes[0]
    bounds = list(range(10, 31))
    styles = {"E4": ("-", "E4_debt_blind_launch", "E4_softwall_reject"),
              "E6b": ("--", "E6b_shadow_launch", "E6b_softwall_reject")}

    def without_unit(case, recovery_bound):
        # The unit is rejected: the required recoveries run from the decision.
        w = cases[case]
        return w["decision_time_ms"] + w["unresolved_debts"] * recovery_bound - w["radio_guard_boundary_ms"]

    backstop_curves = {}
    for case, (style, launched_scenario, backstop_scenario) in styles.items():
        ax.plot(bounds, [excess(case, b) for b in bounds], color=COLORS["red"], linestyle=style, linewidth=1.5)
        # Backstop admits the unit while its certificate holds and rejects it afterward.
        fine = [10 + i * 0.05 for i in range(401)]
        curve = [excess(case, b) if excess(case, b) <= 0 else without_unit(case, b) for b in fine]
        backstop_curves[case] = curve
        ax.plot(fine, curve, color=COLORS["blue"], linestyle=style, linewidth=1.5)
        for scenario, marker, color in ((launched_scenario, "x", COLORS["red"]), (backstop_scenario, "o", COLORS["blue"])):
            measured = [a["last_recovery_complete_ms"] - cases[case]["radio_guard_boundary_ms"] for a in attempts
                        if a["scenario"] == scenario]
            offsets = [((i % 9) - 4) * 0.12 for i in range(len(measured))]
            if marker == "x":
                ax.scatter([bound + o for o in offsets], measured, s=9, marker="x", color=color, linewidths=0.8,
                           zorder=3)
            else:
                ax.scatter([bound + o for o in offsets], measured, s=9, marker="o", facecolor="none",
                           edgecolor=color, linewidths=0.7, zorder=3)
    assert max(max(curve) for curve in backstop_curves.values()) <= 0
    ax.axhspan(0, 30, color=COLORS["light_red"], alpha=0.45, zorder=0)
    ax.axhline(0, color=COLORS["dark"], linewidth=0.8)
    ax.axvline(bound, color=COLORS["blue"], linestyle=":", linewidth=1.1)
    ax.text(bound + 0.55, -20, "qualified bound", ha="left", va="center", fontsize=6.0, color=COLORS["blue"],
            rotation=90)
    ax.axvline(paths[-1], color=COLORS["gray"], linestyle=":", linewidth=1.1)
    ax.text(paths[-1] + 0.55, -43, "largest measured", ha="left", va="center", fontsize=5.8,
            color=COLORS["gray"], rotation=90)
    ax.text(10.6, 26, "radio violation", ha="left", va="top", fontsize=6.3, color=COLORS["red"])
    ax.set_xlim(10, 30)
    ax.set_ylim(-72, 30)
    ax.set_xticks([10, 15, 20, 25, 30])
    ax.set_yticks([-60, -30, 0, 20])
    ax.set_xlabel("recovery bound (ms)")
    ax.set_ylabel("last recovery past\nits deadline (ms)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Complementary CDF of the physical path of every recovery in the
    # two-node campaign, against the thresholds of (a) and the contract.
    ax = axes[1]
    n = len(paths)
    ax.step(paths, [(n - i) / n for i in range(n)], where="post", color=COLORS["blue"], linewidth=1.4)
    ax.text(5.0, 0.3, "Backstop recoveries", ha="left", va="center", fontsize=6.3, color=COLORS["blue"])
    for x, style, label in ((19, "-", "E4 limit"), (24, "--", "E6b limit")):
        ax.axvline(x, color=COLORS["red"], linestyle=style, linewidth=1.0)
        ax.text(x - 0.6, 0.02, label, ha="right", va="center", fontsize=6.3, color=COLORS["red"], rotation=90)
    ax.axvline(bound, color=COLORS["blue"], linestyle=":", linewidth=1.1)
    ax.text(bound + 0.7, 0.02, "qualified bound", ha="left", va="center", fontsize=6.3, color=COLORS["blue"],
            rotation=90)
    ax.set_yscale("log")
    ax.set_xlim(0, 30)
    ax.set_ylim(5e-4, 1.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("measured recovery time (ms)")
    ax.set_ylabel("fraction of\nrecoveries above")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    handles = [Line2D([], [], color=COLORS["red"], linestyle="-", linewidth=1.5),
               Line2D([], [], color=COLORS["blue"], linestyle="-", linewidth=1.5),
               Line2D([], [], color=COLORS["red"], linestyle="--", linewidth=1.5),
               Line2D([], [], color=COLORS["blue"], linestyle="--", linewidth=1.5),
               Line2D([], [], marker="x", linestyle="none", color=COLORS["red"], markersize=4.5),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor=COLORS["blue"], markersize=4.5)]
    figure_legend(fig, [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]],
                  ["E4, launched anyway", "E6b, launched anyway", "measured, launched anyway",
                   "E4, Backstop", "E6b, Backstop", "measured, Backstop"], ncol=2, compact=True)
    save(fig, "softwall_eval_sensitivity")
    return {"excess_at_contract_ms": {case: excess(case, bound) for case in cases},
            "zero_excess_bound_ms": {"E4": 19, "E6b": 24},
            "recovery_path_ms": {"count": n, "median": statistics.median(paths), "max": paths[-1],
                                 "p99": paths[int(0.99 * (n - 1))]}}


BURST_PATTERNS = (("steady", "steady"), ("burstgpt", "BurstGPT"), ("gamma_cv1", "Poisson"),
                  ("gamma_cv2", "CV 2"), ("gamma_cv4", "CV 4"), ("gamma_cv8", "CV 8"))


def make_eval_burst() -> dict:
    """AI service and radio safety under bursty AI arrivals (C176 physical runs and SLO replay)."""
    campaign = load("burst_campaign")
    replay = load("burst_replay")
    runs = {(run["pattern"], run["policy"], run["mode"]): run for run in campaign["runs"]}
    assert campaign["certified_policies_safe"] and campaign["all_runs_completed"]
    patterns = [name for name, _ in BURST_PATTERNS if (name, "backstop", "natural") in runs]
    labels = [label for name, label in BURST_PATTERNS if name in patterns]
    xs = list(range(len(patterns)))
    policies = (("static", "static reservation", COLORS["gray"], ":", "o"),
                ("recovery_first", "recovery-first", COLORS["gray"], "--", "s"),
                ("backstop", "Backstop", COLORS["blue"], "-", "D"),
                ("idle_time", "idle-time admission", COLORS["red"], "-", "^"))

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 1.45), gridspec_kw={"width_ratios": [1.15, 1.15, 1.15]})
    fig.subplots_adjust(wspace=0.62)
    summary = {"on_time_tokens_per_s": {}, "broken_percent": {}, "padded_guard_miss_percent": {},
               "gain_over_recovery_first": {}}

    # (a) On-time AI tokens per second under natural execution. Static
    # reservation holds all four recoveries and admits no unit in this mode.
    ax = axes[0]
    for key, label, color, style, marker in policies:
        if key == "static":
            values = [0.0 for _ in patterns]
        else:
            values = [runs[(name, key, "natural")]["on_time_tokens_per_s"] for name in patterns]
        summary["on_time_tokens_per_s"][label] = dict(zip(patterns, values))
        ax.plot(xs, values, color=color, linestyle=style, marker=marker, markersize=3.4, linewidth=1.4)
    ax.set_xticks(xs, labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlabel("AI arrivals (burstier \u2192)")
    ax.set_ylabel("on-time AI tokens\nper second")
    ax.set_ylim(-30, 820)
    ax.set_yticks([0, 250, 500, 750])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Periods in which the admitted AI breaks the guard at the declared
    # bounds (natural runs), and periods whose last recovery physically ends
    # after the guard when every unit takes its bound (padded runs).
    ax = axes[1]
    for key, label, color, style, marker in policies:
        if key == "static":
            continue
        values = [100.0 * runs[(name, key, "natural")]["contract_broken_periods"]
                  / runs[(name, key, "natural")]["iterations"] for name in patterns]
        summary["broken_percent"][label] = dict(zip(patterns, values))
        ax.plot(xs, values, color=color, linestyle=style, marker=marker, markersize=3.4, linewidth=1.4)
        padded = [(i, 100.0 * runs[(name, key, "padded")]["physical_guard_misses"]
                   / runs[(name, key, "padded")]["iterations"])
                  for i, name in enumerate(patterns) if (name, key, "padded") in runs]
        if padded:
            summary["padded_guard_miss_percent"][label] = {patterns[i]: v for i, v in padded}
            ax.scatter([i for i, _ in padded], [v for _, v in padded], s=26, marker=marker, facecolor="white",
                       edgecolor=color, linewidths=1.1, zorder=4)
    ax.set_xticks(xs, labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_xlabel("AI arrivals (burstier \u2192)")
    ax.set_ylabel("periods missing the\nrecovery deadline (%)")
    ax.set_ylim(-1, 14)
    ax.set_yticks([0, 5, 10])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # (c) Gain of Backstop over recovery-first in on-time AI tokens against
    # the AI SLO: replay mean over the six patterns (line, with the range) and
    # the physical runs at the 200 ms SLO (markers).
    ax = axes[2]
    rows = {(row["pattern"], row["slo_ms"], row["policy"]): row for row in replay["summary"]}
    slos = sorted({row["slo_ms"] for row in replay["summary"]})
    gains = {slo: [100.0 * (rows[(name, slo, "backstop")]["on_time_tokens_per_s"]
                            / rows[(name, slo, "recovery_first")]["on_time_tokens_per_s"] - 1.0)
                   for name, _ in BURST_PATTERNS] for slo in slos}
    mean = [statistics.mean(gains[slo]) for slo in slos]
    sx = list(range(len(slos)))
    ax.fill_between(sx, [min(gains[slo]) for slo in slos], [max(gains[slo]) for slo in slos],
                    color=COLORS["light_gray"], linewidth=0, zorder=1)
    ax.plot(sx, mean, color=COLORS["dark"], marker="s", markersize=3.0, linewidth=1.3, zorder=3)
    physical = [100.0 * (runs[(name, "backstop", "natural")]["on_time_tokens_per_s"]
                         / runs[(name, "recovery_first", "natural")]["on_time_tokens_per_s"] - 1.0)
                for name in patterns]
    slo_ms = runs[(patterns[0], "backstop", "natural")]["slo_ms"]
    ax.scatter([slos.index(slo_ms)] * len(physical), physical, s=18, marker="o", facecolor="white",
               edgecolor=COLORS["blue"], linewidths=1.0, zorder=4)
    summary["gain_over_recovery_first"] = {"replay_mean_by_slo": dict(zip(map(str, slos), mean)),
                                           "replay_range_by_slo": {str(slo): [min(gains[slo]), max(gains[slo])]
                                                                   for slo in slos},
                                           "physical_by_pattern": dict(zip(patterns, physical))}
    ax.axhline(0, color=COLORS["dark"], linewidth=0.7)
    ax.set_xticks(sx, [str(int(slo)) for slo in slos])
    ax.tick_params(axis="x", labelsize=6.6)
    ax.set_xlim(-0.4, len(slos) - 0.6)
    ax.set_xlabel("AI SLO (ms)")
    ax.set_ylabel("Backstop gain over\nrecovery-first (%)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3.4, linewidth=1.4)
               for _, _, color, style, marker in policies]
    extra = [Line2D([], [], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["red"],
                    markersize=4.5),
             Line2D([], [], color=COLORS["dark"], marker="s", markersize=3.0, linewidth=1.3),
             Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["blue"],
                    markersize=4.2)]
    names = [label for _, label, _, _, _ in policies]
    # Column-major order: the first row names the policies, the second the
    # replay line of (c), the measured runs of (c), and the padded runs of (b).
    order = [handles[0], extra[1], handles[1], extra[2], handles[2], extra[0], handles[3]]
    order_names = [names[0], "replay mean", names[1], "measured run", names[2], "padded run", names[3]]
    figure_legend(fig, order, order_names, ncol=4, compact=True)
    save(fig, "softwall_eval_burst")
    return summary


def make_eval_load() -> dict:
    """On-time AI throughput against offered AI load (C176 Poisson runs and replay)."""
    physical = {}
    for run in load("load_campaign")["runs"] + load("burst_campaign")["runs"]:
        if run["pattern"] == "gamma_cv1" and run["mode"] == "natural":
            physical[(round(run["offered_rate_per_s"], 2), run["policy"])] = run["on_time_tokens_per_s"]
    rates = sorted({rate for rate, _ in physical})
    assert all((rate, policy) in physical for rate in rates for policy in ("backstop", "recovery_first", "idle_time"))
    replay = load("load_replay")["summary"]
    policies = (("static", "static reservation", COLORS["gray"], ":", "o"),
                ("recovery_first", "recovery-first", COLORS["gray"], "--", "s"),
                ("backstop", "Backstop", COLORS["blue"], "-", "D"),
                ("idle_time", "idle-time admission", COLORS["red"], "-", "^"))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.4))
    fig.subplots_adjust(wspace=0.4)
    summary = {"physical_200ms": {}, "replay_1000ms": {}}

    # (a) Measured on-time AI throughput with a 200 ms SLO. Static reservation
    # holds all four recoveries and admits no request in this mode.
    ax = axes[0]
    for key, label, color, style, marker in policies:
        values = [0.0 if key == "static" else physical[(rate, key)] for rate in rates]
        summary["physical_200ms"][label] = dict(zip(map(str, rates), values))
        ax.plot(rates, values, color=color, linestyle=style, marker=marker, markersize=3.4, linewidth=1.4)
    ax.set_xlabel("offered AI load (requests/s)")
    ax.set_ylabel("on-time AI tokens\nper second")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-40, 1100)
    ax.set_yticks([0, 500, 1000])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.text(0.7, 1050, "200 ms SLO, measured", ha="left", va="top", fontsize=6.5, color=COLORS["dark"])
    panel_label(ax, "a")

    # (b) Replay with a 1,000 ms SLO: Backstop and its recovery-first ablation
    # fit the same work into each window and reach the same throughput.
    ax = axes[1]
    for key, label, color, style, marker in policies:
        rows = sorted((row["rate_per_s"], row["on_time_tokens_per_s"]) for row in replay
                      if row["slo_ms"] == 1000.0 and row["policy"] == key)
        summary["replay_1000ms"][label] = {f"{rate:.2f}": value for rate, value in rows}
        ax.plot([r for r, _ in rows], [v for _, v in rows], color=color, linestyle=style, marker=marker,
                markersize=3.4, linewidth=1.4)
    ax.set_xlabel("offered AI load (requests/s)")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-80, 2200)
    ax.set_yticks([0, 1000, 2000])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.text(0.7, 2100, "1,000 ms SLO, replay", ha="left", va="top", fontsize=6.5, color=COLORS["dark"])
    panel_label(ax, "b")

    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3.4, linewidth=1.4)
               for _, _, color, style, marker in policies]
    figure_legend(fig, handles, [label for _, label, _, _, _ in policies], ncol=4, compact=True)
    save(fig, "softwall_eval_load")
    return summary


def make_capacity_headroom() -> dict:
    """Capacity model, trace headroom screen, and prespecified lever decomposition."""
    capacity = load("capacity")
    headroom = load("headroom")
    levers = load("levers")

    cs = capacity["summary"]
    hs = headroom["summary"]
    ls = levers["summary"]
    assert capacity["all_pass"] and capacity["exact_checker"]["mismatches"] == 0
    assert cs["grid_points"] == 21384 and cs["mandatory_infeasible_points"] == 7668
    assert hs["gap_at_least_mde_and_positive_all_windows_points"] == 362
    assert ls["candidate_points"] == 362
    assert ls["greedy_within_5pct_of_fixed_oracle_points"] == 361
    assert ls["flexible_recovery_positive_gain_points"] == 203

    fields = capacity["row_fields"]
    rows = [dict(zip(fields, row)) for row in capacity["rows"]]
    representative = sorted(
        (row for row in rows if
         row["cells"] == 4 and row["homes"] == 2 and row["gpus"] == 1 and
         row["mandatory_feasible"] == 1 and row["expiry_ms"] == 155 and
         row["recovery_ms"] == 25 and row["context"] == 64 and
         row["ai_deadline_ms"] == 50 and row["offered"] == 4 and
         row["correlation"] == 0.5),
        key=lambda row: row["success_probability"],
    )
    assert len(representative) == 3

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 1.72))
    fig.subplots_adjust(wspace=0.80, bottom=0.26)

    # (a) Expected request capacity at one declared model point.
    ax = axes[0]
    pp = [row["success_probability"] for row in representative]
    series = [
        ("static", "static_requests", COLORS["gray"], ":", "o"),
        ("recovery-first", "recovery_first_requests", COLORS["gray"], "--", "s"),
        ("Backstop", "softwall_requests", COLORS["blue"], "-", "D"),
    ]
    for label, field, color, line, marker in series:
        ax.plot(pp, [row[field] for row in representative], label=label, color=color,
                linestyle=line, marker=marker, markersize=3.8)
    ax.set_xlabel("NeuralRx success $p$")
    ax.set_ylabel("expected timely\nrequests")
    ax.set_xticks(pp)
    ax.set_ylim(-0.03, 1.55)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.18)
    panel_label(ax, "a")

    # (b) The frozen trace screen locates headroom only at intermediate SLOs.
    ax = axes[1]
    slos = [50, 100, 250, 1000]
    feasible, advanced = [], []
    for slo in slos:
        eligible = [row for row in headroom["rows"]
                    if row["mandatory_feasible"] and row["ai_slo_ms"] == slo]
        feasible.append(len(eligible))
        advanced.append(sum(row["advance_to_lever_evaluation"] for row in eligible))
    assert feasible == [513, 513, 513, 513] and advanced == [0, 298, 64, 0]
    shares = [100.0 * a / f for a, f in zip(advanced, feasible)]
    xx = list(range(len(slos)))
    ax.bar(xx, shares, color=COLORS["blue"], width=0.66)
    ax.axhline(5.0, color=COLORS["gray"], linestyle=":", linewidth=1.0)
    ax.set_xticks(xx, ["50", "100", "250", "1s"])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("AI SLO (ms)")
    ax.set_ylabel("passing points (%)")
    ax.set_ylim(0, 65)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    # (c) Which mechanism closes the 362 prespecified headroom points.
    ax = axes[2]
    labels = ["AI-first", "placement"]
    counts = [ls["greedy_within_5pct_of_fixed_oracle_points"],
              ls["flexible_recovery_positive_gain_points"]]
    shares = [100.0 * count / ls["candidate_points"] for count in counts]
    ax.bar([0, 1], shares, color=[COLORS["blue"], COLORS["light_blue"]], edgecolor=COLORS["blue"],
           linewidth=0.8, width=0.64)
    ax.set_xticks([0, 1], labels)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel("candidate points (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    save(fig, "softwall_capacity_headroom")
    return {
        "representative_success_probability": pp,
        "representative_static": [row["static_requests"] for row in representative],
        "representative_recovery_first": [row["recovery_first_requests"] for row in representative],
        "representative_softwall": [row["softwall_requests"] for row in representative],
        "trace_slo_ms": slos,
        "trace_feasible_points": feasible,
        "trace_advanced_points": advanced,
        "lever_candidate_points": ls["candidate_points"],
        "greedy_within_5pct_points": counts[0],
        "flexible_placement_positive_points": counts[1],
    }


def main() -> None:
    derive_background()
    derive_eval()
    derive_runtime()
    outputs = {
        "execution": make_execution(),
        "debt": make_debt(),
        "eval_capacity": make_eval_capacity(),
        "eval_drivers": make_eval_drivers(),
        "eval_trace": make_eval_trace(),
        "eval_safety": make_eval_safety(),
        "eval_online": make_eval_online(),
        "eval_burst": make_eval_burst(),
        "eval_load": make_eval_load(),
        "eval_refinement": make_eval_refinement(),
        "envelope": make_envelope(),
        "eval_sensitivity": make_eval_sensitivity(),
        "capacity_headroom": make_capacity_headroom(),
    }
    stems = ("softwall_execution", "softwall_debt", "softwall_eval_capacity", "softwall_eval_drivers",
             "softwall_eval_trace", "softwall_eval_safety", "softwall_eval_online", "softwall_eval_burst",
             "softwall_eval_load",
             "softwall_eval_refinement",
             "softwall_envelope", "softwall_eval_sensitivity", "softwall_capacity_headroom")
    manifest = {
        "schema": "softwall-paper-figure-manifest-v1",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()},
        "figures": [f"{stem}.pdf" for stem in stems],
        "derived": outputs,
        "claim_boundary": (
            "All plots reproduce finite-sample audited artifacts. Diagnostic campaigns are not pooled; "
            "sample maxima are not WCET; the 4.5 ms production gate remains failed; and Sionna CDL-D/E "
            "does not qualify Aerial TDL-A or field IQ. The capacity figure is an exact-checked model and "
            "the trace figure covers the prespecified headroom points only."
        ),
    }
    manifest["output_sha256"] = {name: sha256(HERE / name) for stem in stems
                                 for name in (f"{stem}.pdf", f"{stem}.svg")}
    (HERE / "softwall_figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
