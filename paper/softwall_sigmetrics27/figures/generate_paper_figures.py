#!/usr/bin/env python3
"""Generate the additional SoftWall paper figures from audited result JSON.

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


def load(name: str):
    return json.loads(SOURCES[name].read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# matplotlib tab10 palette. SoftWall data is drawn in blue and reference or
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
    fig.savefig(
        HERE / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        HERE / f"{stem}.svg",
        bbox_inches="tight",
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


def top_legend(ax, ncol: int, handles=None, labels=None) -> None:
    """Legend above the axes and above the panel marker."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.13), ncol=ncol, frameon=False,
              handlelength=1.5, handletextpad=0.4, columnspacing=0.9, labelspacing=0.2, borderaxespad=0.0)


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

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3))
    fig.subplots_adjust(wspace=0.45)

    # (a) Radio p99 against the MPS share of a co-running Qwen prefill class.
    # Lines connect the mean over runs; bars span the runs.
    ax = axes[0]
    alone = [value for sweep in sweeps.values() for value in sweep["by_share"]["0"]["p99_ms"]]
    ax.axhspan(min(alone), max(alone), color=COLORS["light_gray"], zorder=1, label="RAN alone")
    ax.axhline(deadline, color=COLORS["red"], linestyle="--", linewidth=1.2, zorder=2, label="deadline")
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
    ax.set_xlabel("AI MPS share (%)")
    ax.set_ylabel("radio p99 (ms)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("light AI", "RAN alone", "heavy AI", "deadline")]
    top_legend(ax, ncol=2, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
    panel_label(ax, "a")

    # (b) Complementary CDF of latency normalized by its own median.
    ax = axes[1]
    series = (("certificate_build", "host build", COLORS["orange"]),
              ("pair_python", "Python pair", COLORS["brown"]),
              ("pair_native", "native pair", COLORS["cyan"]),
              ("kernel", "GPU kernel", COLORS["gray"]))
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
    ax.set_xlabel("latency / median")
    ax.set_ylabel("CCDF")
    ax.grid(which="major", **GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("host build", "native pair", "Python pair", "GPU kernel")]
    top_legend(ax, ncol=2, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
    panel_label(ax, "b")

    save(fig, "softwall_execution")
    return {"deadline_ms": deadline, "p99_range_ms": summary,
            "alone_p99_range_ms": [min(alone), max(alone)], "tails": tails}


def make_debt() -> dict:
    """Receiver outcome uncertainty and the retained debt of a four-TB epoch."""
    data = json.loads(BACKGROUND_DATA.read_text())
    deployment = data["deployment_joined"]
    assert deployment["epochs"] == 600

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3))
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
    ax.set_xlabel("SNR offset (dB)")
    ax.set_ylabel("TBs decoded (%)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=color, linewidth=2.2) for color in channel_colors.values()]
    handles += [Line2D([], [], color=COLORS["dark"], linestyle="-", linewidth=1.5),
                Line2D([], [], color=COLORS["dark"], linestyle="--", linewidth=1.2)]
    labels = list(channel_colors) + ["NeuralRx", "conventional"]
    order = [0, 3, 1, 4, 2]
    top_legend(ax, ncol=3, handles=[handles[i] for i in order], labels=[labels[i] for i in order])
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
        ax.text(p, y, "all 4" if k == 4 else f"$\\geq${k}", ha="center", va=va, fontsize=6.8,
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
    ax.set_xlabel("per-TB failure probability $p$")
    ax.set_ylabel("$P$(recoveries $\\geq k$)")
    ax.grid(which="major", **GRID)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=shades[3], linewidth=1.5),
               Rectangle((0, 0), 1, 1, facecolor=COLORS["light_red"], edgecolor=COLORS["red"], linestyle="--"),
               Line2D([], [], color=shades[1], marker="o", linestyle="none", markeredgecolor=COLORS["dark"],
                      markersize=4.5),
               Line2D([], [], color=shades[1], marker="s", linestyle="none", markeredgecolor=COLORS["dark"],
                      markersize=4.2)]
    top_legend(ax, ncol=2, handles=[handles[0], handles[2], handles[1], handles[3]],
               labels=["independent", "deployment", "correlated", "receiver trials"])
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

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.3), gridspec_kw={"width_ratios": [1.2, 1.0]})
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
                markersize=3.2, linewidth=1.5, label=f"ctx {context}")
    physical = []
    for case in cases:
        if case["context_length"] is None:
            continue
        pending = 4 - case["success_count"]
        name = case["case_id"].split("_")[0]
        x = case["conservative_prediction_time_ms"]
        physical.append({"case": name, "decision_ms": x, "pending": pending, "admit": case["expected_lease"]})
        if case["expected_lease"]:
            ax.scatter(x, pending, marker="o", s=22, facecolor="white", edgecolor=COLORS["green"], linewidth=1.4,
                       zorder=5)
        else:
            ax.scatter(x, pending, marker="x", s=24, color=COLORS["red"], linewidth=1.5, zorder=5)
        # Admitted cases are labeled below their marker and rejected cases above.
        ax.text(x + 1.5, pending + (0.2 if not case["expected_lease"] else -0.2), name, fontsize=6.3,
                color=COLORS["dark"], ha="left", va="bottom" if not case["expected_lease"] else "top")
    ax.set_xlim(40, 120)
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
    labels += ["admitted", "rejected"]
    top_legend(ax, ncol=3, handles=handles, labels=labels)
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
            label="scheduler p99")
    ax.plot(counts, decision_max, color=COLORS["blue"], marker="s", markersize=3.0, linewidth=1.2,
            linestyle="--", label="scheduler max")
    ax.plot(counts, verify_p99, color=COLORS["green"], marker="D", markersize=3.0, linewidth=1.5,
            label="verifier p99")
    ax.axhline(5.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="budget")
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts, [str(n) for n in counts])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel("pending recoveries")
    ax.set_ylabel("latency (ms)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    top_legend(ax, ncol=2)
    panel_label(ax, "b")

    save(fig, "softwall_envelope")
    return {"frontier": frontier, "physical_cases": physical, "decision_p99_ms": decision_p99,
            "decision_max_ms": decision_max, "verify_p99_ms": verify_p99}


def figure_legend(fig, handles, labels, ncol: int) -> None:
    """One legend above all panels of a figure."""
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
               handlelength=1.8, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)


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
                ("softwall_requests", "SoftWall", COLORS["blue"], "-", "D"))

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
    ax.set_xlabel("cells")
    ax.set_ylabel("AI capacity\n(% of oracle)")
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
    ax.set_xlabel("cells")
    ax.set_ylabel("radio violation\nprobability (%)")
    ax.set_ylim(-2, 35)
    ax.set_yticks([0, 15, 30])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    handles = [Line2D([], [], color=color, linestyle=style, marker=marker, markersize=3.6, linewidth=1.5)
               for _, _, color, style, marker in policies]
    handles.append(Line2D([], [], color=COLORS["red"], marker="^", markersize=3.8, linewidth=1.5))
    figure_legend(fig, handles, ["static", "recovery-first", "SoftWall", "idle-time"], ncol=4)
    save(fig, "softwall_eval_capacity")
    summary["feasible_points"] = len(feasible)
    summary["useful_points"] = len(useful)
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
    ax.set_xlabel("GPUs")
    ax.set_ylabel("timely AI value\n(% of oracle)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Distribution of the SoftWall-to-recovery-first value ratio.
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
    ax.set_xlabel("SoftWall / recovery-first value")
    ax.set_ylabel("CDF (%)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    handles = [Rectangle((0, 0), 1, 1, color=COLORS["gray"]), Rectangle((0, 0), 1, 1, color=COLORS["blue"])]
    figure_legend(fig, handles, ["recovery-first", "SoftWall"], ncol=2)
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

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.4), gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.subplots_adjust(wspace=0.38)

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
    ax.text(-0.55, guard - 1.5, "guard", ha="left", va="top", fontsize=6.5, color=COLORS["red"])
    ax.set_xticks([0, 1, 2], ["E4", "E6b", "E6a"])
    ax.set_xlabel("boundary case")
    ax.set_xlim(-0.6, 2.4)
    ax.set_ylim(85, 172)
    ax.set_yticks([100, 125, 150])
    ax.set_ylabel("last recovery\nend (ms)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Qwen units per prompt length in the two-node campaign: executed by
    # SoftWall, and admitted only by idle-time admission because they break
    # the certified recovery bound. Static reservation admits none.
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
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("Qwen units")
    ax.set_ylim(0, 215)
    ax.set_yticks([0, 100, 200])
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
                      markeredgecolor=COLORS["blue"], markersize=4.5),
               Line2D([], [], marker="x", linestyle="none", color=COLORS["red"], markersize=4.5),
               Rectangle((0, 0), 1, 1, color=COLORS["blue"]),
               Rectangle((0, 0), 1, 1, facecolor=COLORS["light_red"], edgecolor=COLORS["red"], hatch="/////")]
    figure_legend(fig, handles, ["SoftWall attempt", "baseline attempt", "executed by SoftWall",
                                 "breaks the bound"], ncol=4)
    save(fig, "softwall_eval_safety")
    return {"boundary": summary, "qwen_executed": dict(zip(map(str, contexts), executed)),
            "qwen_rejected": dict(zip(map(str, contexts), rejected))}


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
        ("SoftWall", "softwall_requests", COLORS["blue"], "-", "D"),
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
    outputs = {
        "execution": make_execution(),
        "debt": make_debt(),
        "eval_capacity": make_eval_capacity(),
        "eval_trace": make_eval_trace(),
        "eval_safety": make_eval_safety(),
        "envelope": make_envelope(),
        "capacity_headroom": make_capacity_headroom(),
    }
    stems = ("softwall_execution", "softwall_debt", "softwall_eval_capacity", "softwall_eval_trace",
             "softwall_eval_safety", "softwall_envelope", "softwall_capacity_headroom")
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
