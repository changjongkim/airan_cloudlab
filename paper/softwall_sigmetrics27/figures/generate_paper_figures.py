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
    ax.set_xlabel("per-TB debt probability $p$")
    ax.set_ylabel("$P$(debts $\\geq k$)")
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


def make_qualification_evidence() -> dict:
    cap = load("mps_cap")
    life = load("mps_lifecycle")
    life_protocol = load("mps_lifecycle_protocol")
    q2 = load("q2")
    q2_protocol = load("q2_protocol")
    fault = load("fault")

    assert cap["by_cap"]["100"]["misses"] == 2
    assert cap["by_cap"]["20"]["misses"] == 1
    assert life["mps_only_misses"] == 12 and life["cpu_only_misses"] == 0
    assert q2["summary"]["actual_nrx_requests"] == 4800
    assert q2["summary"]["qwen_units"] == 1077
    assert fault["summary"]["deadline_misses"] == 0

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.0))
    fig.subplots_adjust(hspace=0.95, wspace=0.62)

    # (a) Diagnostic miss observations from two separate campaigns. The shared
    # per-10k axis is visual only; a gap and group labels keep them unpooled.
    ax = axes[0, 0]
    n_life = len(life["rows"]) * life_protocol["radio"]["iterations_per_condition"]
    raw = [
        ("cap 100", 2, cap["by_cap"]["100"]["releases"]),
        ("cap 20", 1, cap["by_cap"]["20"]["releases"]),
        ("CPU sham", 0, n_life),
        ("MPS client", 12, n_life),
    ]
    xpos = [0, 1, 2.7, 3.7]
    values = [misses / n * 10000 for _, misses, n in raw]
    ax.bar(xpos, values, color=[COLORS["orange"], COLORS["orange"], COLORS["gray"], COLORS["red"]], width=0.7)
    ax.axvline(1.85, color=COLORS["gray"], linestyle=":", linewidth=1.0)
    ax.set_xticks(xpos, [r[0] for r in raw], rotation=25, ha="right")
    for x_center, group in ((0.5, "cap sweep"), (3.2, "lifecycle")):
        ax.text(x_center, 0.97, group, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color=COLORS["dark"])
    ax.set_ylabel("misses per 10k")
    ax.set_ylim(0, 17.5)
    ax.set_xlim(-0.6, 4.3)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    # (b) Class-specific Qwen admission.
    ax = axes[0, 1]
    contexts = [16, 32, 64, 128, 256, 512]
    accepted = [q2["summary"]["qwen_executed_by_class"][str(v)] for v in contexts]
    offered = [q2["summary"]["qwen_offered_by_class"][str(v)] for v in contexts]
    rejected = [o - a for o, a in zip(offered, accepted)]
    ypos = list(range(len(contexts)))
    ax.barh(ypos, accepted, color=COLORS["blue"], label="executed")
    ax.barh(ypos, rejected, left=accepted, color=COLORS["red"], label="certificate reject")
    ax.set_yticks(ypos, [str(v) for v in contexts])
    ax.set_xlabel("Qwen requests per class")
    ax.set_ylabel("context tokens")
    ax.set_xlim(0, 200)
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=2, frameon=False, handlelength=1.2,
              columnspacing=0.9)
    panel_label(ax, "b")

    # (c) Whole-path quantiles versus the declared contract.
    ax = axes[1, 0]
    rows = [
        ("NeuralRx", q2["summary"]["nrx_release_to_complete_ms"], q2_protocol["mode"]["nrx_bound_ms"]),
        ("recovery", q2["summary"]["recovery_path_ms"], q2_protocol["mode"]["conventional_bound_ms"]),
        ("radio\ncommit", q2["summary"]["radio_release_to_commit_ms"], q2_protocol["mode"]["expiry_ms"]),
    ]
    for y, (_, vals, bound) in enumerate(rows):
        ax.plot([vals["p50"], vals["max"]], [y, y], color="#AAB2BD", lw=2.0, zorder=1)
        ax.scatter(vals["p50"], y, marker="o", s=26, color=COLORS["blue"], label="p50" if y == 0 else None, zorder=3)
        ax.scatter(vals["p99"], y, marker="s", s=26, color=COLORS["orange"], label="p99" if y == 0 else None, zorder=3)
        ax.scatter(vals["max"], y, marker="D", s=26, color=COLORS["red"], label="max" if y == 0 else None, zorder=3)
        ax.scatter(bound, y, marker="|", s=170, linewidths=2.4, color=COLORS["green"],
                   label="contract" if y == 0 else None, zorder=4)
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(0, 165)
    ax.set_xlabel("release-to-terminal path (ms)")
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=4, frameon=False, handletextpad=0.3,
              columnspacing=0.8)
    panel_label(ax, "c")

    # (d) Fault coverage. A log scale keeps rare terminal events visible.
    ax = axes[1, 1]
    f = fault["summary"]
    labels = ["NeuralRx", "recoveries", "corr. all-fail", "post-fault epochs", "replay pairs", "terminal faults"]
    counts = [f["actual_nrx_requests"], f["physical_recoveries"], f["correlated_all_fail_recoveries"],
              f["post_terminal_radio_rounds"], f["stale_duplicate_nrx_pairs"] + f["stale_duplicate_recovery_pairs"],
              f["terminal_channel_faults"]]
    y = list(range(len(labels)))
    ax.barh(y, counts, color=COLORS["blue"], height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 5000)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("observed events (log)")
    ax.grid(axis="x", which="major", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "d")

    save(fig, "softwall_qualification_evidence")
    return {
        "mps_diagnostics": [{"label": label, "misses": misses, "requests": n} for label, misses, n in raw],
        "qwen_contexts": contexts,
        "qwen_accepted": accepted,
        "qwen_rejected": rejected,
        "fault_counts": dict(zip((label.replace("\n", " ") for label in labels), counts)),
    }


def make_outcome_boundaries() -> dict:
    baseline = load("baseline")
    lifecycle = load("lifecycle")
    production = load("production")
    channel = load("channel")

    assert baseline["summary"]["totals"]["softwall"]["timely_value_tokens"] == 349387
    assert baseline["summary"]["totals"]["recovery_first_full_bound"]["gap_to_oracle_pct"] < baseline["summary"]["minimum_effect_pct"]
    assert lifecycle["counts"] == {"qualified_or_partial_modes": 5, "total_modes": 10, "unqualified_modes": 5}
    assert production["current_timing_diagnosis"]["cross_gpu_candidates"]["raw_iq_full_remote_persistent_ordered"]["deadline_counts"]["parallel_pair_wall_le_deadline"] == 995
    assert channel["aggregate_low_snr_paired"]["neural_only"] == 31
    assert channel["aggregate_low_snr_paired"]["conventional_only"] == 12

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 4.3))
    fig.subplots_adjust(hspace=1.0, wspace=0.62)

    # (a) Negative throughput result relative to the prespecified minimum effect.
    # A dot plot keeps the zero-gain case visible.
    ax = axes[0, 0]
    gains = [baseline["summary"]["totals"]["recovery_first_empirical"]["gap_to_oracle_pct"],
             baseline["summary"]["totals"]["recovery_first_full_bound"]["gap_to_oracle_pct"]]
    names = ["empirical\nservice", "full-bound\nsensitivity"]
    y = [0, 1]
    threshold = baseline["summary"]["minimum_effect_pct"]
    ax.hlines(y, 0, gains, color=COLORS["blue"], linewidth=1.6)
    ax.scatter(gains, y, s=36, color=COLORS["blue"], zorder=3, label="SoftWall gain")
    ax.axvline(threshold, color=COLORS["red"], linestyle="--", linewidth=1.3, label="prespecified\nminimum effect")
    ax.set_yticks(y, names)
    ax.set_ylim(1.6, -0.6)
    ax.set_xlim(-0.15, 5.5)
    ax.set_xlabel("timely-token gain (%)")
    ax.grid(axis="x", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46), ncol=2, frameon=False, handlelength=1.4,
              columnspacing=0.8)
    panel_label(ax, "a")

    # (b) Claim-scoped lifecycle matrix.
    ax = axes[0, 1]
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 5)
    ax.axis("off")
    mode_layout = [
        ("warm\npersistent", "Q", 0, 4),
        ("30 s idle", "Q", 1, 4),
        ("MPS restart", "P", 0, 3),
        ("Qwen reload", "P", 1, 3),
        ("same-worker\nreconnect", "P", 0, 2),
        ("process or\nmodel cold", "UQ", 1, 2),
        ("5 min idle", "UQ", 0, 1),
        ("30 min idle", "UQ", 1, 1),
        ("GC on", "UQ", 0, 0),
        ("worker\nreplacement", "UQ", 1, 0),
    ]
    style = {
        "Q": (COLORS["light_green"], COLORS["green"], "qualified"),
        "P": (COLORS["light_orange"], COLORS["orange"], "partial"),
        "UQ": (COLORS["light_gray"], COLORS["gray"], "unqualified"),
    }
    for label, status, col, row in mode_layout:
        face, edge, _ = style[status]
        ax.add_patch(Rectangle((col + 0.02, row + 0.07), 0.96, 0.86, facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(col + 0.5, row + 0.5, label, ha="center", va="center", fontsize=7.0, color=COLORS["dark"],
                linespacing=1.1)
    handles = [Rectangle((0, 0), 1, 1, facecolor=face, edgecolor=edge) for face, edge, _ in style.values()]
    ax.legend(handles, [name for _, _, name in style.values()], loc="upper center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, frameon=False, handlelength=1.1, columnspacing=0.8)
    panel_label(ax, "b")

    # (c) Production exit-gate progression over the prespecified 1,000-request
    # runs. Late requests use a log axis because the gate requires zero late
    # requests and the final candidate differs from the others by a small count.
    # The 300-request mechanism ablation is a diagnostic and is not plotted.
    ax = axes[1, 0]
    candidates = production["current_timing_diagnosis"]["cross_gpu_candidates"]
    prod_rows = [
        ("precomputed\nsplit", "precomputed_ls_split"),
        ("raw-IQ P2P", "raw_iq_full_remote"),
        ("same stream", "raw_iq_full_remote_same_stream"),
        ("frozen\ncandidate", "raw_iq_full_remote_persistent_ordered"),
    ]
    late, totals = [], []
    for _, key in prod_rows:
        counts = candidates[key]["deadline_counts"]
        late.append(counts["parallel_pair_wall_gt_deadline"])
        totals.append(counts["parallel_pair_wall_le_deadline"] + counts["parallel_pair_wall_gt_deadline"])
    assert totals == [1000, 1000, 1000, 1000]
    yy = list(range(len(prod_rows)))
    ax.barh(yy, late, color=COLORS["blue"], height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_yticks(yy, [r[0] for r in prod_rows])
    ax.invert_yaxis()
    ax.set_xlabel("requests over 4.5 ms (log)")
    ax.grid(axis="x", which="major", **GRID)
    ax.set_axisbelow(True)
    panel_label(ax, "c")

    # (d) Supported simulated channel result. The four waterfall strata use a
    # numeric Es/No axis; the 10 dB control stratum sits after an axis break and
    # is not joined to the waterfall curves.
    ax = axes[1, 1]
    strata = channel["models"]["D"]["strata"]
    low = [row["esno_db"] for row in strata if row["esno_db"] < 0]
    assert low == [-4.0, -3.6, -3.2, -3.0]
    assert [row["esno_db"] for row in channel["models"]["E"]["strata"]] == [row["esno_db"] for row in strata]
    high_x = -2.55
    line_specs = [
        ("D conventional", "D", "conventional_correct", COLORS["gray"], "o", "-"),
        ("D NeuralRx", "D", "neural_correct", COLORS["blue"], "s", "-"),
        ("E conventional", "E", "conventional_correct", COLORS["gray"], "o", "--"),
        ("E NeuralRx", "E", "neural_correct", COLORS["blue"], "s", "--"),
    ]
    for label, model, field, color, marker, ls in line_specs:
        rows_m = channel["models"][model]["strata"]
        lo = [100 * r[field] / r["trials"] for r in rows_m if r["esno_db"] < 0]
        hi = [100 * r[field] / r["trials"] for r in rows_m if r["esno_db"] >= 0]
        ax.plot(low, lo, label=label, color=color, marker=marker, linestyle=ls, markersize=4.2)
        ax.plot([high_x] * len(hi), hi, color=color, marker=marker, linestyle="none", markersize=4.2)
    ax.set_xticks(low + [high_x], ["-4.0", "-3.6", "-3.2", "-3.0", "10"], rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_xlim(-4.12, -2.43)
    for xb in (-2.79, -2.75):
        ax.plot([xb - 0.02, xb + 0.02], [-0.045, 0.045], transform=ax.get_xaxis_transform(),
                color=COLORS["dark"], linewidth=1.0, clip_on=False)
    ax.set_ylim(-4, 106)
    ax.set_ylabel("decode success (%)")
    ax.set_xlabel("Es/No (dB)")
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, handlelength=1.8,
              columnspacing=0.8)
    panel_label(ax, "d")

    save(fig, "softwall_outcome_boundaries")
    return {
        "throughput_gain_pct": gains,
        "minimum_effect_pct": threshold,
        "production_labels": [r[0].replace("\n", " ") for r in prod_rows],
        "production_late": late,
        "production_totals": totals,
        "channel_aggregate": channel["aggregate_low_snr_paired"],
    }


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
    execution = make_execution()
    debt = make_debt()
    qualification = make_qualification_evidence()
    boundaries = make_outcome_boundaries()
    capacity = make_capacity_headroom()
    manifest = {
        "schema": "softwall-paper-figure-manifest-v1",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()},
        "figures": [
            "softwall_execution.pdf",
            "softwall_debt.pdf",
            "softwall_qualification_evidence.pdf",
            "softwall_outcome_boundaries.pdf",
            "softwall_capacity_headroom.pdf",
        ],
        "derived": {"execution": execution, "debt": debt, "qualification": qualification,
                    "boundaries": boundaries, "capacity": capacity},
        "claim_boundary": (
            "All plots reproduce finite-sample audited artifacts. Diagnostic campaigns are not pooled; "
            "sample maxima are not WCET; the 4.5 ms production gate remains failed; and Sionna CDL-D/E "
            "does not qualify Aerial TDL-A or field IQ."
        ),
    }
    manifest["output_sha256"] = {
        name: sha256(HERE / name)
        for stem in ("softwall_execution", "softwall_debt", "softwall_qualification_evidence",
                     "softwall_outcome_boundaries", "softwall_capacity_headroom")
        for name in (f"{stem}.pdf", f"{stem}.svg")
    }
    (HERE / "softwall_figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
