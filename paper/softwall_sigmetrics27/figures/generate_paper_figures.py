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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


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
}


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


ARROW_LW = 1.2


def rounded_box(ax, xy, width, height, text, face, edge, fontsize=7.6, lw=1.1):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        linewidth=lw,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        linespacing=1.18,
    )
    return box


def arrow(ax, start, end, color=None, style="-|>", lw=ARROW_LW, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=9,
        linewidth=lw,
        color=color or COLORS["gray"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    return patch


def make_airan_background() -> None:
    """Problem-domain background: same-TB dependency and the contended GPU path."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 2.45))
    fig.subplots_adjust(wspace=0.30)

    # (a) One transport block creates an optional result and a conditional but
    # mandatory conventional branch. This is a domain dependency, not a
    # SoftWall component diagram.
    ax = axes[0]
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.00, 0.40), 0.27, 0.20, "uplink TB\nrelease", COLORS["light_blue"], COLORS["blue"], 6.2)
    rounded_box(ax, (0.365, 0.70), 0.27, 0.18, "optional\nNeuralRx", COLORS["light_orange"], COLORS["orange"], 6.5)
    rounded_box(ax, (0.365, 0.12), 0.27, 0.20, "conventional\nrecovery", COLORS["light_red"], COLORS["red"], 6.2)
    rounded_box(ax, (0.73, 0.40), 0.27, 0.20, "single radio\ncommit", COLORS["light_green"], COLORS["green"], 6.2)
    arrow(ax, (0.20, 0.61), (0.365, 0.79), color=COLORS["orange"])
    arrow(ax, (0.20, 0.39), (0.365, 0.22), color=COLORS["red"])
    arrow(ax, (0.635, 0.79), (0.80, 0.61), color=COLORS["green"])
    arrow(ax, (0.635, 0.22), (0.80, 0.39), color=COLORS["red"])
    ax.text(0.86, 0.80, "timely\nsuccess", ha="center", va="center", fontsize=6.2, color=COLORS["dark"])
    ax.text(0.12, 0.20, "failure, late,\nor missing", ha="center", va="center", fontsize=6.2,
            color=COLORS["dark"])
    panel_label(ax, "a")

    # (b) The physical path exposes three scheduling classes and two scopes:
    # local optional endpoints and a globally shared recovery/AI lane.
    ax = axes[1]
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, (0.00, 0.70), 0.22, 0.16, "RAN\nhome A", COLORS["light_blue"], COLORS["blue"], 6.5)
    rounded_box(ax, (0.00, 0.14), 0.22, 0.16, "RAN\nhome B", COLORS["light_blue"], COLORS["blue"], 6.5)
    rounded_box(ax, (0.30, 0.70), 0.29, 0.16, "NRx endpoint\nGPU $g_A$", COLORS["light_orange"], COLORS["orange"], 5.9)
    rounded_box(ax, (0.30, 0.14), 0.29, 0.16, "NRx endpoint\nGPU $g_B$", COLORS["light_orange"], COLORS["orange"], 5.9)
    rounded_box(ax, (0.68, 0.36), 0.32, 0.28, "shared lane\ncuPHY recovery\nexternal AI", COLORS["light_purple"], COLORS["purple"], 5.9)
    arrow(ax, (0.22, 0.78), (0.30, 0.78), color=COLORS["orange"])
    arrow(ax, (0.22, 0.22), (0.30, 0.22), color=COLORS["orange"])
    arrow(ax, (0.11, 0.70), (0.68, 0.53), color=COLORS["red"], rad=0.18)
    arrow(ax, (0.11, 0.30), (0.68, 0.47), color=COLORS["red"], rad=-0.18)
    arrow(ax, (0.59, 0.78), (0.84, 0.64), color=COLORS["gray"])
    arrow(ax, (0.59, 0.22), (0.84, 0.36), color=COLORS["gray"])
    panel_label(ax, "b")

    save(fig, "softwall_airan_background")


def make_scheduling_problem() -> None:
    """Three failure modes that motivate conditional-recovery scheduling."""
    fig, axes = plt.subplots(3, 1, figsize=(FIG_WIDTH, 3.75))
    fig.subplots_adjust(hspace=0.72)

    def base_axis(ax, label, title):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 40, 75, 100], ["release", "decision", "guard", "expiry"])
        ax.tick_params(axis="x", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(75, color=COLORS["red"], linestyle="--", linewidth=1.0)
        ax.axvline(100, color=COLORS["dark"], linestyle=":", linewidth=1.0)
        panel_label(ax, label)

    # Current occupancy omits unresolved recovery, so debt-blind AI consumes
    # the only all-fail recovery interval.
    ax = axes[0]
    base_axis(ax, "a", "Current-idle admission hides future mandatory work")
    ax.add_patch(Rectangle((40, 0.57), 35, 0.25, facecolor=COLORS["light_purple"],
                           edgecolor=COLORS["purple"], linewidth=1.0))
    ax.text(57.5, 0.695, "external AI", ha="center", va="center", fontsize=7.0)
    ax.add_patch(Rectangle((68, 0.17), 25, 0.25, facecolor=COLORS["light_red"],
                           edgecolor=COLORS["red"], linewidth=1.2))
    ax.text(80.5, 0.295, "all-fail recovery", ha="center", va="center", fontsize=7.0, zorder=5,
            bbox=dict(facecolor=COLORS["light_red"], edgecolor="none", pad=0.6))
    ax.annotate("overlap", xy=(71.5, 0.50), xytext=(60, 0.97), ha="center", fontsize=6.8,
                color=COLORS["red"], arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=1.0))

    # Static reservation is safe but cannot reclaim debt removed by successful
    # optional outcomes until too late for the visible AI request.
    ax = axes[1]
    base_axis(ax, "b", "Static reservation protects recovery but strands conditional capacity")
    starts = [43, 51, 59, 67]
    for idx, start in enumerate(starts):
        color = COLORS["light_red"] if idx == 3 else COLORS["light_gray"]
        edge = COLORS["red"] if idx == 3 else COLORS["gray"]
        ax.add_patch(Rectangle((start, 0.48), 8, 0.25, facecolor=color, edgecolor=edge, linewidth=1.0))
    ax.text(55, 0.90, "four recovery reservations", ha="center", va="center", fontsize=7.0)
    ax.add_patch(Rectangle((36, 0.08), 36, 0.24, facecolor="none", edgecolor=COLORS["purple"],
                           linewidth=1.0, linestyle="--"))
    ax.text(54, 0.20, "AI opportunity rejected", ha="center", va="center", fontsize=6.8, color=COLORS["purple"])
    ax.text(84, 0.61, "3 successes\nrelease debt", ha="center", va="center", fontsize=6.6,
            color=COLORS["gray"])

    # Two independently valid local calendars can collide when they target the
    # same cuPHY recovery lane.
    ax = axes[2]
    base_axis(ax, "c", "Local certificates do not compose on a shared recovery lane")
    ax.add_patch(Rectangle((48, 0.59), 24, 0.22, facecolor=COLORS["light_blue"],
                           edgecolor=COLORS["blue"], linewidth=1.0))
    ax.text(60, 0.70, "home A recovery", ha="center", va="center", fontsize=7.0)
    ax.add_patch(Rectangle((58, 0.18), 24, 0.22, facecolor=COLORS["light_orange"],
                           edgecolor=COLORS["orange"], linewidth=1.0))
    ax.text(70, 0.29, "home B recovery", ha="center", va="center", fontsize=7.0, zorder=5,
            bbox=dict(facecolor=COLORS["light_orange"], edgecolor="none", pad=0.6))
    ax.axvspan(58, 72, ymin=0.14, ymax=0.86, color=COLORS["red"], alpha=0.16)
    ax.text(58, 0.97, "shared-lane collision", ha="center", va="center", fontsize=6.8, color=COLORS["red"])

    save(fig, "softwall_scheduling_problem")


def make_design_flow() -> None:
    """One event transaction. The refinement layers are given by the table in Section 3."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    width, height = 0.29, 0.22
    xs = (0.035, 0.3575, 0.68)
    top, bottom = 0.64, 0.30
    nodes = [
        (xs[0], top, "Radio release\nreserve recovery debt", COLORS["light_blue"], COLORS["blue"]),
        (xs[1], top, "Optional NeuralRx\ndispatch, hold credit", COLORS["light_orange"], COLORS["orange"]),
        (xs[2], top, "Atomic outcome batch\nsuccess deletes debt;\nfailure retains it", COLORS["light_orange"], COLORS["orange"]),
        (xs[2], bottom, "Recovery replan\nand candidate\nexternal-AI lease", COLORS["light_purple"], COLORS["purple"]),
        (xs[1], bottom, "Independent verifier\nall-fail schedule\nand ownership", COLORS["light_green"], COLORS["green"]),
        (xs[0], bottom, "Worker enforcement\nlatest-start, fence,\nsingle commit", COLORS["light_blue"], COLORS["blue"]),
    ]
    for x, y, text, face, edge in nodes:
        rounded_box(ax, (x, y), width, height, text, face, edge, fontsize=7.0)

    # The verifier has two outcomes: an accepted candidate continues to worker
    # enforcement, and a rejected candidate ends in the labeled reject box.
    top_mid, bottom_mid = top + height / 2, bottom + height / 2
    arrow(ax, (xs[0] + width, top_mid), (xs[1], top_mid))
    arrow(ax, (xs[1] + width, top_mid), (xs[2], top_mid))
    arrow(ax, (xs[2] + width / 2, top), (xs[2] + width / 2, bottom + height))
    arrow(ax, (xs[2], bottom_mid), (xs[1] + width, bottom_mid))
    arrow(ax, (xs[1], bottom_mid), (xs[0] + width, bottom_mid))

    reject_w, reject_h = 0.27, 0.1
    reject_x = xs[1] + (width - reject_w) / 2
    arrow(ax, (xs[1] + width / 2, bottom), (xs[1] + width / 2, 0.035 + reject_h), color=COLORS["red"])
    rounded_box(ax, (reject_x, 0.035), reject_w, reject_h, "reject: keep prior state $X_t$",
                COLORS["light_red"], COLORS["red"], fontsize=7.0)

    # The next event re-enters outcome processing. The route runs outside all
    # boxes, so it crosses no other arrow.
    route_x, route_y = 0.008, 0.955
    ax.plot([xs[0], route_x, route_x, xs[2] + width / 2], [bottom_mid, bottom_mid, route_y, route_y],
            color=COLORS["gray"], lw=ARROW_LW, solid_capstyle="butt")
    arrow(ax, (xs[2] + width / 2, route_y), (xs[2] + width / 2, top + height))
    ax.text(0.40, route_y + 0.012, "completion or outcome starts the next event", ha="center", va="bottom",
            fontsize=7.5, color=COLORS["dark"])

    save(fig, "softwall_design_flow")


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
    make_airan_background()
    make_scheduling_problem()
    make_design_flow()
    qualification = make_qualification_evidence()
    boundaries = make_outcome_boundaries()
    capacity = make_capacity_headroom()
    manifest = {
        "schema": "softwall-paper-figure-manifest-v1",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()},
        "figures": [
            "softwall_airan_background.pdf",
            "softwall_scheduling_problem.pdf",
            "softwall_design_flow.pdf",
            "softwall_qualification_evidence.pdf",
            "softwall_outcome_boundaries.pdf",
            "softwall_capacity_headroom.pdf",
        ],
        "derived": {"qualification": qualification, "boundaries": boundaries, "capacity": capacity},
        "claim_boundary": (
            "All plots reproduce finite-sample audited artifacts. Diagnostic campaigns are not pooled; "
            "sample maxima are not WCET; the 4.5 ms production gate remains failed; and Sionna CDL-D/E "
            "does not qualify Aerial TDL-A or field IQ."
        ),
    }
    manifest["output_sha256"] = {
        name: sha256(HERE / name)
        for stem in ("softwall_airan_background", "softwall_scheduling_problem", "softwall_design_flow",
                     "softwall_qualification_evidence", "softwall_outcome_boundaries",
                     "softwall_capacity_headroom")
        for name in (f"{stem}.pdf", f"{stem}.svg")
    }
    (HERE / "softwall_figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
