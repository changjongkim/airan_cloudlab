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
}

# Per-epoch records of the two-node warm campaign (Q2). The raw coordinator
# logs are local artifacts; their digests are pinned in the Q2 summary JSON.
# The motivation figure is drawn from a small derived file that records the
# counts and the digests of these logs.
MOTIVATION_RAW = [
    ROOT / "results/softwall_multigpu/raw/confirm159q2f_batch_dev_j58857672_coordinator.json",
    ROOT / "results/softwall_multigpu/raw/confirm159q2g_batch_holdout_j58858194_coordinator.json",
]
MOTIVATION_DATA = HERE / "softwall_motivation_data.json"


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




def derive_motivation() -> None:
    """Count retained debts and certificate lease decisions per measured epoch."""
    if not all(path.exists() for path in MOTIVATION_RAW):
        return
    pinned = load("q2")["artifact_sha256"]
    debt_counts = {str(m): 0 for m in range(5)}
    classes = {}
    epochs = 0
    for path in MOTIVATION_RAW:
        rel = str(path.relative_to(ROOT))
        assert pinned[rel] == sha256(path), f"raw log differs from pinned digest: {rel}"
        for row in json.loads(path.read_text())["rounds"]:
            epochs += 1
            retained = len(row["outcome_transition"]["unresolved_obligations"])
            debt_counts[str(retained)] += 1
            entry = classes.setdefault(str(row["context_length"]), {"offered": 0, "fits": 0, "breaks": 0})
            entry["offered"] += 1
            if row["lease_accepted"]:
                entry["fits"] += 1
            else:
                assert row["lease_reason"] == "lease_breaks_global_certificate", row["lease_reason"]
                entry["breaks"] += 1
    data = {
        "schema": "softwall-motivation-data-v1",
        "scope": ("Two-node warm P180/D155 campaign (development and holdout). Retained debts are "
                  "the unresolved NeuralRx outcomes at the 45 ms common cutoff. A lease 'fits' when "
                  "the certificate admitted it under declared bounds and 'breaks' when it violated "
                  "the all-fail recovery schedule. Static reservation of all four admitted debts "
                  "leaves 8 ms of the 108 ms window, so it admits no AI unit."),
        "raw_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in MOTIVATION_RAW},
        "epochs": epochs,
        "retained_debt_epochs": debt_counts,
        "lease_by_context": classes,
    }
    MOTIVATION_DATA.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_motivation() -> dict:
    """Measured retained debt and the errors of the two existing admission views."""
    data = json.loads(MOTIVATION_DATA.read_text())
    counts = data["retained_debt_epochs"]
    epochs = data["epochs"]
    assert epochs == 1200 and sum(counts.values()) == epochs
    classes = data["lease_by_context"]
    contexts = [16, 32, 64, 128, 256, 512]
    assert all(classes[str(c)]["offered"] == 200 for c in contexts)
    assert classes["256"]["breaks"] == 55 and classes["512"]["breaks"] == 68

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 1.95),
                             gridspec_kw={"width_ratios": [0.85, 1.15]})
    fig.subplots_adjust(wspace=0.42)

    # (a) Retained debts at the common cutoff, against the two fixed views.
    ax = axes[0]
    xs = list(range(5))
    shares = [100.0 * counts[str(m)] / epochs for m in xs]
    ax.bar(xs, shares, width=0.62, color=COLORS["orange"], zorder=3, label="measured epochs")
    ax.axvline(0, color=COLORS["red"], linestyle=":", linewidth=1.4, zorder=2,
               label="current-idle view")
    ax.axvline(4, color=COLORS["gray"], linestyle="--", linewidth=1.4, zorder=2,
               label="static reservation")
    ax.set_xticks(xs)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(0, 60)
    ax.set_xlabel("retained debts at cutoff")
    ax.set_ylabel("epochs (%)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("measured epochs", "current-idle view", "static reservation")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, handlelength=1.6, labelspacing=0.25)
    panel_label(ax, "a")

    # (b) Per AI class: static rejects units that fit; current-idle admits units
    # that break the all-fail recovery schedule.
    ax = axes[1]
    xs = list(range(len(contexts)))
    width = 0.36
    static_wrong = [100.0 * classes[str(c)]["fits"] / classes[str(c)]["offered"] for c in contexts]
    idle_wrong = [100.0 * classes[str(c)]["breaks"] / classes[str(c)]["offered"] for c in contexts]
    ax.bar([x - width / 2 for x in xs], static_wrong, width=width, color=COLORS["gray"], zorder=3,
           label="static rejects a fitting unit")
    ax.bar([x + width / 2 for x in xs], idle_wrong, width=width, color=COLORS["red"], zorder=3,
           label="current-idle admits a breaking unit")
    ax.set_xticks(xs, [str(c) for c in contexts])
    ax.set_ylim(0, 105)
    ax.set_xlabel("AI prompt length (tokens)")
    ax.set_ylabel("epochs of the class (%)")
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False,
              handlelength=1.4, labelspacing=0.25)
    panel_label(ax, "b")

    save(fig, "softwall_motivation")
    return {
        "epochs": epochs,
        "retained_debt_share_pct": dict(zip(["0", "1", "2", "3", "4"], shares)),
        "static_wrong_pct": dict(zip([str(c) for c in contexts], static_wrong)),
        "current_idle_wrong_pct": dict(zip([str(c) for c in contexts], idle_wrong)),
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
    derive_motivation()
    motivation = make_motivation()
    qualification = make_qualification_evidence()
    boundaries = make_outcome_boundaries()
    capacity = make_capacity_headroom()
    manifest = {
        "schema": "softwall-paper-figure-manifest-v1",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()},
        "figures": [
            "softwall_motivation.pdf",
            "softwall_qualification_evidence.pdf",
            "softwall_outcome_boundaries.pdf",
            "softwall_capacity_headroom.pdf",
        ],
        "derived": {"motivation": motivation, "qualification": qualification, "boundaries": boundaries,
                    "capacity": capacity},
        "claim_boundary": (
            "All plots reproduce finite-sample audited artifacts. Diagnostic campaigns are not pooled; "
            "sample maxima are not WCET; the 4.5 ms production gate remains failed; and Sionna CDL-D/E "
            "does not qualify Aerial TDL-A or field IQ."
        ),
    }
    manifest["output_sha256"] = {
        name: sha256(HERE / name)
        for stem in ("softwall_motivation", "softwall_qualification_evidence", "softwall_outcome_boundaries",
                     "softwall_capacity_headroom")
        for name in (f"{stem}.pdf", f"{stem}.svg")
    }
    (HERE / "softwall_figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "outputs": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
