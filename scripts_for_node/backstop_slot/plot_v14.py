#!/usr/bin/env python3
"""Figures of v14 (recovery-loss model, stoppable AI pieces, estimator baseline, kinds of AI work).

usage:
  plot_v14.py model    OUT.png LOSS_MODEL.json [LOSS_MODEL.json ...]   lost candidates: model against measurement
  plot_v14.py tradeoff OUT.png TITLE SWEEP.json [SWEEP.json ...]       AI served against recoveries kept
  plot_v14.py gpus     OUT.png LABEL=SWEEP.json ...                    Antiphase against baselines per condition
  plot_v14.py headline OUT.png LABEL=SWEEP.json ...                    three load patterns, all methods
  plot_v14.py classes  OUT.png CLASSES5.json                           what every kind of AI work got
  plot_v14.py la       OUT.png LINK_ADAPTATION.json                    goodput per MCS with and without recovery
  plot_v14.py timeline OUT.png RUN [SLOTS [FIRST]]                     measured timeline of one run (all GPUs)
  plot_v14.py sensitivity OUT.png SENSITIVITY.json                     lost candidates against the extra run time
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42        # TrueType in PDF figures (no Type 3 fonts)
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, VIOLET, PINK, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#b0408f",
                                                       "#1f2328", "#59636e", "#d9dee4")
GOLD, BROWN = "#c9971c", "#7d5a3c"
NAME = os.environ.get("SCHEME_NAME", "Antiphase")
OUR = os.environ.get("OUR_V14", "wm")
SCALE = float(os.environ.get("FIG_SCALE", "1"))      # paper figures: smaller canvas, same font sizes

_subplots = plt.subplots


def _scaled_subplots(*args, figsize=None, **kwargs):
    if figsize is not None:
        figsize = (figsize[0] * SCALE, figsize[1] * SCALE)
    return _subplots(*args, figsize=figsize, **kwargs)


plt.subplots = _scaled_subplots


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def family(code: str, ours: str = OUR):
    """(legend label, color, marker, point label)"""
    if code == ours:
        return NAME, BLUE, "s", NAME
    if code == "wn":
        return f"{NAME}, largest AI units also next to the conventional receiver", "#8fb8ea", "s", "largest AI units too"
    if code == "wm":
        return f"{NAME}, other settings of the rule", "#8fb8ea", "s", "no largest AI units"
    if code[:2] == "wr":
        return f"{NAME}, other settings of the rule", "#8fb8ea", "s", f"largest AI units too,\nnext to a neural receiver\nwhile {code[2:] or 3} free"
    if code == "vf":
        return "Earlier version (AI pieces cannot stop)", MUTED, "D", "earlier version"
    if code == "yyr":
        return "Share chosen by a reliability estimator", GOLD, "v", "estimator"
    if code == "yyp":
        return "Share chosen by a reliability estimator, AI at low priority", BROWN, "P", "estimator"
    if code[0] == "s" and code[1:].isdigit():
        return "Fixed GPU share", ORANGE, "o", f"{code[1:]}%"
    if code[0] == "p" and code[1:].isdigit():
        return "Fixed GPU share, AI at low priority", AQUA, "^", f"{code[1:]}%" if code != "p100" else "no cap"
    if code[0] in "de" and "l" in code:
        shares = code[1:].split("l")[0].replace("x", "/") + "%"
        if code[0] == "d":
            return "Share follows the radio load", VIOLET, "v", shares
        return "Share follows the radio load, AI at low priority", PINK, "P", shares
    return None


def legend(target, entries, **kwargs) -> None:
    seen, handles = set(), []
    for label, color, marker, _ in entries:
        if label in seen:
            continue
        seen.add(label)
        handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle="", markersize=7,
                                  markeredgecolor="white", label=label))
    target.legend(handles=handles, fontsize=8.5, frameon=False, **kwargs)


# ---------------------------------------------------------------------------------------------------
def model(out: str, paths: list[str]) -> None:
    """(a) event simulation, (b) closed form with dependent arrivals; both against the measurement."""
    split = paths.index("--") if "--" in paths else len(paths)
    sim = [r for p in paths[:split] for r in json.loads(Path(p).read_text())["rows"]]
    closed = [r for p in paths[split + 1:] for r in json.loads(Path(p).read_text())["rows"]]
    kinds = (("Without AI", "white"), ("GPU share policies", ORANGE), ("GPU share policies, AI at low priority", AQUA),
             ("AI pieces granted by rule, cannot stop", MUTED), ("AI pieces granted by rule, can stop", BLUE))
    markers = {1: "^", 2: "D", 4: "o"}

    def kind_of(policy: str) -> int:
        if policy in ("yyr", "yyp"):        # a fixed share, chosen again every second
            return 1 if policy == "yyr" else 2
        return 0 if policy == "n" else 1 if policy[0] in "sd" else 2 if policy[0] in "pe" else 3 if policy[0] in "vi" else 4

    points = {"sim": [], "closed": []}
    seen = set()
    for r in sim:
        key = (r["condition"], r["seed"])
        if key not in seen:
            seen.add(key)
            points["sim"].append((0, int(r["lanes"]), 100.0 * r["lost_no_ai_measured"] / r["candidates"],
                                  100.0 * r["lost_no_ai_model"] / r["candidates"]))
        if r["rule"] != "none":
            points["sim"].append((kind_of(r["policy"]), int(r["lanes"]), 100.0 * r["lost_measured"] / r["candidates"],
                                  100.0 * r["lost_model"] / r["candidates"]))
    for r in closed:
        points["closed"].append((kind_of(r["policy"]), int(r["lanes"]), 100.0 * r["lost_measured"],
                                 100.0 * r.get("lost_model", r.get("lost_chain_markov"))))
    panels = [("sim", "(a) Event simulation with the measured arrivals")]
    if closed:
        panels.append(("closed", "(b) Closed form"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.9))
    axes = np.atleast_1d(axes)
    top = 42.0
    for ax, (key, title) in zip(axes, panels):
        ax.plot([0, top], [0, top], color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
        for kind, lanes, x, y in sorted(points[key], key=lambda item: item[0] == 0):
            color = kinds[kind][1]
            ax.scatter([x], [y], s=30, color=color, marker=markers[lanes], edgecolor=INK if kind == 0 else "white",
                       linewidth=0.9 if kind == 0 else 0.6, zorder=4 if kind else 5, alpha=0.9)
        xs = np.array([item[2] for item in points[key]])
        ys = np.array([item[3] for item in points[key]])
        ax.text(0.04, 0.96, f"{len(xs)} runs\ncorrelation {np.corrcoef(xs, ys)[0, 1]:.3f}\n"
                f"mean absolute error {np.abs(xs - ys).mean():.2f} points", transform=ax.transAxes, fontsize=8.5, color=INK, va="top")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        style(ax, "Candidates that got no neural receiver: measured (%)", "Predicted (%)", title)
        ax.title.set_fontsize(9.5)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=7, markeredgecolor=INK if c == "white" else "white", label=l)
               for l, c in kinds]
    handles += [plt.Line2D([], [], color=MUTED, marker=m, linestyle="", markersize=7, markeredgecolor="white", label=l)
                for m, l in (("^", "One neural receiver"), ("D", "Two neural receivers"), ("o", "Four neural receivers"))]
    fig.legend(handles=handles, fontsize=8.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 0.97))
    fig.savefig(out, dpi=170)


# ---------------------------------------------------------------------------------------------------
def tradeoff(out: str, title: str, paths: list[str]) -> None:
    rows = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            if r["policy"] != "n" and r["policy"][:2] != "wr" and family(r["policy"]) and r.get("recovered_pct") is not None:
                rows.setdefault(r["policy"], r)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    entries = []
    low = min(r["recovered_pct"] for r in rows.values())
    for code, r in rows.items():
        label, color, marker, short = family(code)
        entries.append((label, color, marker, short))
        x, y = r["recovered_pct"], r["ai_slo"] / 1e3
        ax.errorbar([x], [y], xerr=[[x - r.get("recovered_pct_min", x)], [r.get("recovered_pct_max", x) - x]],
                    color=color, linewidth=1.0, zorder=3)
        ax.scatter([x], [y], s=110 if code == OUR else 58, color=color, marker=marker, edgecolor="white", linewidth=1.0, zorder=4)
        dx, dy, ha = {OUR: (-9, -15, "right"), "vf": (-7, 7, "right"), "p30": (-7, -13, "right"),
                      "yyr": (-9, 4, "right"), "wn": (-6, 9, "right"), "wm": (-6, 9, "right")}.get(code, (7, 5, "left"))
        ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha,
                    fontweight="bold" if code == OUR else "normal")
    ax.axvline(100.0, color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
    ax.set_xlim(float(np.floor(low - 0.6)), 100.8)
    ax.set_ylim(0, None)
    style(ax, "Recovered transport blocks kept, of the run without AI (%)", "AI served within the time limit (thousand tokens/s)", title)
    legend(fig, sorted(entries, key=lambda e: e[1] != BLUE), ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=170)


# ---------------------------------------------------------------------------------------------------
def bars(out: str, items: list[str], codes: list[str], ours: dict | None = None, width: float = 11.0) -> None:
    """One group of bars per condition: AI served (top) and recoveries kept (bottom).
    Codes ``d`` and ``e`` stand for the share-follows-load policy of the condition, ``OURS`` for
    the scheme (per condition from ``ours``)."""
    conditions = [(i.split("=", 1)[0], json.loads(Path(i.split("=", 1)[1]).read_text())) for i in items]
    fig, axes = plt.subplots(2, 1, figsize=(width, 6.6), sharex=True)
    entries, n = [], len(codes)
    tops = []
    for ci, (label, rows) in enumerate(conditions):
        by = {r["policy"]: r for r in rows}
        mine = (ours or {}).get(label, OUR)
        present = []
        for code in codes:
            if code == "OURS":
                code = mine
            elif code in ("d", "e"):
                code = next((c for c in by if c[0] == code and c.endswith("l0")), code)
            if by.get(code) is not None and family(code, mine) is not None and by[code].get("recovered_pct") is not None:
                present.append(code)
        for bi, code in enumerate(present):
            r = by[code]
            flabel, color, marker, short = family(code, mine)
            entries.append((flabel, color, "s", short))
            x = ci + (bi - (len(present) - 1) / 2) * (0.86 / n)
            for ax, key, scale in ((axes[0], "ai_slo", 1e-3), (axes[1], "recovered_pct", 1.0)):
                value = r[key] * scale
                ax.bar([x], [value], width=0.86 / n - 0.012, color=color, edgecolor="white", linewidth=0.6, zorder=3)
                lo, hi = r.get(key + "_min", r[key]) * scale, r.get(key + "_max", r[key]) * scale
                ax.plot([x, x], [lo, hi], color=INK, linewidth=0.9, zorder=4)
                if key == "ai_slo":
                    ax.text(x, hi + 0.8, f"{value:.1f}", fontsize=6.8, color=INK, ha="center", va="bottom", rotation=90)
                    tops.append(hi)
                else:
                    ax.text(x, 100.9, f"{value:.1f}", fontsize=6.8, color=INK, ha="center", va="bottom", rotation=90)
            axes[1].text(x, -0.03, "ours" if code == mine else short.replace("\n", " "), fontsize=6.8, color=MUTED, ha="center",
                         va="top", rotation=90, transform=axes[1].get_xaxis_transform())
    axes[1].axhline(100.0, color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
    lows = [r["recovered_pct"] for _, rows in conditions for r in rows if r.get("recovered_pct") is not None and r["policy"] != "n"]
    axes[1].set_ylim(min(90.0, float(np.floor(min(lows) - 1.0))), 103.6)
    axes[0].set_ylim(0, max(tops) * 1.2)
    style(axes[0], "", "AI served within the time limit\n(thousand tokens/s)")
    style(axes[1], "", "Recovered transport blocks kept,\nof the run without AI (%)")
    for ax in axes:
        ax.grid(False, axis="x")
    for ax in axes:
        ax.set_xticks(range(len(conditions)))
        ax.tick_params(axis="x", length=0)
    axes[1].set_xticklabels([])
    axes[0].set_xticklabels([])
    for ci, (label, _) in enumerate(conditions):
        axes[0].text(ci, 1.03, label.replace("\\n", "\n"), fontsize=9, color=INK, ha="center", va="bottom",
                     transform=axes[0].get_xaxis_transform())
    legend(fig, sorted(entries, key=lambda e: e[1] != BLUE), ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.12, 1, 0.97), h_pad=1.2)
    fig.subplots_adjust(hspace=0.16)
    fig.savefig(out, dpi=170)


# ---------------------------------------------------------------------------------------------------
def classes(out: str, path: str) -> None:
    """One panel per kind of AI work, one bar per method; first panel: recoveries kept."""
    data = json.loads(Path(path).read_text())
    mine = next(c for c in ("wm", "wn", "wr") if c in data)
    order = [c for c in (mine, "s10", "s30", "p30", "p70", "p100") if c in data]
    first = next(iter(data.values()))["classes"]
    shown = {"chat": "Chat", "large": "Larger LLM", "embed": "Text embedding", "vision": "Image classification",
             "filler": "Work without a time limit"}
    panels = [("Recovered transport blocks kept (%)", lambda r: r["recovered_pct"])]
    for name in sorted(first, key=lambda n: ("chat", "prefill", "encoder", "batch").index(first[n]["kind"])):
        kind, label = first[name]["kind"], shown.get(name, name)
        if kind == "chat":
            panels.append((f"{label}: output tokens on time (per s)", lambda r, n=name: r["classes"][n]["output_tokens_on_time_per_s"]))
            panels.append((f"{label}: responses fully on time (%)", lambda r, n=name: r["classes"][n]["responses_on_time_pct"]))
        elif kind == "encoder":
            panels.append((f"{label}: items within limit (per s)", lambda r, n=name: r["classes"][n]["within_limit_per_s"]))
        elif kind == "batch":
            panels.append((f"{label} (thousand tokens/s)", lambda r, n=name: r["classes"][n]["within_limit_per_s"] / 1e3))
        else:
            panels.append((f"{label}: prompt tokens within limit (thousand/s)",
                           lambda r, n=name: r["classes"][n]["within_limit_per_s"] / 1e3))
    cols = 4 if len(panels) > 4 else 2
    rows_n = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.1 * rows_n))
    axes = np.atleast_1d(axes).ravel()
    entries = []
    ticks = {"wn": "never", "wr": "3 free", "p100": "no cap"}
    for ax, (title, value) in zip(axes, panels):
        for i, code in enumerate(order):
            label, color, marker, short = family(code, mine)
            entries.append((label, color, "s", short))
            try:
                v = float(value(data[code]))
            except KeyError:
                continue
            ax.bar([i], [v], width=0.72, color=color, edgecolor="white", linewidth=0.6, zorder=3)
            ax.text(i, v, f"{v:.1f}" if v < 100 else f"{v:.0f}", fontsize=7.5, color=INK, ha="center", va="bottom")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["ours" if c == mine else ticks.get(c, family(c, mine)[3]) for c in order], fontsize=7.5, color=MUTED)
        style(ax, "", "", title)
        ax.grid(False, axis="x")
        ax.title.set_fontsize(8.5)
        if "kept" in title:
            low = min(float(data[c]["recovered_pct"]) for c in order)
            ax.set_ylim(min(88.0, low - 2.0), 102.5)
            ax.axhline(100.0, color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    for ax in axes[len(panels):]:
        ax.axis("off")
    legend(fig, sorted(entries, key=lambda e: e[1] != BLUE), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=170)


def la(out: str, path: str) -> None:
    """(a) first-transmission error rate against Es/No per MCS, with and without the recovery path;
    (b) goodput of the MCS that a 10%-target rate controller picks, and of the best MCS."""
    data = json.loads(Path(path).read_text())
    points = {float(e): p for e, p in data["points"].items()}
    xs = sorted(points)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    shown = [int(m) for m in os.environ.get("LA_MCS", "13,14,15").split(",")]      # three MCS, lowest first
    shades = dict(zip(shown, ("#f2a27f", ORANGE, "#a8431c")))
    blues = dict(zip(shown, ("#8fb8ea", BLUE, "#184f94")))
    span = xs[-1] - xs[0]
    for mcs in shown:
        pts = [(e, points[e]["by_mcs"][str(mcs)]) for e in xs if str(mcs) in points[e]["by_mcs"]]
        ex = [e for e, _ in pts]
        ax.plot(ex, [max(100 * r["bler_conventional"], 0.2) for _, r in pts], color=shades[mcs], marker="o", markersize=5,
                markeredgecolor="white", linewidth=1.8)
        ax.plot(ex, [max(100 * r["bler_with_recovery_timed"], 0.2) for _, r in pts], color=blues[mcs], marker="s", markersize=5,
                markeredgecolor="white", linewidth=1.8, linestyle="--")
        first = pts[0]
        ax.annotate(f"MCS {mcs}", (first[0], max(100 * first[1]["bler_conventional"], 0.2)), xytext=(-8, 5 * (shown.index(mcs) - 1)), textcoords="offset points",
                    fontsize=8, color=INK, ha="right", va="center")
    ax.axhline(10.0, color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
    ax.text(xs[0] - span * 0.17, 10.6, "10% target", fontsize=8, color=INK, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 100)
    ax.set_yticks([0.2, 1, 10, 100])
    ax.set_yticklabels(["0.2 or less", "1", "10", "100"])
    ax.set_xlim(xs[0] - span * 0.18, xs[-1] + span * 0.05)
    style(ax, "Es/No (dB)", "Transport blocks not decoded at the first transmission (%)", "(a) Error rate per MCS")
    ax.legend(handles=[plt.Line2D([], [], color=ORANGE, marker="o", linewidth=1.8, markeredgecolor="white", label="Conventional receiver only"),
                       plt.Line2D([], [], color=BLUE, marker="s", linewidth=1.8, linestyle="--", markeredgecolor="white", label="With the recovery path")],
              fontsize=8.5, frameon=False, loc="lower left")
    ax = axes[1]
    series = (("target10_goodput_conventional", "target10_mcs_conventional", ORANGE, "o", "-", "Conventional receiver only"),
              ("target10_goodput_same_mcs_timed", "target10_mcs_conventional", "#8fb8ea", "s", ":", "Recovery path, same MCS"),
              ("target10_goodput_with_recovery_timed", "target10_mcs_with_recovery_timed", BLUE, "s", "--", "Recovery path, MCS chosen with it"))
    for key, mkey, color, marker, line, label in series:
        ys = [points[e][key] / 1e3 for e in xs]
        ax.plot(xs, ys, color=color, marker=marker, markersize=6, markeredgecolor="white", linewidth=2, linestyle=line, label=label)
        if key != "target10_goodput_same_mcs_timed":
            for e, y in zip(xs, ys):
                up = key.endswith("with_recovery_timed")
                ax.annotate(str(points[e][mkey]), (e, y), xytext=(0, 8 if up else -14), textcoords="offset points", fontsize=7.5,
                            color=color, ha="center")
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.06 * (high - low), high + 0.04 * (high - low))
    style(ax, "Es/No (dB)", "Goodput per uplink slot (thousand bits)", "(b) Goodput with a 10% error-rate target (numbers: MCS chosen)")
    ax.title.set_fontsize(9.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=170)


def sensitivity(out: str, path: str) -> None:
    """(a) candidates that got no neural receiver and (b) runs that span three uplink slots, against
    the extra run time of the neural receiver: closed form (line) and measured policies (points)."""
    data = json.loads(Path(path).read_text())
    curve = data["curve"]
    xs = [c["extra_run_ms"] for c in curve]
    shown = [c.strip() for c in os.environ.get("CODES", "wm,wr3,vf,s10,p30,s30,p50,p70,wr1,p100").split(",")]
    short = {"wm": NAME, "wr3": "AI next to a neural receiver\nwhile 3 are free", "wr1": "while 1 is free", "vf": "earlier version"}
    place = {"wm": (8, -4, "left"), "wr3": (-2, 9, "center"), "s10": (0, -14, "center"), "vf": (-6, 6, "right"), "p30": (8, -4, "left"),
             "p50": (-8, 2, "right"), "s30": (-8, 2, "right"), "p70": (-8, 3, "right"), "wr1": (6, -12, "left"), "p100": (6, 5, "left")}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    panels = (("lost", "lost_with_known_shift", "lost", "Candidates that got no neural receiver (%)", "(a) Lost candidates"),
              ("three_slot", "three_slot_with_known_shift", "three_slot", "Neural receiver runs that span three uplink slots (%)",
               "(b) Runs that span three uplink slots"))
    entries = []
    for ax, (key, key_shift, point_key, ylabel, title) in zip(axes, panels):
        ax.plot(xs, [100 * c[key] for c in curve], color=INK, linewidth=1.8, label="Closed form")
        ax.plot(xs, [100 * c[key_shift] for c in curve], color=INK, linewidth=1.4, linestyle=(0, (4, 2)),
                label=f"Closed form, conventional result {data['known_shift_ms']:.1f} ms later")
        for point in data["points"]:
            code = point["policy"]
            if code not in shown:
                continue
            label, color, marker, name = family(code)
            entries.append((label if code[:2] != "wr" else f"{NAME}, other settings of the rule", color, marker, name))
            x, y = point["extra_run_ms"], 100 * point[point_key]
            ax.scatter([x], [y], s=110 if code == OUR else 58, color=color, marker=marker, edgecolor="white", linewidth=1.0, zorder=4)
            dx, dy, ha = place.get(code, (7, 5, "left"))
            ax.annotate(short.get(code, name), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK, ha=ha,
                        fontweight="bold" if code == OUR else "normal")
        ax.set_xlim(-0.08, 2.0)
        ax.set_ylim(0, None)
        style(ax, "Extra run time of the neural receiver next to AI, median (ms)", ylabel, title)
    lines = [plt.Line2D([], [], color=INK, linewidth=1.8, label="Closed form, constant extra run time"),
             plt.Line2D([], [], color=INK, linewidth=1.4, linestyle=(0, (4, 2)),
                        label=f"Closed form, conventional result also {data['known_shift_ms']:.1f} ms later")]
    seen, handles = set(), []
    for label, color, marker, _ in sorted(entries, key=lambda e: e[1] != BLUE):
        if label not in seen:
            seen.add(label)
            handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle="", markersize=7, markeredgecolor="white", label=label))
    fig.legend(handles=lines + handles, fontsize=8.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=170)


def timeline(out: str, run: str, count: int = 10, first: int | None = None) -> None:
    """Measured timeline of one run: neural receiver runs and AI pieces on every GPU.
    Shows the rule: the AI pieces of a GPU stop when a neural receiver starts there and resume
    when it ends; the other GPUs keep running AI."""
    raw = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
    work = raw / f"{run}_work"
    config = json.loads((raw / f"{run}.json").read_text())["config"]
    gpus = int(config["num_gpus"])
    period = int(float(config["period_ms"]) * 1e6)
    head = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = head[2] - head[0] * period
    nrx = {g: [(r[2], r[3]) for r in json.loads((work / f"lane{g}.json").read_text())["records"]] for g in range(gpus)}
    ai = {g: [(q[0], q[1]) for q in json.loads((work / f"ai{g}.json").read_text())["pieces"]] for g in range(gpus)}
    span = count * period
    if first is None:           # a window with three or four runs on different GPUs, none cut by the window edge
        best, best_score = 40, -1.0
        for k in range(40, int(config["periods"]) - count, 3):
            a, b = epoch + k * period, epoch + k * period + span
            inside = [(g, s, d) for g in range(gpus) for s, d in nrx[g] if s > a + 1e6 and d < b - 1e6]
            cut = sum(1 for g in range(gpus) for s, d in nrx[g] if (s < a < d) or (s < b < d))
            pieces = sum(1 for g in range(gpus) for s, d in ai[g] if s > a and d < b)
            score = len({g for g, _, _ in inside}) * 2 + min(len(inside), 4) - 3 * cut + min(pieces, 40) / 40
            if score > best_score:
                best, best_score = k, score
        first = best
    origin = epoch + first * period
    ms = lambda t: (t - origin) / 1e6
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    for g in range(gpus):
        y = gpus - 1 - g
        for start, done in nrx[g]:
            if done > origin and start < origin + span:
                ax.barh(y + 0.2, ms(done) - ms(start), left=ms(start), height=0.3, color=AQUA, linewidth=0)
                if ms(start) > 0 and ms(done) < span / 1e6:
                    ax.annotate(f"{ms(done) - ms(start):.1f} ms", (ms(start) + 0.15, y + 0.2), va="center", fontsize=7.5,
                                color="white", fontweight="bold")
        for start, done in ai[g]:
            if done > origin and start < origin + span:
                ax.barh(y - 0.2, ms(done) - ms(start), left=ms(start), height=0.3, color=BLUE, linewidth=0.4, edgecolor="white")
    for k in range(count + 1):
        ax.axvline(k * period / 1e6, color=MUTED, linestyle=(0, (1, 2)), linewidth=0.8, zorder=0)
    ax.set_yticks(range(gpus))
    ax.set_yticklabels([f"GPU {gpus - 1 - y}" for y in range(gpus)], fontsize=8.5, color=INK)
    ax.set_ylim(-0.6, gpus - 0.4)
    ax.set_xlim(0, span / 1e6)
    ax.set_xlabel("Time (ms); dotted lines: uplink slots arrive", fontsize=8.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["Neural receiver run (run time inside)", "AI pieces"], fontsize=8, frameon=False, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title(f"{NAME}: measured run, uplink slots {first}-{first + count - 1}", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out, "first period", first)


def closedloop(out: str, items: list[str]) -> None:
    """Closed-loop link adaptation, one panel per condition ("title=la_closed_*.json"): AI served against the
    goodput of the two-user cells relative to the server without AI."""
    conditions = [(i.split("=", 1)[0], json.loads(Path(i.split("=", 1)[1]).read_text())["policies"]) for i in items]
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.3 * len(conditions), 4.4), sharey=True, squeeze=False)
    entries = []
    shown = lambda c: c not in ("x", "n", "xp100") and c[:2] != "wr" and family(c) is not None
    low = min(min(100.0 * (g / n - 1.0) for g, n in zip(r["goodput_by_seed"], rows["n"]["goodput_by_seed"]))
              for _, rows in conditions for c, r in rows.items() if shown(c))
    for ax, (title, rows) in zip(axes[0], conditions):
        for code, r in rows.items():
            if not shown(code):
                continue
            label, color, marker, short = family(code)
            entries.append((label, color, marker, short))
            x, y = r["ai_slo"] / 1e3, r["vs_recovery_no_ai_pct"]
            ys = [100.0 * (g / n - 1.0) for g, n in zip(r["goodput_by_seed"], rows["n"]["goodput_by_seed"])]
            ax.plot([x, x], [min(ys), max(ys)], color=color, linewidth=1.0, zorder=3)
            ax.scatter([x], [y], s=110 if code == OUR else 58, color=color, marker=marker, edgecolor="white", linewidth=1.0, zorder=4)
            if code == OUR:
                dx, dy, ha = (0, 10, "center") if x > 8 else (2, 10, "left")
            elif code == "p70" or (code == "p100" and x > 44):
                dx, dy, ha = -7, -3, "right"
            elif code == "p100":
                dx, dy, ha = 7, -3, "left"
            else:
                dx, dy, ha = 7, -11, "left"
            ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK, ha=ha,
                        fontweight="bold" if code == OUR else "normal")
        ax.axhline(0.0, color=INK, linewidth=1.0, linestyle=(0, (1, 2)))
        gain = rows["n"]["vs_no_recovery_pct"]
        style(ax, "AI served within the time limit\n(thousand tokens/s)", "", f"{title}\nRecovery path: {gain:+.0f}% goodput")
        ax.set_xlim(0, 56)
    axes[0][0].set_ylim(float(np.floor(low - 0.4)), 1.2)
    axes[0][0].set_ylabel("Goodput of the two-user cells,\nagainst the server without AI (%)", fontsize=9, color=INK)
    legend(fig, sorted(entries, key=lambda e: e[1] != BLUE), ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=170)


def main() -> None:
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "closedloop":
        return closedloop(out, sys.argv[3:])
    if mode == "timeline":
        return timeline(out, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10,
                        int(sys.argv[5]) if len(sys.argv) > 5 else None)
    if mode == "classes":
        return classes(out, sys.argv[3])
    if mode == "la":
        return la(out, sys.argv[3])
    if mode == "sensitivity":
        return sensitivity(out, sys.argv[3])
    if mode == "model":
        model(out, sys.argv[3:])
    elif mode == "tradeoff":
        tradeoff(out, sys.argv[3], sys.argv[4:])
    elif mode == "gpus":
        ours = dict(pair.split(":") for pair in os.environ.get("OURS_BY_LABEL", "").split(";") if pair)
        bars(out, sys.argv[3:], ["OURS", "s10", "p30", "p70"], ours, width=9.6)
    elif mode == "headline":
        codes = os.environ.get("CODES", "OURS,s10,p30,p70,d,e,yyr,yyp").split(",")
        bars(out, sys.argv[3:], codes)


if __name__ == "__main__":
    main()
