#!/usr/bin/env python3
"""Paper-size versions of the v14 figures (canvas 5.5 in wide, fonts at their final size).

usage:
  plot_paper.py model       OUT.pdf SIM.json [SIM.json ...] -- CLOSED.json [CLOSED.json ...]
  plot_paper.py sensitivity OUT.pdf SENSITIVITY.json
  plot_paper.py tradeoff    OUT.pdf SWEEP.json
  plot_paper.py bars        OUT.pdf CODES LABEL=SWEEP.json [LABEL=SWEEP.json ...]
  plot_paper.py la          OUT.pdf LINK_ADAPTATION.json
The timeline figure is plot_v14.py timeline with FIG_SCALE=0.75.
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

from plot_v14 import AQUA, BLUE, BROWN, GOLD, GRID, INK, MUTED, NAME, ORANGE, OUR, PINK, VIOLET

LIGHT = "#8fb8ea"
FS = 7.2            # labels and legends
SMALL = 6.3         # annotations


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=FS, color=INK, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FS, color=INK, labelpad=2)
    if title:
        ax.set_title(title, fontsize=FS + 0.6, color=INK, loc="left", pad=4)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FS - 0.6, length=2, pad=2)


def family(code: str, ours: str = OUR):
    """(legend label, color, marker, point label)"""
    if code == ours:
        return NAME, BLUE, "s", NAME
    if code == "wn" or code[:2] == "wr":
        return "AI next to a neural receiver while 3 or 1 are free", LIGHT, "s", f"{code[2:]} free"
    if code == "vf":
        return "AI pieces that cannot stop", MUTED, "D", ""
    if code == "yyr":
        return "Share chosen by an estimator", GOLD, "v", "estimator"
    if code == "yyp":
        return "Share chosen by an estimator, low-priority AI", BROWN, "P", "estimator"
    if code[0] == "s" and code[1:].isdigit():
        return "Fixed share", ORANGE, "o", f"{code[1:]}%"
    if code[0] == "p" and code[1:].isdigit():
        return "Fixed share, low-priority AI", AQUA, "^", f"{code[1:]}%" if code != "p100" else "no cap"
    if code[0] in "de" and "l" in code:
        shares = code[1:].split("l")[0].replace("x", "/") + "%"
        if code[0] == "d":
            return "Share follows the load", VIOLET, "v", shares
        return "Share follows the load, low-priority AI", PINK, "P", shares
    return None


def handles_of(entries, size: float = 5.0):
    seen, handles = set(), []
    for label, color, marker, _ in entries:
        if label not in seen:
            seen.add(label)
            handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle="", markersize=size,
                                      markeredgecolor="white", markeredgewidth=0.5, label=label))
    return handles


# ---------------------------------------------------------------------------------------------------
def model(out: str, paths: list[str]) -> None:
    split = paths.index("--")
    sim = [r for p in paths[:split] for r in json.loads(Path(p).read_text())["rows"]]
    closed = [r for p in paths[split + 1:] for r in json.loads(Path(p).read_text())["rows"]]
    kinds = (("Without AI", "white"), ("GPU share policies", ORANGE), ("GPU share policies, low-priority AI", AQUA),
             ("AI pieces granted by rule, cannot stop", MUTED), ("AI pieces granted by rule, can stop", BLUE))
    markers = {1: "^", 2: "D", 4: "o"}

    def kind_of(policy: str) -> int:
        if policy in ("yyr", "yyp"):
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
        points["closed"].append((kind_of(r["policy"]), int(r["lanes"]), 100.0 * r["lost_measured"], 100.0 * r["lost_model"]))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.35))
    top = 42.0
    for ax, (key, title) in zip(axes, (("sim", "(a) Event simulation"), ("closed", "(b) Closed form"))):
        ax.plot([0, top], [0, top], color=INK, linewidth=0.8, linestyle=(0, (1, 2)))
        for kind, lanes, x, y in sorted(points[key], key=lambda item: item[0] == 0):
            ax.scatter([x], [y], s=11, color=kinds[kind][1], marker=markers[lanes], edgecolor=INK if kind == 0 else "white",
                       linewidth=0.6 if kind == 0 else 0.3, zorder=4 if kind else 5, alpha=0.9)
        xs = np.array([item[2] for item in points[key]])
        ys = np.array([item[3] for item in points[key]])
        ax.text(0.04, 0.97, f"{len(xs)} runs\ncorrelation {np.corrcoef(xs, ys)[0, 1]:.3f}\nmean absolute error\n{np.abs(xs - ys).mean():.2f} points",
                transform=ax.transAxes, fontsize=SMALL + 0.3, color=INK, va="top", linespacing=1.25)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        style(ax, "Lost candidates, measured (%)", "Lost candidates, predicted (%)", title)
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=4.6, markeredgewidth=0.6,
                          markeredgecolor=INK if c == "white" else "white", label=l) for l, c in kinds]
    handles += [plt.Line2D([], [], color=MUTED, marker=m, linestyle="", markersize=4.6, markeredgecolor="white", label=l)
                for m, l in (("^", "One neural receiver"), ("D", "Two neural receivers"), ("o", "Four neural receivers"))]
    fig.legend(handles=handles, fontsize=FS - 0.4, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
    fig.savefig(out, dpi=300)


# ---------------------------------------------------------------------------------------------------
def sensitivity(out: str, path: str) -> None:
    data = json.loads(Path(path).read_text())
    curve = data["curve"]
    xs = [c["extra_run_ms"] for c in curve]
    shown = [c.strip() for c in os.environ.get("CODES", "wm,wr3,vf,s10,p30,s30,p50,p70,wr1,p100").split(",")]
    place = {"wm": (2, -11, "left"), "wr3": (-1, 5, "center"), "s10": (5, -7, "left"), "vf": (-5, 2, "right"), "p30": (5, -3, "left"),
             "p50": (-5, 1, "right"), "s30": (-5, 2, "right"), "p70": (-5, 2, "right"), "wr1": (5, -8, "left"), "p100": (3, 4, "left")}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.95))
    panels = (("lost", "lost_with_known_shift", "Lost candidates (%)", "(a) Candidates that find no free neural receiver"),
              ("three_slot", "three_slot_with_known_shift", "Runs over three uplink slots (%)", "(b) Neural receiver runs over three slots"))
    entries = []
    for ax, (key, key_shift, ylabel, title) in zip(axes, panels):
        ax.plot(xs, [100 * c[key] for c in curve], color=INK, linewidth=1.3)
        ax.plot(xs, [100 * c[key_shift] for c in curve], color=INK, linewidth=1.0, linestyle=(0, (4, 2)))
        for point in data["points"]:
            code = point["policy"]
            if code not in shown:
                continue
            label, color, marker, name = family(code)
            entries.append((label, color, marker, name))
            x, y = point["extra_run_ms"], 100 * point[key]
            ax.scatter([x], [y], s=44 if code == OUR else 24, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=4)
            dx, dy, ha = place.get(code, (5, 3, "left"))
            if name:
                ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=SMALL, color=INK, ha=ha,
                            fontweight="bold" if code == OUR else "normal")
        ax.set_xlim(-0.08, 2.0)
        ax.set_ylim(0, None)
        style(ax, "Extra run time of the neural receiver (ms)", ylabel, title)
        ax.title.set_fontsize(FS)
    lines = [plt.Line2D([], [], color=INK, linewidth=1.3, label="Closed form, constant extra run time"),
             plt.Line2D([], [], color=INK, linewidth=1.0, linestyle=(0, (4, 2)),
                        label=f"Closed form, conventional result also {data['known_shift_ms']:.1f} ms later")]
    order = sorted(entries, key=lambda e: (e[1] != BLUE, e[1] == LIGHT))
    fig.legend(handles=lines + handles_of(order), fontsize=FS - 0.7, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=1.2)
    fig.savefig(out, dpi=300)


# ---------------------------------------------------------------------------------------------------
def tradeoff(out: str, path: str) -> None:
    skip = set(os.environ.get("SKIP", "vf,wn,n").split(","))
    rows = {}
    for r in json.loads(Path(path).read_text()):
        if r["policy"] not in skip and r["policy"][:2] != "wr" and family(r["policy"]) and r.get("recovered_pct") is not None:
            rows.setdefault(r["policy"], r)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    entries = []
    low = min(r["recovered_pct"] for r in rows.values())
    place = {OUR: (-7, -3, "right"), "p30": (-5, -9, "right"), "yyr": (-6, -8, "right"), "yyp": (5, 4, "left"), "s10": (0, -10, "center")}
    for code, r in rows.items():
        label, color, marker, short = family(code)
        entries.append((label, color, marker, short))
        x, y = r["recovered_pct"], r["ai_slo"] / 1e3
        ax.errorbar([x], [y], xerr=[[x - r.get("recovered_pct_min", x)], [r.get("recovered_pct_max", x) - x]],
                    color=color, linewidth=0.8, zorder=3)
        ax.scatter([x], [y], s=52 if code == OUR else 28, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=4)
        dx, dy, ha = place.get(code, (5, 4, "left"))
        ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=SMALL, color=INK, ha=ha,
                    fontweight="bold" if code == OUR else "normal")
    ax.axvline(100.0, color=INK, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.set_xlim(float(np.floor(low - 0.6)), 100.8)
    ax.set_ylim(0, None)
    style(ax, "Recoveries kept (% of the run without AI)", "AI served within 200 ms (thousand tokens/s)")
    fig.legend(handles=handles_of(sorted(entries, key=lambda e: e[1] != BLUE)), fontsize=FS - 0.4, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=300)


# ---------------------------------------------------------------------------------------------------
def bars(out: str, codes: list[str], items: list[str]) -> None:
    conditions = [(i.split("=", 1)[0], json.loads(Path(i.split("=", 1)[1]).read_text())) for i in items]
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.7), sharex=True)
    entries, n, tops = [], len(codes), []
    for ci, (label, rows) in enumerate(conditions):
        by = {r["policy"]: r for r in rows}
        present = []
        for code in codes:
            if code == "OURS":
                code = OUR
            elif code in ("d", "e"):
                code = next((c for c in by if c[0] == code and c.endswith("l0")), code)
            if by.get(code) is not None and family(code) is not None and by[code].get("recovered_pct") is not None:
                present.append(code)
        for bi, code in enumerate(present):
            r = by[code]
            flabel, color, _, short = family(code)
            entries.append((flabel, color, "s", short))
            x = ci + (bi - (len(present) - 1) / 2) * (0.88 / n)
            for ax, key, scale in ((axes[0], "ai_slo", 1e-3), (axes[1], "recovered_pct", 1.0)):
                value = r[key] * scale
                ax.bar([x], [value], width=0.88 / n - 0.012, color=color, edgecolor="white", linewidth=0.4, zorder=3)
                lo, hi = r.get(key + "_min", r[key]) * scale, r.get(key + "_max", r[key]) * scale
                ax.plot([x, x], [lo, hi], color=INK, linewidth=0.7, zorder=4)
                if key == "ai_slo":
                    ax.text(x, hi + 0.8, f"{value:.1f}", fontsize=5.4, color=INK, ha="center", va="bottom", rotation=90)
                    tops.append(hi)
                else:
                    ax.text(x, 100.7, f"{value:.1f}", fontsize=5.4, color=INK, ha="center", va="bottom", rotation=90)
            axes[1].text(x, -0.03, NAME if code == OUR else short, fontsize=5.4, color=MUTED, ha="center", va="top", rotation=90,
                         transform=axes[1].get_xaxis_transform())
    axes[1].axhline(100.0, color=INK, linewidth=0.8, linestyle=(0, (1, 2)))
    lows = [r["recovered_pct"] for _, rows in conditions for r in rows if r.get("recovered_pct") is not None and r["policy"] != "n"]
    axes[1].set_ylim(min(90.0, float(np.floor(min(lows) - 1.0))), 104.6)
    axes[0].set_ylim(0, max(tops) * 1.27)
    style(axes[0], "", "AI served\n(thousand tokens/s)")
    style(axes[1], "", "Recoveries\nkept (%)")
    for ax in axes:
        ax.grid(False, axis="x")
        ax.set_xticks(range(len(conditions)))
        ax.tick_params(axis="x", length=0)
        ax.set_xticklabels([])
    for ci, (label, _) in enumerate(conditions):
        axes[0].text(ci, 1.03, label.replace("\\n", "\n"), fontsize=FS - 0.6, color=INK, ha="center", va="bottom",
                     transform=axes[0].get_xaxis_transform(), linespacing=1.15)
    fig.legend(handles=handles_of(sorted(entries, key=lambda e: e[1] != BLUE), 5.0), fontsize=FS - 0.4, frameon=False,
               ncol=int(os.environ.get("LEGEND_COLS", "2")), loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handletextpad=0.3)
    rows_in_legend = int(np.ceil(len({e[0] for e in entries}) / int(os.environ.get("LEGEND_COLS", "2"))))
    fig.tight_layout(rect=(0, 0.02 + 0.037 * rows_in_legend, 1, 0.98), h_pad=0.8)
    fig.subplots_adjust(hspace=0.14)
    fig.savefig(out, dpi=300)


# ---------------------------------------------------------------------------------------------------
def la(out: str, path: str) -> None:
    data = json.loads(Path(path).read_text())
    points = {float(e): p for e, p in data["points"].items()}
    xs = sorted(points)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
    ax = axes[0]
    shown = [int(m) for m in os.environ.get("LA_MCS", "13,14,15").split(",")]      # three MCS, lowest first
    shades = dict(zip(shown, ("#f2a27f", ORANGE, "#a8431c")))
    blues = dict(zip(shown, (LIGHT, BLUE, "#184f94")))
    span = xs[-1] - xs[0]
    ticks = xs[::2] if len(xs) > 4 else xs
    for mcs in shown:
        pts = [(e, points[e]["by_mcs"][str(mcs)]) for e in xs if str(mcs) in points[e]["by_mcs"]]
        ex = [e for e, _ in pts]
        ax.plot(ex, [max(100 * r["bler_conventional"], 0.2) for _, r in pts], color=shades[mcs], marker="o", markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.2)
        ax.plot(ex, [max(100 * r["bler_with_recovery_timed"], 0.2) for _, r in pts], color=blues[mcs], marker="s", markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.2, linestyle="--")
        first = pts[0]
        ax.annotate(f"MCS {mcs}", (first[0], max(100 * first[1]["bler_conventional"], 0.2)), xytext=(-5, 5 * (shown.index(mcs) - 1)), textcoords="offset points",
                    fontsize=SMALL, color=INK, ha="right", va="center")
    ax.axhline(10.0, color=INK, linewidth=0.8, linestyle=(0, (1, 2)))
    ax.set_yscale("log")
    ax.set_ylim(0.2, 100)
    ax.set_yticks([0.2, 1, 10, 100])
    ax.set_yticklabels(["0.2 or less", "1", "10", "100"])
    ax.set_xlim(xs[0] - span / 3, xs[-1] + span / 15)
    ax.set_xticks(ticks)
    style(ax, "Es/No (dB)", "TBs not decoded at the\nfirst transmission (%)", "(a) Error rate per MCS")
    ax = axes[1]
    series = (("target10_goodput_conventional", "target10_mcs_conventional", ORANGE, "o", "-"),
              ("target10_goodput_same_mcs_timed", "target10_mcs_conventional", LIGHT, "s", ":"),
              ("target10_goodput_with_recovery_timed", "target10_mcs_with_recovery_timed", BLUE, "s", "--"))
    for key, mkey, color, marker, line in series:
        ys = [points[e][key] / 1e3 for e in xs]
        ax.plot(xs, ys, color=color, marker=marker, markersize=3.6, markeredgecolor="white", markeredgewidth=0.4, linewidth=1.3, linestyle=line)
        if key != "target10_goodput_same_mcs_timed":
            for e, y in zip(xs, ys):
                up = key.endswith("with_recovery_timed")
                ax.annotate(str(points[e][mkey]), (e, y), xytext=(0, 5 if up else -10), textcoords="offset points", fontsize=SMALL - 0.4,
                            color=color, ha="center")
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.08 * (high - low), high + 0.06 * (high - low))
    ax.set_xticks(ticks)
    style(ax, "Es/No (dB)", "Goodput per uplink slot\n(thousand bits)", "(b) Goodput with a 10% error target")
    ax.title.set_fontsize(FS)
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", markersize=3.6, linewidth=1.3, markeredgecolor="white", label="Conventional receiver only"),
               plt.Line2D([], [], color=BLUE, marker="s", markersize=3.6, linewidth=1.3, linestyle="--", markeredgecolor="white",
                          label="With the recovery path"),
               plt.Line2D([], [], color=LIGHT, marker="s", markersize=3.6, linewidth=1.3, linestyle=":", markeredgecolor="white",
                          label="Recovery path, MCS unchanged (b)")]
    fig.legend(handles=handles, fontsize=FS - 0.6, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    fig.savefig(out, dpi=300)


def main() -> None:
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "model":
        model(out, sys.argv[3:])
    elif mode == "sensitivity":
        sensitivity(out, sys.argv[3])
    elif mode == "tradeoff":
        tradeoff(out, sys.argv[3])
    elif mode == "bars":
        bars(out, sys.argv[3].split(","), sys.argv[4:])
    elif mode == "la":
        la(out, sys.argv[3])


if __name__ == "__main__":
    main()
