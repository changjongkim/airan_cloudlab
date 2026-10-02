#!/usr/bin/env python3
"""Figures of the v13 sweeps (inputs: analyze_sweep.py / analyze_series.py JSON files).

usage:
  plot_v13.py aiload   SWEEP.json OUT.png             AI served, recoveries kept, L1 late vs offered AI load
  plot_v13.py frontier SWEEP.json RATE OUT.png [title] AI served vs recoveries kept at one AI load
  plot_v13.py axis     OUT.png XLABEL  X1=SWEEP1.json X2=SWEEP2.json ...   one condition per x value
  plot_v13.py series   SERIES.json SEED OUT.png        load, AI served and recoveries over time
The scheme name comes from SCHEME_NAME (default "Our Scheme").
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#1f2328", "#59636e", "#d9dee4"
NAME = os.environ.get("SCHEME_NAME", "Our Scheme")
TOKENS_PER_REQUEST, GPUS = 414.8, 4
OURS = "vf"
RATE = int(os.environ.get("AI_RATE", "32"))     # AI load shown where a figure has one condition per x value


def family(code: str):
    """(family label, color, marker, line style, short label)"""
    if code == OURS:
        return NAME, BLUE, "s", "-", NAME
    if code == "i":
        return "AI only while the GPU has no radio work", MUTED, "D", ":", "idle only"
    if code[0] == "s":
        return "Fixed GPU share", ORANGE, "o", "-", f"{code[1:]}%"
    if code[0] == "p":
        return "Fixed GPU share, AI at low priority", AQUA, "^", "--", f"{code[1:]}%"
    shares, lag = code[1:].split("l")
    label = shares.replace("x", "/") + "%" + ("" if lag == "0" else ", 1 s late")
    line = "-." if lag == "0" else ":"
    if code[0] == "d":
        return "Share follows the radio load", VIOLET, "v", line, label
    return "Share follows the radio load, AI at low priority", "#b0408f", "P", line, label


def known(code: str) -> bool:
    return code in (OURS, "i") or (code[0] in "sp" and code[1:].isdigit()) or (code[0] in "de" and "l" in code)


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


def family_legend(ax_or_fig, codes, without_ai=False, **kwargs) -> None:
    seen, handles = set(), []
    for code in codes:
        label, color, marker, line, _ = family(code)
        if label not in seen:
            seen.add(label)
            handles.append(plt.Line2D([], [], color=color, marker=marker, linestyle=line, linewidth=2 if code == OURS else 1.4,
                                      markersize=7, markeredgecolor="white", label=label))
    if without_ai:
        handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (1, 2)), linewidth=1.2, label="Without AI (panel c)"))
    ax_or_fig.legend(handles=handles, fontsize=8.5, frameon=False, **kwargs)


def spread_labels(values, gap, low, high):
    """Label positions near ``values`` that keep ``gap`` between neighbours and stay inside [low, high]."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ys = [values[i] for i in order]
    gap = min(gap, (high - low) / max(1, len(ys)))
    for _ in range(400):
        moved = False
        for i in range(1, len(ys)):
            short = gap - (ys[i] - ys[i - 1])
            if short > 1e-9:
                ys[i - 1] -= short / 2
                ys[i] += short / 2
                moved = True
        if ys and ys[-1] > high:
            ys = [y - (ys[-1] - high) for y in ys]
        if ys and ys[0] < low:
            ys = [y + (low - ys[0]) for y in ys]
        if not moved:
            break
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = ys[rank]
    return out


def lines(ax, rows, codes, key, xs_of, scale=1.0, log=False, labels=True):
    import math
    ends = []
    for code in codes:
        mine = sorted((r for r in rows if r["policy"] == code and r.get(key) is not None), key=xs_of)
        if not mine:
            continue
        _, color, marker, line, short = family(code)
        x, y = [xs_of(r) for r in mine], [r[key] * scale for r in mine]
        ax.plot(x, y, color=color, marker=marker, linestyle=line, linewidth=2.4 if code == OURS else 1.4,
                markersize=7 if code == OURS else 5.5, markeredgecolor="white", markeredgewidth=1.0,
                zorder=4 if code == OURS else 2)
        ends.append((x[-1], y[-1], short, color))
    if not (labels and ends):
        return
    low, high = ax.get_ylim()
    left, right = ax.get_xlim()
    x_end = max(e[0] for e in ends)
    if ax.get_xscale() == "log":
        x_text = 10 ** (math.log10(x_end) + 0.12 * (math.log10(right) - math.log10(left)))
    else:
        x_text = x_end + 0.07 * (right - left)
    to = (lambda v: math.log10(max(v, 1e-9))) if log else (lambda v: v)
    back = (lambda v: 10 ** v) if log else (lambda v: v)
    span = to(high) - to(low)
    placed = spread_labels([to(e[1]) for e in ends], span / 15, to(low) + span * 0.02, to(high) - span * 0.04)
    for (x, y, text, color), where in zip(ends, placed):
        ax.annotate(text, xy=(x, y), xytext=(x_text, back(where)), fontsize=8, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, shrinkA=0, shrinkB=3))


def aiload(path: str, out: str) -> None:
    rows = json.loads(Path(path).read_text())
    codes = [c for c in ("s10", "s30", "s50", "s100", "p30", "p50", "p70", "p100", OURS) if any(r["policy"] == c for r in rows)]
    ref = next((r for r in rows if r["policy"] == "n"), None)
    offered = lambda r: r["ai_rate"] * GPUS * TOKENS_PER_REQUEST / 1e3
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax in axes:
        ax.set_xlim(0, max(offered(r) for r in rows) * 1.25)
    top = max(r.get("ai_slo", 0) for r in rows) / 1e3
    axes[0].set_ylim(0, top * 1.08)
    lines(axes[0], rows, codes, "ai_slo", offered, 1e-3)
    style(axes[0], "AI offered (thousand tokens/s)", "AI served within 200 ms (thousand tokens/s)", "(a) AI served")
    # The uncapped share loses far more than the others: it is named in a note, not drawn.
    shown = [c for c in codes if c != "s100"]
    low = min(r["recovered_pct"] for r in rows if r["policy"] in shown and r.get("recovered_pct") is not None)
    axes[1].set_ylim(low - 1.5, 100.5)
    lines(axes[1], rows, shown, "recovered_pct", offered)
    off = [r["recovered_pct"] for r in rows if r["policy"] == "s100" and r.get("recovered_pct") is not None]
    if off:
        axes[1].text(0.03, 0.04, f"Fixed 100% keeps {min(off):.0f}-{max(off):.0f}% (below the axis)", transform=axes[1].transAxes,
                     fontsize=8, color=ORANGE)
    style(axes[1], "AI offered (thousand tokens/s)", "Recovered TBs kept (% of the run without AI)", "(b) Recoveries kept")
    axes[2].set_yscale("log")
    values = [r["l1_late_pct"] for r in rows if r["policy"] in codes]
    axes[2].set_ylim(max(0.004, min(values) * 0.6), max(values) * 1.6)
    if ref:
        axes[2].axhline(ref["l1_late_pct"], color=INK, linewidth=1.2, linestyle=(0, (1, 2)))
    lines(axes[2], rows, codes, "l1_late_pct", offered, log=True)
    style(axes[2], "AI offered (thousand tokens/s)", "TBs that miss the layer-1 deadline (%)", "(c) Layer-1 deadline")
    family_legend(fig, codes, without_ai=True, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    print(out)


def frontier(path: str, rate: int, out: str, title: str = "") -> None:
    rows = [r for r in json.loads(Path(path).read_text()) if r["ai_rate"] == rate and r.get("recovered_pct") is not None
            and known(r["policy"]) and r["policy"] != "s100"]
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    groups = {}
    for r in rows:
        groups.setdefault(family(r["policy"])[0], []).append(r)
    for label, items in groups.items():
        items.sort(key=lambda r: r["ai_slo"])
        _, color, marker, line, _ = family(items[0]["policy"])
        ours = items[0]["policy"] == OURS
        ax.plot([r["ai_slo"] / 1e3 for r in items], [r["recovered_pct"] for r in items], color=color, marker=marker,
                linestyle=line if len(items) > 1 and items[0]["policy"][0] in "sp" else "", linewidth=1.6,
                markersize=11 if ours else 8,
                markeredgecolor="white", markeredgewidth=1.5, label=label, zorder=5 if ours else 3)
        for r in items:
            if "ai_slo_min" in r and len(r["seeds"]) > 1:
                ax.plot([r["ai_slo_min"] / 1e3, r["ai_slo_max"] / 1e3], [r["recovered_pct"]] * 2, color=color, linewidth=1, alpha=0.6)
                ax.plot([r["ai_slo"] / 1e3] * 2, [r["recovered_pct_min"], r["recovered_pct_max"]], color=color, linewidth=1, alpha=0.6)
            if not ours and r["policy"] != "i":
                late = "late" in family(r["policy"])[4]
                ax.annotate(family(r["policy"])[4], (r["ai_slo"] / 1e3, r["recovered_pct"]), textcoords="offset points",
                            xytext=(-8, -4) if late else (6, 5), ha="right" if late else "left", fontsize=8, color=MUTED)
    low = min(r.get("recovered_pct_min", r["recovered_pct"]) for r in rows)
    ax.set_ylim(low - 0.8, 100.6)
    ax.set_xlim(0, max(r.get("ai_slo_max", r["ai_slo"]) for r in rows) / 1e3 * 1.12)
    style(ax, "AI served within 200 ms (thousand tokens/s)", "Recovered TBs kept (% of the run without AI)", title)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


def axis(out: str, xlabel: str, pairs: list[str]) -> None:
    rows = []
    for pair in pairs:
        x, path = pair.split("=", 1)
        for r in json.loads(Path(path).read_text()):
            if r["ai_rate"] in (0, RATE):
                rows.append(dict(r, x=float(x)))
    present = {r["policy"] for r in rows}
    codes = [c for c in ("s10", "s30", "s50", "p30", "p50", "p70", "d10x50l0", "d10x50l400", "e30x70l0", "e30x70l400", OURS)
             if c in present]
    xs = sorted({r["x"] for r in rows})
    # A policy is drawn only if it was run at the last x value: its label sits at the end of its line.
    codes = [c for c in codes if any(r["policy"] == c and r["x"] == xs[-1] for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    for ax in axes:
        if xs[-1] / max(xs[0], 1e-9) > 20 and xs[0] > 0:
            ax.set_xscale("log")
            ax.set_xlim(xs[0] * 0.8, xs[-1] * 3.2)
        else:
            ax.set_xlim(xs[0] - 0.05 * (xs[-1] - xs[0]), xs[-1] + 0.3 * (xs[-1] - xs[0]))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.minorticks_off()
    axes[0].set_ylim(0, max(r.get("ai_slo", 0) for r in rows) / 1e3 * 1.08)
    lines(axes[0], rows, codes, "ai_slo", lambda r: r["x"], 1e-3)
    style(axes[0], xlabel, "AI served within 200 ms (thousand tokens/s)", "(a) AI served")
    kept = [r["recovered_pct"] for r in rows if r["policy"] in codes and r.get("recovered_pct") is not None]
    if kept:
        axes[1].set_ylim(min(kept) - 1.5, 101)
        lines(axes[1], rows, codes, "recovered_pct", lambda r: r["x"])
    style(axes[1], xlabel, "Recovered TBs kept (% of the run without AI)", "(b) Recoveries kept")
    axes[2].set_yscale("log")
    values = [r["l1_late_pct"] for r in rows if r["policy"] in codes or r["policy"] == "n"]
    axes[2].set_ylim(max(0.004, min(values) * 0.6), max(values) * 1.6)
    refs = sorted((r for r in rows if r["policy"] == "n"), key=lambda r: r["x"])
    if refs:
        axes[2].plot([r["x"] for r in refs], [max(r["l1_late_pct"], 0.004) for r in refs], color=INK, linewidth=1.2,
                     linestyle=(0, (1, 2)), marker="x", markersize=5)
    lines(axes[2], rows, codes, "l1_late_pct", lambda r: r["x"], log=True)
    style(axes[2], xlabel, "TBs that miss the layer-1 deadline (%)", "(c) Layer-1 deadline")
    family_legend(fig, codes, without_ai=True, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=3.0)
    fig.savefig(out, dpi=150)
    print(out)


def series(path: str, seed: str, out: str) -> None:
    data = json.loads(Path(path).read_text())
    wanted = ("s10", "p70", "e30x50x70l0", "e30x50x70l400", "e30x70l0", "e30x70l400", OURS)
    codes = [c for c in wanted if c in data]
    ref = data["n"]["series"][seed]
    t = ref["t_s"]
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.8), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})
    axes[0].fill_between(t, ref["active_cells"], step="post", color=GRID)
    axes[0].plot(t, ref["active_cells"], drawstyle="steps-post", color=MUTED, linewidth=1.2)
    style(axes[0], "", "Active cells", "(a) Radio load: cells that carry a TB in an uplink slot")
    handles = []
    for code in codes:
        label, color, marker, line, short = family(code)
        one = data[code]["series"][seed]
        text = label if code == OURS else f"{label} ({short})"
        width = 2.4 if code == OURS else 1.4
        handles += axes[1].plot(t, [v / 1e3 for v in one["ai_tokens_per_s"]], color=color, linestyle=line, linewidth=width,
                                label=text, zorder=4 if code == OURS else 2)
        lost, total = [], 0.0
        for a, b in zip(ref["recovered"], one["recovered"]):
            total += a - b
            lost.append(total)
        axes[2].plot(t, lost, color=color, linestyle=line, linewidth=width, zorder=4 if code == OURS else 2)
    style(axes[1], "", "Thousand tokens/s", "(b) AI served within 200 ms")
    style(axes[2], "Time (s)", "TBs", "(c) Recovered TBs lost so far, against the run without AI")
    fig.legend(handles=handles, fontsize=8.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04 + 0.022 * ((len(handles) + 1) // 2), 1, 1))
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "aiload":
        aiload(sys.argv[2], sys.argv[3])
    elif kind == "frontier":
        frontier(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "")
    elif kind == "axis":
        axis(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif kind == "series":
        series(sys.argv[2], sys.argv[3], sys.argv[4])
