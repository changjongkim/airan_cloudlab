#!/usr/bin/env python3
"""Evaluation figures: one metric per panel, one color per policy, Antiphase first.

usage:
  plot_eval.py headline OUT FA.json FB.json FC.json          AI served and recoveries lost, three load patterns
  plot_eval.py tradeoff OUT FA.json                          AI served against recoveries lost, steady full load
  plot_eval.py scale    OUT gpus=A,B,C cells=A,B,C,D demand=A,B,C,D     scaling with GPUs, cells, two-user cells
  plot_eval.py use      OUT OPTIMUM_GAP.json                 share of the safe GPU time that becomes AI
  plot_eval.py closed   OUT "LABEL=LA_CLOSED.json" ...       goodput lost with a rate controller in the loop
  plot_eval.py protect  OUT "TITLE=SCHED.json" ...           latency that AI adds to the radio work and latency of
                                                             the AI requests (SCHED.json: output of analyze_sched.py)
  plot_eval.py frontier OUT "TITLE=LA_CLOSED.json" ...       AI served against goodput lost with a rate controller in the
                                                             loop: settings of the scheme and caps of the low-priority baseline
  plot_eval.py slice    OUT RESULTS_DIR                      AI on a slice of the SMs (CUDA green context): goodput lost against
                                                             the run time of the neural receiver and against the AI served
Sweep files are the output of analyze_sweep.py.  The scheme is policy code OUR_V14 (default wm); in the sweeps
with more than four cells per GPU it is wd.  Names of the baselines:
  Static               fixed 10% MPS share
  Estimator            share chosen every second by a reliability estimator
  Priority             low-priority AI with a 30% cap
  Estimator+Priority   the estimator with low-priority AI
  Follow+Priority      share that follows the radio load without lag, low-priority AI
  Priority-70 / Priority-max   low-priority AI with a 70% cap / without a cap (they lose recoveries)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                            "hatch.linewidth": 0.7, "axes.linewidth": 0.9})
import matplotlib.pyplot as plt

NAME = os.environ.get("SCHEME_NAME", "Antiphase")
OUR = os.environ.get("OUR_V14", "wm")
INK = "#111111"
# name -> (policy code, color, marker, loses recoveries)
POLICIES = {
    NAME: ("OURS", "#1f77b4", "o", False),
    "Static": ("s10", "#2ca02c", "^", False),
    "Estimator": ("yyr", "#17becf", "P", False),
    "Priority": ("p30", "#ff7f0e", "s", False),
    "Estimator+Priority": ("yyp", "#d62728", "v", False),
    "Follow+Priority": ("e", "#9467bd", "D", False),
    "Priority-70": ("p70", "#8c564b", "X", True),          # in line plots: dashed
    "Priority-max": ("p100", "#7f7f7f", "*", True),
}
FS = float(os.environ.get("FONT", "10.5"))


def rows_of(path: str) -> dict:
    return {r["policy"]: r for r in json.loads(Path(path).read_text())}


def pick(by: dict, code: str):
    if code == "OURS":
        return by.get("wd") or by.get(OUR)        # wd: the rule in the sweeps of v14h.sh (wm there is a rejected variant)
    if code == "e":
        return next((by[c] for c in by if c[0] == "e" and c.endswith("l0")), None)
    return by.get(code)


def style(ax, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_ylabel(ylabel, fontsize=FS + 1, color=INK)
    ax.set_xlabel(xlabel, fontsize=FS + 1, color=INK)
    ax.grid(axis="y", color="#bbbbbb", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS, colors=INK, length=3)


def bar_kind(color: str, lossy: bool) -> dict:
    if lossy:
        return dict(facecolor="white", edgecolor=color, linewidth=1.3, hatch="////")
    return dict(facecolor=color, edgecolor="black", linewidth=0.8)


def legend(fig, names, ncol=None, y=1.0) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kind(POLICIES[n][1], POLICIES[n][3])) for n in names]
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(names), frameon=False,
               fontsize=FS, handlelength=1.3, columnspacing=1.2, handletextpad=0.5)


# ---------------------------------------------------------------------------------------------------
def headline(out: str, paths: list[str]) -> None:
    names = [NAME, "Static", "Estimator", "Priority", "Estimator+Priority", "Follow+Priority"]
    titles = ["(a) Steady Full Load", "(b) Load Changes Every 2 s", "(c) Random Load Steps"]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.3), sharex="col")
    for col, path in enumerate(paths):
        by = rows_of(path)
        present = [(n, pick(by, POLICIES[n][0])) for n in names]
        present = [(n, r) for n, r in present if r is not None]
        ours = pick(by, "OURS")
        for i, (n, r) in enumerate(present):
            color, lossy = POLICIES[n][1], POLICIES[n][3]
            value = r["ai_slo"] / 1e3
            axes[0][col].bar([i], [value], width=0.74, zorder=3, **bar_kind(color, lossy))
            axes[0][col].plot([i, i], [r["ai_slo_min"] / 1e3, r["ai_slo_max"] / 1e3], color="black", linewidth=1.0, zorder=4)
            top = r["ai_slo_max"] / 1e3 + 0.8
            if n == NAME:
                axes[0][col].text(i, top, f"{value:.1f}k", fontsize=FS, color=color, ha="center", va="bottom")
            else:
                axes[0][col].text(i, top, f"{ours['ai_slo'] / r['ai_slo']:.1f}×", fontsize=FS, color=INK, ha="center", va="bottom")
            lost = max(0.0, 100.0 - r["recovered_pct"])
            low, high = max(0.0, 100.0 - r["recovered_pct_max"]), max(0.0, 100.0 - r["recovered_pct_min"])
            axes[1][col].bar([i], [lost], width=0.74, zorder=3, **bar_kind(color, lossy))
            axes[1][col].plot([i, i], [low, high], color="black", linewidth=1.0, zorder=4)
            axes[1][col].text(i, high + 0.12, f"{lost:.1f}", fontsize=FS, color=color if n == NAME else INK, ha="center", va="bottom")
        axes[0][col].set_title(titles[col], fontsize=FS + 1, color=INK, pad=6)
        axes[0][col].set_ylim(0, 52)
        axes[1][col].set_ylim(0, 7.2)
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([])
            ax.set_xlim(-0.7, len(names) - 0.3)
        style(axes[0][col], "AI Served\n(k tokens/s)" if col == 0 else "")
        style(axes[1][col], "Recoveries\nLost (%)" if col == 0 else "")
    legend(fig, names, ncol=6, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0, h_pad=0.8)
    fig.savefig(out, dpi=200)


# ---------------------------------------------------------------------------------------------------
def protect(out: str, items: list[str]) -> None:
    """One row per load pattern, one panel per metric: what AI adds to the latency of the radio work (the
    run without AI is the zero or the dashed line) and the latency of the AI requests against their limit.
    The L1 misses are those of all seeds together, per million TBs, with their 95% interval."""
    names = [NAME, "Static", "Estimator", "Priority", "Estimator+Priority", "Follow+Priority", "Priority-70", "Priority-max"]
    # label, key, value is the difference to the run without AI, reference line, digits, scale, keys of the interval
    panels = [("L1 Latency Added\n(ms, 99.9th Pct.)", "l1_p999_ms", True, None, 2, 1.0, None),
              ("L1 Deadlines Missed\n(per Million TBs)", "l1_late_pooled_pct", False, "No AI", 0, 1e4, ("l1_late_low_pct", "l1_late_high_pct")),
              ("NeuralRx Run Time\nAdded (ms, Median)", "nrx_run_p50_ms", True, ("Slack", 1.2), 2, 1.0, None),
              ("AI Request Latency\n(ms, 99th Pct.)", "ai_p99_ms", False, ("Limit", 200.0), 0, 1.0, None)]
    fig, axes = plt.subplots(len(items), len(panels), figsize=(10.6, 2.45 * len(items) + 0.5), squeeze=False)
    for row, item in enumerate(items):
        title, path = item.split("=", 1)
        by = rows_of(path)
        none = by["n"]
        present = [(i, n, pick(by, POLICIES[n][0])) for i, n in enumerate(names)]
        present = [(i, n, r) for i, n, r in present if r is not None]
        for col, (label, key, added, line, digits, scale, interval) in enumerate(panels):
            ax = axes[row][col]
            base = none[key] if added else 0.0
            ours = (pick(by, "OURS")[key] - base) * scale
            top = 0.0
            for i, n, r in present:
                color, lossy = POLICIES[n][1], POLICIES[n][3]
                low_key, high_key = interval or (key + "_min", key + "_max")
                value, low, high = (r[key] - base) * scale, (r[low_key] - base) * scale, (r[high_key] - base) * scale
                ax.bar([i], [value], width=0.74, zorder=3, **bar_kind(color, lossy))
                ax.plot([i, i], [low, high], color="black", linewidth=1.0, zorder=4)
                text = f"{value:.{digits}f}" if (n == NAME or not added) else f"{value / ours:.0f}×"
                ax.text(i, max(high, value), " " + text, fontsize=FS - 1.5, color=color if n == NAME else INK, ha="center", va="bottom",
                        rotation=90)
                top = max(top, high, value)
            if line == "No AI":
                ax.axhline(none[key] * scale, color=INK, linewidth=1.3, linestyle=(0, (4, 2)), zorder=5)
                top = max(top, none[key] * scale)
            elif line:
                ax.axhline(line[1], color="#c00000", linewidth=1.3, linestyle=(0, (4, 2)), zorder=5)
                top = max(top, line[1])
            ax.set_ylim(0, max(top, 1e-9) * 1.42)
            ax.set_xticks([])
            ax.set_xlim(-0.7, len(names) - 0.3)
            style(ax, label if row == len(items) - 1 else "")
            if row == len(items) - 1:
                ax.set_ylabel("")
                ax.set_xlabel(label, fontsize=FS + 0.5, color=INK)
            if col == 0:
                ax.set_ylabel(title.replace("\\n", "\n"), fontsize=FS + 0.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kind(POLICIES[n][1], POLICIES[n][3])) for n in names]
    handles += [plt.Line2D([0], [0], color=INK, linewidth=1.3, linestyle=(0, (4, 2))),
                plt.Line2D([0], [0], color="#c00000", linewidth=1.3, linestyle=(0, (4, 2)))]
    labels = names + ["Server without AI", "Slack (1.2 ms), Limit (200 ms)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, fontsize=FS,
               handlelength=1.5, columnspacing=1.2, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.5 / (2.45 * len(items) + 0.5) - 0.02), w_pad=0.8, h_pad=0.6)
    fig.savefig(out, dpi=200)


# ---------------------------------------------------------------------------------------------------
def tradeoff(out: str, path: str) -> None:
    by = rows_of(path)
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    def point(code):
        r = by[code]
        return r["ai_slo"] / 1e3, max(0.0, 100.0 - r["recovered_pct"])
    static = [point(c) for c in ("s10", "s30") if c in by]
    priority = [point(c) for c in ("p30", "p50", "p70", "p100") if c in by]
    ax.plot(*zip(*static), color=POLICIES["Static"][1], marker="^", markersize=10, linewidth=2.6, markeredgecolor="white", zorder=3,
            label="Static (10%, 30% share)")
    ax.plot(*zip(*priority), color=POLICIES["Priority"][1], marker="s", markersize=9, linewidth=2.6, markeredgecolor="white", zorder=3,
            label="Priority (30%, 50%, 70% cap, no cap)")
    for name in ("Estimator", "Estimator+Priority"):
        code, color, marker, _ = POLICIES[name]
        x, y = point(code)
        ax.plot([x], [y], color=color, marker=marker, markersize=11, linestyle="", markeredgecolor="white", zorder=4, label=name)
    ox, oy = point(OUR)
    ax.plot([ox], [oy], color=POLICIES[NAME][1], marker="o", markersize=15, linestyle="", markeredgecolor="white", zorder=5, label=NAME)
    px, py = point("p30")
    ax.annotate("", xy=(ox - 1.0, py + 0.05), xytext=(px + 1.0, py + 0.05), arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.6))
    ax.text((px + ox) / 2, py + 0.28, f"{ox / px:.1f}× more AI", fontsize=FS, color=INK, ha="center")
    qx, qy = point("p50")
    ax.annotate("", xy=(ox + 0.3, oy + 0.35), xytext=(ox + 0.3, qy - 0.1), arrowprops=dict(arrowstyle="->", color=INK, linewidth=1.6))
    ax.text(ox + 1.0, (qy + oy) / 2 + 0.3, f"{qy:.1f}% → {oy:.1f}%\nlost", fontsize=FS, color=INK, ha="left", va="center")
    ax.set_xlim(0, 53)
    ax.set_ylim(-0.4, 7.2)
    style(ax, "Recoveries Lost (%)", "AI Served (k tokens/s)")
    ax.grid(axis="x", color="#bbbbbb", linewidth=0.6, linestyle=(0, (4, 3)))
    ax.legend(fontsize=FS - 0.5, frameon=False, loc="upper left", handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


# ---------------------------------------------------------------------------------------------------
def scale(out: str, groups: list[str]) -> None:
    names = [NAME, "Static", "Priority", "Priority-70"]
    spec = {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in groups}
    axes_of = {"gpus": ("GPUs (4 Cells Each)", ["1", "2", "4"], "(a) Server Size"),
               "cells": ("Cells on Four GPUs", ["16", "20", "32", "48"], "(b) Number of Cells"),
               "demand": ("Two-User Cells of 16", ["2", "4", "6", "8"], "(c) Neural Receiver Demand")}
    fig, axes = plt.subplots(2, len(spec), figsize=(10.2, 5.2), sharex="col")
    for col, (key, paths) in enumerate(spec.items()):
        xlabel, ticks, title = axes_of[key]
        tables = [rows_of(p) for p in paths]
        for n in names:
            code, color, marker, lossy = POLICIES[n]
            rows = [pick(by, code) for by in tables]
            xs = [i for i, r in enumerate(rows) if r is not None and r.get("recovered_pct") is not None]
            if not xs:
                continue
            kind = dict(color=color, marker=marker, markersize=9 if n != NAME else 11, linewidth=2.8 if n == NAME else 2.2,
                        markeredgecolor="white", linestyle=(0, (3, 2)) if lossy else "-", zorder=5 if n == NAME else 3)
            axes[0][col].plot(xs, [rows[i]["ai_slo"] / 1e3 for i in xs], **kind)
            axes[1][col].plot(xs, [max(0.0, 100.0 - rows[i]["recovered_pct"]) for i in xs], **kind)
        axes[0][col].set_title(title, fontsize=FS + 1, color=INK, pad=6)
        axes[1][col].set_xticks(range(len(ticks)))
        axes[1][col].set_xticklabels(ticks)
        axes[0][col].set_ylim(0, None)
        axes[1][col].set_ylim(-0.6, None)
        style(axes[0][col], "AI Served\n(k tokens/s)" if col == 0 else "")
        style(axes[1][col], "Recoveries\nLost (%)" if col == 0 else "", xlabel)
    handles = [plt.Line2D([], [], color=POLICIES[n][1], marker=POLICIES[n][2], markersize=9, linewidth=2.4, markeredgecolor="white",
                          linestyle=(0, (3, 2)) if POLICIES[n][3] else "-") for n in names]
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False, fontsize=FS,
               handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0, h_pad=0.8)
    fig.savefig(out, dpi=200)


# ---------------------------------------------------------------------------------------------------
def use(out: str, path: str, condition: str = "fa c16") -> None:
    c = json.loads(Path(path).read_text())["conditions"][condition]
    bound = c["clairvoyant_tokens"]
    names = [NAME, "Static", "Estimator", "Priority", "Estimator+Priority"]
    values = {NAME: c["rule_tokens"] / bound}
    for n in names[1:]:
        values[n] = c["others"][POLICIES[n][0]]["tokens"] / bound
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for i, n in enumerate(names):
        color = POLICIES[n][1]
        ax.bar([i], [100 * values[n]], width=0.7, zorder=3, **bar_kind(color, False))
        label = f"{100 * values[n]:.0f}%" if n == NAME else f"{values[NAME] / values[n]:.1f}×"
        ax.text(i, 100 * values[n] + 1.5, label, fontsize=FS, color=color if n == NAME else INK, ha="center", va="bottom")
    ax.axhline(100.0, color=INK, linewidth=1.0, linestyle=(0, (2, 2)))
    ax.text(len(names) - 0.55, 97.5, "offline schedule that knows the future", fontsize=FS - 1.5, color=INK, ha="right", va="top", style="italic",
            fontweight="normal")
    ax.set_xticks([])
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.6, len(names) - 0.4)
    style(ax, "Safe GPU Time\nUsed for AI (%)")
    legend(fig, names, ncol=3, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(out, dpi=200)


# ---------------------------------------------------------------------------------------------------
def closed(out: str, items: list[str]) -> None:
    names = [NAME, "Static", "Priority", "Priority-70", "Priority-max"]
    codes = {NAME: "wm", "Static": "s10", "Priority": "p30", "Priority-70": "p70", "Priority-max": "p100"}
    conditions = [(i.rsplit("=", 1)[0], json.loads(Path(i.rsplit("=", 1)[1]).read_text())["policies"]) for i in items]
    fig, ax = plt.subplots(figsize=(10.2, 3.9))
    width = 0.84 / len(names)
    top = 0.0
    for ci, (label, rows) in enumerate(conditions):
        base = rows["n"]["goodput_by_seed"]
        for bi, n in enumerate(names):
            r = rows.get(codes[n])
            if r is None:
                continue
            color, lossy = POLICIES[n][1], POLICIES[n][3]
            loss = max(0.0, -r["vs_recovery_no_ai_pct"])
            seeds = [max(0.0, 100.0 * (1.0 - g / b)) for g, b in zip(r["goodput_by_seed"], base)]
            x = ci + (bi - (len(names) - 1) / 2) * width
            ax.bar([x], [loss], width=width - 0.02, zorder=3, **bar_kind(color, False))
            ax.plot([x, x], [min(seeds), max(seeds)], color="black", linewidth=1.0, zorder=4)
            ax.text(x, max(seeds) + 0.15, f"{loss:.1f}", fontsize=FS - 1.5, color=color if n == NAME else INK, ha="center", va="bottom")
            top = max(top, max(seeds))
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([label.replace("\\n", "\n") for label, _ in conditions], fontsize=FS)
    ax.set_ylim(0, top + 1.2)
    style(ax, "Goodput Lost (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kind(POLICIES[n][1], False)) for n in names]
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False, fontsize=FS,
               handlelength=1.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=200)


def frontier(out: str, items: list[str]) -> None:
    """AI served against goodput lost, one panel per condition: the settings of the rule and the caps of the
    low-priority baseline (items: "TITLE=LA_CLOSED.json")."""
    conditions = [(i.rsplit("=", 1)[0], json.loads(Path(i.rsplit("=", 1)[1]).read_text())["policies"]) for i in items]
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.45 * len(conditions), 3.7), squeeze=False)
    blue, orange, green = POLICIES[NAME][1], POLICIES["Priority"][1], POLICIES["Static"][1]
    for ax, (title, rows) in zip(axes[0], conditions):
        point = lambda c: (rows[c]["ai_slo"] / 1e3, -rows[c]["vs_recovery_no_ai_pct"])
        caps = [c for c in ("p20", "p30", "p40", "p50", "p60", "p70", "p100") if c in rows]
        rule = [c for c in ("wm", "wr3", "wr2", "wr1") if c in rows]
        ax.axhline(0.0, color="#666666", linewidth=0.8, zorder=1)
        ax.plot(*zip(*[point(c) for c in caps]), color=orange, marker="s", markersize=7, linewidth=2.2, markeredgecolor="white", zorder=3)
        for c in caps:
            x, y = point(c)
            ax.annotate("max" if c == "p100" else c[1:], (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=FS - 2.5, color=orange, fontweight="normal")
        if "s10" in rows:
            ax.plot(*point("s10"), color=green, marker="^", markersize=9, linestyle="", markeredgecolor="white", zorder=3)
        ax.plot(*zip(*[point(c) for c in rule]), color=blue, marker="o", markersize=6, linewidth=2.2, markeredgecolor="white", zorder=4)
        ax.plot(*point("wm"), color=blue, marker="o", markersize=13, linestyle="", markeredgecolor="white", zorder=5)
        for c in rule[1:]:
            x, y = point(c)
            ax.annotate(c[2:], (x, y), textcoords="offset points", xytext=(9, -11), ha="center", fontsize=FS - 2.5, color=blue,
                        fontweight="normal")
        ys = [point(c)[1] for c in caps + rule + (["s10"] if "s10" in rows else [])]
        ax.set_ylim(min(-0.35, min(ys) - 0.25), max(1.2, max(ys) * 1.18))
        ax.set_xlim(0, max(point(c)[0] for c in caps + rule) * 1.12)
        ax.set_title(title.replace("\\n", "\n"), fontsize=FS, color=INK)
        style(ax, "Goodput Lost (%)" if ax is axes[0][0] else "", "AI Served (k tokens/s)")
        ax.grid(axis="x", color="#bbbbbb", linewidth=0.6, linestyle=(0, (4, 3)))
    handles = [plt.Line2D([], [], color=blue, marker="o", markersize=12, linestyle="", markeredgecolor="white"),
               plt.Line2D([], [], color=blue, marker="o", markersize=6, linewidth=2.2, markeredgecolor="white"),
               plt.Line2D([], [], color=orange, marker="s", markersize=7, linewidth=2.2, markeredgecolor="white"),
               plt.Line2D([], [], color=green, marker="^", markersize=9, linestyle="", markeredgecolor="white")]
    labels = [NAME, f"{NAME}, AI next to NeuralRx while 3 / 2 / 1 are free", "Priority, cap 20-70% and no cap", "Static"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=FS - 0.5,
               handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=200)


def slice_option(out: str, results: str) -> None:
    """AI on a slice of the SMs of a GPU (v21_slice.sh, v22_slice2.sh): goodput lost against the median run of the
    neural receiver (top) and against the AI served (bottom), one column per condition."""
    R = Path(results)
    load = lambda name: json.loads((R / name).read_text())
    times = [load("slice_time_s1.json"), load("slice_time_s2.json"), load("slice_time_off.json")]
    blue, orange = POLICIES[NAME][1], POLICIES["Priority"][1]
    purple, red, gray = "#9467bd", "#d62728", "#555555"
    # family -> (policy codes, color, marker, size, labels next to the points, offset of a label in points);
    # a slice of more SMs has a larger marker
    by_sms = {"12": 5.5, "16": 8.0, "28": 11.0}
    families = [("rule", ["wm"], blue, "o", 13, None, None),
                ("shared", ["ws12", "ws16", "ws28"], purple, "D", by_sms, None, None),
                ("apart", ["wd16", "wd28"], red, "D", by_sms, None, None),
                ("always", ["g28", "gn28", "gd28"], gray, "^", 9, None, None),
                ("share", ["p30", "p70"], orange, "s", 8, lambda c: c[1:], (9, 6))]
    conditions = [("4 Two-User Cells\nTarget 10%", "lsa", "lta"), ("4 Two-User Cells\nTarget 1%", "lsb", "ltb"),
                  ("8 Two-User Cells\nTarget 1%", "lsc", "ltc")]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.7), squeeze=False)
    for col, (title, shared_tag, apart_tag) in enumerate(conditions):
        points = []                    # (family index, code, neural receiver run ms, AI k tokens/s, goodput lost %)
        for tag in (shared_tag, apart_tag):
            rows = load(f"la_closed_{tag}.json")["policies"]
            for k, (_, codes, *_rest) in enumerate(families):
                for code in codes:
                    if code not in rows:
                        continue
                    runs = [r["nrx_run_ms"]["p50"] for t in times for r in t.get(tag, {}).get("r32" + code, {}).values()]
                    points.append((k, code, sum(runs) / len(runs), rows[code]["ai_slo"] / 1e3, -rows[code]["vs_recovery_no_ai_pct"]))
        for row, pick_x in ((0, lambda q: q[2]), (1, lambda q: q[3])):
            ax = axes[row][col]
            ax.axhline(0.0, color="#666666", linewidth=0.8, zorder=1)
            for k, (_, codes, color, marker, size, label, offset) in enumerate(families):
                mine = [q for q in points if q[0] == k]
                for q in mine:
                    ax.plot([pick_x(q)], [q[4]], color=color, marker=marker, linestyle="", markeredgecolor="white",
                            markersize=size[q[1][2:]] if isinstance(size, dict) else size, zorder=5 if k == 0 else 3)
                for q in mine:
                    if label:
                        ax.annotate(label(q[1]), (pick_x(q), q[4]), textcoords="offset points", xytext=offset, ha="center",
                                    va="center", fontsize=FS - 2.5, color=color, fontweight="normal")
            ys = [q[4] for q in points]
            ax.set_ylim(min(-0.35, min(ys) - 0.25), max(1.2, max(ys) * 1.22))
            if row == 0:
                ax.axvline(7.5, color="#666666", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
                ax.set_xlim(6.2, 8.05)
                if col == 0:
                    ax.annotate("3 Uplink Periods", (7.5, ax.get_ylim()[1]), textcoords="offset points", xytext=(-5, -4),
                                ha="right", va="top", fontsize=FS - 2.5, color="#444444", fontweight="normal")
                ax.set_title(title.replace("\\n", "\n"), fontsize=FS, color=INK)
                style(ax, "Goodput Lost (%)" if col == 0 else "", "NeuralRx Run (ms, Median)")
            else:
                ax.set_xlim(0, max(q[3] for q in points) * 1.12)
                style(ax, "Goodput Lost (%)" if col == 0 else "", "AI Served (k tokens/s)")
            ax.grid(axis="x", color="#bbbbbb", linewidth=0.6, linestyle=(0, (4, 3)))
    handles = [plt.Line2D([], [], color=c, marker=m, markersize=8 if isinstance(z, dict) else min(z, 10), linestyle="",
                          markeredgecolor="white") for _, _, c, m, z, _, _ in families]
    labels = [NAME, f"{NAME} + AI on 12 / 16 / 28 SMs During NeuralRx (Small to Large)",
              "The Same on 16 / 28 SMs, NeuralRx on the Other SMs", "AI Always on 28 SMs", "Priority, Cap 30% and 70%"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=FS - 0.5,
               handlelength=1.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=200)


def main() -> None:
    mode, out = sys.argv[1], sys.argv[2]
    if mode == "headline":
        headline(out, sys.argv[3:6])
    elif mode == "tradeoff":
        tradeoff(out, sys.argv[3])
    elif mode == "scale":
        scale(out, sys.argv[3:])
    elif mode == "use":
        use(out, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "fa c16")
    elif mode == "closed":
        closed(out, sys.argv[3:])
    elif mode == "protect":
        protect(out, sys.argv[3:])
    elif mode == "frontier":
        frontier(out, sys.argv[3:])
    elif mode == "slice":
        slice_option(out, sys.argv[3])


if __name__ == "__main__":
    main()
