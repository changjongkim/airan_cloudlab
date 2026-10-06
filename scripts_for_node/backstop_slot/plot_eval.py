#!/usr/bin/env python3
"""Evaluation figures: one metric per panel, one color per policy, Antiphase first.

usage:
  plot_eval.py headline OUT FA.json FB.json FC.json          AI served and recoveries lost, three load patterns
  plot_eval.py tradeoff OUT FA.json                          AI served against recoveries lost, steady full load
  plot_eval.py scale    OUT gpus=A,B,C cells=A,B,C,D demand=A,B,C,D     scaling with GPUs, cells, two-user cells
  plot_eval.py use      OUT OPTIMUM_GAP.json                 share of the safe GPU time that becomes AI
  plot_eval.py closed   OUT "LABEL=LA_CLOSED.json" ...       goodput lost with a rate controller in the loop
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


if __name__ == "__main__":
    main()
