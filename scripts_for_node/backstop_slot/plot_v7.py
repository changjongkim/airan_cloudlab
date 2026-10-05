#!/usr/bin/env python3
"""v7 (large NeuralRx as rescue): rescued TBs and L1 misses against AI served, one AI load.

usage: plot_v7.py ANALYSIS.json OUT.png AI_LOAD
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#59636e", "#d9dee4"

data = json.loads(Path(sys.argv[1]).read_text())
load = int(sys.argv[3])
rows = [r for r in data["rows"] if r["ai_rate"] == load]
fixed = sorted((r for r in rows if r["key"].startswith("s")), key=lambda r: int(r["key"][1:]))
ours = [r for r in rows if r["key"].startswith("v")]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
panels = (("TBs rescued by NeuralRx (% of the run without AI)", lambda r: 100 * r["rescue_mean"], False),
          ("TBs that missed the L1 deadline (%, mean of seeds)", lambda r: 100 * r["l1_mean"], True))
for ax, (label, value, log) in zip(axes, panels):
    ax.plot([r["slo"] / 1e3 for r in fixed], [value(r) for r in fixed], color=ORANGE, linewidth=2, marker="o",
            markersize=8, markeredgecolor="white", markeredgewidth=1.5, label="Fixed GPU share", zorder=2)
    ax.plot([r["slo"] / 1e3 for r in ours], [value(r) for r in ours], color=BLUE, linewidth=0, marker="s",
            markersize=10, markeredgecolor="white", markeredgewidth=1.5, label=os.environ.get("SCHEME_NAME", "Antiphase"), zorder=3)
    for i, r in enumerate(fixed):
        if log and i < 3:
            offset, align = (-7, 7), "right"
        elif not log:
            offset, align = (-2, -15), "right"
        else:
            offset, align = (7, 5), "left"
        ax.annotate(f"{r['key'][1:]}%", (r["slo"] / 1e3, value(r)), textcoords="offset points",
                    xytext=offset, ha=align, fontsize=9, color=MUTED)
    if log:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 12)
        ax.axhline(0.05, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate("target 0.05%", (ax.get_xlim()[1], 0.05), textcoords="offset points", xytext=(-4, 4),
                    ha="right", fontsize=8, color=MUTED)
        ax.set_yticks([0.01, 0.1, 1, 10])
        ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    else:
        lowest = min(value(r) for r in fixed + ours)
        ax.set_ylim(5 * (lowest // 5) - 2, 100.8)
    ax.set_xlim(0, max(r["slo"] for r in fixed + ours) / 1e3 * 1.12)
    ax.set_ylabel(label, fontsize=9, color=INK)
    ax.set_xlabel("AI tokens served within the time limit (thousand/s)", fontsize=9, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[0].legend(fontsize=9, frameon=False, loc="lower left")
title = sys.argv[4] if len(sys.argv) > 4 else f"Large NeuralRx as rescue, 16 cells, {load} AI requests/s per GPU"
fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(sys.argv[2], dpi=150)
print(sys.argv[2])
