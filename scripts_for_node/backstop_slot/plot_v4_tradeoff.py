#!/usr/bin/env python3
"""AI served versus rescued TBs kept, one panel per cell count (analysis JSON of analyze_v3.py).

usage: plot_v4_tradeoff.py ANALYSIS.json OUT.png [title suffix]
Each panel uses the AI load at which Our Scheme served the most; the line joins the fixed shares.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#59636e", "#d9dee4"

data = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
suffix = sys.argv[3] if len(sys.argv) > 3 else ""
cells = sorted({r["cells"] for r in data["rows"]})
fig, axes = plt.subplots(1, len(cells), figsize=(4.6 * len(cells), 4.2), squeeze=False)
for ax, count in zip(axes[0], cells):
    rows = [r for r in data["rows"] if r["cells"] == count]
    ours_all = [r for r in rows if r["key"].startswith("v")]
    load = max(ours_all, key=lambda r: r["slo"])["ai_rate"]
    fixed = sorted((r for r in rows if r["ai_rate"] == load and r["key"].startswith("s")),
                   key=lambda r: int(r["key"][1:]))
    ax.plot([r["slo"] / 1e3 for r in fixed], [100 * r["rescue_mean"] for r in fixed], color=ORANGE,
            linewidth=2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.5,
            label="Fixed GPU share", zorder=2)
    ours = [r for r in ours_all if r["ai_rate"] == load]
    ax.plot([r["slo"] / 1e3 for r in ours], [100 * r["rescue_mean"] for r in ours], color=BLUE, linewidth=0,
            marker="s", markersize=10, markeredgecolor="white", markeredgewidth=1.5, label="Our Scheme", zorder=3)
    top = max(r["slo"] for r in fixed + ours) / 1e3
    ax.set_xlim(0, top * 1.12)
    # Share labels: to the upper right; to the upper left when the next point sits on top of it.
    fig.canvas.draw()
    spots = [ax.transData.transform((r["slo"] / 1e3, 100 * r["rescue_mean"])) for r in fixed]
    for i, r in enumerate(fixed):
        crowded = i + 1 < len(fixed) and abs(spots[i + 1][0] - spots[i][0]) < 45 and abs(spots[i + 1][1] - spots[i][1]) < 30
        ax.annotate(f"{r['key'][1:]}%", (r["slo"] / 1e3, 100 * r["rescue_mean"]), textcoords="offset points",
                    xytext=(-7, 6) if crowded else (7, 6), ha="right" if crowded else "left", fontsize=9, color=MUTED)
    ax.axhline(100 * data["targets"]["rescue"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("target 99%", (0, 100 * data["targets"]["rescue"]), textcoords="offset points", xytext=(4, -11),
                fontsize=8, color=MUTED)
    ax.set_title(f"{count} cells ({count // 4} per GPU), {load} requests/s per GPU{suffix}", fontsize=10.5,
                 color=INK, loc="left")
    ax.set_xlabel("AI tokens served within the time limit (thousand/s)", fontsize=9, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[0][0].set_ylabel("TBs rescued by NeuralRx (% of the run without AI)", fontsize=9, color=INK)
axes[0][0].legend(fontsize=9, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(out, dpi=150)
print(out)
