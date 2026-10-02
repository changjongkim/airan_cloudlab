#!/usr/bin/env python3
"""Whole-run AI served versus rescued TBs when the load changes within the run:
Our Scheme, fixed GPU shares, and shares switched with the load (analyze_phases.py output).

usage: plot_v11.py PHASES.json OUT.png "title"
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#1f2328", "#59636e", "#d9dee4"

rows = json.loads(Path(sys.argv[1]).read_text())
fig, ax = plt.subplots(figsize=(6.6, 4.4))
fixed = sorted((r for r in rows if re.fullmatch(r"s\d+", r["policy"])), key=lambda r: int(r["policy"][1:]))
ours = [r for r in rows if r["policy"].startswith("v")]
dyn = [r for r in rows if r["policy"].startswith("d")]
ax.plot([r["ai"] / 1e3 for r in fixed], [r["rescue"] for r in fixed], color=ORANGE, linewidth=2, marker="o",
        markersize=8, markeredgecolor="white", markeredgewidth=1.5, label="Fixed GPU share", zorder=2)
for r in fixed:
    ax.annotate(f"{r['policy'][1:]}%", (r["ai"] / 1e3, r["rescue"]), textcoords="offset points", xytext=(-2, -15),
                ha="right", fontsize=9, color=MUTED)
ax.plot([r["ai"] / 1e3 for r in dyn], [r["rescue"] for r in dyn], color=AQUA, linewidth=0, marker="^", markersize=9,
        markeredgecolor="white", markeredgewidth=1.5, label="Share switched with the load", zorder=3)
for r in dyn:
    m = re.fullmatch(r"d(\d+)x(\d+)l(\d+)", r["policy"])
    late = "" if m.group(3) == "0" else ", sees load 1 s late"
    ax.annotate(f"{m.group(1)}/{m.group(2)}%{late}", (r["ai"] / 1e3, r["rescue"]), textcoords="offset points",
                xytext=(-6, -16) if late else (8, -4), ha="right" if late else "left", fontsize=9, color=MUTED)
ax.plot([r["ai"] / 1e3 for r in ours], [r["rescue"] for r in ours], color=BLUE, linewidth=0, marker="s", markersize=10,
        markeredgecolor="white", markeredgewidth=1.5, label=os.environ.get("SCHEME_NAME", "Our Scheme"), zorder=4)
lowest = min(r["rescue"] for r in rows)
ax.set_ylim(lowest - 0.6, 100.2)
ax.set_xlim(0, max(r["ai"] for r in rows) / 1e3 * 1.18)
ax.set_xlabel("AI tokens served within the time limit (thousand/s)", fontsize=9, color=INK)
ax.set_ylabel("TBs rescued by NeuralRx (% of the run without AI)", fontsize=9, color=INK)
ax.set_title(sys.argv[3] if len(sys.argv) > 3 else "", fontsize=10, color=INK, loc="left")
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=8)
ax.legend(fontsize=9, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=150)
print(sys.argv[2])
