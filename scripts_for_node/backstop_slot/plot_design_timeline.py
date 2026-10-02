#!/usr/bin/env python3
"""Measured timeline of one GPU around one recovery, for the scheme and a fixed share.

usage: plot_design_timeline.py OURS_RUN FIXED_RUN GPU FIRST_PERIOD COUNT OUT [fixed-share label]
Both runs use the same seed, so the same TB fails and goes to the same neural receiver.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
GRAY, AQUA, BLUE, INK, MUTED, GRID, RED = "#8a939d", "#1baf7a", "#2a78d6", "#1f2328", "#59636e", "#d9dee4", "#c0392b"


def load(run: str, gpu: int, first: int, count: int):
    work = RAW / f"{run}_work"
    config = json.loads((RAW / f"{run}.json").read_text())["config"]
    period_ns = int(float(config["period_ms"]) * 1e6)
    cells = [c for c in config["cells"] if c["gpu"] == gpu]
    epoch = None
    conv, release = [], {}
    for cell in config["cells"]:
        for r in json.loads((work / f"conv{cell['cell']}.json").read_text())["records"]:
            epoch = r[2] - r[0] * period_ns if epoch is None else epoch
            release[(cell["cell"], r[0])] = r[2]
            if cell["gpu"] == gpu and first <= r[0] < first + count:
                conv.append((cells.index(cell), r[3], r[4]))
    origin = epoch + first * period_ns
    end = origin + count * period_ns
    nrx = []
    for r in json.loads((work / f"lane{gpu}.json").read_text())["records"]:
        if r[3] > origin and r[2] < end:
            nrx.append((r[2], r[3], release[(r[0], r[1])]))
    pieces = []
    for path in work.glob(f"ai{gpu}*.json"):
        pieces += [(p[0], p[1]) for p in json.loads(path.read_text())["pieces"] if p[1] > origin and p[0] < end]
    return config, conv, nrx, pieces, origin, period_ns


def draw(ax, run: str, title: str, gpu: int, first: int, count: int) -> None:
    config, conv, nrx, pieces, origin, period_ns = load(run, gpu, first, count)
    ms = lambda t: (t - origin) / 1e6
    span = count * period_ns / 1e6
    for index, start, done in conv:
        ax.barh(2.0 + 0.2 * (1.5 - index), ms(done) - ms(start), left=ms(start), height=0.17, color=GRAY, linewidth=0)
    rescue = float(config["rescue_deadline_ms"])
    for start, done, arrival in nrx:
        ax.barh(1.0, ms(done) - ms(start), left=ms(start), height=0.5, color=AQUA, linewidth=0)
        due = ms(arrival) + rescue
        if 0 <= due <= span:
            ax.plot([due, due], [0.62, 1.38], color=RED, linewidth=1.6)
            ax.annotate("recovery deadline", (due, 1.38), textcoords="offset points", xytext=(3, 2), fontsize=7.5, color=RED)
        if 0 <= ms(arrival) <= span:
            ax.plot([ms(arrival)], [1.42], marker="v", color=INK, markersize=5)
            ax.annotate("failed TB arrived", (ms(arrival), 1.42), textcoords="offset points", xytext=(5, 0), fontsize=7.5,
                        color=INK, va="center")
        ax.annotate(f"{ms(done) - ms(start):.1f} ms", (ms(start) + 0.15, 1.0), va="center", fontsize=8, color="white",
                    fontweight="bold")
    for start, done in pieces:
        ax.barh(0.0, ms(done) - ms(start), left=ms(start), height=0.5, color=BLUE, linewidth=0.4, edgecolor="white")
    for k in range(count + 1):
        ax.axvline(k * period_ns / 1e6, color=MUTED, linestyle=(0, (1, 2)), linewidth=0.8, zorder=0)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["AI pieces", "NeuralRx\n(recovery)", "Conventional\nreceivers (4 cells)"], fontsize=8.5, color=INK)
    ax.set_xlim(0, span)
    ax.set_ylim(-0.45, 2.6)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    ours, fixed = sys.argv[1], sys.argv[2]
    gpu, first, count = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    name = os.environ.get("SCHEME_NAME", "Our Scheme")
    other = sys.argv[7] if len(sys.argv) > 7 else "Fixed GPU share"
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), sharex=True)
    draw(axes[0], ours, f"(a) {name}: AI stops while the neural receiver runs", gpu, first, count)
    draw(axes[1], fixed, f"(b) {other}: AI runs next to the neural receiver", gpu, first, count)
    axes[1].set_xlabel("Time (ms); dotted lines: uplink slots arrive", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(sys.argv[6], dpi=200)
    print(sys.argv[6])


if __name__ == "__main__":
    main()
