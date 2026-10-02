#!/usr/bin/env python3
"""Measured timeline of the whole server: neural receiver runs and AI pieces on every GPU.

usage: plot_reserve_timeline.py RUN FIRST_PERIOD COUNT OUT     (FIRST_PERIOD = auto picks a window)
Shows the free-neural-receiver rule of the scheme: AI keeps running next to a neural receiver
while it is the only one running, and stops on the GPUs with a neural receiver once a second
one starts.  The top panel counts the free neural receivers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
GRAY, AQUA, BLUE, INK, MUTED, GRID, RED = "#8a939d", "#1baf7a", "#2a78d6", "#1f2328", "#59636e", "#d9dee4", "#c0392b"


def load(run: str):
    work = RAW / f"{run}_work"
    config = json.loads((RAW / f"{run}.json").read_text())["config"]
    gpus = int(config["num_gpus"])
    period_ns = int(float(config["period_ms"]) * 1e6)
    first = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = first[2] - first[0] * period_ns
    nrx = {g: [(r[2], r[3]) for r in json.loads((work / f"lane{g}.json").read_text())["records"]] for g in range(gpus)}
    ai = {g: [(p[0], p[1]) for p in json.loads((work / f"ai{g}.json").read_text())["pieces"]] for g in range(gpus)}
    return config, gpus, period_ns, epoch, nrx, ai


def pick(nrx, ai, gpus, period_ns, epoch, count, periods) -> int:
    """A window with one neural receiver running alone and then a second one, in which the AI
    pieces that were granted before the second one started end soon after it starts."""
    step = 100_000
    span = count * period_ns
    best, best_score = 40, -1.0
    for first in range(40, periods - count, 2):
        origin = epoch + first * period_ns
        grid = np.arange(origin, origin + span, step)
        busy = np.zeros(len(grid), dtype=int)
        on = np.zeros((gpus, len(grid)), dtype=bool)
        running = np.zeros((gpus, len(grid)), dtype=bool)
        for g in range(gpus):
            for start, done in nrx[g]:
                if done > origin and start < origin + span:
                    on[g] |= (grid >= start) & (grid < done)
            for start, done in ai[g]:
                if done > origin and start < origin + span:
                    running[g] |= (grid >= start) & (grid < done)
        busy = on.sum(axis=0)
        if busy.max() != 2:
            continue
        one, two = (busy == 1).mean(), (busy == 2).mean()
        first_two = int(np.argmax(busy == 2))
        scarce = on & (busy >= 2)[None, :]
        leak = (running & scarce).sum() / max(1, scarce.sum())
        alone = on & (busy == 1)[None, :]
        beside = (running & alone).sum() / max(1, alone.sum())
        score = min(one, 2 * two) + 0.3 * (busy[:first_two] == 1).mean() - leak + 0.3 * beside \
            if first_two > len(grid) // 5 else -1
        if score > best_score:
            best, best_score = first, score
    return best


def main() -> None:
    run, count, out = sys.argv[1], int(sys.argv[3]), sys.argv[4]
    name = os.environ.get("SCHEME_NAME", "Our Scheme")
    config, gpus, period_ns, epoch, nrx, ai = load(run)
    first = pick(nrx, ai, gpus, period_ns, epoch, count, int(config["periods"])) if sys.argv[2] == "auto" else int(sys.argv[2])
    origin, span = epoch + first * period_ns, count * period_ns
    ms = lambda t: (t - origin) / 1e6
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.3), sharex=True, gridspec_kw={"height_ratios": [1, 3.4]})
    grid = np.arange(origin, origin + span, 50_000)
    busy = np.zeros(len(grid), dtype=int)
    for g in range(gpus):
        for start, done in nrx[g]:
            busy += (grid >= start) & (grid < done)
    axes[0].step([ms(t) for t in grid], gpus - busy, where="post", color=INK, linewidth=1.4)
    axes[0].axhline(gpus - 1, color=RED, linewidth=1, linestyle=(0, (3, 2)))
    axes[0].annotate("AI next to a neural receiver needs three free", (span / 1e6, gpus - 1), textcoords="offset points",
                     xytext=(-2, -10), ha="right", fontsize=7.5, color=RED)
    axes[0].set_ylim(gpus - 3.4, gpus + 0.9)
    axes[0].set_yticks(range(gpus - 3, gpus + 1))
    axes[0].set_ylabel("Free neural\nreceivers", fontsize=8.5, color=INK)
    for g in range(gpus):
        y = gpus - 1 - g
        for start, done in nrx[g]:
            if done > origin and start < origin + span:
                axes[1].barh(y + 0.2, ms(done) - ms(start), left=ms(start), height=0.3, color=AQUA, linewidth=0)
                if ms(start) > 0 and ms(done) < span / 1e6:
                    axes[1].annotate(f"{ms(done) - ms(start):.1f} ms", (ms(start) + 0.15, y + 0.2), va="center", fontsize=7.5,
                                     color="white", fontweight="bold")
        for start, done in ai[g]:
            if done > origin and start < origin + span:
                axes[1].barh(y - 0.2, ms(done) - ms(start), left=ms(start), height=0.3, color=BLUE, linewidth=0.4,
                             edgecolor="white")
    axes[1].set_yticks(range(gpus))
    axes[1].set_yticklabels([f"GPU {gpus - 1 - y}" for y in range(gpus)], fontsize=8.5, color=INK)
    axes[1].set_ylim(-0.6, gpus - 0.4)
    axes[1].set_xlabel("Time (ms); dotted lines: uplink slots arrive", fontsize=8.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    axes[1].legend(handles, ["Neural receiver run (run time inside)", "AI pieces"], fontsize=8, frameon=False, ncol=2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for ax in axes:
        for k in range(count + 1):
            ax.axvline(k * period_ns / 1e6, color=MUTED, linestyle=(0, (1, 2)), linewidth=0.8, zorder=0)
        ax.set_xlim(0, span / 1e6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_title(f"{name}: measured run, uplink slots {first}-{first + count - 1}", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(out, "first period", first)


if __name__ == "__main__":
    main()
