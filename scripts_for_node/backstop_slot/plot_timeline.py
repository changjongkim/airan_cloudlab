#!/usr/bin/env python3
"""Measured timeline of one GPU over a few uplink periods, for two policies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def load(run: str, gpu: int, first: int, count: int):
    work = RAW / f"{run}_work"
    config = json.loads((RAW / f"{run}.json").read_text())["config"]
    cells = [c for c in config["cells"] if c["gpu"] == gpu]
    epoch = None
    rows = []
    for cell in cells:
        conv = json.loads((work / f"conv{cell['cell']}.json").read_text())
        for r in conv["records"]:
            if r[0] == 0:
                epoch = r[2]
            if first <= r[0] < first + count:
                rows.append(("conventional", r[3], r[4], cell["profile"], bool(r[5])))
    period_ns = int(float(config["period_ms"]) * 1e6)
    lane = work / f"lane{gpu}.json"
    if lane.is_file():
        for r in json.loads(lane.read_text())["records"]:
            if first <= r[1] < first + count:
                rows.append(("NeuralRx", r[2], r[3], f"cell {r[0]}", bool(r[4])))
    ai = json.loads((work / f"ai{gpu}.json").read_text())
    lo = epoch + first * period_ns
    hi = epoch + (first + count) * period_ns + 2_000_000
    for p in ai["pieces"]:
        if lo <= p[0] <= hi:
            rows.append(("AI", p[0], p[1], f"{p[2]} units", True))
    return rows, lo, period_ns, count, len(cells)


def draw(ax, run: str, title: str, gpu: int, first: int, count: int) -> None:
    rows, origin, period_ns, count, n_cells = load(run, gpu, first, count)
    lanes = {"conventional": 2, "NeuralRx": 1, "AI": 0}
    colors = {"conventional": "tab:gray", "NeuralRx": "tab:green", "AI": "tab:blue"}
    conv_slots = {}
    for kind, start, end, label, ok in sorted(rows, key=lambda r: r[1]):
        y = lanes[kind]
        if kind == "conventional":
            slot = conv_slots.setdefault(start // 1_000_000, len(conv_slots) % 4)
            y = 2 + 0.2 * (conv_slots[start // 1_000_000] - 1.5)
        ax.barh(y, (end - start) / 1e6, left=(start - origin) / 1e6, height=0.18 if kind == "conventional" else 0.6,
                color=colors[kind], alpha=0.85 if ok or kind != "NeuralRx" else 0.45,
                edgecolor="black", linewidth=0.3)
    for k in range(count + 1):
        ax.axvline(k * period_ns / 1e6, color="black", linestyle=":", linewidth=0.8)
        ax.axvline(k * period_ns / 1e6 + 4.0, color="tab:red", linestyle="--", linewidth=0.6)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["AI (Qwen units)", "NeuralRx lane", f"conventional\n({n_cells} cells)"])
    ax.set_xlim(0, count * period_ns / 1e6 + 1.5)
    ax.set_title(title, fontsize=10)


def main() -> None:
    ours, fixed = sys.argv[1], sys.argv[2]
    gpu, first, count = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    out = Path(sys.argv[6])
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True)
    titles = sys.argv[7:9] if len(sys.argv) >= 9 else ["Our Scheme", "same NeuralRx rule + fixed GPU share"]
    draw(axes[0], ours, titles[0], gpu, first, count)
    draw(axes[1], fixed, titles[1], gpu, first, count)
    axes[1].set_xlabel("time from uplink data arrival (ms); dotted: data arrival, dashed red: L1 deadline")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
