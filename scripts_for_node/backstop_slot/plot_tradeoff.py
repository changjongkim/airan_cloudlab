#!/usr/bin/env python3
"""Rescued TBs versus AI tokens within SLO for each policy point (16 cells)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
JOB = sys.argv[1] if len(sys.argv) > 1 else "59103692"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW.parent / "tradeoff_c16.png"


def load(tag: str, kind: str) -> dict | None:
    path = RAW / f"{tag}_c16_rescue_{kind}_j{JOB}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    h = d["headline"]
    ai = h.get("ai_total") or {}
    return {"rescues": h["nrx_rescues_on_time"], "slo": ai.get("tokens_within_slo_per_s", 0.0),
            "late": h["late_tbs"], "nrx_late": h.get("nrx_late")}


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, rate in zip(axes, (4, 8)):
    series = [
        ("fixed MPS share", "static", [20, 30, 50, 70], "s", "tab:gray", "o"),
        ("Antiphase (co-run bound p99)", "backstop_corun", [50, 70, 100], "c", "tab:blue", "s"),
        ("Antiphase (co-run bound p90)", "backstop_corun", [50, 70], "q", "tab:cyan", "^"),
    ]
    for label, kind, shares, key, color, marker in series:
        points = []
        for share in shares:
            p = load(f"m4r{rate}{key}{share}", kind)
            if p:
                points.append((p["rescues"], p["slo"] / 1000, share))
        if not points:
            continue
        ax.plot([p[0] for p in points], [p[1] for p in points], marker=marker,
                color=color, label=label)
        for x, y, share in points:
            ax.annotate(f"{share}%", (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=color)
    ax.set_title(f"16 cells, AI load {rate} requests/s per GPU")
    ax.set_xlabel("TBs rescued by NeuralRx on time")
    ax.set_ylabel("AI tokens within SLO (thousand/s)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
