#!/usr/bin/env python3
"""Rescue-deadline sweep (16 cells, AI 12 req/s, 2 seeds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1] if len(sys.argv) > 1 else "59115134"
D2 = ["4.0", "5.0", "6.5", "11.5", "21.5", "41.5"]
POLICIES = [
    ("Our Scheme", "x{s}d{t}_c16_rescue_value_backstop_corun"),
    ("Our Scheme, 3 ms AI pieces", "x{s}d{t}p3_c16_rescue_value_backstop_corun"),
    ("fixed 50% GPU share", "x{s}d{t}_c16_rescue_value_static"),
    ("fixed 70% GPU share", "x{s}d{t}s70_c16_rescue_value_static"),
]


def load(stem):
    path = ROOT / "raw" / f"{stem}_j{JOB}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    h = d["headline"]
    ai = h.get("ai_total") or {}
    return {"rescues": h["nrx_rescues_on_time"], "nrx_late": h.get("nrx_late") or 0,
            "late_tbs": h["late_tbs"], "retx": h.get("retransmissions"),
            "slo": ai.get("tokens_within_slo_per_s", 0.0), "nrx_runs": h["nrx_runs"]}


def main() -> None:
    rows = []
    ref = [load(f"x{s}n_c16_rescue_value_none") for s in (1, 2)]
    ref = [r for r in ref if r]
    ref_rescues = float(np.mean([r["rescues"] for r in ref])) if ref else None
    for label, pattern in POLICIES:
        for d2 in D2:
            runs = [load(pattern.format(s=s, t=d2.replace(".", ""))) for s in (1, 2)]
            runs = [r for r in runs if r]
            if not runs:
                continue
            m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            rows.append({"policy": label, "rescue_deadline_ms": float(d2), "seeds": len(runs), **m})
    (ROOT / f"d2_sweep_j{JOB}.json").write_text(json.dumps(
        {"no_ai_rescues_at_41_5ms": ref_rescues, "rows": rows}, indent=2))
    print(f"no AI, rescue deadline 41.5 ms: rescues {ref_rescues}")
    print("| policy | rescue deadline | rescues | NRx late | L1 late | retransmissions | SLO tokens/s |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['policy']} | {r['rescue_deadline_ms']} | {r['rescues']:.0f} | {r['nrx_late']:.0f} | "
              f"{r['late_tbs']:.0f} | {r['retx']:.0f} | {r['slo']:.0f} |")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"Our Scheme": "tab:blue", "Our Scheme, 3 ms AI pieces": "tab:cyan",
              "fixed 50% GPU share": "tab:orange", "fixed 70% GPU share": "tab:red"}
    for label, _ in POLICIES:
        pts = sorted((r["rescue_deadline_ms"], r["rescues"], r["slo"]) for r in rows if r["policy"] == label)
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=colors[label], label=label)
        axes[1].plot([p[0] for p in pts], [p[2] / 1000 for p in pts], marker="o", color=colors[label], label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([4, 5, 6.5, 11.5, 21.5, 41.5])
        ax.set_xticklabels(["4", "5", "6.5", "11.5", "21.5", "41.5"])
        ax.set_xlabel("NeuralRx rescue deadline after data arrival (ms)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("TBs rescued by NeuralRx in time")
    axes[1].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
    axes[0].legend(fontsize=8)
    fig.suptitle("16 cells, AI 12 requests/s per GPU", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / f"d2_sweep_j{JOB}.png", dpi=160)
    print(ROOT / f"d2_sweep_j{JOB}.png")


if __name__ == "__main__":
    main()
