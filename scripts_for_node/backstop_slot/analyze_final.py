#!/usr/bin/env python3
"""Aggregate the final slot-scale campaign (two seeds) and draw its figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1] if len(sys.argv) > 1 else "59103692"

POLICIES = [
    ("rescue_value", "backstop_corun", "Antiphase"),
    ("rescue_value", "static", "our NeuralRx rule + fixed GPU share"),
    ("parallel", "static", "always both receivers + fixed GPU share"),
    ("parallel_admit", "static", "both receivers at arrival, deadline drop + fixed GPU share"),
    ("off", "static", "no NeuralRx + fixed GPU share"),
    ("rescue_value", "none", "no AI"),
]


def metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    h, a = d["headline"], d["all"]
    ai = h.get("ai_total") or {}
    ttft = [v["ttft_ms"] for v in (d.get("ai") or {}).values() if v and v["ttft_ms"].get("n")]
    return {
        "tbs": h["tbs"], "late_tbs": h["late_tbs"],
        "decoded_on_time": h["decoded_on_time"],
        "rescues": h["nrx_rescues_on_time"], "nrx_runs": h["nrx_runs"],
        "nrx_late": h.get("nrx_late") or 0,
        "conv_p99_ms": a["conv_done_ms"]["p99"],
        "ai_tokens_per_s": ai.get("tokens_per_s", 0.0),
        "ai_slo_tokens_per_s": ai.get("tokens_within_slo_per_s", 0.0),
        "ai_completed": ai.get("completed", 0), "ai_arrived": ai.get("arrived", 0),
        "ttft_p50_ms": float(np.mean([t["p50"] for t in ttft])) if ttft else None,
        "ttft_p99_ms": float(max(t["p99"] for t in ttft)) if ttft else None,
    }


def main() -> None:
    rows = []
    for cells in (8, 16):
        for rate in (4, 8):
            for nrx, ai, label in POLICIES:
                runs = []
                for seed in (1, 2):
                    tag = f"f{seed}n" if ai == "none" else f"f{seed}r{rate}"
                    path = ROOT / "raw" / f"{tag}_c{cells}_{nrx}_{ai}_j{JOB}.json"
                    if path.is_file():
                        runs.append(metrics(path))
                if not runs:
                    continue
                mean = {k: (float(np.mean([r[k] for r in runs])) if runs[0][k] is not None else None)
                        for k in runs[0]}
                rows.append({"cells": cells, "ai_rate_per_gpu": rate, "nrx_policy": nrx,
                             "ai_policy": ai, "label": label, "seeds": len(runs),
                             "mean": mean, "per_seed": runs})
    out = ROOT / f"final_campaign_j{JOB}.json"
    out.write_text(json.dumps({
        "schema": "backstop-slot-final-v1",
        "setting": "4x A100-SXM4-80GB, DDDSU 30 kHz, 4.0 ms from data arrival (T0+4.5 ms), "
                   "50% weak rank-1 cells, strong cells rank 2; 4,000 UL periods per run; "
                   "AI = Qwen2.5-1.5B prefill, BurstGPT lengths, Poisson arrivals, SLO 200 ms",
        "rows": rows,
    }, indent=2))
    cols = ["late_tbs", "decoded_on_time", "rescues", "nrx_runs", "nrx_late", "conv_p99_ms",
            "ai_slo_tokens_per_s", "ai_tokens_per_s", "ttft_p50_ms", "ttft_p99_ms"]
    print("| cells | AI load | policy | " + " | ".join(cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for r in rows:
        vals = []
        for c in cols:
            v = r["mean"][c]
            vals.append("-" if v is None else (f"{100*v:.2f}%" if c == "decoded_on_time"
                                               else f"{v:.2f}" if c == "conv_p99_ms"
                                               else f"{v:.0f}"))
        print(f"| {r['cells']} | {r['ai_rate_per_gpu']} | {r['label']} | " + " | ".join(vals) + " |")

    colors = {"Antiphase": "tab:blue", "our NeuralRx rule + fixed GPU share": "tab:orange",
              "always both receivers + fixed GPU share": "tab:red",
              "both receivers at arrival, deadline drop + fixed GPU share": "tab:purple",
              "no NeuralRx + fixed GPU share": "tab:gray"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (cells, rate) in zip(axes, [(8, 4), (8, 8), (16, 4), (16, 8)]):
        for r in rows:
            if r["cells"] != cells or r["ai_rate_per_gpu"] != rate or r["label"] not in colors:
                continue
            for seed_run in r["per_seed"]:
                ax.scatter(seed_run["rescues"], seed_run["ai_slo_tokens_per_s"] / 1000,
                           color=colors[r["label"]], s=28, alpha=0.5)
            ax.scatter(r["mean"]["rescues"], r["mean"]["ai_slo_tokens_per_s"] / 1000,
                       color=colors[r["label"]], s=90, marker="*", label=r["label"])
        ax.set_title(f"{cells} cells, {rate} AI requests/s per GPU", fontsize=10)
        ax.set_xlabel("TBs rescued by NeuralRx in time")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(ROOT / f"final_tradeoff_j{JOB}.png", dpi=160)
    print(ROOT / f"final_tradeoff_j{JOB}.png")


if __name__ == "__main__":
    main()
