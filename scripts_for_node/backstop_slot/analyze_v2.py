#!/usr/bin/env python3
"""Aggregate the scheme-v2 campaign (rescue deadline 6.5 ms, AI admission)."""

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

POLICIES = [
    ("Antiphase", "v{s}r{r}_c{c}_rescue_value_backstop_corun"),
    ("Antiphase, rescue by the L1 deadline", "v{s}r{r}d40_c{c}_rescue_value_backstop_corun"),
    ("fixed 50% GPU share", "v{s}r{r}_c{c}_rescue_value_static"),
    ("fixed 30% GPU share", "v{s}r{r}s30_c{c}_rescue_value_static"),
    ("AI only on idle GPU", "v{s}r{r}_c{c}_rescue_value_backstop"),
    ("both receivers at arrival + fixed 50%", "v{s}r{r}_c{c}_parallel_admit_static"),
    ("no NeuralRx + fixed 50%", "v{s}r{r}_c{c}_off_static"),
]


def metrics(path: Path) -> dict:
    d = json.loads(path.read_text())
    h, a = d["headline"], d["all"]
    ai = h.get("ai_total") or {}
    per_gpu = [v for v in (d.get("ai") or {}).values() if v]
    rejected = sum(v.get("rejected", 0) for v in per_gpu)
    ttft = [v["ttft_ms"] for v in per_gpu if v["ttft_ms"].get("n")]
    return {
        "tbs": h["tbs"], "late_tbs": h["late_tbs"],
        "decoded_final": h.get("decoded_final"), "retransmissions": h.get("retransmissions"),
        "rescues": h["nrx_rescues_on_time"], "nrx_runs": h["nrx_runs"],
        "nrx_late": h.get("nrx_late") or 0, "conv_p99_ms": a["conv_done_ms"]["p99"],
        "ai_slo_tokens_per_s": ai.get("tokens_within_slo_per_s", 0.0),
        "ai_tokens_per_s": ai.get("tokens_per_s", 0.0),
        "ai_rejected_fraction": rejected / ai["arrived"] if ai.get("arrived") else 0.0,
        "ttft_p50_ms": float(np.mean([t["p50"] for t in ttft])) if ttft else None,
        "ttft_p99_ms": float(max(t["p99"] for t in ttft)) if ttft else None,
    }


def mean(stems):
    runs = [metrics(ROOT / "raw" / f"{s}_j{JOB}.json") for s in stems
            if (ROOT / "raw" / f"{s}_j{JOB}.json").is_file()]
    if not runs:
        return None
    out = {k: (float(np.mean([r[k] for r in runs])) if runs[0][k] is not None else None) for k in runs[0]}
    out["seeds"] = len(runs)
    return out


def main() -> None:
    rows = []
    for cells in (16, 24, 32):
        ref = mean([f"v{s}n_c{cells}_rescue_value_none" for s in (1, 2)])
        for rate in (8, 12):
            for label, pattern in POLICIES:
                m = mean([pattern.format(s=s, r=rate, c=cells) for s in (1, 2)])
                if not m:
                    continue
                m["rescues_vs_no_ai"] = m["rescues"] / ref["rescues"] if ref and ref["rescues"] else None
                m["late_tbs_fraction"] = m["late_tbs"] / m["tbs"]
                rows.append({"cells": cells, "ai_rate": rate, "policy": label, **m})
            if ref:
                rows.append({"cells": cells, "ai_rate": rate, "policy": "no AI", **ref,
                             "rescues_vs_no_ai": 1.0, "late_tbs_fraction": ref["late_tbs"] / ref["tbs"]})
    (ROOT / f"v2_campaign_j{JOB}.json").write_text(json.dumps({
        "schema": "backstop-slot-v2-campaign",
        "setting": "L1 deadline 4.0 ms from data (T0+4.5 ms); NeuralRx rescue deadline 6.5 ms "
                   "(retransmission-grant decision); AI admission for every AI policy; lanes per GPU "
                   "1 (16 cells) or 2 (24/32 cells); 4,000 UL periods, 2 seeds",
        "rows": rows}, indent=2))
    print("| cells | AI | policy | L1 late | decoded | retx | rescues (vs no AI) | NRx late | SLO tokens/s | rejected | TTFT p50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cells']} | {r['ai_rate']} | {r['policy']} | {r['late_tbs']:.0f} ({100*r['late_tbs_fraction']:.3f}%) | "
              f"{100*r['decoded_final']:.2f}% | {r['retransmissions']:.0f} | {r['rescues']:.0f} ({100*r['rescues_vs_no_ai']:.0f}%) | "
              f"{r['nrx_late']:.0f} | {r['ai_slo_tokens_per_s']:.0f} | {100*r['ai_rejected_fraction']:.0f}% | "
              f"{(r['ttft_p50_ms'] or 0):.0f} |")

    labels = ["Antiphase", "fixed 30% GPU share", "fixed 50% GPU share", "AI only on idle GPU",
              "both receivers at arrival + fixed 50%"]
    colors = ["tab:blue", "tab:olive", "tab:orange", "tab:gray", "tab:purple"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    for col, rate in enumerate((8, 12)):
        for i, (label, color) in enumerate(zip(labels, colors)):
            xs, rescue, slo = [], [], []
            for cells in (16, 24, 32):
                r = next((r for r in rows if r["cells"] == cells and r["ai_rate"] == rate and r["policy"] == label), None)
                if r:
                    xs.append(cells)
                    rescue.append(100 * r["rescues_vs_no_ai"])
                    slo.append(r["ai_slo_tokens_per_s"] / 1000)
            offset = (i - 2) * 1.2
            axes[0][col].bar([x + offset for x in xs], rescue, width=1.2, color=color, label=label)
            axes[1][col].bar([x + offset for x in xs], slo, width=1.2, color=color, label=label)
        axes[0][col].set_title(f"AI load {rate} requests/s per GPU", fontsize=10)
        axes[0][col].set_ylabel("TBs rescued by NeuralRx\n(% of no AI)")
        axes[0][col].set_ylim(0, 110)
        axes[1][col].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
        for ax in (axes[0][col], axes[1][col]):
            ax.set_xticks([16, 24, 32])
            ax.set_xlabel("cells")
            ax.grid(axis="y", alpha=0.3)
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT / f"v2_campaign_j{JOB}.png", dpi=160)
    print(ROOT / f"v2_campaign_j{JOB}.png")


if __name__ == "__main__":
    main()
