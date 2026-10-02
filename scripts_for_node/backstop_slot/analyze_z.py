#!/usr/bin/env python3
"""Z campaign: AI throughput under the same radio targets.

Radio targets per (cells, rescue deadline): L1 misses at most ``L1_TARGET`` of all TBs, and
NeuralRx rescues at least ``RESCUE_TARGET`` of the no-AI run with the same seed.  The table
reports every policy; the summary picks, per setting, the policy with the most AI tokens
within the SLO among those that meet both targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1] if len(sys.argv) > 1 else "59120865"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "z"
SETTINGS = ([(int(c), d) for c, d in (x.split(":") for x in sys.argv[3].split(","))]
            if len(sys.argv) > 3 else [(16, "6.5"), (24, "6.5"), (32, "6.5"), (16, "5.0")])
L1_TARGET = 0.0005          # 0.05% of TBs
RESCUE_TARGET = 0.99

POLICIES = [
    ("Our Scheme", "{p}{s}d{t}u_c{c}_rescue_value_backstop_units"),
    ("Our Scheme, no AI during conventional", "{p}{s}d{t}v_c{c}_rescue_value_backstop_units"),
    ("fixed 30% share", "{p}{s}d{t}s30_c{c}_rescue_value_static"),
    ("fixed 50% share", "{p}{s}d{t}s50_c{c}_rescue_value_static"),
    ("fixed 70% share", "{p}{s}d{t}s70_c{c}_rescue_value_static"),
    ("fixed 100% share", "{p}{s}d{t}s100_c{c}_rescue_value_static"),
    ("both receivers at arrival + fixed 50%", "{p}{s}d{t}s50_c{c}_parallel_admit_static"),
]


def load(stem: str) -> dict | None:
    path = ROOT / "raw" / f"{stem}_j{JOB}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    h = d["headline"]
    ai = h.get("ai_total") or {}
    per = [v for v in (d.get("ai") or {}).values() if v]
    return {
        "tbs": h["tbs"], "late": h["late_tbs"], "rescues": h["nrx_rescues_on_time"],
        "nrx_late": h.get("nrx_late") or 0, "nrx_runs": h["nrx_runs"],
        "false": h.get("false_crc_pass") or 0, "retx": h.get("retransmissions"),
        "decoded": h.get("decoded_final"),
        "slo": ai.get("tokens_within_slo_per_s", 0.0), "served": ai.get("tokens_per_s", 0.0),
        "rejected": (sum(v.get("rejected", 0) for v in per) / ai["arrived"]) if ai.get("arrived") else 0.0,
        "ttft_p50": float(np.mean([v["ttft_ms"]["p50"] for v in per if v["ttft_ms"].get("n")])) if per else None,
    }


def main() -> None:
    rows, best = [], []
    for cells, d2 in SETTINGS:
        tag = d2.replace(".", "")
        refs = {s: load(f"{PREFIX}{s}d{tag}n_c{cells}_rescue_value_none") for s in (1, 2)}
        for label, pattern in POLICIES:
            runs = []
            for s in (1, 2):
                r = load(pattern.format(p=PREFIX, s=s, t=tag, c=cells))
                if r and refs.get(s):
                    r["rescue_ratio"] = r["rescues"] / max(1, refs[s]["rescues"])
                    r["late_fraction"] = r["late"] / r["tbs"]
                    runs.append(r)
            if not runs:
                continue
            m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if runs[0][k] is not None}
            m["meets_targets"] = all(r["late_fraction"] <= L1_TARGET and r["rescue_ratio"] >= RESCUE_TARGET
                                     for r in runs)
            rows.append({"cells": cells, "rescue_deadline_ms": float(d2), "policy": label,
                         "seeds": len(runs), **m})
        ref = [r for r in refs.values() if r]
        if ref:
            rows.append({"cells": cells, "rescue_deadline_ms": float(d2), "policy": "no AI", "seeds": len(ref),
                         **{k: float(np.mean([r[k] for r in ref])) for k in ref[0] if ref[0][k] is not None},
                         "rescue_ratio": 1.0, "late_fraction": float(np.mean([r["late"] / r["tbs"] for r in ref])),
                         "meets_targets": True})
        ok = [r for r in rows if r["cells"] == cells and r["rescue_deadline_ms"] == float(d2)
              and r["policy"] != "no AI" and r["meets_targets"]]
        if ok:
            top = max(ok, key=lambda r: r["slo"])
            best.append({"cells": cells, "rescue_deadline_ms": float(d2), "policy": top["policy"],
                         "ai_slo_tokens_per_s": top["slo"]})
    (ROOT / f"{PREFIX}_campaign_j{JOB}.json").write_text(json.dumps({
        "targets": {"l1_miss_fraction_max": L1_TARGET, "rescue_ratio_min": RESCUE_TARGET},
        "rows": rows, "best_meeting_targets": best}, indent=2))
    print("| cells | rescue deadline | policy | L1 late | rescues (vs no AI) | NRx late | SLO tokens/s | rejected | meets targets |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['cells']} | {r['rescue_deadline_ms']} | {r['policy']} | {r['late']:.0f} ({100*r['late_fraction']:.3f}%) | "
              f"{r['rescues']:.0f} ({100*r['rescue_ratio']:.1f}%) | {r['nrx_late']:.0f} | {r['slo']:.0f} | "
              f"{100*r.get('rejected', 0):.0f}% | {'yes' if r['meets_targets'] else 'no'} |")
    print("\nbest AI throughput that meets both radio targets:")
    for b in best:
        print(f"  {b['cells']} cells, rescue deadline {b['rescue_deadline_ms']} ms: {b['policy']} ({b['ai_slo_tokens_per_s']:.0f} tokens/s)")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    colors = {"Our Scheme": "tab:blue", "Our Scheme, no AI during conventional": "tab:cyan",
              "fixed 30% share": "tab:olive", "fixed 50% share": "tab:orange", "fixed 70% share": "tab:red",
              "fixed 100% share": "tab:brown", "both receivers at arrival + fixed 50%": "tab:purple"}
    for ax, (cells, d2) in zip(axes, [(c, float(d)) for c, d in SETTINGS]):
        for r in rows:
            if r["cells"] != cells or r["rescue_deadline_ms"] != d2 or r["policy"] == "no AI":
                continue
            ax.scatter(100 * r["rescue_ratio"], r["slo"] / 1000, s=70, color=colors[r["policy"]],
                       marker="o" if r["meets_targets"] else "x", label=r["policy"])
        ax.axvline(100 * RESCUE_TARGET, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{cells} cells, rescue deadline {d2} ms", fontsize=10)
        ax.set_xlabel("TBs rescued by NeuralRx (% of no AI)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
    handles, labels = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(ROOT / f"{PREFIX}_campaign_j{JOB}.png", dpi=160)
    print(ROOT / f"{PREFIX}_campaign_j{JOB}.png")


if __name__ == "__main__":
    main()
