#!/usr/bin/env python3
"""Capacity region: the most AI a policy serves while keeping the radio targets.

For each (cells, rescue deadline) and policy, runs at several AI loads are compared with the
no-AI run of the same seed.  A load is compliant when, on every seed, L1 misses stay at or
below ``L1_TARGET`` of all TBs and NeuralRx rescues reach ``RESCUE_TARGET`` of no AI.  The
compliant capacity is the largest AI throughput (tokens within the SLO) over compliant loads.
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
JOB = sys.argv[1] if len(sys.argv) > 1 else "59124351"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "c"
REF_PREFIX = "c"          # no-AI references come from the capacity campaign
L1_TARGET, RESCUE_TARGET = 0.0005, 0.99
SETTINGS = ([(24, "6.5", (4, 8, 12, 16, 20)), (32, "11.5", (4, 8, 12, 16)), (16, "6.5", (12, 16, 20))]
            if PREFIX == "c" else [(24, "6.5", (12, 16, 20)), (32, "11.5", (12, 16, 20))])
POLICIES = [("Antiphase", "u"), ("Antiphase, 512-token units allowed during conventional", "w"),
            ("fixed 30% share", "s30"), ("fixed 50% share", "s50"),
            ("fixed 70% share", "s70"), ("fixed 100% share", "s100")]


def load(stem):
    path = ROOT / "raw" / f"{stem}_j{JOB}.json"
    if not path.is_file():
        return None
    h = json.loads(path.read_text())["headline"]
    ai = h.get("ai_total") or {}
    return {"tbs": h["tbs"], "late": h["late_tbs"], "rescues": h["nrx_rescues_on_time"],
            "nrx_late": h.get("nrx_late") or 0, "slo": ai.get("tokens_within_slo_per_s", 0.0),
            "served": ai.get("tokens_per_s", 0.0)}


def main() -> None:
    table, capacity = [], []
    for cells, d2, rates in SETTINGS:
        tag = d2.replace(".", "")
        refs = {}
        for s in (1, 2):
            # Reference = mean of the no-AI runs of this seed (a second run measures the noise).
            pair = [r for r in (load(f"{REF_PREFIX}{s}d{tag}n_c{cells}_rescue_value_none"),
                                load(f"{REF_PREFIX}{s}d{tag}m_c{cells}_rescue_value_none")) if r]
            if pair:
                refs[s] = {"rescues": float(np.mean([r["rescues"] for r in pair])),
                           "runs": [r["rescues"] for r in pair]}
        for label, key in POLICIES:
            kind = "backstop_units" if key in ("u", "w") else "static"
            best, best_mean = None, None
            for rate in rates:
                runs = []
                for s in (1, 2):
                    r = load(f"{PREFIX}{s}d{tag}r{rate}{key}_c{cells}_rescue_value_{kind}")
                    if r and refs.get(s):
                        r["rescue_ratio"] = r["rescues"] / max(1, refs[s]["rescues"])
                        r["late_fraction"] = r["late"] / r["tbs"]
                        runs.append(r)
                if len(runs) < 2:
                    continue
                ok = all(r["late_fraction"] <= L1_TARGET and r["rescue_ratio"] >= RESCUE_TARGET for r in runs)
                ok_mean = (all(r["late_fraction"] <= L1_TARGET for r in runs)
                           and float(np.mean([r["rescue_ratio"] for r in runs])) >= RESCUE_TARGET)
                row = {"cells": cells, "rescue_deadline_ms": float(d2), "policy": label, "ai_rate": rate,
                       "slo": float(np.mean([r["slo"] for r in runs])),
                       "rescue_ratio_min": float(min(r["rescue_ratio"] for r in runs)),
                       "rescue_ratio_mean": float(np.mean([r["rescue_ratio"] for r in runs])),
                       "late_fraction_max": float(max(r["late_fraction"] for r in runs)),
                       "nrx_late": float(np.mean([r["nrx_late"] for r in runs])), "compliant": ok,
                       "compliant_mean": ok_mean}
                table.append(row)
                if ok and (best is None or row["slo"] > best["slo"]):
                    best = row
                if ok_mean and (best_mean is None or row["slo"] > best_mean["slo"]):
                    best_mean = row
            capacity.append({"cells": cells, "rescue_deadline_ms": float(d2), "policy": label,
                             "compliant_ai_tokens_per_s": best["slo"] if best else 0.0,
                             "at_ai_rate": best["ai_rate"] if best else None,
                             "compliant_mean_ai_tokens_per_s": best_mean["slo"] if best_mean else 0.0,
                             "at_ai_rate_mean": best_mean["ai_rate"] if best_mean else None})
    (ROOT / f"capacity_region_{PREFIX}_j{JOB}.json").write_text(json.dumps(
        {"targets": {"l1_miss_fraction_max": L1_TARGET, "rescue_ratio_min": RESCUE_TARGET},
         "rows": table, "capacity": capacity}, indent=2))
    print("| cells | rescue deadline | policy | AI load | SLO tokens/s | rescues min (mean) | L1 late max | NRx late | compliant |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in table:
        print(f"| {r['cells']} | {r['rescue_deadline_ms']} | {r['policy']} | {r['ai_rate']} | {r['slo']:.0f} | "
              f"{100*r['rescue_ratio_min']:.1f}% ({100*r['rescue_ratio_mean']:.1f}%) | {100*r['late_fraction_max']:.3f}% | "
              f"{r['nrx_late']:.0f} | {'yes' if r['compliant'] else 'no'} |")
    print("\nno-AI reference runs per seed:", {f"{c}/{d}": None for c, d, _ in SETTINGS})
    print("\ncompliant AI capacity (tokens within SLO per second): every seed >= 99% | seed mean >= 99%")
    for c in capacity:
        print(f"  {c['cells']} cells, {c['rescue_deadline_ms']} ms, {c['policy']}: {c['compliant_ai_tokens_per_s']:.0f} "
              f"(load {c['at_ai_rate']}) | {c['compliant_mean_ai_tokens_per_s']:.0f} (load {c['at_ai_rate_mean']})")

    fig, axes = plt.subplots(1, len(SETTINGS), figsize=(5 * len(SETTINGS), 3.8))
    colors = {"Antiphase": "tab:blue", "fixed 30% share": "tab:olive", "fixed 50% share": "tab:orange",
              "fixed 70% share": "tab:red", "fixed 100% share": "tab:brown",
              "Antiphase, 512-token units allowed during conventional": "tab:cyan"}
    for ax, (cells, d2, _) in zip(axes, SETTINGS):
        for label, _ in POLICIES:
            pts = [r for r in table if r["cells"] == cells and r["rescue_deadline_ms"] == float(d2) and r["policy"] == label]
            if not pts:
                continue
            ax.plot([p["ai_rate"] for p in pts], [p["slo"] / 1000 for p in pts], color=colors[label], label=label)
            for p in pts:
                ax.scatter(p["ai_rate"], p["slo"] / 1000, color=colors[label], marker="o" if p["compliant"] else "x", s=50)
        ax.set_title(f"{cells} cells, rescue deadline {d2} ms", fontsize=10)
        ax.set_xlabel("offered AI requests per second per GPU")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AI tokens within 200 ms\n(thousand per second)")
    axes[0].legend(fontsize=8)
    fig.suptitle("o: radio targets kept on every seed, x: missed", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / f"capacity_region_{PREFIX}_j{JOB}.png", dpi=160)
    print(ROOT / f"capacity_region_{PREFIX}_j{JOB}.png")


if __name__ == "__main__":
    main()
