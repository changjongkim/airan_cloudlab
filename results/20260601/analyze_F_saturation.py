#!/usr/bin/env python3
"""F saturation matrix analyzer — L1 frame time stats from JSON.

각 condition에 대해 n=5 (or n=10 baseline) JSON 모아서:
- mean ± SD, p99 mean ± SD, max mean ± SD
- vs baseline (F_0_alone) Δ%
- 95% CI 오버랩 (significance proxy)
- 그룹별 (memory/compute/intensity/stacking) ranking
"""
import json, re, glob, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent / "F_saturation"
OUT = Path(__file__).parent / "analysis_F"
OUT.mkdir(parents=True, exist_ok=True)


def load_runs():
    """Group JSON files by condition (strip _runN)."""
    agg = defaultdict(list)
    for jf in sorted(ROOT.glob("realL1_*.json")):
        m = re.match(r"realL1_(.+?)_run\d+_\d{8}_\d{6}\.json", jf.name)
        if not m:
            continue
        cond = m.group(1)
        try:
            d = json.loads(jf.read_text())
            agg[cond].append(d)
        except json.JSONDecodeError:
            pass
    return agg


def cond_block(name):
    """Categorize condition by its block prefix."""
    if name.startswith("F_0"): return "baseline"
    if name.startswith("F_B_D2D"): return "B_D2D"
    if name.startswith("F_C_H2D"): return "C_H2D"
    if name.startswith("F_D_GEMM"): return "D_GEMM"
    if name.startswith("F_E_chanpred"): return "E_chanpred"
    if name.startswith("F_E_resnet"): return "E_resnet"
    if name.startswith("F_E_forecaster"): return "E_forecaster"
    if name.startswith("F_F_stack_chanpred"): return "F_stack_chanpred"
    if name.startswith("F_F_stack_resnet"): return "F_stack_resnet"
    if name.startswith("F_G_kitchen"): return "G_kitchen"
    return "other"


def stats(runs, key):
    vals = np.array([r[key] for r in runs if key in r])
    return float(vals.mean()), float(vals.std()), len(vals)


def main():
    agg = load_runs()
    print(f"loaded {len(agg)} conditions")
    base_cond = "F_0_alone"
    if base_cond not in agg:
        print(f"!!! no baseline {base_cond}")
        return
    bm_mean, bm_sd, bm_n = stats(agg[base_cond], "mean_ms")
    bp99_mean, bp99_sd, _ = stats(agg[base_cond], "p99_ms")
    bmax_mean, bmax_sd, _ = stats(agg[base_cond], "max_ms")
    print(f"baseline ({base_cond}, n={bm_n}): "
          f"mean={bm_mean:.2f}±{bm_sd:.2f}ms  p99={bp99_mean:.2f}±{bp99_sd:.2f}ms  max={bmax_mean:.2f}±{bmax_sd:.2f}ms\n")

    # Per-condition table with deltas
    rows = []
    for cond in sorted(agg.keys()):
        runs = agg[cond]
        m_mean, m_sd, n = stats(runs, "mean_ms")
        p99_mean, p99_sd, _ = stats(runs, "p99_ms")
        max_mean, max_sd, _ = stats(runs, "max_ms")
        # 95% CI on p99
        p99_se = p99_sd / max(1, np.sqrt(n))
        ci_lo = p99_mean - 1.96 * p99_se
        ci_hi = p99_mean + 1.96 * p99_se
        bp99_se = bp99_sd / max(1, np.sqrt(bm_n))
        b_lo = bp99_mean - 1.96 * bp99_se
        b_hi = bp99_mean + 1.96 * bp99_se
        sig = "SEPARATE" if (ci_lo > b_hi or ci_hi < b_lo) else "OVERLAP"
        rows.append({
            "condition": cond,
            "block": cond_block(cond),
            "n": n,
            "mean_ms": m_mean, "mean_sd": m_sd,
            "mean_delta_pct": 100*(m_mean-bm_mean)/bm_mean,
            "p99_ms": p99_mean, "p99_sd": p99_sd,
            "p99_ci_lo": ci_lo, "p99_ci_hi": ci_hi,
            "p99_delta_pct": 100*(p99_mean-bp99_mean)/bp99_mean,
            "max_ms": max_mean, "max_sd": max_sd,
            "max_delta_pct": 100*(max_mean-bmax_mean)/bmax_mean,
            "p99_significant": sig,
        })

    # Print by block
    print(f"{'condition':<35} {'block':<18} {'n':>3}  {'mean Δ':>8}  {'p99 Δ':>8}  {'max Δ':>8}  {'sig'}")
    print("-" * 105)
    for r in sorted(rows, key=lambda x: (x["block"], -x["p99_delta_pct"])):
        if r["condition"] == base_cond:
            print(f"{r['condition']:<35} {r['block']:<18} {r['n']:>3}  "
                  f"{r['mean_ms']:>5.1f}ms  {r['p99_ms']:>5.1f}ms  {r['max_ms']:>5.1f}ms  baseline")
        else:
            print(f"{r['condition']:<35} {r['block']:<18} {r['n']:>3}  "
                  f"{r['mean_delta_pct']:+>7.1f}%  {r['p99_delta_pct']:+>7.1f}%  {r['max_delta_pct']:+>7.1f}%  {r['p99_significant']}")

    # CSV out
    cols = ["condition", "block", "n", "mean_ms", "mean_sd", "mean_delta_pct",
            "p99_ms", "p99_sd", "p99_ci_lo", "p99_ci_hi", "p99_delta_pct",
            "max_ms", "max_sd", "max_delta_pct", "p99_significant"]
    with open(OUT / "F_summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\nWrote {OUT / 'F_summary.csv'}")

    # TOP conditions by p99 inflation
    print("\n=== TOP 10 p99 inflation ===")
    top = sorted([r for r in rows if r["condition"] != base_cond], key=lambda x: -x["p99_delta_pct"])[:10]
    for r in top:
        sig = "*" if r["p99_significant"] == "SEPARATE" else ""
        print(f"  {r['p99_delta_pct']:+>7.1f}%  {r['condition']:<40} {sig}")

    # By block: highest impact per block
    print("\n=== BLOCK SUMMARY (highest p99 inflation per block) ===")
    by_block = defaultdict(list)
    for r in rows:
        if r["condition"] == base_cond: continue
        by_block[r["block"]].append(r)
    for blk in sorted(by_block):
        sub = sorted(by_block[blk], key=lambda x: -x["p99_delta_pct"])
        worst = sub[0]
        mean_p99_delta = np.mean([r["p99_delta_pct"] for r in sub])
        print(f"  {blk:<20}  n_conds={len(sub):>2}  mean Δp99={mean_p99_delta:+.1f}%  worst={worst['p99_delta_pct']:+.1f}% ({worst['condition']})")


if __name__ == "__main__":
    main()
