#!/usr/bin/env python3
"""G NeuralRx coloc analyzer — vs F baseline + per-coloc effect."""
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent / "G_coloc"
OUT = Path(__file__).parent / "analysis_G"
OUT.mkdir(parents=True, exist_ok=True)


def load_runs():
    agg = defaultdict(list)
    for jf in sorted(ROOT.glob("realL1_*.json")):
        m = re.match(r"realL1_(.+?)_run\d+_\d{8}_\d{6}\.json", jf.name)
        if not m: continue
        cond = m.group(1)
        try:
            agg[cond].append(json.loads(jf.read_text()))
        except json.JSONDecodeError:
            pass
    return agg


def stats(runs, key):
    vals = np.array([r[key] for r in runs if key in r])
    return float(vals.mean()), float(vals.std()), len(vals)


def block(name):
    if name.startswith("G_0a"): return "alone_3g"
    if name.startswith("G_0b"): return "alone_4g"
    if name.startswith("G_0c"): return "alone_2g"
    if name.startswith("G_1a"): return "coloc_3g"
    if name.startswith("G_1b"): return "coloc_4g"
    if name.startswith("G_1c"): return "coloc_2g"
    if name.startswith("G_2_3gColoc"): return "coloc_3g_ext"
    if name.startswith("G_3_3gColoc_het"): return "coloc_3g_het"
    if name.startswith("G_4_3gColoc_homo"): return "coloc_3g_homo"
    if name.startswith("G_5"): return "coloc_4g_ext"
    if name.startswith("G_6"): return "coloc_2g_ext"
    return "other"


def main():
    agg = load_runs()
    print(f"loaded {len(agg)} conditions\n")

    print(f"{'condition':<40} {'block':<18} {'n':>3}  {'mean':>7}  {'p99':>7}  {'max':>7}")
    print("-" * 90)
    rows = []
    for cond in sorted(agg.keys()):
        runs = agg[cond]
        m_mean, m_sd, n = stats(runs, "mean_ms")
        p99_mean, p99_sd, _ = stats(runs, "p99_ms")
        max_mean, max_sd, _ = stats(runs, "max_ms")
        rows.append((cond, block(cond), n, m_mean, m_sd, p99_mean, p99_sd, max_mean, max_sd))
        print(f"{cond:<40} {block(cond):<18} {n:>3}  {m_mean:>6.1f}  {p99_mean:>6.1f}  {max_mean:>6.1f}")

    # CRITICAL COMPARISONS
    print("\n" + "="*80)
    print("CRITICAL: Coloc vs Alone (pure NeuralRx co-location effect)")
    print("="*80)
    for partition in ["3g", "4g", "2g"]:
        alone_name = f"G_0{'abc'[['3g','4g','2g'].index(partition)]}_{partition}_alone"
        coloc_name = f"G_1{'abc'[['3g','4g','2g'].index(partition)]}_{partition}_coloc"
        alone_runs = next((r for r in rows if r[0] == alone_name), None)
        coloc_runs = next((r for r in rows if r[0] == coloc_name), None)
        if not (alone_runs and coloc_runs):
            print(f"  [{partition}] missing alone or coloc data")
            continue
        a_mean, a_sd, c_mean, c_sd = alone_runs[3], alone_runs[4], coloc_runs[3], coloc_runs[4]
        a_p99, c_p99 = alone_runs[5], coloc_runs[5]
        a_max, c_max = alone_runs[7], coloc_runs[7]
        print(f"\n  [{partition}]:")
        print(f"    alone   : mean={a_mean:.2f}±{a_sd:.2f}  p99={a_p99:.2f}  max={a_max:.2f}")
        print(f"    +coloc  : mean={c_mean:.2f}±{c_sd:.2f}  p99={c_p99:.2f}  max={c_max:.2f}")
        print(f"    Δ       : mean={100*(c_mean-a_mean)/a_mean:+.1f}%  p99={100*(c_p99-a_p99)/a_p99:+.1f}%  max={100*(c_max-a_max)/a_max:+.1f}%")

    # G_2 (3g coloc + external AI variants) — show all external AI types
    print("\n" + "="*80)
    print("G_2 sweep: 3g (L1+NeuralRx coloc) + external AI (2g)")
    print("="*80)
    coloc_alone = next((r for r in rows if r[0] == "G_1a_3g_coloc"), None)
    if coloc_alone:
        ca_mean = coloc_alone[3]; ca_p99 = coloc_alone[5]; ca_max = coloc_alone[7]
        print(f"  baseline coloc alone: mean={ca_mean:.2f}  p99={ca_p99:.2f}  max={ca_max:.2f}")
        print()
        for r in sorted(rows):
            if not r[0].startswith("G_2_3gColoc"):
                continue
            ext = r[0].replace("G_2_3gColoc_", "")
            d_mean = 100*(r[3]-ca_mean)/ca_mean
            d_p99 = 100*(r[5]-ca_p99)/ca_p99
            d_max = 100*(r[7]-ca_max)/ca_max
            print(f"  +{ext:<14}: mean={r[3]:.2f}({d_mean:+.1f}%)  p99={r[5]:.2f}({d_p99:+.1f}%)  max={r[7]:.2f}({d_max:+.1f}%)")

    # G_3/G_4 multi-AI in coloc context
    print("\n=== Multi-AI external in coloc 3g ===")
    for r in rows:
        if r[0].startswith("G_3_3gColoc_het") or r[0].startswith("G_4_3gColoc_homo"):
            if coloc_alone:
                d_p99 = 100*(r[5]-coloc_alone[5])/coloc_alone[5]
                print(f"  {r[0]}: p99={r[5]:.2f} ({d_p99:+.1f}% vs coloc alone)")

    # G_5 / G_6 (4g/2g coloc + chanpred)
    print("\n=== Partition size effect in coloc + chanpred ===")
    for r in rows:
        if r[0] in ("G_5_4gColoc_chanpred", "G_6_2gColoc_chanpred_3g"):
            print(f"  {r[0]}: mean={r[3]:.2f}  p99={r[5]:.2f}  max={r[7]:.2f}")

    # Write CSV
    cols = ["condition", "block", "n", "mean_ms", "mean_sd", "p99_ms", "p99_sd", "max_ms", "max_sd"]
    with open(OUT / "G_summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"\nWrote {OUT / 'G_summary.csv'}")


if __name__ == "__main__":
    main()
