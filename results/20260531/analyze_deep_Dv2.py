#!/usr/bin/env python3
"""Dv2 analyzer — n=10 phase decomposition (D2D vs H2D + compute + launch + chanpred)."""
import os, sqlite3, re
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_Dv2")
OUT = Path("/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_Dv2_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def stats(db):
    con = sqlite3.connect(db); cur = con.cursor()
    try:
        cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    if len(rows) < 100:
        return None
    rows = rows[1000:]  # skip warmup
    gaps = np.array([(rows[i+1][0]-rows[i][1])/1000 for i in range(len(rows)-1) if rows[i+1][0]>rows[i][1]])
    if len(gaps) < 100:
        return None
    return {
        "n_kern": len(rows),
        "p50_gap_us": float(np.median(gaps)),
        "p95_gap_us": float(np.percentile(gaps, 95)),
        "p99_gap_us": float(np.percentile(gaps, 99)),
        "p999_gap_us": float(np.percentile(gaps, 99.9)),
        "max_gap_us": float(gaps.max()),
    }


def main():
    agg = defaultdict(list)
    for db in sorted(ROOT.glob("*.sqlite")):
        m = re.match(r"(.+?)_run(\d+)$", db.stem)
        if not m:
            continue
        scenario, run = m.group(1), int(m.group(2))
        s = stats(db)
        if s:
            agg[scenario].append(s)

    print("=== Dv2 phase decomposition (n=10, with CI) ===\n")
    print(f"{'Scenario':<25} {'n':>3}  {'p99 mean±SD':>15}  {'p999 mean±SD':>17}  {'95% CI p99':>20}")
    print("-" * 90)
    rows = []
    for sc, items in sorted(agg.items()):
        if not items:
            continue
        n = len(items)
        p99s = np.array([x["p99_gap_us"] for x in items])
        p999s = np.array([x["p999_gap_us"] for x in items])
        maxs = np.array([x["max_gap_us"] for x in items])
        p99_mean = p99s.mean(); p99_sd = p99s.std()
        # 95% CI = mean ± 1.96 * SE
        p99_se = p99_sd / np.sqrt(n)
        ci_lo = p99_mean - 1.96 * p99_se
        ci_hi = p99_mean + 1.96 * p99_se
        p999_mean = p999s.mean(); p999_sd = p999s.std()
        print(f"{sc:<25} {n:>3}  {p99_mean:>7.0f} ± {p99_sd:>4.0f}us  {p999_mean:>8.0f} ± {p999_sd:>5.0f}us  [{ci_lo:>6.0f}, {ci_hi:>6.0f}]")
        rows.append({"scenario": sc, "n": n, "p99_mean": p99_mean, "p99_sd": p99_sd,
                     "ci_lo": ci_lo, "ci_hi": ci_hi, "p999_mean": p999_mean,
                     "p999_sd": p999_sd, "max_mean": maxs.mean(), "max_sd": maxs.std()})

    with open(OUT / "Dv2_summary.csv", "w") as f:
        f.write("scenario,n,p99_mean,p99_sd,p99_ci_lo,p99_ci_hi,p999_mean,p999_sd,max_mean,max_sd\n")
        for r in rows:
            f.write(f"{r['scenario']},{r['n']},{r['p99_mean']:.1f},{r['p99_sd']:.1f},"
                    f"{r['ci_lo']:.1f},{r['ci_hi']:.1f},{r['p999_mean']:.1f},{r['p999_sd']:.1f},"
                    f"{r['max_mean']:.1f},{r['max_sd']:.1f}\n")

    # Baseline = Dv2_0_alone
    base = next((r for r in rows if "Dv2_0_alone" in r["scenario"]), None)
    if base:
        print(f"\n=== vs baseline Dv2_0_alone (p99={base['p99_mean']:.0f}us) ===")
        print(f"{'Scenario':<25} {'Δp99':>10}  {'p99_CI_overlap?':>18}  {'Δp999':>10}  {'sig?'}")
        for r in rows:
            if r["scenario"] == base["scenario"]: continue
            dp99 = r["p99_mean"] - base["p99_mean"]
            dp99_pct = 100*dp99/base["p99_mean"]
            # CIs overlap?
            overlap = "OVERLAP" if (r["ci_lo"] <= base["ci_hi"] and r["ci_hi"] >= base["ci_lo"]) else "SEPARATE"
            dp999 = r["p999_mean"] - base["p999_mean"]
            sig = "***" if overlap == "SEPARATE" else "ns"
            print(f"{r['scenario']:<25} {dp99_pct:+>9.1f}%  {overlap:>18}  {100*dp999/base['p999_mean']:+>9.1f}%  {sig}")


if __name__ == "__main__":
    main()
