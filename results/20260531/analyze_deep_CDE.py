#!/usr/bin/env python3
"""
Deep-dive C/D/E: unified analyzer that reads any *_run*.sqlite or *_l1.sqlite
under the input dir and computes per-capture statistics:
- L1 kernel count, total duration
- Gap distribution: p50/p95/p99/p999/max
- Burst share (top 1% gaps as % of total idle)
- Long-tail ratio (p99/p50, p999/p99)
For C: groups by batch size; outputs dose-response CSV.
For D: groups by phase type (memcpy/compute/launch/full); outputs decomposition CSV.
For E: groups by multi-AI configuration; outputs smoothing comparison.
"""
import os, sys, sqlite3, re, glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np

STAGE = sys.argv[1] if len(sys.argv) > 1 else "C"
ROOT = Path(f"/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_{STAGE}")
OUT = Path(f"/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_{STAGE}_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def stats(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    if len(rows) < 10:
        return None
    durs = np.array([(r[1] - r[0]) / 1000 for r in rows])
    gaps = np.array([(rows[i + 1][0] - rows[i][1]) / 1000 for i in range(len(rows) - 1) if rows[i + 1][0] > rows[i][1]])
    if len(gaps) < 5:
        return None
    total_kern = float(durs.sum())
    total_idle = float(gaps.sum())
    p99g = float(np.percentile(gaps, 99))
    top1_thresh = p99g
    burst_share = float(gaps[gaps >= top1_thresh].sum()) / max(1.0, total_idle)
    return {
        "n_kernels": len(rows),
        "med_dur_us": float(np.median(durs)),
        "p99_dur_us": float(np.percentile(durs, 99)),
        "med_gap_us": float(np.median(gaps)),
        "p95_gap_us": float(np.percentile(gaps, 95)),
        "p99_gap_us": p99g,
        "p999_gap_us": float(np.percentile(gaps, 99.9)),
        "max_gap_us": float(gaps.max()),
        "total_kern_us": total_kern,
        "total_idle_us": total_idle,
        "idle_pct": 100 * total_idle / (total_kern + total_idle),
        "burst_share_top1": burst_share,
        "longtail_p99_p50": p99g / max(0.001, float(np.median(gaps))),
        "longtail_p999_p99": float(np.percentile(gaps, 99.9)) / max(0.001, p99g),
    }


def main():
    if not ROOT.exists():
        print(f"no input {ROOT}")
        return
    rows = []
    for db in sorted(ROOT.glob("*.sqlite")):
        # Accept multiple naming forms:
        # D: <scenario>_run<N>.sqlite
        # C: <scenario>_run<N>.sqlite (after run_# was in label)
        # E: <scenario>_l1.sqlite (single capture per scenario)
        if "_ai" in db.stem:
            continue  # skip AI captures, only L1
        m = re.match(r"(.+?)_run(\d+)$", db.stem)
        m2 = re.match(r"(.+?)_l1$", db.stem)
        if m:
            scenario, run = m.group(1), m.group(2)
        elif m2:
            scenario, run = m2.group(1), "1"
        else:
            scenario, run = db.stem, "1"
        s = stats(db)
        if s is None:
            continue
        s["scenario"] = scenario
        s["run"] = run
        rows.append(s)
    if not rows:
        print("no rows")
        return

    # aggregate by scenario (median over runs)
    agg = defaultdict(list)
    for r in rows:
        agg[r["scenario"]].append(r)
    out_csv = OUT / f"deep_{STAGE}_stats.csv"
    cols = ["scenario", "n_runs", "n_kernels", "med_dur_us", "p99_dur_us",
            "med_gap_us", "p95_gap_us", "p99_gap_us", "p999_gap_us", "max_gap_us",
            "idle_pct", "burst_share_top1", "longtail_p99_p50", "longtail_p999_p99"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for sc, items in sorted(agg.items()):
            n = len(items)
            row = [sc, n]
            for c in cols[2:]:
                vals = [it[c] for it in items]
                row.append(f"{np.median(vals):.2f}")
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"wrote {out_csv}")

    # If C: parse batch size, build dose-response table
    if STAGE == "C":
        dose = OUT / "dose_response.csv"
        with open(dose, "w") as f:
            f.write("config,batch_or_dim,p99_gap_us,p999_gap_us,idle_pct,burst_share\n")
            for sc, items in sorted(agg.items()):
                m = re.search(r"_(b|d)(\d+)$", sc)
                if not m:
                    continue
                kind, val = m.group(1), int(m.group(2))
                cfg = sc.rsplit("_", 1)[0]
                p99 = np.median([i["p99_gap_us"] for i in items])
                p999 = np.median([i["p999_gap_us"] for i in items])
                ip = np.median([i["idle_pct"] for i in items])
                bs = np.median([i["burst_share_top1"] for i in items])
                f.write(f"{cfg},{val},{p99:.1f},{p999:.1f},{ip:.2f},{bs:.3f}\n")
        print(f"wrote {dose}")

    # If D: phase decomposition relative to D0 baseline
    if STAGE == "D":
        if "D0_3gL1_alone" in agg:
            base = np.median([i["p99_gap_us"] for i in agg["D0_3gL1_alone"]])
            base_p999 = np.median([i["p999_gap_us"] for i in agg["D0_3gL1_alone"]])
            decomp = OUT / "phase_decomposition.csv"
            with open(decomp, "w") as f:
                f.write("phase,p99_gap_us,delta_pct,p999_gap_us,delta_p999_pct,burst_share,longtail_p99_p50\n")
                for sc, items in sorted(agg.items()):
                    p99 = np.median([i["p99_gap_us"] for i in items])
                    p999 = np.median([i["p999_gap_us"] for i in items])
                    bs = np.median([i["burst_share_top1"] for i in items])
                    lt = np.median([i["longtail_p99_p50"] for i in items])
                    f.write(f"{sc},{p99:.1f},{100*(p99-base)/base:+.1f}%,{p999:.1f},{100*(p999-base_p999)/base_p999:+.1f}%,{bs:.3f},{lt:.1f}\n")
            print(f"wrote {decomp}")

    print("DONE")


if __name__ == "__main__":
    main()
