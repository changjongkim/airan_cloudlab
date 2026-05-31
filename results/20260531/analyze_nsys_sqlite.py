#!/usr/bin/env python3
"""
Detailed nsys analysis via SQLite — per-kernel-type gap inflation.

For each scenario:
  1. Identify top cuPHY L1 kernels by total time spent
  2. For each kernel type: measure gap distribution (median, p99) AFTER it completes
  3. Compare alone vs with-AI: which kernel's POST-gap inflates the most
  4. Identify "vulnerable kernels" — the bottlenecks of inter-kernel scheduling

Hypothesis: certain kernels (e.g. memory-bound channel estimation) have their
post-completion gap inflate more under AI co-tenant than others.
"""
import sqlite3
import statistics
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent / "nsys_sqlite_v2"

# cuPHY L1-specific kernel name prefixes (filter out memcpy/python noise)
CUPHY_KERNEL_KEYWORDS = [
    "channel_eq", "ch_est", "noise_intf", "ldpc", "pusch", "crc",
    "windowed", "eqMmse", "noiseIntf", "ldpcDeRate", "ldpcDecode", "crcCheck",
    "scrambling", "descrambling", "demap", "softDemap",
]


def is_cuphy_kernel(name):
    if not name:
        return False
    low = name.lower()
    return any(k.lower() in low for k in CUPHY_KERNEL_KEYWORDS)


def load_kernels(db_path):
    """Return list of (start_ns, end_ns, name) for all kernels."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT k.start, k.end, COALESCE(s_demangled.value, s_short.value, 'unknown') as name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s_demangled ON k.demangledName = s_demangled.id
        LEFT JOIN StringIds s_short ON k.shortName = s_short.id
        ORDER BY k.start
    """).fetchall()
    conn.close()
    return rows


def analyze_per_kernel(rows):
    """Group kernels by name; compute count, total time, post-gap distribution."""
    by_kernel = defaultdict(lambda: {"durations": [], "post_gaps": []})
    for i in range(len(rows)):
        start, end, name = rows[i]
        dur = end - start
        by_kernel[name]["durations"].append(dur)
        # Post-gap: time until next kernel starts
        if i + 1 < len(rows):
            next_start = rows[i + 1][0]
            gap = next_start - end
            if gap >= 0:
                by_kernel[name]["post_gaps"].append(gap)
    # Compute summary stats per kernel
    summary = {}
    for name, data in by_kernel.items():
        durs = data["durations"]
        gaps = data["post_gaps"]
        if not gaps:
            continue
        durs.sort()
        gaps.sort()

        def pct(arr, p):
            return arr[int(len(arr) * p / 100)] if arr else 0

        summary[name] = {
            "count": len(durs),
            "total_dur_us": sum(durs) / 1000.0,
            "median_dur_us": pct(durs, 50) / 1000.0,
            "median_post_gap_us": pct(gaps, 50) / 1000.0,
            "p95_post_gap_us": pct(gaps, 95) / 1000.0,
            "p99_post_gap_us": pct(gaps, 99) / 1000.0,
            "max_post_gap_us": gaps[-1] / 1000.0,
            "mean_post_gap_us": statistics.mean(gaps) / 1000.0,
        }
    return summary


def shorten(name, limit=60):
    if len(name) <= limit:
        return name
    return name[:limit - 3] + "..."


def main():
    # Group sqlite files by scenario
    by_scenario = defaultdict(list)
    for f in sorted(ROOT.glob("*.sqlite")):
        stem = f.stem  # e.g. S5_3g_alone_run1
        m = stem.rsplit("_run", 1)
        if len(m) != 2:
            continue
        scenario = m[0]
        by_scenario[scenario].append(str(f))

    # Aggregate across runs per scenario
    print(f"Found {sum(len(v) for v in by_scenario.values())} sqlite files across {len(by_scenario)} scenarios")
    print()

    scenarios_data = {}
    for scenario, paths in sorted(by_scenario.items()):
        merged = defaultdict(lambda: {"durations": [], "post_gaps": []})
        for path in paths:
            rows = load_kernels(path)
            # Filter cuPHY only
            cuphy_rows = [r for r in rows if is_cuphy_kernel(r[2])]
            if not cuphy_rows:
                continue
            for i in range(len(cuphy_rows)):
                start, end, name = cuphy_rows[i]
                merged[name]["durations"].append(end - start)
                if i + 1 < len(cuphy_rows):
                    gap = cuphy_rows[i + 1][0] - end
                    if gap >= 0:
                        merged[name]["post_gaps"].append(gap)
        # Compute summary
        summary = {}
        for name, data in merged.items():
            durs = data["durations"]
            gaps = data["post_gaps"]
            if not gaps:
                continue
            durs.sort()
            gaps.sort()

            def pct(arr, p):
                return arr[int(len(arr) * p / 100)] if arr else 0

            summary[name] = {
                "count": len(durs),
                "total_dur_us": sum(durs) / 1000.0,
                "median_dur_us": pct(durs, 50) / 1000.0,
                "median_post_gap_us": pct(gaps, 50) / 1000.0,
                "p95_post_gap_us": pct(gaps, 95) / 1000.0,
                "p99_post_gap_us": pct(gaps, 99) / 1000.0,
                "max_post_gap_us": gaps[-1] / 1000.0,
            }
        scenarios_data[scenario] = summary
        print(f"  {scenario}: {len(summary)} cuPHY kernel types")

    # Print top kernel types in S5 (baseline)
    baseline = "S5_3g_alone"
    if baseline not in scenarios_data:
        print("No baseline data")
        return
    base = scenarios_data[baseline]
    top_kernels = sorted(base.items(), key=lambda kv: kv[1]["total_dur_us"], reverse=True)[:15]

    print(f"\n{'='*120}")
    print(f"TOP 15 cuPHY kernels in S5 baseline (by total time):")
    print(f"{'='*120}\n")
    print(f"  {'Kernel':<55}{'Count':>8}{'TotalDur(us)':>14}{'MedDur(us)':>12}{'MedPostGap(us)':>16}{'p99PostGap(us)':>16}")
    print("  " + "-" * 121)
    for name, s in top_kernels:
        print(f"  {shorten(name, 55):<55}{s['count']:>8}{s['total_dur_us']:>14.0f}{s['median_dur_us']:>12.2f}{s['median_post_gap_us']:>16.2f}{s['p99_post_gap_us']:>16.2f}")

    # Compare scenarios for each TOP kernel: post-gap inflation
    comparison_scenarios = [
        ("S5_3g_alone", "baseline"),
        ("S6_3g_qwen", "+Qwen"),
        ("S7_3g_neuralrx", "+NeuralRx"),
        ("S9_3g_3AI_1g", "+3 AI"),
        ("S13_3g_sat_compute", "+sat_compute"),
        ("S24_3g_2sat", "+2 sat"),
    ]

    print(f"\n{'='*120}")
    print(f"POST-GAP INFLATION per kernel (3g L1 + various AI vs S5 baseline)")
    print(f"{'='*120}\n")

    # For each top kernel, show p99 post-gap across scenarios
    print(f"  {'Kernel':<55}" + "".join(f"{label:>16}" for _, label in comparison_scenarios))
    print("  " + "-" * (55 + 16 * len(comparison_scenarios)))
    for name, _ in top_kernels[:10]:
        cells = []
        for sname, label in comparison_scenarios:
            if sname not in scenarios_data:
                cells.append(f"{'N/A':>16}")
                continue
            if name in scenarios_data[sname]:
                p99 = scenarios_data[sname][name]["p99_post_gap_us"]
                if sname == "S5_3g_alone":
                    cells.append(f"{p99:>15.1f}us")
                else:
                    base_p99 = base[name]["p99_post_gap_us"]
                    delta = (p99 - base_p99) / base_p99 * 100 if base_p99 else 0
                    cells.append(f"{delta:>+15.1f}%")
            else:
                cells.append(f"{'N/A':>16}")
        print(f"  {shorten(name, 55):<55}" + "".join(cells))

    # Memory operations analysis (memcpy/memset)
    print(f"\n{'='*120}")
    print(f"MEMORY OPERATIONS (memcpy/memset) — total time and count")
    print(f"{'='*120}\n")
    print(f"  {'Scenario':<22}{'Kernels':>12}{'Total kernel time (ms)':>26}{'Idle (ms)':>14}")
    for sname, _ in comparison_scenarios:
        if sname not in by_scenario:
            continue
        # Sum kernel time + count across runs
        for path in by_scenario[sname][:1]:  # just first run
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            kernel_count = cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
            kernel_time = cur.execute("SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] or 0
            first_start, last_end = cur.execute("""
                SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL
            """).fetchone()
            total = last_end - first_start if first_start else 0
            idle = total - kernel_time
            conn.close()
            print(f"  {sname:<22}{kernel_count:>12}{kernel_time/1e6:>25.2f} {idle/1e6:>13.2f}")


if __name__ == "__main__":
    main()
