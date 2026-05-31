#!/usr/bin/env python3
"""5/31 mid-analysis. Extract per-experiment stats from N=20 JSON files."""
import json, os, glob, statistics
from pathlib import Path

ROOT = Path(__file__).parent

import re
_LOG_PATTERN = re.compile(r"mean=([0-9.]+)ms\s+p95=([0-9.]+)ms\s+p99=([0-9.]+)ms")

def load_runs(dir_path):
    runs = []
    # Prefer JSON if present (Phase1/2/3/4 dirs via run_n20)
    json_files = sorted(glob.glob(str(dir_path / "run_*.json")))
    if json_files:
        for jf in json_files:
            try:
                with open(jf) as f: runs.append(json.load(f))
            except Exception:
                pass
        return runs
    # Fall back to log files (baselines via run_fullgpu/mig_baselines)
    for lf in sorted(glob.glob(str(dir_path / "run_*.log"))):
        try:
            with open(lf) as f: content = f.read()
            m = _LOG_PATTERN.search(content)
            if m:
                runs.append({
                    "mean_ms": float(m.group(1)),
                    "p95_ms": float(m.group(2)),
                    "p99_ms": float(m.group(3)),
                })
        except Exception:
            pass
    return runs

def summary(name, runs):
    n = len(runs)
    if n == 0: return f"{name}: NO DATA"
    means = [r["mean_ms"] for r in runs]
    p99s = [r["p99_ms"] for r in runs]
    p95s = [r["p95_ms"] for r in runs]
    return {
        "name": name, "N": n,
        "mean_of_means": statistics.mean(means),
        "p99_of_p99s": statistics.mean(p99s),
        "mean_of_p95s": statistics.mean(p95s),
        "stdev_means": statistics.stdev(means) if n >= 2 else 0,
        "min_mean": min(means), "max_mean": max(means),
        # bimodal: gap between cluster medians
        "first_third_mean": statistics.mean(sorted(means)[:n//3]) if n >= 3 else means[0],
        "last_third_mean": statistics.mean(sorted(means)[-(n//3):]) if n >= 3 else means[-1],
    }

# Baselines
BASELINES = [
    ("Full_GPU_no_MIG", "n20_baseline_fullGPU_v2"),
    ("7g_MIG_single",   "n20_baseline_7g_single"),
    ("4g_MIG_alone",    "n20_baseline_4g_alone"),
    ("3g_MIG_alone",    "n20_baseline_3g_alone"),
    ("2g_MIG_alone",    "n20_baseline_2g_alone"),
]

PHASE1 = [
    ("phase1_qwen7b_stress",   "n20_phase1_qwen7b_stress"),
    ("phase1_qwen7b_prefill",  "n20_phase1_qwen7b_prefill"),
    ("phase1_qwen7b_decode",   "n20_phase1_qwen7b_decode"),
    ("phase1_qwen_small",      "n20_phase1_qwen_small"),
]

PHASE4 = [
    ("phase4_neuralrx",        "n20_phase4_neuralrx"),
    ("phase4_chanpred",        "n20_phase4_chanpred"),
    ("phase4_xapp",            "n20_phase4_xapp"),
]

PHASE2 = [
    ("phase2_M1_3way_balanced",  "n20_phase2_M1_3way_balanced"),
    ("phase2_M2_3way_L1small",   "n20_phase2_M2_3way_L1small"),
    ("phase2_M3_3way_asym",      "n20_phase2_M3_3way_asym"),
    ("phase2_M4_4way_1L1_3AI",   "n20_phase2_M4_4way_1L1_3AI"),
]

PHASE3 = [
    ("phase3_D1_L1_starved",    "n20_phase3_D1_L1_starved"),
    ("phase3_D2_L1_boosted",    "n20_phase3_D2_L1_boosted"),
]

def print_table(title, entries):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    print(f"{'name':<30} {'N':>3} {'mean':>7} {'p95':>7} {'p99':>7} {'stdev':>7} {'min/max':>15} {'bimodal_gap':>10}")
    print("-" * 100)
    for label, dirname in entries:
        d = ROOT / dirname
        runs = load_runs(d)
        s = summary(label, runs)
        if isinstance(s, str): print(s); continue
        bimodal_gap = s["last_third_mean"] - s["first_third_mean"]
        print(f"{label:<30} {s['N']:>3} {s['mean_of_means']:>6.2f} {s['mean_of_p95s']:>6.2f} {s['p99_of_p99s']:>6.2f} {s['stdev_means']:>6.2f} {s['min_mean']:>5.1f}/{s['max_mean']:>5.1f}  {bimodal_gap:>9.2f}")

print_table("BASELINES (L1 alone, no AI co-tenant)", BASELINES)
print_table("PHASE 1 — Qwen variants on split-50-50 (3g L1 + 3g AI)", PHASE1)
print_table("PHASE 4 — Real AI-RAN on split-50-50", PHASE4)
print_table("PHASE 2 — Multi-partition layouts", PHASE2)
print_table("PHASE 3 — D1/D2 stress configs", PHASE3)

# Baseline integrity check
print(f"\n{'='*80}\nBASELINE INTEGRITY CHECK\n{'='*80}")
expected_baselines = {
    "Full_GPU_no_MIG": (32, 42),   # expected mean range
    "7g_MIG_single":   (32, 42),
    "4g_MIG_alone":    (35, 45),
    "3g_MIG_alone":    (38, 48),
    "2g_MIG_alone":    (45, 60),
}
for label, dirname in BASELINES:
    d = ROOT / dirname
    runs = load_runs(d)
    s = summary(label, runs)
    if isinstance(s, str):
        print(f"  [{label}] NO DATA — FAIL")
        continue
    if s["N"] < 20:
        print(f"  [{label}] N={s['N']} (< 20) — INCOMPLETE")
        continue
    lo, hi = expected_baselines.get(label, (0, 1e9))
    in_range = lo <= s["mean_of_means"] <= hi
    flag = "OK" if in_range else f"OUT OF EXPECTED RANGE [{lo},{hi}]"
    print(f"  [{label}] N={s['N']} mean={s['mean_of_means']:.2f}ms — {flag}")

# AI throughput v2 summary
print(f"\n{'='*80}\nAI THROUGHPUT V2 (split-60-40, L1=3g + AI=2g, N=5)\n{'='*80}")
v2_root = ROOT / "ai_throughput_v2"
if v2_root.exists():
    for ai_dir in sorted(v2_root.iterdir()):
        if not ai_dir.is_dir(): continue
        tp_file = ai_dir / "throughput.txt"
        if not tp_file.exists(): continue
        rates = []
        for line in open(tp_file):
            m = re.search(r"([0-9.]+)\s*(it/s|inf/s|pred/s|GB/s|tokens/s)", line)
            if m: rates.append(float(m.group(1)))
        if rates:
            print(f"  {ai_dir.name:<30} N={len(rates):>2} mean={statistics.mean(rates):>8.2f} range=[{min(rates):.1f}, {max(rates):.1f}]")

# ai_full_matrix summary
print(f"\n{'='*80}\nAI FULL MATRIX (19 cells: AI workload throughput × partitions, with/without L1)\n{'='*80}")
full_root = ROOT / "ai_full_matrix"
if full_root.exists():
    for cell_dir in sorted(full_root.iterdir()):
        if not cell_dir.is_dir(): continue
        for sub in ["alone", "with_l1"]:
            sub_dir = cell_dir / sub
            if not sub_dir.exists(): continue
            rates = []
            for lf in sorted(sub_dir.glob("run_*.log")):
                try:
                    content = open(lf).read()
                    m = re.search(r"([0-9.]+)\s*(it/s|inf/s|pred/s|GB/s|tflops=([0-9.]+))", content)
                    if m:
                        if m.group(2).startswith("tflops"):
                            rates.append(float(m.group(3)))
                        else:
                            rates.append(float(m.group(1)))
                    # Also try tflops directly
                    m2 = re.search(r"tflops=([0-9.]+)", content)
                    if m2 and not m: rates.append(float(m2.group(1)))
                    m3 = re.search(r"bw=([0-9.]+)GB/s", content)
                    if m3 and not m and not m2: rates.append(float(m3.group(1)))
                except Exception:
                    pass
            if rates:
                print(f"  {cell_dir.name+'/'+sub:<40} N={len(rates):>2} mean={statistics.mean(rates):>8.2f} range=[{min(rates):.1f}, {max(rates):.1f}]")
import re
