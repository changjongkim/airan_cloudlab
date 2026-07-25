#!/usr/bin/env python3
"""Chain 18 Parts 3-7 unified analysis.

Assumes results synced to results/20260725/chain18/ containing nsys-rep files
labelled p3_..., p4_..., p5_..., p6_..., p7_...

Outputs a mega set of figures:
  Part 3: f15_p3_nsweep_extended.png (nrx N=5..16, memcpy/embed N=1..8)
  Part 4: f16_p4_fine_thread_pct.png (10 pct steps × 4 wl)
  Part 5: f17_p5_thread_vs_process.png (multi-thread vs multi-proc controlled)
  Part 6: f18_p6_multigpu.png (L1 GPU0 / AI GPU1 baseline)
  Part 7: f19_p7_longwindow.png + f20_p7_statistical.png (10-trial stats)
"""
import os, re, json, subprocess, glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
CH18 = os.path.join(BASE, "chain18")
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

# ---- extract kernel activity from nsys-rep via nsys stats ----
def nsys_launch_count(path):
    """Return per-second CUDA kernel launch count average from nsys-rep."""
    try:
        r = subprocess.run(["nsys","stats","--report","cuda_gpu_kern_sum","--format","csv","--force-export","true",path],
                            capture_output=True, text=True, timeout=120)
        # parse Instances column
        n = 0
        for ln in r.stdout.splitlines():
            parts = ln.split(",")
            if len(parts) >= 3 and parts[1].strip().isdigit():
                n += int(parts[1].strip())
        return n
    except Exception as e:
        return 0

def nsys_kernel_duration_dist(path):
    """Return list of (kernel_name, avg_ns) for L1 kernels."""
    try:
        r = subprocess.run(["nsys","stats","--report","cuda_gpu_kern_sum","--format","csv","--force-export","true",path],
                            capture_output=True, text=True, timeout=120)
        rows = []
        header_seen = False
        for ln in r.stdout.splitlines():
            if "Name" in ln and "Instances" in ln: header_seen = True; continue
            if not header_seen: continue
            parts = [p.strip('" ') for p in ln.split(",")]
            if len(parts) < 5: continue
            try:
                avg_ns = float(parts[3])
                rows.append((parts[-1], avg_ns))
            except: pass
        return rows
    except: return []

# ============================================================
# Part 3 — N-sweep extended
# ============================================================
def analyze_part3():
    """Aggregate launch counts per (workload, N, MPS)."""
    data = defaultdict(list)  # (wl, N, mps) -> [launches]
    for f in sorted(glob.glob(f"{CH18}/p3_*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        m = re.match(r"p3_(\w+?)N(\d+)_MPS(\w+)_t(\d+)", base)
        if not m: continue
        wl, N, mps, t = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        launches = nsys_launch_count(f)
        data[(wl, N, mps)].append(launches)
    return data

def plot_part3(data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    workloads = ["nrx","memcpy","embed"]
    Ns_map = {"nrx": [5,7,10,12,16], "memcpy": [1,2,4,6,8], "embed": [1,2,4,6,8]}
    for ax, wl in zip(axes, workloads):
        Ns = Ns_map[wl]
        for mps, color in [("off","#dc2626"),("on","#10b981")]:
            vals = [np.mean(data.get((wl,N,mps),[0])) for N in Ns]
            errs = [np.std(data.get((wl,N,mps),[0])) for N in Ns]
            ax.errorbar(Ns, vals, yerr=errs, fmt="o-", color=color, label=f"MPS {mps}",
                        linewidth=2.5, markersize=10, capsize=5)
        ax.set_xlabel(f"N ({wl} processes)"); ax.set_ylabel("L1 kernel launches")
        ax.set_title(f"{wl.upper()} N-sweep", fontweight="bold")
        ax.grid(alpha=0.3); ax.legend()
    plt.suptitle("Figure 15. Chain 18 Part 3 — Extended N-process sweep on Config A\n"
                 "L1 kernel launches vs. concurrent AI processes",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG}/f15_p3_nsweep_extended.png", dpi=140, bbox_inches='tight')
    plt.close()

# ============================================================
# Part 4 — Fine MPS thread%
# ============================================================
def analyze_part4():
    data = defaultdict(list)  # (wl, pct) -> [launches]
    for f in sorted(glob.glob(f"{CH18}/p4_*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        m = re.match(r"p4_(\w+)_pct(\d+)_t(\d+)", base)
        if not m: continue
        wl, pct, t = m.group(1), int(m.group(2)), int(m.group(3))
        launches = nsys_launch_count(f)
        data[(wl, pct)].append(launches)
    return data

def plot_part4(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    workloads = ["nrx4","ranai","memcpy4","embed4"]
    pcts = [10,20,30,40,50,60,70,80,90,100]
    palette = ["#3b82f6","#10b981","#f59e0b","#dc2626"]
    for wl, color in zip(workloads, palette):
        vals = [np.mean(data.get((wl,pct),[0])) for pct in pcts]
        ax.plot(pcts, vals, "o-", color=color, label=wl, linewidth=2.5, markersize=10)
    ax.set_xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (%)")
    ax.set_ylabel("L1 kernel launches (30s)")
    ax.set_title("Figure 16. Chain 18 Part 4 — Fine MPS thread% sweep (Config A)\n"
                 "Sensitivity of L1 throughput to same-partition AI cap",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG}/f16_p4_fine_thread_pct.png", dpi=140, bbox_inches='tight')
    plt.close()

# ============================================================
# Part 5 — Multi-thread vs Multi-process
# ============================================================
def analyze_part5():
    thr_data = defaultdict(list)   # beam -> [launches]
    proc_data = defaultdict(list)  # N -> [launches]
    for f in sorted(glob.glob(f"{CH18}/p5_*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        m = re.match(r"p5_thrOnly_beam(\d+)_t(\d+)", base)
        if m:
            thr_data[int(m.group(1))].append(nsys_launch_count(f)); continue
        m = re.match(r"p5_procOnly_N(\d+)_t(\d+)", base)
        if m:
            proc_data[int(m.group(1))].append(nsys_launch_count(f))
    return thr_data, proc_data

def plot_part5(thr_data, proc_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    # Threads: total = 6 + beam
    beams = sorted(thr_data.keys())
    totals = [6+b for b in beams]
    ax1.errorbar(totals, [np.mean(thr_data[b]) for b in beams],
                 yerr=[np.std(thr_data[b]) for b in beams],
                 fmt="o-", color="#3b82f6", label="Multi-thread (1 proc)", linewidth=2.5, markersize=10, capsize=5)
    Ns = sorted(proc_data.keys())
    totals_p = [14*N for N in Ns]
    ax2.errorbar(totals_p, [np.mean(proc_data[N]) for N in Ns],
                 yerr=[np.std(proc_data[N]) for N in Ns],
                 fmt="s-", color="#dc2626", label="Multi-process (14 thr each)", linewidth=2.5, markersize=10, capsize=5)
    ax1.set_xlabel("Total concurrent AI threads"); ax1.set_ylabel("L1 kernel launches")
    ax1.set_title("Multi-thread scaling (same process)", fontweight="bold"); ax1.grid(alpha=0.3); ax1.legend()
    ax2.set_xlabel("Total AI threads across processes"); ax2.set_ylabel("L1 kernel launches")
    ax2.set_title("Multi-process scaling", fontweight="bold"); ax2.grid(alpha=0.3); ax2.legend()
    plt.suptitle("Figure 17. Chain 18 Part 5 — Multi-thread vs Multi-process controlled\n"
                 "Same total work, single vs. many CUDA contexts",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG}/f17_p5_thread_vs_process.png", dpi=140, bbox_inches='tight')
    plt.close()

# ============================================================
# Part 6 — Multi-GPU baseline
# ============================================================
def analyze_part6():
    data = defaultdict(list)  # wl -> [launches]
    for f in sorted(glob.glob(f"{CH18}/p6_*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        m = re.match(r"p6_multiGPU_(\w+)_t(\d+)", base)
        if m:
            data[m.group(1)].append(nsys_launch_count(f))
    return data

def plot_part6(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    wls = list(data.keys())
    vals = [np.mean(data[wl]) for wl in wls]
    errs = [np.std(data[wl]) for wl in wls]
    ax.bar(range(len(wls)), vals, yerr=errs, capsize=6,
           color="#10b981", alpha=0.85, edgecolor="#111", lw=1.2)
    ax.set_xticks(range(len(wls))); ax.set_xticklabels(wls, rotation=25, ha="right")
    ax.set_ylabel("L1 kernel launches (30s)")
    ax.set_title("Figure 18. Chain 18 Part 6 — Multi-GPU baseline (L1 on GPU0, AI on GPU1)\n"
                 "Cross-GPU has no shared driver contention",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG}/f18_p6_multigpu.png", dpi=140, bbox_inches='tight')
    plt.close()

# ============================================================
# Part 7 — Long-window + statistical
# ============================================================
def analyze_part7():
    long_data = defaultdict(list)
    stat_data = defaultdict(list)
    for f in sorted(glob.glob(f"{CH18}/p7_*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        launches = nsys_launch_count(f)
        m = re.match(r"p7_long_(\w+)_t(\d+)", base)
        if m: long_data[m.group(1)].append(launches); continue
        m = re.match(r"p7_stat_nrxN(\d+)_MPSon_t(\d+)", base)
        if m: stat_data[int(m.group(1))].append(launches)
    return long_data, stat_data

def plot_part7(long_data, stat_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    # Long-window bar
    keys = list(long_data.keys())
    means = [np.mean(long_data[k]) for k in keys]
    stds = [np.std(long_data[k]) for k in keys]
    ax1.bar(range(len(keys)), means, yerr=stds, capsize=6,
            color=["#3b82f6","#10b981","#dc2626"], alpha=0.85, edgecolor="#111")
    ax1.set_xticks(range(len(keys))); ax1.set_xticklabels(keys, rotation=15, ha="right")
    ax1.set_ylabel("L1 kernel launches over 300s")
    ax1.set_title("Long-window (300s) — did steady-state break?", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    # 10-trial statistical box plot
    Ns = sorted(stat_data.keys())
    ax2.boxplot([stat_data[N] for N in Ns], positions=range(len(Ns)),
                widths=0.6, patch_artist=True, medianprops=dict(color="#111",lw=1.5))
    for N in Ns:
        vals = stat_data[N]
        ax2.text(Ns.index(N), max(vals), f"n={len(vals)}", ha="center", fontsize=9)
    ax2.set_xticks(range(len(Ns))); ax2.set_xticklabels([f"N={N}" for N in Ns])
    ax2.set_ylabel("L1 kernel launches (30s)")
    ax2.set_title("10-trial statistical at breakpoint (NRx, MPS on)", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.suptitle("Figure 19+20. Chain 18 Part 7 — Long-window + statistical robustness",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG}/f19_p7_longwindow_statistical.png", dpi=140, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Parts 3-7 analysis")
    if not os.path.isdir(CH18):
        print(f"Missing {CH18} — waiting for Parts 3-7 to complete"); exit()
    p3 = analyze_part3(); print(f"Part 3: {len(p3)} conditions")
    if p3: plot_part3(p3); print("saved f15")
    p4 = analyze_part4(); print(f"Part 4: {len(p4)} conditions")
    if p4: plot_part4(p4); print("saved f16")
    p5_t, p5_p = analyze_part5(); print(f"Part 5: {len(p5_t)} thr + {len(p5_p)} proc")
    if p5_t or p5_p: plot_part5(p5_t, p5_p); print("saved f17")
    p6 = analyze_part6(); print(f"Part 6: {len(p6)} conditions")
    if p6: plot_part6(p6); print("saved f18")
    p7_l, p7_s = analyze_part7(); print(f"Part 7: {len(p7_l)} long + {len(p7_s)} stat")
    if p7_l or p7_s: plot_part7(p7_l, p7_s); print("saved f19")
    # dump stats
    dump = {}
    for name, d in [("p3",p3),("p4",p4),("p5_thr",p5_t),("p5_proc",p5_p),("p6",p6),("p7_long",p7_l),("p7_stat",p7_s)]:
        dump[name] = {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                      for k, v in d.items()}
    with open(f"{BASE}/parts3to7_stats.json","w") as fp:
        json.dump(dump, fp, indent=2)
    print("Wrote parts3to7_stats.json")
