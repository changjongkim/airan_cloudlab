"""Figures for 20260614 measurement campaign on d8545.

Builds 6 figures saved under figures/:
  fig_t01_partition_alone_vs_coloc.png  — partition sweep alone slope vs coloc flat
  fig_t02_cross_partition_matrix.png    — all single AI + stacking, MIG cross-partition flat
  fig_t03_coloc_workload_invariant.png  — chanpred/NeuralRx/mixed all collapse to ~360ms
  fig_t04_three_regimes_cdf.png         — alone vs cross-partition AI vs same-part coloc CDF
  fig_t05_7g_coloc_equals_no_mig.png    — 7g (=full GPU) coloc matches Perlmutter no-MIG
  fig_t06_ncu_dram_alone_vs_coloc.png   — DRAM throughput from C1 NCU
"""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

def load_ms(pat):
    ms = []
    for f in sorted(glob.glob(pat)):
        try:
            ms += json.load(open(f)).get("raw_ms", [])
        except Exception:
            pass
    return np.array(ms) if ms else None

def stats(ms):
    if ms is None or len(ms) == 0:
        return None
    n = len(ms)
    return dict(n=n, p50=float(np.percentile(ms, 50)),
                mean=float(np.mean(ms)), p99=float(np.percentile(ms, 99)),
                max=float(np.max(ms)), min=float(np.min(ms)),
                std=float(np.std(ms)))

# --------------------------------------------------------------------------
# Figure 1: partition sweep alone vs +NeuralRx coloc
# --------------------------------------------------------------------------
def fig_t01():
    sizes = ["2g", "3g", "4g", "7g"]
    alone_p50, alone_p99 = [], []
    coloc_p50, coloc_p99 = [], []
    for s in sizes:
        a = stats(load_ms(f"{ROOT}/E5_alone_partition/{s}/realL1_*.json"))
        c = stats(load_ms(f"{ROOT}/E6_coloc_neuralrx/{s}/realL1_*.json"))
        alone_p50.append(a["p50"]); alone_p99.append(a["p99"])
        coloc_p50.append(c["p50"]); coloc_p99.append(c["p99"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                    gridspec_kw={"width_ratios":[1,1]})
    x = np.arange(len(sizes))
    w = 0.35
    ax1.bar(x - w/2, alone_p50, w, label="alone p50", color="#3b82f6")
    ax1.bar(x + w/2, alone_p99, w, label="alone p99", color="#1e40af")
    ax1.set_xticks(x); ax1.set_xticklabels(sizes)
    ax1.set_ylabel("L1 frame time (ms)")
    ax1.set_title("L1 alone — partition size slope")
    ax1.legend()
    for i, (a, b) in enumerate(zip(alone_p50, alone_p99)):
        ax1.text(i - w/2, a+0.3, f"{a:.1f}", ha="center", fontsize=9)
        ax1.text(i + w/2, b+0.3, f"{b:.1f}", ha="center", fontsize=9)
    ax1.set_ylim(0, max(alone_p99)*1.25)

    ax2.bar(x - w/2, coloc_p50, w, label="coloc p50", color="#f87171")
    ax2.bar(x + w/2, coloc_p99, w, label="coloc p99", color="#991b1b")
    ax2.set_xticks(x); ax2.set_xticklabels(sizes)
    ax2.set_ylabel("L1 frame time (ms)")
    ax2.set_title("L1 + NeuralRx same-partition coloc — FLAT ~360ms")
    ax2.legend(loc="lower left")
    for i, (a, b) in enumerate(zip(coloc_p50, coloc_p99)):
        ax2.text(i - w/2, a+3, f"{a:.0f}", ha="center", fontsize=9)
        ax2.text(i + w/2, b+3, f"{b:.0f}", ha="center", fontsize=9)
    ax2.set_ylim(0, max(coloc_p99)*1.15)

    plt.suptitle("Partition size matters for alone (1.5x); doesn't matter under coloc",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t01_partition_alone_vs_coloc.png", dpi=140, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------
# Figure 2: cross-partition matrix — single AI + stacking all flat
# --------------------------------------------------------------------------
def fig_t02():
    conds = [
        ("alone (3g)",           f"{ROOT}/E0_baseline_3g/realL1_*.json"),
        ("+ chanpred",           f"{ROOT}/E2_chanpred/realL1_*.json"),
        ("+ NeuralRx",           f"{ROOT}/E1_neuralrx/realL1_*.json"),
        ("+ xapp",               f"{ROOT}/E4_misc/realL1_E4_xapp_*.json"),
        ("+ sat_compute",        f"{ROOT}/E4_misc/realL1_E4_sat_compute_*.json"),
        ("+ sat_hbm",            f"{ROOT}/E4_misc/realL1_E4_sat_hbm_*.json"),
        ("+ chanpred ×4",        f"{ROOT}/A1_stacking/chanpred_x4/realL1_*.json"),
        ("+ ResNet ×2",          f"{ROOT}/A1_stacking/resnet_x2/realL1_*.json"),
        ("+ kitchen (cp+mc+gm)", f"{ROOT}/A1_stacking/kitchen/realL1_*.json"),
    ]
    labels, p50s, p99s = [], [], []
    for label, pat in conds:
        s = stats(load_ms(pat))
        if s is None: continue
        labels.append(label); p50s.append(s["p50"]); p99s.append(s["p99"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, p50s, w, label="p50", color="#10b981")
    ax.bar(x + w/2, p99s, w, label="p99", color="#047857")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("L1 frame time (ms)")
    ax.set_title("MIG cross-partition: every workload, every stack — L1 stays ~40-50ms")
    ax.legend()
    ax.axhline(50, ls="--", color="grey", alpha=0.5, label="50ms reference")
    for i, (a, b) in enumerate(zip(p50s, p99s)):
        ax.text(i - w/2, a+0.5, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, b+0.5, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(p99s)*1.25)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t02_cross_partition_matrix.png", dpi=140, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------
# Figure 3: same-partition coloc workload-invariance
# --------------------------------------------------------------------------
def fig_t03():
    conds = [
        ("chanpred (E3)",                     f"{ROOT}/E3_coloc/realL1_*.json"),
        ("NeuralRx (E6-3g)",                  f"{ROOT}/E6_coloc_neuralrx/3g/realL1_*.json"),
        ("chanpred+ResNet mixed (A2)",        f"{ROOT}/A2_mixed_coloc/chanpred_resnet/realL1_*.json"),
    ]
    labels, p50s, p99s, maxs = [], [], [], []
    for label, pat in conds:
        s = stats(load_ms(pat))
        if s is None: continue
        labels.append(label); p50s.append(s["p50"]); p99s.append(s["p99"]); maxs.append(s["max"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, p50s, w, label="p50", color="#fca5a5")
    ax.bar(x,     p99s, w, label="p99", color="#dc2626")
    ax.bar(x + w, maxs, w, label="max", color="#7f1d1d")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("L1 frame time (ms)")
    ax.set_title("Same-partition coloc on 3g — workload doesn't matter, always ~360ms")
    ax.legend()
    for i, (a, b, c) in enumerate(zip(p50s, p99s, maxs)):
        ax.text(i - w, a+3, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i,     b+3, f"{b:.0f}", ha="center", fontsize=9)
        ax.text(i + w, c+3, f"{c:.0f}", ha="center", fontsize=9)
    ax.set_ylim(0, max(maxs)*1.15)
    ax.axhline(43, ls="--", color="grey", alpha=0.5)
    ax.text(0.02, 0.08, "alone baseline ~43ms", color="grey",
            transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t03_coloc_workload_invariant.png", dpi=140, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------
# Figure 4: Three-regime CDF
# --------------------------------------------------------------------------
def fig_t04():
    sources = [
        ("alone (3g)",          load_ms(f"{ROOT}/E0_baseline_3g/realL1_*.json"), "#10b981", "-"),
        ("cross-part + NeuralRx", load_ms(f"{ROOT}/E1_neuralrx/realL1_*.json"), "#3b82f6", "-"),
        ("cross-part + chanpred×4", load_ms(f"{ROOT}/A1_stacking/chanpred_x4/realL1_*.json"), "#3b82f6", "--"),
        ("coloc 3g + chanpred", load_ms(f"{ROOT}/E3_coloc/realL1_*.json"), "#dc2626", "-"),
        ("coloc 3g + NeuralRx", load_ms(f"{ROOT}/E6_coloc_neuralrx/3g/realL1_*.json"), "#dc2626", "--"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, ms, color, ls in sources:
        if ms is None: continue
        ms_sorted = np.sort(ms)
        cdf = np.arange(1, len(ms_sorted)+1) / len(ms_sorted)
        ax.plot(ms_sorted, cdf, label=label, color=color, ls=ls, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("L1 frame time (ms, log scale)")
    ax.set_ylabel("CDF")
    ax.set_title("Three regimes — clean isolation, cross-partition, or catastrophic coloc")
    ax.set_xlim(30, 500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.axvline(43, ls=":", color="grey", alpha=0.7)
    ax.text(43, 0.05, " alone", fontsize=9, color="grey")
    ax.axvline(360, ls=":", color="grey", alpha=0.7)
    ax.text(360, 0.05, " coloc", fontsize=9, color="grey")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t04_three_regimes_cdf.png", dpi=140, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------
# Figure 5: 7g coloc = no-MIG equivalence (compare with Perlmutter)
# --------------------------------------------------------------------------
def fig_t05():
    # Local CloudLab MIG data
    e5_7g = stats(load_ms(f"{ROOT}/E5_alone_partition/7g/realL1_*.json"))
    e6_7g = stats(load_ms(f"{ROOT}/E6_coloc_neuralrx/7g/realL1_*.json"))
    e6_3g = stats(load_ms(f"{ROOT}/E6_coloc_neuralrx/3g/realL1_*.json"))

    # Perlmutter no-MIG reference (from PART F: alone=124ms p99, NeuralRx=389ms p99)
    perl_alone_p99 = 124.0
    perl_nrx_p99 = 389.0

    conds = [
        ("CloudLab\n7g alone\n(full GPU isolated)", e5_7g["p99"], "#10b981"),
        ("Perlmutter\nno-MIG alone\n(no MIG, time-slice)", perl_alone_p99, "#3b82f6"),
        ("CloudLab\n3g coloc + NeuralRx\n(same MIG partition)", e6_3g["p99"], "#f59e0b"),
        ("CloudLab\n7g coloc + NeuralRx\n(full GPU, 2 processes)", e6_7g["p99"], "#dc2626"),
        ("Perlmutter\nno-MIG + NeuralRx\n(no MIG, 2 processes)", perl_nrx_p99, "#7f1d1d"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(conds))
    colors = [c[2] for c in conds]
    vals = [c[1] for c in conds]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v+8, f"{v:.0f} ms", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds], fontsize=9)
    ax.set_ylabel("L1 p99 frame time (ms)")
    ax.set_title("7g coloc (full GPU + 2 processes) ≈ no-MIG NeuralRx — both \"shared context contention floor\"")
    ax.set_ylim(0, max(vals)*1.2)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t05_7g_coloc_equals_no_mig.png", dpi=140, bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------
# Figure 6: NCU DRAM throughput alone vs coloc
# --------------------------------------------------------------------------
def fig_t06():
    import csv
    def parse_ncu_csv(path, metric):
        """NCU csv has many rows per kernel; return mean of given metric across kernels."""
        vals = []
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Metric Name", "")
                    val = row.get("Metric Value", "")
                    if metric in name and val:
                        try:
                            v = float(val.replace(",", ""))
                            vals.append(v)
                        except ValueError:
                            pass
        except Exception:
            pass
        return vals

    alone_dram = parse_ncu_csv(f"{ROOT}/C1_ncu/alone.csv", "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    coloc_dram = parse_ncu_csv(f"{ROOT}/C1_ncu/coloc_nrx.csv", "dram__throughput.avg.pct_of_peak_sustained_elapsed")
    alone_l2   = parse_ncu_csv(f"{ROOT}/C1_ncu/alone.csv", "lts__t_sector_hit_rate")
    coloc_l2   = parse_ncu_csv(f"{ROOT}/C1_ncu/coloc_nrx.csv", "lts__t_sector_hit_rate")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if alone_dram and coloc_dram:
        # box plot
        ax1.boxplot([alone_dram, coloc_dram], tick_labels=["alone (3g)", "coloc + NeuralRx"])
        ax1.set_ylabel("DRAM throughput (% peak sustained)")
        ax1.set_title("DRAM throughput per kernel (NCU replay)")
        ax1.text(1, max(alone_dram)*0.95, f"n={len(alone_dram)} kernels", ha="center", fontsize=9, color="grey")
        ax1.text(2, max(coloc_dram)*0.95, f"n={len(coloc_dram)} kernels", ha="center", fontsize=9, color="grey")

    if alone_l2 and coloc_l2:
        ax2.boxplot([alone_l2, coloc_l2], tick_labels=["alone (3g)", "coloc + NeuralRx"])
        ax2.set_ylabel("L2 hit rate (%)")
        ax2.set_title("L2 sector hit rate per kernel")

    plt.suptitle("NCU per-kernel — alone vs +NeuralRx coloc (3g MIG)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig_t06_ncu_dram_alone_vs_coloc.png", dpi=140, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Building figures...")
    fig_t01(); print(" t01 partition alone vs coloc")
    fig_t02(); print(" t02 cross-partition matrix")
    fig_t03(); print(" t03 coloc workload-invariance")
    fig_t04(); print(" t04 three-regime CDF")
    fig_t05(); print(" t05 7g coloc = no-MIG equivalence")
    fig_t06(); print(" t06 NCU DRAM alone vs coloc")
    print("Done — figures under:", FIGDIR)
