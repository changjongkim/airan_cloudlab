"""Generate all visualization charts from v1 (light) + v2-v5 (heavy) data.

Critical-framing charts showing MIG limitations.
Outputs PNGs into charts/ subdir. Uses matplotlib + numpy only.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Critical color palette
C_NOMIG = "#d62728"    # red — no MIG
C_MIG_50 = "#2ca02c"   # green — symmetric (best case)
C_MIG_ASYM = "#ff7f0e" # orange — asymmetric (leakage)
C_MIG_SMALL = "#9467bd" # purple — small L1 partition
C_BASELINE = "#1f77b4" # blue — L1 alone

# ============================================================================
# v1 LIGHT WORKLOAD CHARTS
# ============================================================================

def chart_v1_celling_4mig():
    """Cell-count scaling at 4 MIG presets + no-mig baseline (v1, GPT-2)."""
    cells = [1, 4, 10, 20, 40]
    nomig    = [1.97,  6.79, 189.32, 375.76, 747.92]
    s_5050   = [2.18,  8.37,  17.29,  34.64,  68.96]
    s_6040   = [1.87,  7.02,  17.39,  40.84,  69.08]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # linear
    ax = axes[0]
    ax.plot(cells, nomig,  "o-", color=C_NOMIG, label="no-mig (shared GPU)", lw=2, ms=8)
    ax.plot(cells, s_5050, "s-", color=C_MIG_50, label="MIG split-50-50", lw=2, ms=8)
    ax.plot(cells, s_6040, "^-", color=C_MIG_ASYM, label="MIG split-60-40", lw=2, ms=8)
    ax.set_xlabel("Number of cells (L1 load)")
    ax.set_ylabel("L1 mean latency (ms)")
    ax.set_title("(linear) Cell-count scaling under GPT-2 interference")
    ax.legend()
    ax.set_xticks(cells)

    # log
    ax = axes[1]
    ax.plot(cells, nomig,  "o-", color=C_NOMIG, label="no-mig", lw=2, ms=8)
    ax.plot(cells, s_5050, "s-", color=C_MIG_50, label="MIG split-50-50", lw=2, ms=8)
    ax.plot(cells, s_6040, "^-", color=C_MIG_ASYM, label="MIG split-60-40", lw=2, ms=8)
    ax.set_yscale("log"); ax.set_xticks(cells)
    ax.set_xlabel("Number of cells"); ax.set_ylabel("L1 mean (ms, log)")
    ax.set_title("(log) Saturation point visible at cells≥10 for no-mig")
    ax.annotate("HBM BW saturation\n→ super-linear",
                xy=(10, 189), xytext=(15, 60),
                arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
    ax.legend()

    fig.suptitle("v1 light AI (GPT-2 124M): MIG linear vs no-mig polynomial explosion", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_01_cell_scaling_4presets.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_ai_type():
    """v1 split-50-50 cells=20: AI workload type comparison."""
    # Aggregated from 47 cells=20 runs
    ai_types = ["GPT-2 124M", "ResNet-50 bs=16", "HBM stress 1GB"]
    means = [41.19, 41.19, 34.53]  # representative runs
    p99s = [42.63, 42.26, 35.43]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ai_types)); w = 0.35
    ax.bar(x - w/2, means, w, label="mean", color=C_MIG_50)
    ax.bar(x + w/2, p99s, w, label="p99", color="#1abc9c")
    ax.set_xticks(x); ax.set_xticklabels(ai_types)
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("v1 split-50-50 cells=20: AI workload type comparison\n"
                 "(All AI types give similar L1 latency under MIG — but only because v1 workloads are too small)")
    ax.legend()
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.text(i - w/2, m + 1, f"{m:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, p + 1, f"{p:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_02_ai_type_compare.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_hbm_sweep():
    """v1: HBM stress alloc sweep (0.5-16GB) at split-50-50."""
    hbm = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    l1 = [40.83, 39.20, 39.10, 40.83, 41.25, 41.16]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hbm, l1, "o-", color=C_MIG_50, lw=2, ms=10)
    ax.set_xscale("log")
    ax.set_xticks(hbm); ax.set_xticklabels([str(x) for x in hbm])
    ax.set_xlabel("HBM stress alloc (GB, log scale)")
    ax.set_ylabel("L1 mean latency (ms)")
    ax.set_ylim(35, 45)
    ax.set_title("v1: L1 latency stable across 32× HBM stress sweep\n"
                 "BUT: this looks great because workload is small enough that AI HBM access doesn't conflict")
    for x, y in zip(hbm, l1):
        ax.text(x, y + 0.5, f"{y:.1f}", ha="center", fontsize=9)
    avg = np.mean(l1)
    ax.axhline(avg, color="gray", ls="--", alpha=0.5, label=f"avg {avg:.2f} ms (±{max(l1)-min(l1):.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_03_hbm_intensity.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_resnet_batch():
    """v1: ResNet batch sweep at split-50-50."""
    bs = [8, 16, 32, 64]
    l1 = [34.65, 34.52, 41.28, 41.33]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(bs)), l1, color=C_MIG_50, alpha=0.8)
    for i, (b, y) in enumerate(zip(bs, l1)):
        ax.text(i, y + 0.5, f"{y:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(bs))); ax.set_xticklabels([f"bs={b}" for b in bs])
    ax.set_xlabel("ResNet batch size"); ax.set_ylabel("L1 mean (ms)")
    ax.set_title("v1: ResNet batch 8× sweep — L1 jumps slightly at bs=32+\n"
                 "But still in 34-41ms range. MIG \"isolates\" but range still bimodal even within one AI workload.")
    ax.set_ylim(30, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_04_resnet_batch.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_prb_sweep():
    """v1: PRB allocation sweep at split-50-50 + GPT-2."""
    prbs = [51, 106, 217, 273]
    l1 = [34.91, 34.34, 32.62, 34.45]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(prbs, l1, "o-", color=C_MIG_50, lw=2, ms=10)
    ax.set_xlabel("Number of PRBs (L1 frequency-domain load)")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_ylim(30, 40)
    ax.set_title("v1: PRB sweep 51→273 — L1 nearly flat (33-35 ms)\n"
                 "cuPHY fixed-overhead dominates; PRB allocation barely affects mean")
    for x, y in zip(prbs, l1):
        ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_05_prb_sweep.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_antenna_sweep():
    """v1: MIMO antenna sweep at split-50-50."""
    cfgs = ["2T2R", "4T4R", "8T8R"]
    l1 = [31.84, 41.29, 46.27]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(cfgs, l1, color=[C_MIG_50, C_BASELINE, C_MIG_ASYM])
    for i, y in enumerate(l1):
        ax.text(i, y + 0.5, f"{y:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("MIMO config"); ax.set_ylabel("L1 mean (ms)")
    ax.set_ylim(25, 55)
    ax.set_title("v1: Antenna scaling at split-50-50\n"
                 "8T8R is 1.5× slower — channel estimation work scales with antenna count")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_06_antenna_sweep.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_mcs_sweep():
    """v1: MCS sweep at split-50-50."""
    mcs = [0, 2, 7, 16, 24]
    l1 = [34.62, 39.25, 34.57, 39.19, 40.82]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(mcs, l1, "o-", color=C_MIG_50, lw=2, ms=10)
    ax.set_xlabel("MCS index (LDPC intensity)")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("v1: MCS sweep — L1 latency irregular (34-41ms)\n"
                 "Non-monotonic; suggests measurement variance dominates over MCS effect")
    for x, y in zip(mcs, l1):
        ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=9)
    ax.set_ylim(30, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_07_mcs_sweep.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_multimodal():
    """v1 split-50-50 cells=20 all runs - shows multimodal distribution.

    Aggregated from all 47 cells=20 runs, sorted by mean. Shows clusters.
    """
    # data from CSV inspection
    means = [41.187, 34.533, 34.696, 41.192, 34.639, 40.833, 39.196, 39.101, 40.831, 41.247,
             41.161, 34.648, 34.517, 41.275, 41.328, 34.912, 34.344, 32.622, 34.455, 31.837,
             41.291, 46.267, 41.129, 39.490, 34.624, 39.248, 34.574, 39.187, 40.824]
    ais = ["gpt2","hbm","hbm","resnet","gpt2","hbm","hbm","hbm","hbm","hbm",
           "hbm","resnet","resnet","resnet","resnet","gpt2","gpt2","gpt2","gpt2","gpt2",
           "gpt2","gpt2","gpt2","gpt2","gpt2","gpt2","gpt2","gpt2","gpt2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # histogram
    ax = axes[0]
    ax.hist(means, bins=15, color=C_MIG_50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("L1 mean latency (ms)")
    ax.set_ylabel("Run count")
    ax.set_title("v1 split-50-50 cells=20: histogram of 29 sub-sweep runs\nMulti-modal distribution visible — MIG isolation is NOT deterministic")
    # markers for clusters
    for cluster_center, name in [(32, "cluster A"), (35, "cluster B"), (40, "cluster C"), (46, "outlier")]:
        if cluster_center > min(means)-1 and cluster_center < max(means)+1:
            ax.axvline(cluster_center, color="red", ls=":", alpha=0.4)

    # strip plot by AI
    ax = axes[1]
    color_map = {"gpt2": "#1f77b4", "hbm": "#ff7f0e", "resnet": "#2ca02c"}
    for i, (m, a) in enumerate(zip(means, ais)):
        ax.scatter([i], [m], color=color_map[a], s=70, alpha=0.8, edgecolor="black", lw=0.5)
    # legend
    for a, c in color_map.items():
        ax.scatter([], [], color=c, label=a)
    ax.set_xlabel("Run index (sorted by time)")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("Strip plot: same preset, same cells, different sub-params\nspread 31.8 → 46.3 ms (45% variance)")
    ax.legend(title="AI workload")

    fig.suptitle("MIG split-50-50 is NOT a single point — wide spread across configs",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_08_multimodal.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_oom_failures():
    """v1: 14 OOM failures all on 1g.5gb partition."""
    presets = ["split-20-80\n(L1=1g.5gb)", "four-way-eq\n(L1=1g.5gb)", "seven-1g\n(L1=1g.5gb)"]
    counts = [11, 1, 2]   # OOM occurrence counts in v1
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(presets, counts, color=C_NOMIG, alpha=0.85)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 0.2, f"{c} runs", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("OOM failure count")
    ax.set_title("v1: ALL 14 failures = L1 OOM on 1g.5gb partition\n"
                 "cuPHY LdpcDeRateMatch alone needs >5GB HBM → MIG smallest slice unusable")
    ax.set_ylim(0, max(counts) + 2)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_09_oom_1g5gb.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_stability():
    """v1: Long-duration stability at split-50-50 + GPT-2."""
    durations = ["50 iters\n(~2s)", "300 iters\n(~12s)", "1000 iters\n(~40s)"]
    means = [40.82, 39.49, 41.13]
    p99s  = [42.11, 41.39, 41.36]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(durations)); w = 0.35
    ax.bar(x - w/2, means, w, label="mean", color=C_MIG_50)
    ax.bar(x + w/2, p99s, w, label="p99", color="#1abc9c")
    ax.set_xticks(x); ax.set_xticklabels(durations)
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("v1: Stability across run durations (split-50-50 + GPT-2)\n"
                 "Variation <2% across 20× duration range → MIG ISO stable in time")
    ax.legend()
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.text(i - w/2, m + 0.3, f"{m:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, p + 0.3, f"{p:.1f}", ha="center", fontsize=9)
    ax.set_ylim(35, 45)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_10_stability.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v1_partition_at_20cells():
    """v1: L1 latency by L1 partition size (cells=20, with GPT-2 AI)."""
    parts = ["1g.5gb\n(L1 alone)\n[OOM]", "2g.10gb\n(split-40-60\nL1 part)", "3g.20gb\n(split-50-50)",
             "3g.20gb\n(split-60-40\nL1 part)", "full GPU\n(no-mig)"]
    means = [None, 47.14, 41.19, 34.47, 375.76]  # gpt2 cases (representative)
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(parts))
    bars = ax.bar(xs, [v if v else 0 for v in means],
                  color=[C_NOMIG if v is None else (C_NOMIG if v > 100 else C_MIG_50) for v in means])
    for i, (b, v) in enumerate(zip(bars, means)):
        if v is None:
            ax.text(i, 5, "OOM", ha="center", fontsize=10, color="red", fontweight="bold")
        else:
            ax.text(i, v + 8, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(parts, fontsize=9)
    ax.set_ylabel("L1 mean (ms, GPT-2 concurrent)")
    ax.set_title("v1: L1 latency by partition size — note 9× jump to no-mig\n"
                 "But this 9× is SM contention from light AI, not HBM bandwidth (see v2 for correction)")
    ax.set_ylim(0, 420)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v1_11_partition_v1.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# v2-v5 HEAVY WORKLOAD CHARTS
# ============================================================================

def chart_v2_partition_baselines():
    """v2: L1 alone baselines on each partition size (8T8R, no AI)."""
    parts = ["1g.5gb\n(OOM)", "2g.10gb (B2)", "3g.20gb (B)\n[sweet spot]", "4g.20gb (B4)", "full GPU (A)\n[not measured]"]
    lat = [None, 59.27, 46.14, 52.77, None]
    sm = ["1", "2", "3", "4", "7"]
    hbm_bw = ["1/8", "2/8", "4/8", "4/8", "8/8"]
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(parts))
    bars = ax.bar(xs, [v or 0 for v in lat],
                  color=[C_NOMIG if v is None else (C_MIG_50 if v == 46.14 else C_BASELINE) for v in lat])
    for i, v in enumerate(lat):
        if v is None:
            label = "OOM" if i == 0 else "driver bug"
            ax.text(i, 5, label, ha="center", fontsize=10, color="red", fontweight="bold")
        else:
            ax.text(i, v + 1.5, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i, -7, f"SM={sm[i]}\nHBM BW={hbm_bw[i]}", ha="center", fontsize=8, color="gray")
    ax.set_xticks(xs); ax.set_xticklabels(parts, fontsize=9)
    ax.set_ylim(-12, 75)
    ax.set_ylabel("L1 alone latency (ms, 8T8R 20 cells, no AI)")
    ax.set_title("v2 baselines: L1 by partition — cuPHY is HBM-bound\n"
                 "3g.20gb is OPTIMAL (50% HBM BW saturates 8T8R 20-cell); 4g.20gb (same BW) gives NO speedup")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_01_partition_baselines.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v2_split_with_qwen():
    """v2: Effect of different MIG splits with Qwen-7B."""
    configs = ["B (alone)\n3g.20gb", "split-40-60\nL1=2g.10gb", "split-50-50\nL1=3g.20gb", "split-60-40\nL1=3g.20gb"]
    means = [46.14, 66.55, 46.47, 49.7]  # split-60-40 avg of 4 (52.80+46.67+46.53+52.86)/4 = 49.71
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [C_BASELINE, C_MIG_SMALL, C_MIG_50, C_MIG_ASYM]
    ax.bar(np.arange(len(configs)), means, color=colors)
    for i, v in enumerate(means):
        ax.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(len(configs))); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("v2: MIG splits with heavy AI (Qwen-7B)\n"
                 "small L1 partition (2g) penalizes; symmetric 50:50 best; 60:40 has bimodal leakage (avg shown)")
    ax.set_ylim(40, 75)
    # callout
    ax.annotate("Small L1\npartition cap",
                xy=(1, 66.55), xytext=(0.6, 72),
                arrowprops=dict(arrowstyle="->", color=C_MIG_SMALL),
                fontsize=9, color=C_MIG_SMALL)
    ax.annotate("Bimodal 50%/50%\nintermittent leak",
                xy=(3, 49.7), xytext=(2.3, 60),
                arrowprops=dict(arrowstyle="->", color=C_MIG_ASYM),
                fontsize=9, color=C_MIG_ASYM)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_02_splits_with_qwen.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v2_bimodal_detail():
    """v2: split-60-40 + Qwen N=4 bimodal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    runs_labels = ["Run 1\n(orig)", "Run 2\n(repro)", "Run 3\n(repro)", "Run 4\n(repro)"]
    means = [52.80, 46.67, 46.53, 52.86]
    p99s  = [53.84, 47.98, 48.09, 54.26]
    colors = [C_NOMIG if m > 50 else C_MIG_50 for m in means]
    ax = axes[0]
    xs = np.arange(len(runs_labels))
    ax.bar(xs, means, color=colors, alpha=0.85)
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.text(i, m + 0.4, f"{m:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(46.14, color="black", ls="--", alpha=0.5, label="B baseline (46.14)")
    ax.set_xticks(xs); ax.set_xticklabels(runs_labels)
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("split-60-40 + Qwen-7B: 4 sequential measurements\nBimodal: 50% HIGH (+6.7ms), 50% LOW")
    ax.set_ylim(40, 60)
    ax.legend(loc="lower right", fontsize=9)

    ax = axes[1]
    low_mode = [m for m in means if m < 50]
    high_mode = [m for m in means if m >= 50]
    ax.bar(["LOW\n(baseline)", "HIGH\n(+6.7 ms)"],
           [len(low_mode), len(high_mode)],
           color=[C_MIG_50, C_NOMIG])
    ax.set_ylabel("Run count (N=4)")
    ax.set_title("Bimodal: 50:50 across N=4\n(small N — needs N>=20 to confirm)")
    for i, c in enumerate([len(low_mode), len(high_mode)]):
        ax.text(i, c + 0.05, str(c), ha="center", fontsize=12, fontweight="bold")
    fig.suptitle("CRITICAL FINDING: MIG asymmetric split has INTERMITTENT leakage",
                 y=1.02, fontsize=12, fontweight="bold", color=C_NOMIG)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_03_bimodal_detail.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v2_mig_vs_nomig_heavy():
    """v2: MIG vs no-mig heavy workload (the key chart)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    workloads = ["Qwen-7B", "HBM 16GB"]
    no_mig_mean = [57.89, 48.03]
    no_mig_p99 = [172.60, 49.91]
    mig_mean = [46.47, 46.55]
    mig_p99 = [48.23, 49.26]
    x = np.arange(len(workloads)); w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, no_mig_mean, w, label="no-mig", color=C_NOMIG)
    ax.bar(x + w/2, mig_mean, w, label="MIG split-50-50", color=C_MIG_50)
    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("mean: only ~1.24× advantage\n→ MIG marginal benefit on mean")
    ax.legend()
    for i, (n, m) in enumerate(zip(no_mig_mean, mig_mean)):
        ax.text(i - w/2, n + 1, f"{n:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, m + 1, f"{m:.1f}", ha="center", fontsize=9)

    ax = axes[1]
    ax.bar(x - w/2, no_mig_p99, w, label="no-mig", color=C_NOMIG)
    ax.bar(x + w/2, mig_p99, w, label="MIG split-50-50", color=C_MIG_50)
    ax.set_xticks(x); ax.set_xticklabels(workloads)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("p99: 3.6× advantage\n→ MIG's ONLY decisive benefit is tail latency")
    ax.legend()
    for i, (n, m) in enumerate(zip(no_mig_p99, mig_p99)):
        ax.text(i - w/2, n + 4, f"{n:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w/2, m + 4, f"{m:.1f}", ha="center", fontsize=9)
    ax.annotate("Qwen p99 spike", xy=(0 + w/2 - 0.18, 172), xytext=(0.6, 130),
                arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")

    fig.suptitle("v2 heavy AI: MIG mean benefit is small. Only p99 (jitter) is dramatically different.", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_04_mig_vs_nomig_heavy.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v2_percentile_fan():
    """v2 percentile distribution fan."""
    configs = ["no-mig\n+ Qwen", "MIG 50:50\n+ Qwen", "no-mig\n+ HBM 16GB", "MIG 50:50\n+ HBM 16GB"]
    p50 = [46.97, 46.44, 47.95, 46.44]
    p95 = [78.60, 46.82, 48.13, 46.53]
    p99 = [172.60, 48.29, 49.91, 49.26]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(configs)); w = 0.25
    ax.bar(x - w, p50, w, label="p50 (median)", color="#4daf4a")
    ax.bar(x, p95, w, label="p95", color="#ff7f00")
    ax.bar(x + w, p99, w, label="p99 (worst 1%)", color="#e41a1c")
    ax.set_xticks(x); ax.set_xticklabels(configs)
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("v2 heavy: p50/p95/p99 percentile fan\nno-mig + Qwen has p99 jitter spike. MIG 50:50 keeps all percentiles tight.")
    ax.legend()
    for i, (a, b, c) in enumerate(zip(p50, p95, p99)):
        ax.text(i - w, a + 3, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i, b + 3, f"{b:.0f}", ha="center", fontsize=8)
        ax.text(i + w, c + 3, f"{c:.0f}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_05_percentile_fan.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# v2 COVERAGE / MISSING DATA CHARTS
# ============================================================================

def chart_v2_coverage():
    """Coverage matrix of v2 experiments (what we have vs what we don't)."""
    presets = ["split-20-80\n(L1=1g.5gb)", "split-40-60\n(L1=2g.10gb)",
               "split-50-50\n(L1=3g.20gb)", "split-60-40\n(L1=3g.20gb)",
               "no-mig\n(shared GPU)", "B baseline\n(L1 alone)"]
    workloads = ["none\n(L1 alone)", "Qwen-7B", "HBM 16GB", "ResNet-50", "Multi-AI mix"]

    # rows = presets, cols = workloads; values = number of measurements (0 = no data)
    coverage = np.array([
        # none, Qwen, HBM16, ResNet, MultiAI
        [0,    0,    0,    0,    0    ],  # split-20-80 (always OOM)
        [0,    1,    0,    0,    0    ],  # split-40-60 (only 1 Qwen run)
        [1,    4,    1,    0,    0    ],  # split-50-50 (B + 3 Qwen + 1 HBM + cell=1)
        [0,    4,    0,    0,    0    ],  # split-60-40 (4 Qwen runs — bimodal)
        [0,    1,    1,    0,    0    ],  # no-mig (Qwen + HBM)
        [3,    0,    0,    0,    0    ],  # B baselines (B + B2 + B4)
    ])

    fig, ax = plt.subplots(figsize=(11, 6))
    # use red for 0, green gradient for >=1
    cmap = plt.cm.RdYlGn
    im = ax.imshow(coverage, cmap=cmap, vmin=0, vmax=4, aspect="auto")
    ax.set_xticks(np.arange(len(workloads)))
    ax.set_xticklabels(workloads)
    ax.set_yticks(np.arange(len(presets)))
    ax.set_yticklabels(presets)
    for i in range(len(presets)):
        for j in range(len(workloads)):
            v = coverage[i, j]
            label = "—" if v == 0 else f"N={v}"
            color = "white" if v == 0 else ("white" if v >= 3 else "black")
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    ax.set_title("v2 heavy workload data coverage — most cells are EMPTY\n"
                 "We over-sampled split-50-50; under-sampled asymmetric and other AI types",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, label="number of measurements")
    cbar.set_ticks([0, 1, 2, 3, 4])
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_06_coverage_matrix.png"), bbox_inches="tight")
    plt.close(fig)


def chart_v2_all_splits():
    """Every v2 split measurement, with N annotations and error bars."""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # categories: each measurement family
    cats = [
        ("L1 alone\n2g.10gb (B2)",  [59.27],                       "L1 alone",  C_MIG_SMALL),
        ("L1 alone\n3g.20gb (B)",   [46.142],                      "L1 alone",  C_BASELINE),
        ("L1 alone\n4g.20gb (B4)",  [52.77],                       "L1 alone",  C_MIG_ASYM),
        ("split-40-60\n+ Qwen",     [66.555],                      "with Qwen", C_MIG_SMALL),
        ("split-50-50\n+ Qwen",     [46.335, 46.529, 46.53],       "with Qwen", C_MIG_50),
        ("split-50-50\n+ HBM 16G",  [46.555],                      "with HBM",  C_MIG_50),
        ("split-60-40\n+ Qwen",     [52.799, 46.67, 46.53, 52.86], "with Qwen", C_MIG_ASYM),
        ("no-mig\n+ Qwen",          [57.89],                       "with Qwen", C_NOMIG),
        ("no-mig\n+ HBM 16G",       [48.03],                       "with HBM",  C_NOMIG),
    ]
    xs = np.arange(len(cats))

    # bar chart of mean ± range
    for i, (label, vals, _, color) in enumerate(cats):
        avg = np.mean(vals)
        ax.bar(i, avg, color=color, alpha=0.85)
        # individual points as scatter
        for v in vals:
            ax.scatter(i, v, color="black", s=40, zorder=5, alpha=0.7)
        # N label
        ax.text(i, avg + 1.5, f"N={len(vals)}", ha="center", fontsize=9, color="black")
        # value label
        ax.text(i, avg / 2, f"{avg:.1f}", ha="center", color="white", fontsize=9, fontweight="bold")
        # for split-60-40 mark bimodal
        if "60-40" in label and "Qwen" in label:
            ax.annotate("BIMODAL!\n46.5 / 52.8", xy=(i, 60), xytext=(i + 0.6, 75),
                        arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cats], fontsize=8)
    ax.set_ylabel("L1 mean (ms)")
    ax.set_ylim(0, 85)
    ax.set_title("ALL v2 measurements (every datapoint shown)\n"
                 "split-40-60 has only N=1; split-60-40 has N=4 with bimodal spread")
    # Horizontal line at B baseline
    ax.axhline(46.14, color="black", ls=":", alpha=0.4, label="B baseline 46.14")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "v2_07_all_splits.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# v1 vs v2 CRITICAL COMPARISON
# ============================================================================

def chart_v1_vs_v2_comparison():
    """The central critical chart: v1 light vs v2 heavy MIG advantage."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: same metric (mean isolation ratio = no-mig / MIG)
    workloads_v1 = ["GPT-2", "HBM 1GB", "HBM 8GB", "ResNet"]
    nomig_v1 = [375.74, 398.85, 398.48, 379.67]
    mig_v1 = [41.19, 34.53, 34.70, 41.19]
    ratio_v1 = [n/m for n, m in zip(nomig_v1, mig_v1)]

    workloads_v2 = ["Qwen-7B", "HBM 16GB"]
    nomig_v2 = [57.89, 48.03]
    mig_v2 = [46.47, 46.55]
    ratio_v2 = [n/m for n, m in zip(nomig_v2, mig_v2)]

    ax = axes[0]
    x1 = np.arange(len(workloads_v1)); w = 0.8
    ax.bar(x1, ratio_v1, w, color=C_MIG_50, label="v1 light AI")
    x2 = np.arange(len(workloads_v1), len(workloads_v1) + len(workloads_v2))
    ax.bar(x2, ratio_v2, w, color=C_NOMIG, label="v2 heavy AI")
    ax.set_xticks(np.concatenate([x1, x2]))
    ax.set_xticklabels(workloads_v1 + workloads_v2, rotation=15)
    ax.set_ylabel("MIG mean isolation factor\n(no-mig mean / MIG mean)")
    ax.set_title("Mean isolation: v1 claimed 9-12×, v2 reveals only 1.2×")
    ax.legend()
    for i, r in enumerate(ratio_v1):
        ax.text(i, r + 0.3, f"{r:.1f}×", ha="center", fontsize=10, fontweight="bold", color=C_MIG_50)
    for i, r in enumerate(ratio_v2):
        ax.text(len(workloads_v1)+i, r + 0.3, f"{r:.2f}×", ha="center", fontsize=10, fontweight="bold", color=C_NOMIG)
    ax.axhline(1, color="black", ls="-", alpha=0.4)

    # Right: workload size comparison
    ax = axes[1]
    sizes = {
        "GPT-2 124M\n(v1)": 0.5,        # ~500 MB
        "ResNet-50\n(v1)": 0.1,         # ~100 MB
        "HBM 1GB stress\n(v1)": 1.0,
        "HBM 8GB stress\n(v1)": 8.0,
        "HBM 16GB stress\n(v2)": 16.0,
        "Qwen-7B fp16\n(v2)": 14.0,
    }
    names = list(sizes.keys())
    vals = list(sizes.values())
    colors = [C_MIG_50 if "v1" in n else C_NOMIG for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xscale("log")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v}GB", va="center", fontsize=9)
    ax.set_xlabel("AI model / stress size (GB, log)")
    ax.set_title("Workload size: v1 mostly tiny (≤1GB); v2 realistic (14-16GB)\nv1 didn't actually stress HBM — that's why MIG looked great")
    ax.invert_yaxis()
    fig.suptitle("Why v1 was misleading: workload size matters",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "X_v1_vs_v2_critical.png"), bbox_inches="tight")
    plt.close(fig)


def chart_mig_bottlenecks_summary():
    """Critical-framing summary chart: MIG limitations."""
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

    # (a) v1 vs v2 mean ratio
    ax = fig.add_subplot(gs[0, 0])
    wls = ["GPT-2\n(v1)", "Qwen-7B\n(v2)"]
    ratios = [375.74/41.19, 57.89/46.47]
    ax.bar(wls, ratios, color=[C_MIG_50, C_NOMIG])
    ax.axhline(1, color="black", ls="-", alpha=0.4)
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.3, f"{r:.1f}×", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("MIG isolation factor (mean)")
    ax.set_title("(a) v1 overclaim → v2 reality\n(9× → 1.2×)")

    # (b) Bimodal leakage
    ax = fig.add_subplot(gs[0, 1])
    runs = [1, 2, 3, 4]
    means = [52.80, 46.67, 46.53, 52.86]
    ax.bar(runs, means, color=[C_NOMIG if m > 50 else C_MIG_50 for m in means])
    ax.axhline(46.14, color="black", ls="--", alpha=0.5)
    ax.set_xticks(runs); ax.set_xlabel("repetition")
    ax.set_ylabel("L1 mean (ms)")
    ax.set_title("(b) Bimodal leakage @ split-60-40\nIntermittent, not consistent")
    ax.set_ylim(40, 60)
    for r, m in zip(runs, means):
        ax.text(r, m + 0.4, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")

    # (c) Partition size hard cap
    ax = fig.add_subplot(gs[0, 2])
    parts = ["1g\n(OOM)", "2g", "3g\nopt", "4g"]
    lats = [None, 59.27, 46.14, 52.77]
    ax.bar(parts, [v or 0 for v in lats],
           color=[C_NOMIG, C_BASELINE, C_MIG_50, C_BASELINE])
    for i, v in enumerate(lats):
        if v is None:
            ax.text(i, 5, "OOM", ha="center", fontsize=10, color="red", fontweight="bold")
        else:
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 75)
    ax.set_ylabel("L1 alone (ms)")
    ax.set_title("(c) Partition size hard cap\n4g same as 3g, 1g OOM")

    # (d) No-mig p99 spike
    ax = fig.add_subplot(gs[1, 0])
    cfgs = ["no-mig\nQwen", "MIG 50:50\nQwen"]
    p99 = [172.60, 48.29]
    mean = [57.89, 46.47]
    x = np.arange(len(cfgs)); w = 0.35
    ax.bar(x - w/2, mean, w, label="mean", color="#4daf4a")
    ax.bar(x + w/2, p99, w, label="p99", color="#e41a1c")
    ax.set_xticks(x); ax.set_xticklabels(cfgs)
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("(d) Real MIG value = p99 not mean")
    ax.legend(fontsize=8)
    for i, (m, p) in enumerate(zip(mean, p99)):
        ax.text(i - w/2, m + 5, f"{m:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, p + 5, f"{p:.0f}", ha="center", fontsize=8, fontweight="bold")

    # (e) Cell-count explosion vs linear
    ax = fig.add_subplot(gs[1, 1])
    cells = [1, 4, 10, 20, 40]
    ax.plot(cells, [1.97, 6.79, 189.32, 375.76, 747.92], "o-", color=C_NOMIG, label="no-mig")
    ax.plot(cells, [2.18, 8.37, 17.29, 34.64, 68.96], "s-", color=C_MIG_50, label="MIG 50:50")
    ax.set_yscale("log")
    ax.set_xlabel("cells"); ax.set_ylabel("L1 (ms, log)")
    ax.set_title("(e) MIG linear vs no-mig explode\n(v1 only — heavy AI needs retest)")
    ax.legend(fontsize=8)

    # (f) Verdict text
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    verdict = (
        "VERDICT: MIG is NOT optimal\n"
        "─────────────────────────\n"
        "Bottlenecks remain:\n\n"
        "  ✗ Heavy AI: mean benefit\n"
        "    only 1.2× (not 9×)\n\n"
        "  ✗ Asymmetric splits leak\n"
        "    intermittently (~50% time)\n\n"
        "  ✗ 1g.5gb partition unusable\n"
        "    (cuPHY OOM)\n\n"
        "  ✗ Beyond 3g.20gb gives no\n"
        "    speedup (HBM-BW capped)\n\n"
        "  ✗ Heavy AI cell-count untested\n\n"
        "  ✓ Only redeeming feature:\n"
        "    p99 jitter isolation (5G SLA)\n\n"
        "MIG ≠ replacement for AI-RAN\n"
        "orchestration. Software still\n"
        "needed."
    )
    ax.text(0.0, 1.0, verdict, ha="left", va="top",
            fontsize=10, family="monospace", transform=ax.transAxes,
            color="#333333")

    fig.suptitle("MIG limitations — a critical view of the data",
                 fontsize=14, fontweight="bold", y=1.0, color=C_NOMIG)
    fig.savefig(os.path.join(CHARTS, "X_critical_summary.png"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"Generating charts in {CHARTS}")
    print()
    print("--- v1 light workload (11 charts) ---")
    chart_v1_celling_4mig();         print("  v1_01_cell_scaling_4presets.png")
    chart_v1_ai_type();              print("  v1_02_ai_type_compare.png")
    chart_v1_hbm_sweep();            print("  v1_03_hbm_intensity.png")
    chart_v1_resnet_batch();         print("  v1_04_resnet_batch.png")
    chart_v1_prb_sweep();            print("  v1_05_prb_sweep.png")
    chart_v1_antenna_sweep();        print("  v1_06_antenna_sweep.png")
    chart_v1_mcs_sweep();            print("  v1_07_mcs_sweep.png")
    chart_v1_multimodal();           print("  v1_08_multimodal.png")
    chart_v1_oom_failures();         print("  v1_09_oom_1g5gb.png")
    chart_v1_stability();            print("  v1_10_stability.png")
    chart_v1_partition_at_20cells(); print("  v1_11_partition_v1.png")
    print()
    print("--- v2 heavy workload (7 charts) ---")
    chart_v2_partition_baselines();  print("  v2_01_partition_baselines.png")
    chart_v2_split_with_qwen();      print("  v2_02_splits_with_qwen.png")
    chart_v2_bimodal_detail();       print("  v2_03_bimodal_detail.png")
    chart_v2_mig_vs_nomig_heavy();   print("  v2_04_mig_vs_nomig_heavy.png")
    chart_v2_percentile_fan();       print("  v2_05_percentile_fan.png")
    chart_v2_coverage();             print("  v2_06_coverage_matrix.png")
    chart_v2_all_splits();           print("  v2_07_all_splits.png")
    print()
    print("--- Critical comparison (2 charts) ---")
    chart_v1_vs_v2_comparison();     print("  X_v1_vs_v2_critical.png")
    chart_mig_bottlenecks_summary(); print("  X_critical_summary.png")
    print()
    print("Done. 18 charts in", CHARTS)
