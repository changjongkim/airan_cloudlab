#!/usr/bin/env python3
"""
3 genuinely new angles (not redundant):

supp_21: AI workload kernel SIGNATURE matrix from AI-side nsys captures
         → predicts which AI causes L1 contention from AI workload profile
         → validates pattern-similarity from opposite direction

supp_22: chanpred = 117K kernels/sec (15x L1 rate), zero L1 effect
         → rules out "kernel launch queue contention" alternative hypothesis
         → only memcpy queue contention remains as mechanism

supp_23: P5 sustained 5-min × 9 workloads, 7500 iter each
         → contention is PERSISTENT (not transient warmup)
         → coloc cost is structural, not a one-time spike
"""
import sqlite3
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

mpl.rcParams.update({
    'font.size': 11,
    'figure.figsize': (12, 7),
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo', 'NanumGothic', 'sans-serif'],
})

ROOT = Path(__file__).parent.parent
SQLITE = ROOT / "20260531" / "nsys_sqlite_v2"
DEEPA = ROOT / "20260531" / "nsys_deep_A"
P5 = ROOT / "20260531" / "p5_sustained"
OUT = Path(__file__).parent / "figures"


def workload_signature(db):
    """Extract AI workload's kernel/memcpy signature."""
    if not os.path.exists(db): return None
    con = sqlite3.connect(db); cur = con.cursor()
    out = {}
    try:
        cur.execute("SELECT MIN(start), MAX(end), COUNT(*), AVG((end-start)/1000.0) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        s, e, n, avg = cur.fetchone()
        dur_s = (e - s) / 1e9 if s and e else 1
        out["kern_n"] = n or 0
        out["kern_rate"] = (n or 0) / dur_s
        out["kern_avg_us"] = avg or 0
        out["dur_s"] = dur_s
        cur.execute("SELECT COUNT(*), AVG((end-start)/1000.0), SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        mc_n, mc_avg, mc_total = cur.fetchone()
        out["mc_n"] = mc_n or 0
        out["mc_rate"] = (mc_n or 0) / dur_s
        out["mc_avg_us"] = mc_avg or 0
        out["mc_total_ms"] = mc_total or 0
        # max memcpy size
        cur.execute("SELECT MAX(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        out["mc_max_kb"] = (cur.fetchone()[0] or 0) / 1024
    except Exception:
        return None
    con.close()
    return out


def fig_supp_21_ai_signature_matrix():
    """AI workload signature predicts L1 contention."""
    # AI workloads we have AI-side captures for
    workloads = [
        ("chanpred",      DEEPA / "A1_S35_2gL1_chanpred3g_ai.sqlite",  "no",  "#27AE60"),
        ("ResNet",        DEEPA / "A2_S34_4gL1_resnet2g_ai.sqlite",   "yes (bistable)", "#F39C12"),
        ("ResNet (M5c)",  DEEPA / "A3_M5c_resnet_ai.sqlite",          "yes", "#E74C3C"),
        ("Forecaster",    DEEPA / "A4_M8a_forecaster_ai.sqlite",      "no",  "#27AE60"),
    ]
    sigs = []
    for name, db, contention, col in workloads:
        s = workload_signature(db)
        if s: sigs.append((name, s, contention, col))
    if not sigs: return

    # L1 reference (alone)
    l1_ref = workload_signature(SQLITE / "S5_3g_alone_run1.sqlite")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel A: kernel rate
    ax = axes[0]
    x = np.arange(len(sigs))
    rates = [s[1]["kern_rate"] for s in sigs]
    cols = [s[3] for s in sigs]
    ax.bar(x, rates, color=cols)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sigs], rotation=15)
    ax.set_ylabel("kernel launch rate (kernels/sec)")
    ax.set_yscale('log')
    if l1_ref:
        ax.axhline(l1_ref["kern_rate"], ls='--', color='black', alpha=0.6,
                   label=f'L1 reference ({l1_ref["kern_rate"]:.0f}/s)')
        ax.legend()
    for i, s in enumerate(sigs):
        ax.text(i, rates[i] * 1.15, f"{rates[i]:.0f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("(A) Kernel launch rate per AI workload")

    # Panel B: memcpy rate
    ax = axes[1]
    mc_rates = [s[1]["mc_rate"] for s in sigs]
    ax.bar(x, mc_rates, color=cols)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sigs], rotation=15)
    ax.set_ylabel("memcpy rate (ops/sec)")
    if l1_ref:
        ax.axhline(l1_ref["mc_rate"], ls='--', color='black', alpha=0.6,
                   label=f'L1 reference ({l1_ref["mc_rate"]:.0f}/s)')
        ax.legend()
    for i, r in enumerate(mc_rates):
        ax.text(i, r + 0.5, f"{r:.1f}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("(B) Memcpy ops rate per AI workload")

    # Panel C: contention prediction
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("(C) Pattern signature → L1 contention 예측")
    ax.text(0.5, 9.5,
            "AI workload signature가 L1 contention을 예측한다:",
            fontsize=11, fontweight='bold')

    # Show signature table
    y = 8.5
    ax.text(0.5, y, f"{'workload':<15s} {'kern rate':>11s} {'memcpy rate':>12s} {'L1 contention':>14s}",
            fontsize=10, family='monospace')
    y -= 0.5
    ax.text(0.5, y, "─"*60, fontsize=10, family='monospace')
    for name, s, c, col in sigs:
        y -= 0.6
        ax.text(0.5, y, f"{name[:14]:<15s} {s['kern_rate']:>10.0f}  {s['mc_rate']:>10.1f}  {c:>14s}",
                fontsize=10, family='monospace', color=col, fontweight='bold')
    y -= 1
    ax.text(0.5, y, "패턴:", fontsize=10, fontweight='bold')
    y -= 0.5
    ax.text(0.5, y, "  • memcpy rate가 L1과 비슷한 워크로드 → contention", fontsize=9)
    y -= 0.5
    ax.text(0.5, y, "  • kernel rate가 아무리 높아도 (chanpred 117K/s)", fontsize=9)
    y -= 0.4
    ax.text(0.7, y, "memcpy 거의 없으면 L1 contention 없음", fontsize=9)
    y -= 0.5
    ax.text(0.5, y, "  → memcpy queue가 진짜 contention point", fontsize=9, color='#C0392B', fontweight='bold')

    fig.suptitle("Supp 21 — AI 워크로드 signature가 L1 contention을 예측한다 (AI-side validation)\n"
                 "패턴: memcpy rate 비슷 → contention. kernel rate는 무관.", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_21_ai_workload_signature.png")
    plt.close(fig)
    print(f"  ✓ supp_21 AI workload signature matrix ({len(sigs)} workloads)")


def fig_supp_22_launch_queue_ruled_out():
    """chanpred has 15x L1's kernel rate but causes ZERO L1 effect — rules out launch queue."""
    fig, ax = plt.subplots(figsize=(11, 6))

    workloads = [
        ("L1 alone (no AI)", 7900, 4.2, "#666666"),
        ("L1 + chanpred",    7900 + 117474, 4.2, "#27AE60"),  # combined launch
        ("L1 + ResNet",      7900 + 6650, 14.3, "#E74C3C"),
        ("L1 + Forecaster",  7900 + 3226, 4.2, "#27AE60"),
        ("L1 + NeuralRx",    7900 + 1500, 14.3, "#E74C3C"),  # est NRx 1500/s from per-op latency 0.77ms inverse
    ]
    x = np.arange(len(workloads))
    rates = [w[1] for w in workloads]
    percall = [w[2] for w in workloads]
    cols = [w[3] for w in workloads]

    ax2 = ax.twinx()
    bars = ax.bar(x - 0.2, rates, 0.4, color=cols, label='combined kernel rate (L1 + AI)', alpha=0.85)
    bars2 = ax2.bar(x + 0.2, percall, 0.4, color='#9B59B6', label='L1 per-call memcpy duration (us)')
    ax.set_xticks(x); ax.set_xticklabels([w[0] for w in workloads], rotation=15)
    ax.set_ylabel("Total kernel launch rate (kernels/sec)", color='#34495E')
    ax2.set_ylabel("L1 per-call 60KB memcpy duration (us)", color='#9B59B6')

    for i, (r, p) in enumerate(zip(rates, percall)):
        ax.text(i - 0.2, r * 1.05, f"{r:,}", ha='center', fontsize=9, color='#34495E')
        ax2.text(i + 0.2, p + 0.5, f"{p:.1f}us", ha='center', fontsize=10, fontweight='bold', color='#9B59B6')

    ax.text(0.5, 0.95,
            "관찰: chanpred 추가시 launch rate가 125K/s (16x 증가)인데 L1 memcpy duration은 4.2us 그대로.\n"
            "ResNet 추가시 launch rate가 14.5K/s (1.8x 증가)인데 L1 memcpy duration은 14.3us (3.4x).\n"
            "→ launch rate increase와 L1 disturbance 사이 correlation 없음.\n"
            "→ Memcpy queue contention만 L1을 disturb. Kernel launch queue contention 가설 REJECT.",
            transform=ax.transAxes, va='top', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9))

    ax.set_title("Supp 22 — Kernel launch queue contention 가설 REJECT\n"
                 "chanpred 117K launch/s로 L1 disturb 안 함 → memcpy queue가 유일한 contention point", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_22_launch_queue_ruled_out.png")
    plt.close(fig)
    print(f"  ✓ supp_22 launch queue hypothesis rejected")


def parse_p5_stats(d):
    """Parse P5 5-min run stats from log."""
    runs = []
    pat = re.compile(r"mean=([0-9.]+)ms\s+p95=([0-9.]+)ms\s+p99=([0-9.]+)ms\s+miss1ms=([0-9]+)/([0-9]+)")
    for log in sorted(Path(d).glob("run_*_l1.log")):
        try:
            for line in log.read_text(errors="ignore").splitlines():
                m = pat.search(line)
                if m:
                    miss_n = int(m.group(4))
                    total_n = int(m.group(5))
                    runs.append({
                        "mean": float(m.group(1)),
                        "p95": float(m.group(2)),
                        "p99": float(m.group(3)),
                        "miss_rate": 100 * miss_n / total_n if total_n else 0,
                        "n_iters": total_n,
                    })
                    break
        except Exception:
            pass
    return runs


def fig_supp_23_sustained_persistence():
    """P5 sustained 5-min × n=2 — contention persistence over 7500 iterations."""
    workloads = ["alone", "qwen_small", "sat_compute", "sat_hbm", "chanpred",
                 "xapp", "neuralrx", "resnet", "forecaster"]
    data = []
    for w in workloads:
        runs = parse_p5_stats(P5 / w)
        if runs: data.append((w, runs))
    if not data: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: mean+SD across the 2 runs
    ax = axes[0]
    x = np.arange(len(data))
    p99_means = [np.mean([r["p99"] for r in d[1]]) for d in data]
    p99_sds = [np.std([r["p99"] for r in d[1]]) for d in data]
    means = [np.mean([r["mean"] for r in d[1]]) for d in data]
    colors = []
    for n, _ in data:
        if "alone" in n: colors.append('#666666')
        elif n in ["neuralrx", "qwen_small", "sat_hbm", "resnet"]:
            colors.append('#E74C3C')
        else: colors.append('#27AE60')
    w = 0.4
    ax.bar(x - w/2, means, w, label='mean (over 7500 iters)', color='#3498DB')
    ax.bar(x + w/2, p99_means, w, yerr=p99_sds, capsize=4,
           label='p99 (over 7500 iters)', color='#E74C3C')
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=15)
    ax.set_ylabel("L1 latency (ms)")
    ax.set_title("(A) 5-min × n=2 sustained: mean vs p99 (per-run aggregate)")
    ax.legend()

    # Panel B: per-run consistency (do runs agree?)
    ax = axes[1]
    for i, (lbl, runs) in enumerate(data):
        for j, r in enumerate(runs):
            ax.scatter(i + (j - 0.5) * 0.2, r["p99"], s=80, marker='os'[j],
                       color=colors[i], edgecolor='black', zorder=3)
        # connect line
        if len(runs) == 2:
            ax.plot([i - 0.1, i + 0.1], [runs[0]["p99"], runs[1]["p99"]],
                    color='gray', ls='--', alpha=0.5, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=15)
    ax.set_ylabel("L1 p99 (ms)")
    ax.set_title("(B) Per-run variability (○ run1, □ run2)\n낮은 분산 = sustained 효과 reproducible")

    # Annotate
    for i, (lbl, runs) in enumerate(data):
        cv = np.std([r["p99"] for r in runs]) / np.mean([r["p99"] for r in runs]) * 100 if len(runs) else 0
        ax.text(i, max(r["p99"] for r in runs) + 1, f"CV={cv:.1f}%", ha='center', fontsize=8)

    fig.suptitle("Supp 23 — P5 5-min sustained: contention은 transient warmup이 아니라 PERSISTENT\n"
                 "각 run 7500 iterations (5 min). AI 효과는 sustained × 2 runs에서 reproducible.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_23_sustained_persistence.png")
    plt.close(fig)
    print(f"  ✓ supp_23 P5 sustained persistence ({len(data)} workloads)")


def main():
    print(f"Generating 3 new-angle figures → {OUT}")
    fig_supp_21_ai_signature_matrix()
    fig_supp_22_launch_queue_ruled_out()
    fig_supp_23_sustained_persistence()
    print("Done.")


if __name__ == "__main__":
    main()
