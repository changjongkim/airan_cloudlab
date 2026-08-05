#!/usr/bin/env python3
"""Split verdict into two vertical bar figures — L1 latency + AI throughput.

Replaces old F08 that hardcoded "30% / 100%" AI throughput. Now uses
measured BeamPred + CsiNet aggregate iter/s from Chain 19 logs.

Findings:
  · Full GPU + MPS · N=6 diverse: AI 11,439 iter/s + L1 62 ms worst
  · CP + MPS · N=6:               AI 22,342 iter/s + L1 45 ms worst (2x AI!)
  · CP + MPS · N=16:              AI 41,194 iter/s + L1 44 ms worst
  · SP-4g conditions:              L1 catastrophic, AI throughput not directly
                                   logged (NRx containers only)
"""
import json, glob, re, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"; COL_BASE="#334155"
COL_AI="#2563eb"

def apply_font(lang):
    if lang == "ko":
        plt.rcParams["font.family"] = ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"]

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUT, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# =========================
# Load L1 latency
# =========================
def collect_l1(dirname, prefix_exact):
    trials = []
    for f in glob.glob(f"{BASE_19}/{dirname}/realL1_{prefix_exact}_t*.json"):
        d = json.load(open(f))
        trials.append(d["p99_ms"])
    return trials

# =========================
# Load AI throughput
# =========================
def parse_rate(path):
    vals = []
    try:
        for line in open(path, errors="ignore"):
            m = re.search(r"rate=(\d+)/s", line)
            if m: vals.append(int(m.group(1)))
    except: pass
    return max(vals) if vals else None

def agg_ai(exp, prefix):
    """Aggregate BeamPred + CsiNet rate per trial, then mean/max across trials."""
    trials = defaultdict(int)
    for f in glob.glob(f"{BASE_19}/{exp}/{prefix}_t*.log"):
        b = os.path.basename(f)
        m = re.search(rf"{prefix}_(t\d+)_", b)
        if not m: continue
        trial = m.group(1)
        if "beampred" in b or "csinet" in b:
            v = parse_rate(f)
            if v: trials[trial] += v
    if not trials:
        return None, None, None
    vs = list(trials.values())
    return np.mean(vs), min(vs), max(vs)

# =========================
# Conditions (unified across both figures)
# =========================
def build_conditions(lang):
    ko = (lang == "ko")
    C = []
    # baseline L1 alone
    l1_baseline = collect_l1("chain19_exp5", "e5_baseline")
    C.append({
        "label": "Baseline\n(L1 단독)" if ko else "Baseline\n(L1 alone)",
        "l1": l1_baseline,
        "ai_agg": None, "ai_min": None, "ai_max": None,
        "ai_note": ("AI 없음" if ko else "no AI"),
        "color": COL_BASE, "cat": "base",
    })
    # CP + MPS on · N=6
    ai6 = agg_ai("chain19_exp5", "e5_cpN6")
    C.append({
        "label": "CP + MPS on\nN=6 diverse AI" if ko else "CP + MPS on\nN=6 diverse AI",
        "l1": collect_l1("chain19_exp5", "e5_cpN6"),
        "ai_agg": ai6[0], "ai_min": ai6[1], "ai_max": ai6[2],
        "ai_note": ("BeamPred+CsiNet 합" if ko else "BeamPred+CsiNet sum"),
        "color": COL_GOOD, "cat": "win",
    })
    # CP + MPS on · N=16
    ai16 = agg_ai("chain19_exp5", "e5_cpN16")
    C.append({
        "label": "CP + MPS on\nN=16 diverse AI" if ko else "CP + MPS on\nN=16 diverse AI",
        "l1": collect_l1("chain19_exp5", "e5_cpN16"),
        "ai_agg": ai16[0], "ai_min": ai16[1], "ai_max": ai16[2],
        "ai_note": ("BeamPred+CsiNet 합" if ko else "BeamPred+CsiNet sum"),
        "color": COL_GOOD, "cat": "win",
    })
    # Full GPU + MPS · N=6
    aif6 = agg_ai("chain19_exp1", "e1_cfgB_diverseN6")
    C.append({
        "label": "Full GPU + MPS on\nN=6 diverse AI" if ko else "Full GPU + MPS on\nN=6 diverse AI",
        "l1": collect_l1("chain19_exp1", "e1_cfgB_diverseN6"),
        "ai_agg": aif6[0], "ai_min": aif6[1], "ai_max": aif6[2],
        "ai_note": ("BeamPred+CsiNet 합" if ko else "BeamPred+CsiNet sum"),
        "color": COL_WARN, "cat": "warn",
    })
    # Full GPU + MPS · N=12
    aif12 = agg_ai("chain19_exp1", "e1_cfgB_diverseN12")
    C.append({
        "label": "Full GPU + MPS on\nN=12 diverse AI" if ko else "Full GPU + MPS on\nN=12 diverse AI",
        "l1": collect_l1("chain19_exp1", "e1_cfgB_diverseN12"),
        "ai_agg": aif12[0], "ai_min": aif12[1], "ai_max": aif12[2],
        "ai_note": ("BeamPred+CsiNet 합" if ko else "BeamPred+CsiNet sum"),
        "color": COL_WARN, "cat": "warn",
    })
    # SP-4g + MPS pct=30 · N=6 (NRx)
    C.append({
        "label": "SP-4g + MPS pct=30\nN=6 NRx" if ko else "SP-4g + MPS pct=30\nN=6 NRx",
        "l1": collect_l1("chain19_exp11", "e11_pct30_N6"),
        "ai_agg": None, "ai_min": None, "ai_max": None,
        "ai_note": ("NRx는 throughput 로그 없음" if ko else "NRx has no throughput log"),
        "color": COL_BAD, "cat": "fail",
    })
    # SP-4g + MPS pct=100 · N=6 (NRx)
    C.append({
        "label": "SP-4g + MPS pct=100\nN=6 NRx" if ko else "SP-4g + MPS pct=100\nN=6 NRx",
        "l1": collect_l1("chain19_exp11", "e11_pct100_N6"),
        "ai_agg": None, "ai_min": None, "ai_max": None,
        "ai_note": ("NRx는 throughput 로그 없음" if ko else "NRx has no throughput log"),
        "color": COL_BAD, "cat": "fail",
    })
    return C

# =========================
# Figure A · L1 latency (vertical bars)
# =========================
def fig_l1(lang):
    apply_font(lang)
    C = build_conditions(lang)
    fig, ax = plt.subplots(figsize=(15, 7))
    xs = np.arange(len(C))
    means = [np.mean(c["l1"]) if c["l1"] else 0 for c in C]
    maxes = [max(c["l1"]) if c["l1"] else 0 for c in C]
    mins  = [min(c["l1"]) if c["l1"] else 0 for c in C]
    colors = [c["color"] for c in C]
    ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.65)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        ax.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        # value label
        offset = mx * 0.03 if mx < 100 else mx * 0.02
        ax.text(xs[i], mx + offset*3,
                f"{'mean' if lang=='en' else 'mean'} {m:.0f}\n{'worst' if lang=='en' else 'worst'} {mx:.0f}",
                ha="center", fontsize=11, color=colors[i], fontweight="bold")
    ax.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    ax.text(len(C)-0.5, 55, "5G L1 SLA proxy 50 ms", ha="right",
            fontsize=11, color=INK, style="italic")
    ax.set_yscale("log")
    ax.set_ylim(20, 800)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in C], fontsize=11)
    ax.set_ylabel("L1 p99 지연 (ms · 로그 스케일)" if lang=="ko" else "L1 p99 latency (ms · log scale)")
    ax.set_title("Fig · L1 p99 지연 · 조건별 실측 (mean + 3-trial worst whisker)"
                 if lang=="ko" else
                 "Fig · L1 p99 latency · measured per condition (mean + 3-trial worst whisker)",
                 fontweight="bold", pad=16, loc="left")
    ax.grid(axis="y", alpha=0.5, which="both")
    fig.text(0.02, 0.008,
             "모든 값 · 3 trial · Chain 19 realL1_*.json per-iter 측정. 오차 막대 = min-max 범위."
             if lang=="ko" else
             "All values from 3-trial Chain 19 realL1_*.json per-iter measurements. Whisker = min-max range.",
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G06A_L1_LATENCY{suffix}.png"); plt.close()
    print(f"F_G06A_L1_LATENCY{suffix}")

# =========================
# Figure B · AI throughput (vertical bars)
# =========================
def fig_ai(lang):
    apply_font(lang)
    C = build_conditions(lang)
    fig, ax = plt.subplots(figsize=(15, 7))
    xs = np.arange(len(C))
    means = [c["ai_agg"]/1000 if c["ai_agg"] else 0 for c in C]  # convert to k iter/s
    colors = [c["color"] if c["ai_agg"] else INK_MUT for c in C]
    ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.65)
    for i, c in enumerate(C):
        if c["ai_agg"]:
            ax.plot([xs[i], xs[i]], [c["ai_min"]/1000, c["ai_max"]/1000],
                    color=INK, linewidth=2, alpha=0.85)
            ax.plot([xs[i]-0.1, xs[i]+0.1], [c["ai_min"]/1000, c["ai_min"]/1000],
                    color=INK, linewidth=2, alpha=0.85)
            ax.plot([xs[i]-0.1, xs[i]+0.1], [c["ai_max"]/1000, c["ai_max"]/1000],
                    color=INK, linewidth=2, alpha=0.85)
            ax.text(xs[i], c["ai_max"]/1000 + 1,
                    f"{c['ai_agg']/1000:.1f}k iter/s\n({c['ai_note']})",
                    ha="center", fontsize=10.5, color=colors[i], fontweight="bold")
        else:
            ax.text(xs[i], 1.5, "N/A",
                    ha="center", fontsize=13, color=INK_MUT, fontweight="bold")
            ax.text(xs[i], 0.5, c["ai_note"],
                    ha="center", fontsize=9.5, color=INK_MUT, style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in C], fontsize=11)
    ax.set_ylabel("AI 집계 처리량 (k iter/s · BeamPred+CsiNet 합)"
                  if lang=="ko" else
                  "AI aggregate throughput (k iter/s · BeamPred+CsiNet sum)")
    ax.set_title("Fig · AI 처리량 · 조건별 실측 (BeamPred+CsiNet 컨테이너 iter/s 합)"
                 if lang=="ko" else
                 "Fig · AI throughput · measured per condition (BeamPred+CsiNet iter/s sum)",
                 fontweight="bold", pad=16, loc="left")
    ax.set_ylim(0, max([c["ai_max"]/1000 if c["ai_max"] else 5 for c in C]) * 1.2)
    ax.grid(axis="y", alpha=0.5)
    note_ko = ("실측 · 컨테이너별 rate 로그가 있는 BeamPred+CsiNet의 3-trial 합. "
               "NRx는 로그에 rate 없음 → SP 조건은 N/A. Baseline은 AI 없음. "
               "CP N=16 (41.2k) 이 최고 · Full GPU N=6 (11.4k) 의 3.6배 · MIG+MPS 결합 우수성이 처리량에서도 드러남.")
    note_en = ("Measured per-container rate summed across BeamPred+CsiNet, 3-trial mean. "
               "NRx logs no rate → SP conditions N/A. Baseline has no AI. "
               "CP N=16 (41.2k) is best · 3.6× Full GPU N=6 (11.4k) · MIG+MPS wins on throughput too.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10.5, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G06B_AI_THROUGHPUT{suffix}.png"); plt.close()
    print(f"F_G06B_AI_THROUGHPUT{suffix}")

for lang in ["ko", "en"]:
    fig_l1(lang)
    fig_ai(lang)
print("Done.")
