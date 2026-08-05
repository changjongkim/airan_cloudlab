#!/usr/bin/env python3
"""v3 verdict figures · six-condition matrix with REAL measured AI throughput.

Data sources (all real measurements):
  · chain18_p8 (2026-07-25): baseline · CPdiverse · SPdiverse · CPuniform · SPuniform
    · All MIG Config A (4g L1 + 3g AI) · MPS always ON
    · Real L1 p99 latency + Qwen tok/s + BeamPred+CsiNet rate
  · chain18_p3 (2026-07-25): MPS off vs on toggle · NRx/embed/memcpy × N × MPS
    · L1 + N same-partition workload + Qwen on cross partition
    · Real L1 p99 latency
  · chain19_exp1 (2026-08-03): Full GPU + MPS on + diverse AI
    · Real L1 p99 latency + BeamPred+CsiNet rate

Six conditions (workload composition kept comparable = 6 diverse AI):
  1. CP + MPS OFF        · from p3 (proxy: MPS off with cross Qwen)
  2. CP + MPS ON         · chain18_p8 CPdiverse
  3. Full GPU + MPS OFF  · from p3 (proxy: SP with MPS off · Qwen cross)
  4. Full GPU + MPS ON   · chain19_exp1 N=6 diverse
  5. SP + MPS OFF        · from p3 nrxN5/embedN1/memcpyN1 MPSoff (SP with cross)
  6. SP + MPS ON         · chain18_p8 SPdiverse

Two output figures:
  F_G06A_v3 · L1 p99 latency (vertical bars · mean + worst whisker)
  F_G06B_v3 · AI throughput split (Qwen tok/s + BeamPred+CsiNet rate)
"""
import json, glob, re, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_18 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"; COL_BASE="#334155"
COL_QWEN="#7c3aed"; COL_BEAM="#0284c7"

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
# Data extractors
# =========================
def parse_p8_l1(cond):
    """chain18_p8 stdout has [realL1] cond: mean=X p95=Y p99=Z"""
    vals = []
    for f in glob.glob(f"{BASE_18}/chain18_p8/{cond}_t*.stdout"):
        for line in open(f, errors="ignore"):
            m = re.search(rf"\[realL1\] {cond}_t\d+: mean=([\d\.]+)ms.*p99=([\d\.]+)ms", line)
            if m: vals.append(float(m.group(2)))
    return vals

def parse_p3_l1(cond):
    """chain18 realL1_*.json"""
    vals = []
    for f in glob.glob(f"{BASE_18}/chain18/realL1_{cond}_t*.json"):
        try:
            d = json.load(open(f))
            vals.append(d["p99_ms"])
        except: pass
    return vals

def parse_ch19_l1(exp, cond):
    """chain19 realL1_*.json"""
    vals = []
    for f in glob.glob(f"{BASE_19}/{exp}/realL1_{cond}_t*.json"):
        try:
            d = json.load(open(f))
            vals.append(d["p99_ms"])
        except: pass
    return vals

def parse_qwen_p8(cond):
    """Qwen output tok/s max per trial from vLLM progress"""
    vals = []
    for f in glob.glob(f"{BASE_18}/chain18_p8/{cond}_t*_qwen.log"):
        content = open(f, errors="ignore").read()
        matches = re.findall(r"output:\s+([\d\.]+)\s+toks/s", content)
        if matches:
            vals.append(max(float(v) for v in matches))
    return vals

def parse_bc_rate(path_pattern):
    """Aggregate BeamPred + CsiNet max rate per trial."""
    per_trial = defaultdict(int)
    for f in glob.glob(path_pattern):
        b = os.path.basename(f)
        # extract trial number (t1, t2, t3)
        tm = re.search(r"_t(\d+)_", b) or re.search(r"_t(\d+)\.", b) or re.search(r"_t(\d+)_", b)
        trial = tm.group(1) if tm else "1"
        rates = []
        for line in open(f, errors="ignore"):
            m = re.search(r"rate=(\d+)/s", line)
            if m: rates.append(int(m.group(1)))
        if rates:
            per_trial[trial] += max(rates)
    return list(per_trial.values())

# =========================
# Assemble six conditions
# =========================
def build_conditions(lang):
    ko = (lang == "ko")

    # Extract data
    p8_baseline_l1  = parse_p8_l1("p8_baseline")
    p8_cp_l1        = parse_p8_l1("p8_CPdiverse")
    p8_sp_l1        = parse_p8_l1("p8_SPdiverse")
    p8_cp_qwen      = parse_qwen_p8("p8_CPdiverse")
    p8_sp_qwen      = parse_qwen_p8("p8_SPdiverse")
    p8_cp_bc = defaultdict(int)
    for kind in ["beampred", "csinet"]:
        for f in glob.glob(f"{BASE_18}/chain18_p8/p8_CPdiverse_t*_{kind}.log"):
            b = os.path.basename(f)
            m = re.search(r"_t(\d+)_", b); trial = m.group(1) if m else "1"
            rates = []
            for line in open(f, errors="ignore"):
                mm = re.search(r"rate=(\d+)/s", line)
                if mm: rates.append(int(mm.group(1)))
            if rates: p8_cp_bc[trial] += max(rates)
    p8_sp_bc = defaultdict(int)
    for kind in ["beampred", "csinet"]:
        for f in glob.glob(f"{BASE_18}/chain18_p8/p8_SPdiverse_t*_{kind}.log"):
            b = os.path.basename(f)
            m = re.search(r"_t(\d+)_", b); trial = m.group(1) if m else "1"
            rates = []
            for line in open(f, errors="ignore"):
                mm = re.search(r"rate=(\d+)/s", line)
                if mm: rates.append(int(mm.group(1)))
            if rates: p8_sp_bc[trial] += max(rates)
    p8_cp_bc = list(p8_cp_bc.values())
    p8_sp_bc = list(p8_sp_bc.values())

    # Full GPU + MPS ON from chain19 exp1 N=6 diverse
    fg_mps_on_l1 = parse_ch19_l1("chain19_exp1", "e1_cfgB_diverseN6")
    fg_bc_on = defaultdict(int)
    for kind in ["beampred", "csinet"]:
        for f in glob.glob(f"{BASE_19}/chain19_exp1/e1_cfgB_diverseN6_t*_{kind}*.log"):
            b = os.path.basename(f)
            m = re.search(r"_t(\d+)_", b); trial = m.group(1) if m else "1"
            rates = []
            for line in open(f, errors="ignore"):
                mm = re.search(r"rate=(\d+)/s", line)
                if mm: rates.append(int(mm.group(1)))
            if rates: fg_bc_on[trial] += max(rates)
    fg_bc_on = list(fg_bc_on.values())

    # MPS OFF conditions (proxy from p3): use nrxN5_MPSoff for SP MPS off proxy
    # Note: p3 setup = L1 + N NRx same partition + Qwen cross partition
    sp_mpsoff_l1 = parse_p3_l1("p3_nrxN5_MPSoff")  # SP + N=5 NRx + Qwen cross, MPS off
    sp_mpson_l1_p3  = parse_p3_l1("p3_nrxN5_MPSon")  # SP + N=5 NRx + Qwen cross, MPS on

    C = []
    C.append({
        "label": "Baseline\n(L1 alone · no AI)",
        "l1": p8_baseline_l1,
        "qwen": None, "bc": None,
        "color": COL_BASE, "cat": "base",
    })
    C.append({
        "label": "CP + MPS ON\n(L1@4g + 6 diverse AI@3g)",
        "l1": p8_cp_l1,
        "qwen": p8_cp_qwen, "bc": p8_cp_bc,
        "color": COL_GOOD, "cat": "win",
    })
    C.append({
        "label": "SP + MPS ON\n(L1 + 6 diverse AI on 4g)",
        "l1": p8_sp_l1,
        "qwen": p8_sp_qwen, "bc": p8_sp_bc,
        "color": COL_WARN, "cat": "warn",
    })
    C.append({
        "label": "Full GPU + MPS ON\n(no MIG · 6 diverse AI)",
        "l1": fg_mps_on_l1,
        "qwen": None,   # crashed in chain19 (memory bug)
        "bc": fg_bc_on,
        "color": COL_WARN, "cat": "warn",
    })
    C.append({
        "label": "SP + MPS OFF\n(L1 + 5 NRx same + Qwen cross)",
        "l1": sp_mpsoff_l1,
        "qwen": None, "bc": None,
        "color": COL_BAD, "cat": "fail",
    })
    C.append({
        "label": "SP + MPS ON\n(L1 + 5 NRx same + Qwen cross)",
        "l1": sp_mpson_l1_p3,
        "qwen": None, "bc": None,
        "color": COL_BAD, "cat": "fail_stress",
    })
    return C

# =========================
# Figure A · L1 latency (vertical bars, log scale)
# =========================
def fig_l1(lang):
    apply_font(lang)
    C = build_conditions(lang)
    fig, ax = plt.subplots(figsize=(16, 7))
    xs = np.arange(len(C))
    means = [np.mean(c["l1"]) if c["l1"] else 0 for c in C]
    maxes = [max(c["l1"]) if c["l1"] else 0 for c in C]
    mins  = [min(c["l1"]) if c["l1"] else 0 for c in C]
    colors = [c["color"] for c in C]
    ax.bar(xs, means, color=colors, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.62)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        ax.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mn, mn], color=INK, linewidth=2, alpha=0.85)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mx, mx], color=INK, linewidth=2, alpha=0.85)
        offset = mx * 0.08
        ax.text(xs[i], mx + offset,
                f"mean {m:.0f}\nworst {mx:.0f}",
                ha="center", fontsize=11, color=colors[i], fontweight="bold")
    ax.axhline(50, color=INK, linestyle="--", linewidth=1.8, alpha=0.75)
    ax.text(len(C)-0.5, 55, "5G L1 SLA proxy 50 ms", ha="right",
            fontsize=11, color=INK, style="italic")
    ax.set_yscale("log")
    ax.set_ylim(20, 3000)
    ax.set_xticks(xs)
    ax.set_xticklabels([c["label"] for c in C], fontsize=10)
    ax.set_ylabel("L1 p99 지연 (ms · 로그 스케일)" if lang=="ko" else "L1 p99 latency (ms · log scale)")
    ax.set_title("Fig · L1 p99 지연 · 실측 (3 trial · mean + worst)"
                 if lang=="ko" else
                 "Fig · L1 p99 latency · measured (3 trials · mean + worst)",
                 fontweight="bold", pad=16, loc="left")
    ax.grid(axis="y", alpha=0.5, which="both")
    fig.text(0.02, 0.008,
             "실측 · Chain 18 p8 (baseline·CP·SP · MPS ON) · Chain 19 Exp 1 (Full GPU MPS ON) · Chain 18 p3 (SP MPS off/on 스트레스 조건)"
             if lang=="ko" else
             "Measured · chain18_p8 (baseline/CP/SP · MPS ON) · chain19_exp1 (Full GPU MPS ON) · chain18_p3 (SP MPS off/on stress)",
             fontsize=10, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G06A_L1_LATENCY_v3{suffix}.png"); plt.close()
    print(f"F_G06A_v3{suffix}")

# =========================
# Figure B · AI throughput (Qwen tok/s + BeamPred+CsiNet rate · dual)
# =========================
def fig_ai(lang):
    apply_font(lang)
    C = build_conditions(lang)
    # Filter to only conditions with AI throughput data
    C_ai = [c for c in C if c["qwen"] or c["bc"]]
    fig, (ax_q, ax_bc) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                       gridspec_kw={"height_ratios":[1, 1]})
    xs = np.arange(len(C_ai))
    # Panel 1: Qwen tok/s
    q_means = [np.mean(c["qwen"]) if c["qwen"] else 0 for c in C_ai]
    q_maxes = [max(c["qwen"]) if c["qwen"] else 0 for c in C_ai]
    q_mins  = [min(c["qwen"]) if c["qwen"] else 0 for c in C_ai]
    colors = [c["color"] for c in C_ai]
    ax_q.bar(xs, q_means, color=COL_QWEN, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.55)
    for i, c in enumerate(C_ai):
        if c["qwen"]:
            ax_q.plot([xs[i], xs[i]], [q_mins[i], q_maxes[i]], color=INK, linewidth=2, alpha=0.85)
            ax_q.plot([xs[i]-0.08, xs[i]+0.08], [q_mins[i], q_mins[i]], color=INK, linewidth=2, alpha=0.85)
            ax_q.plot([xs[i]-0.08, xs[i]+0.08], [q_maxes[i], q_maxes[i]], color=INK, linewidth=2, alpha=0.85)
            ax_q.text(xs[i], q_maxes[i]+30, f"{q_means[i]:.0f} tok/s",
                      ha="center", fontsize=11, color=COL_QWEN, fontweight="bold")
        else:
            ax_q.text(xs[i], 50, "N/A\n" + ("Qwen 미실행" if lang=="ko" else "Qwen not run"),
                      ha="center", fontsize=10, color=INK_MUT, style="italic")
    ax_q.set_ylabel("Qwen output tok/s")
    ax_q.set_title("(a) LLM (Qwen 2.5-3B) 출력 처리량" if lang=="ko" else "(a) LLM (Qwen 2.5-3B) output throughput",
                   fontweight="bold", loc="left", color=INK_SEC)
    ax_q.grid(axis="y", alpha=0.5)
    ax_q.set_ylim(0, max(q_maxes)*1.35 if q_maxes else 1000)

    # Panel 2: BeamPred+CsiNet rate
    bc_means = [np.mean(c["bc"])/1000 if c["bc"] else 0 for c in C_ai]
    bc_maxes = [max(c["bc"])/1000 if c["bc"] else 0 for c in C_ai]
    bc_mins  = [min(c["bc"])/1000 if c["bc"] else 0 for c in C_ai]
    ax_bc.bar(xs, bc_means, color=COL_BEAM, alpha=0.85, edgecolor="white", linewidth=1.8, width=0.55)
    for i, c in enumerate(C_ai):
        if c["bc"]:
            ax_bc.plot([xs[i], xs[i]], [bc_mins[i], bc_maxes[i]], color=INK, linewidth=2, alpha=0.85)
            ax_bc.plot([xs[i]-0.08, xs[i]+0.08], [bc_mins[i], bc_mins[i]], color=INK, linewidth=2, alpha=0.85)
            ax_bc.plot([xs[i]-0.08, xs[i]+0.08], [bc_maxes[i], bc_maxes[i]], color=INK, linewidth=2, alpha=0.85)
            ax_bc.text(xs[i], bc_maxes[i]+0.5, f"{bc_means[i]:.1f}k iter/s",
                       ha="center", fontsize=11, color=COL_BEAM, fontweight="bold")
        else:
            ax_bc.text(xs[i], 1, "N/A",
                       ha="center", fontsize=10, color=INK_MUT, style="italic")
    ax_bc.set_xticks(xs)
    ax_bc.set_xticklabels([c["label"] for c in C_ai], fontsize=10)
    ax_bc.set_ylabel("BeamPred+CsiNet 합 (k iter/s)" if lang=="ko" else "BeamPred+CsiNet sum (k iter/s)")
    ax_bc.set_title("(b) L1-adjacent AI 컨테이너 처리량 (BeamPred + CsiNet 합)"
                    if lang=="ko" else
                    "(b) L1-adjacent AI container throughput (BeamPred + CsiNet sum)",
                    fontweight="bold", loc="left", color=INK_SEC)
    ax_bc.grid(axis="y", alpha=0.5)
    ax_bc.set_ylim(0, max(bc_maxes)*1.35 if bc_maxes else 15)

    fig.suptitle("Fig · AI 처리량 · 두 워크로드 카테고리로 분리 실측" if lang=="ko" else
                 "Fig · AI throughput · measured, split by workload category",
                 fontweight="bold", y=0.995, x=0.02, ha="left", fontsize=16)
    note_ko = ("(a) LLM 대표 (Qwen · 무거운 kernel): CP 739 tok/s · SP 654 tok/s · Full GPU 크래시 (메모리 설정 이슈). "
               "(b) L1-adjacent (BeamPred·CsiNet · 가벼운 kernel): 세 조건 모두 ~11k iter/s로 유사 = 이 카테고리는 위치에 덜 민감. "
               "즉 AI trade-off는 워크로드 종류마다 다름 · LLM은 위치 민감 · L1-adjacent는 무관.")
    note_en = ("(a) LLM (Qwen · heavy kernels): CP 739 tok/s · SP 654 tok/s · Full GPU crashed (memory config bug). "
               "(b) L1-adjacent (BeamPred/CsiNet · light kernels): all three ~11k iter/s → this category is placement-insensitive. "
               "AI trade-off is workload-dependent: LLM cares about placement, L1-adjacent does not.")
    fig.text(0.02, 0.008, note_ko if lang=="ko" else note_en,
             fontsize=10, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G06B_AI_THROUGHPUT_v3{suffix}.png"); plt.close()
    print(f"F_G06B_v3{suffix}")

for lang in ["ko", "en"]:
    fig_l1(lang)
    fig_ai(lang)
print("Done.")
