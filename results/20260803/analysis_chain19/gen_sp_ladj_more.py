#!/usr/bin/env python3
"""Additional SP + L1-adjacent evidence figures.

The single F_G07 wasn't sufficient. These four thicken the case:
  F_G08 · SP + NRx · MPS off vs on across A/B/C configs (MPS necessity)
  F_G09 · SP + NRx · L1 launch rate stability across N and configs
  F_G10 · SP + NRx · 3-trial variance for MPS ON (reliability)
  F_G11 · comprehensive SLA table: SP+NRx / SP+diverse / Full GPU / CP

Both KO and EN variants generated.
"""
import os, json, re
import numpy as np
import matplotlib.pyplot as plt

BASE_17 = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
BASE_19 = "/Users/changjongkim/New_research/cloudlab_results/results/20260803"
FIG     = f"{BASE_19}/analysis_chain19/figures/mig_mps"

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"
COL_MPS_ON="#059669"; COL_MPS_OFF="#b91c1c"
COL_A="#7c3aed"; COL_B="#dc6803"; COL_C="#0284c7"

def apply_font(lang):
    if lang == "ko":
        plt.rcParams["font.family"] = ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = ["Helvetica Neue","Helvetica","Arial","DejaVu Sans"]

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUT, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

ch17 = json.load(open(f"{BASE_17}/chain17_all_stats.json"))
def ch17_key(cfg, N, mps, key):
    ks = [k for k in ch17 if re.match(f"cfg{cfg}_A_nrxN{N}_MPS{mps}_t\\d+", k)]
    return [ch17[k][key] for k in ks]
def ch17_mean(cfg, N, mps, key):
    vs = ch17_key(cfg, N, mps, key); return np.mean(vs) if vs else None

# =========================
# F_G08 · MPS necessity for SP + NRx (gap p99 across A/B/C)
# =========================
def g08(lang):
    apply_font(lang)
    Ns = [1, 2, 3, 4, 6, 8]
    configs = [("A", "MIG 4g (SP)", COL_A),
               ("C", "MIG 3g (SP)",  COL_C)]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    for cfg, name, color in configs:
        off = [ch17_mean(cfg, N, "off", "gap_p99")/1e6 for N in Ns]
        on  = [ch17_mean(cfg, N, "on",  "gap_p99")/1e6 for N in Ns]
        axL.plot(Ns, off, "s-", color=color, linewidth=2.5, markersize=10,
                 markerfacecolor="white", markeredgewidth=2, label=name)
        axR.plot(Ns, on,  "o-", color=color, linewidth=2.5, markersize=10,
                 markerfacecolor="white", markeredgewidth=2, label=name)
    for ax, title_ko, title_en, ymax in [
        (axL, "(a) MPS OFF · gap p99 급등 (5–17 ms)", "(a) MPS OFF — gap p99 explodes (5–17 ms)", 20),
        (axR, "(b) MPS ON · gap p99 ≤ 2.5 ms", "(b) MPS ON — gap p99 stays ≤ 2.5 ms", 3)]:
        ax.set_xlabel("N (동일 NRx replica)" if lang=="ko" else "N (identical NRx replicas)")
        ax.set_ylabel("L1 kernel gap p99 (ms)")
        ax.set_title(title_ko if lang=="ko" else title_en,
                     fontweight="bold", loc="left", color=INK_SEC)
        ax.set_xticks(Ns); ax.grid(alpha=0.5); ax.set_ylim(0, ymax)
    axL.legend(frameon=True, loc="upper left")
    if lang == "ko":
        fig.suptitle("F_G08 · SP + NRx · MPS on/off — MPS가 SP+NRx를 실현가능하게 만드는 결정 요소",
                     fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
        fig.text(0.02, 0.008,
                 "같은 워크로드 (identical NRx) · 같은 위상 (SP) · MPS 유무만 다름. MPS OFF에서 tail이 8~15배 상승. MPS는 SP + L1-adjacent 조합의 필수 도구.",
                 fontsize=11, color=INK_SEC, style="italic")
    else:
        fig.suptitle("F_G08 · SP + NRx · MPS on/off — MPS is the decisive enabler of SP + L1-adjacent",
                     fontweight="bold", y=1.02, x=0.02, ha="left", fontsize=17)
        fig.text(0.02, 0.008,
                 "Same workload (identical NRx), same topology (SP), only MPS toggle differs. MPS OFF inflates the tail 8–15×. MPS is mandatory for SP + L1-adjacent.",
                 fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G08_SP_MPS_NECESSITY{suffix}.png"); plt.close()
    print(f"F_G08{suffix}")

# =========================
# F_G09 · L1 launch rate stability (SP + NRx · MPS on)
# =========================
def g09(lang):
    apply_font(lang)
    Ns = [1, 2, 3, 4, 6, 8]
    configs = [("A", "MIG 4g (SP)", COL_A),
               ("C", "MIG 3g (SP)",  COL_C)]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for cfg, name, color in configs:
        rates = [ch17_mean(cfg, N, "on", "launch_rate")/1000 for N in Ns]
        ax.plot(Ns, rates, "o-", color=color, linewidth=3, markersize=12,
                markerfacecolor="white", markeredgewidth=2.5, label=name)
        for N, r in zip(Ns, rates):
            ax.text(N, r+0.3, f"{r:.1f}k", ha="center", fontsize=9.5, color=color, fontweight="bold")
    ax.set_xticks(Ns); ax.grid(alpha=0.5); ax.legend(frameon=True, loc="upper right")
    ax.set_xlabel("N (동일 NRx replica · MPS on)" if lang=="ko" else "N (identical NRx replicas · MPS on)")
    ax.set_ylabel("L1 launch rate (k kernels/s)")
    if lang == "ko":
        ax.set_title("F_G09 · SP + NRx · MPS on — L1 launch rate 유지 (L1이 굶지 않음)",
                     fontweight="bold", pad=18, loc="left")
        fig.text(0.02, 0.008,
                 "L1 launch rate가 N=1의 12k 대비 N=8에서도 2~5k 유지. Slide 3에서 본 MPS OFF의 붕괴 (1k 이하) 와 대비. L1 kernel이 정상 발사됨.",
                 fontsize=11, color=INK_SEC, style="italic")
    else:
        ax.set_title("F_G09 · SP + NRx · MPS on — L1 launch rate holds (no starvation)",
                     fontweight="bold", pad=18, loc="left")
        fig.text(0.02, 0.008,
                 "L1 launch rate holds at 2–5k kernels/s at N=8 (vs 12k at N=1). Contrast with the MPS OFF collapse (<1k) shown in Slide 3.",
                 fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G09_SP_LAUNCH_RATE{suffix}.png"); plt.close()
    print(f"F_G09{suffix}")

# =========================
# F_G10 · 3-trial variance (SP + NRx MPS on · Config A)
# =========================
def g10(lang):
    apply_font(lang)
    Ns = [1, 2, 3, 4, 6, 8]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    means, mins, maxes = [], [], []
    for N in Ns:
        vs = [v/1e6 for v in ch17_key("A", N, "on", "gap_p99")]
        means.append(np.mean(vs)); mins.append(min(vs)); maxes.append(max(vs))
    xs = np.arange(len(Ns)); w = 0.55
    ax.bar(xs, means, w, color=COL_GOOD, alpha=0.85, edgecolor="white", linewidth=2)
    for i, (mn, mx, m) in enumerate(zip(mins, maxes, means)):
        ax.plot([xs[i], xs[i]], [mn, mx], color=INK, linewidth=2, alpha=0.75)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mn, mn], color=INK, linewidth=2, alpha=0.75)
        ax.plot([xs[i]-0.1, xs[i]+0.1], [mx, mx], color=INK, linewidth=2, alpha=0.75)
        ax.text(i, max(mx, m)+0.2, f"mean {m:.2f}\nmax {mx:.2f}",
                ha="center", fontsize=10, color=COL_GOOD, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.axhline(2.0, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.6)
    ax.text(-0.4, 2.05, "gap p99 = 2 ms " + ("참조선" if lang=="ko" else "reference"),
            color=INK_MUT, fontsize=10, style="italic")
    ax.set_ylim(0, 3)
    ax.set_ylabel("L1 kernel gap p99 (ms) · Config A · MPS on")
    ax.grid(axis="y", alpha=0.5)
    if lang == "ko":
        ax.set_title("F_G10 · SP + NRx · Config A · 3 trial 편차 — mean과 worst 모두 안정",
                     fontweight="bold", pad=18, loc="left")
        fig.text(0.02, 0.008,
                 "각 조건 3회 반복. 오차 막대(min–max) 좁음 → 재현성 확인. worst 값도 2.3ms 아래 → SLA 여유 충분.",
                 fontsize=11, color=INK_SEC, style="italic")
    else:
        ax.set_title("F_G10 · SP + NRx · Config A · 3-trial variance — mean and worst both stable",
                     fontweight="bold", pad=18, loc="left")
        fig.text(0.02, 0.008,
                 "3 trials per condition. Tight min–max bars → reproducible. Worst value stays under 2.3 ms → comfortable SLA margin.",
                 fontsize=11, color=INK_SEC, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G10_SP_TRIAL_VAR{suffix}.png"); plt.close()
    print(f"F_G10{suffix}")

# =========================
# F_G11 · Comprehensive comparison table figure
# =========================
def g11(lang):
    apply_font(lang)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.axis("off")
    if lang == "ko":
        header = ["시나리오", "위상", "워크로드", "N=6 지연 지표", "SLA 상태", "출처"]
        rows = [
            ("Baseline",                   "L1 단독",       "-",                                       "42 ms L1 p99",         "✓ (기준)",         "Exp 5"),
            ("SP + L1-adjacent",           "MIG SP-4g",     "L1 + 6× identical NRx",                   "1.5 ms gap p99",        "✓ (SLA proxy 통과)","Chain 17"),
            ("CP + loose AI (검증됨)",     "MIG CP (4g/3g)","L1 + 6 diverse AI (Qwen · Whisper 등)",  "42 ms L1 p99",         "✓",                "Exp 5"),
            ("SP + diverse AI (실패)",     "MIG SP-4g",     "L1 + 6 diverse AI mix (LLM 포함)",       "146 ms L1 p99",        "✗ (SLA 3배 초과)", "Exp 11"),
            ("Full GPU + MPS (bimodal)",   "No MIG",        "L1 + 6 diverse AI",                       "54 ms mean · 62 ms worst","△ worst 실패",   "Exp 1"),
        ]
    else:
        header = ["Scenario", "Topology", "Workload", "N=6 latency metric", "SLA status", "Source"]
        rows = [
            ("Baseline",                     "L1 alone",         "-",                                          "42 ms L1 p99",             "✓ (reference)",     "Exp 5"),
            ("SP + L1-adjacent",             "MIG SP-4g",        "L1 + 6× identical NRx",                      "1.5 ms gap p99",           "✓ (SLA proxy)",     "Chain 17"),
            ("CP + loose AI (verified)",     "MIG CP (4g/3g)",   "L1 + 6 diverse AI (Qwen, Whisper, …)",       "42 ms L1 p99",             "✓",                 "Exp 5"),
            ("SP + diverse AI (fails)",      "MIG SP-4g",        "L1 + 6 diverse AI mix (LLMs included)",      "146 ms L1 p99",            "✗ (3× over SLA)",   "Exp 11"),
            ("Full GPU + MPS (bimodal)",     "No MIG",           "L1 + 6 diverse AI",                          "54 ms mean · 62 ms worst", "△ worst fails",     "Exp 1"),
        ]
    row_colors = [INK_MUT, COL_GOOD, COL_GOOD, COL_BAD, COL_WARN]

    cell_h = 0.8
    cell_w = [2.6, 2.0, 3.8, 2.7, 2.2, 1.4]
    xs = [sum(cell_w[:i]) for i in range(len(cell_w)+1)]
    y_top = 5.0

    for j, h in enumerate(header):
        ax.add_patch(plt.Rectangle((xs[j], y_top), cell_w[j], cell_h, facecolor=INK, alpha=0.15, edgecolor="white"))
        ax.text(xs[j] + cell_w[j]/2, y_top + cell_h/2, h,
                ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    for i, row in enumerate(rows):
        y = y_top - (i+1)*cell_h
        rc = row_colors[i]
        for j, cell in enumerate(row):
            ax.add_patch(plt.Rectangle((xs[j], y), cell_w[j], cell_h,
                                        facecolor=rc if j>0 else INK, alpha=0.1, edgecolor="white"))
            fs = 10.5 if j == 2 else 11
            ax.text(xs[j] + cell_w[j]/2, y + cell_h/2, cell,
                    ha="center", va="center", fontsize=fs,
                    color=rc if j>0 else INK, fontweight="bold" if j==0 else "normal")

    ax.set_xlim(0, sum(cell_w))
    ax.set_ylim(-0.7, 6.0)
    title = "F_G11 · Workload × Topology 종합표 — SP는 워크로드 종류에 따라 성패가 갈림" if lang == "ko" \
        else "F_G11 · Workload × Topology matrix — SP outcome depends on which AI you co-locate"
    ax.text(sum(cell_w)/2, 5.9, title, ha="center", fontsize=16, fontweight="bold")
    note_ko = "지연 지표: gap p99 (kernel 사이 tail 간격, Chain 17) 와 L1 p99 (per-iter latency, Chain 19). 두 지표 직접 비교 어렵지만 심각도 대비는 뚜렷."
    note_en = "Latency metrics: gap p99 (inter-kernel tail, Chain 17) and L1 p99 (per-iter latency, Chain 19). Not directly comparable, but severity contrast is clear."
    ax.text(0, -0.5, note_ko if lang == "ko" else note_en,
            fontsize=10, style="italic", color=INK_SEC)

    plt.tight_layout()
    suffix = "_EN" if lang == "en" else ""
    plt.savefig(f"{FIG}/F_G11_SP_MATRIX{suffix}.png"); plt.close()
    print(f"F_G11{suffix}")

for lang in ["ko", "en"]:
    g08(lang); g09(lang); g10(lang); g11(lang)
print("Done.")
