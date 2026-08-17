#!/usr/bin/env python3
"""Generate the measurement-backed figures used by RESEARCH_WALKTHROUGH_KO.md.

Every plotted value is loaded from a preserved CSV/JSON/SQLite result.  The script does
not modify raw data and intentionally keeps the scope of each experiment visible
in the figure subtitle.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "current" / "figures"

DRAIN = ROOT / "results" / "20260813_drain_free"
MIG_CAUSAL = (
    ROOT
    / "results"
    / "isca_v2"
    / "mig_causal_20260813T1138Z"
)
PLACEMENT = ROOT / "results" / "20260813_nrx_placement"
GDR_POOL = ROOT / "task1_final" / "gdr_pool_20260814T014651Z" / "analysis"
RADIO = ROOT / "task1_final" / "dart_rx_radio_pool" / "analysis"
FIVEWAY = MIG_CAUSAL / "05b_fiveway_absolute_rates"
MIG_MPS_QUOTA = (
    ROOT
    / "results"
    / "isca_v2"
    / "day1_20260813T0523Z"
    / "13_mig_mps_gdr_matrix"
)
HOST_BLOCKING = ROOT / "cuPHY_mitigation_shims" / "results"
MPS_MULTI_NRX = ROOT / "results" / "20260724" / "chain17"
MPS_KERNEL_GAPS = ROOT / "results" / "20260725" / "kernel_gap_stats.json"
RADIO_NSYS = (
    ROOT
    / "task1_final"
    / "dart_rx_radio_pool"
    / "dart_radio_pool_e3_round_robin_all_t34_20260814T093833Z"
    / "nsys_l1.sqlite"
)
KOREAN_FONT = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
ARCHITECTURE_MAP_SOURCE = (
    ROOT / "docs" / "current" / "assets" / "00_architecture_map_supplied.png"
)
GDR_EVOLUTION_SOURCE = (
    ROOT / "docs" / "current" / "assets" / "00a_gdr_evolution_supplied.png"
)
DART_RX_ARCHITECTURE_SOURCE = (
    ROOT / "docs" / "current" / "assets" / "00d_dart_rx_overall_architecture_supplied.png"
)

COLORS = {
    "navy": "#16324f",
    "blue": "#2878b5",
    "cyan": "#4bb3a7",
    "green": "#4c956c",
    "orange": "#f4a261",
    "red": "#d1495b",
    "purple": "#7b6fd0",
    "gray": "#8d99ae",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _plain_log_tick(value, _position):
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dfe4ea", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    if axis.get_yscale() == "log":
        axis.yaxis.set_major_formatter(FuncFormatter(_plain_log_tick))


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(axis, bars, fmt="{:.2f}", *, scale=1.0, pad=3):
    for bar in bars:
        value = bar.get_height() * scale
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def _placement_rows() -> dict[str, dict[str, str]]:
    return {row["config"]: row for row in read_csv(PLACEMENT / "PLACEMENT_SUMMARY.csv")}


def _mig_mps_quota_medians() -> list[tuple[int, int, float, float]]:
    configs = [
        ("l1_30_nrx_70", 30, 70),
        ("l1_50_nrx_50", 50, 50),
        ("l1_70_nrx_30", 70, 30),
    ]
    result = []
    for path, l1_share, nrx_share in configs:
        trials = []
        for json_path in sorted((MIG_MPS_QUOTA / path).glob("trial*/*.json")):
            data = read_json(json_path)
            trials.append((float(data["mean_ms"]), float(data["p99_ms"])))
        assert len(trials) == 3, (path, trials)
        result.append(
            (
                l1_share,
                nrx_share,
                float(np.median([trial[0] for trial in trials])),
                float(np.median([trial[1] for trial in trials])),
            )
        )
    return result


def figure_00_architecture_map():
    """Draw the five measured placement families with one visual grammar."""

    # This presentation-quality architecture map was supplied by the project
    # author.  Keep it as the canonical asset so regenerating the measurement
    # figures cannot silently replace it with the older matplotlib draft.
    if ARCHITECTURE_MAP_SOURCE.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ARCHITECTURE_MAP_SOURCE, OUT / "00_architecture_map.png")
        return

    fig, axes = plt.subplots(2, 3, figsize=(17.8, 10.0))

    def setup(axis, title):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.set_title(title, fontsize=12.2, fontweight="bold", pad=7)

    def box(
        axis,
        x,
        y,
        width,
        height,
        label,
        *,
        face="#ffffff",
        edge=COLORS["navy"],
        linewidth=1.5,
        linestyle="-",
        fontsize=9.2,
        text_color="#172033",
        zorder=2,
    ):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            fontweight="bold",
            zorder=zorder + 1,
        )
        return patch

    def arrow(
        axis,
        start,
        end,
        *,
        color,
        label=None,
        label_xy=None,
        linestyle="-",
        connectionstyle="arc3,rad=0",
        mutation_scale=13,
        linewidth=2.0,
        zorder=4,
    ):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="<->",
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        if label:
            if label_xy is None:
                label_xy = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            axis.text(
                *label_xy,
                label,
                ha="center",
                va="center",
                fontsize=8.1,
                color=color,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
                zorder=zorder + 1,
            )
        return patch

    def gpu_shell(axis, y=0.13, height=0.72):
        outer = FancyBboxPatch(
            (0.045, y),
            0.91,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#f7f9fc",
            edgecolor=COLORS["navy"],
            linewidth=2.0,
            zorder=0,
        )
        axis.add_patch(outer)
        axis.text(0.07, y + height - 0.045, "Physical A100 GPU", fontsize=8.3, color=COLORS["navy"], fontweight="bold")
        return outer

    # (a) Full GPU + MPS: all work shares one physical scheduling/resource domain.
    axis = axes[0, 0]
    setup(axis, "(a) MPS: 하나의 GPU를 모두 공유")
    gpu_shell(axis)
    box(axis, 0.10, 0.24, 0.80, 0.47, "공유 GPU 자원\n(SM · HBM · 작업 대기열)", face="#fff5f5", edge=COLORS["red"], linewidth=1.8)
    box(axis, 0.14, 0.51, 0.27, 0.13, "L1", face="#dceeff", edge=COLORS["blue"])
    box(axis, 0.59, 0.51, 0.27, 0.13, "NRx", face="#e9e4ff", edge=COLORS["purple"])
    box(axis, 0.36, 0.29, 0.28, 0.13, "Background AI", face="#fff0d9", edge=COLORS["orange"])
    axis.text(0.50, 0.17, "하드웨어 벽 없음 → 높은 활용률, 약한 L1 보호", ha="center", fontsize=8.5, color=COLORS["red"], fontweight="bold")

    # (b) MIG local: sibling background is isolated, L1 and NRx are not.
    axis = axes[0, 1]
    setup(axis, "(b) MIG: L1과 NRx는 같은 4g")
    gpu_shell(axis)
    box(axis, 0.08, 0.23, 0.56, 0.48, "", face="#eef6ff", edge=COLORS["blue"], linewidth=1.8)
    axis.text(0.11, 0.66, "4g MIG", fontsize=8.4, color=COLORS["blue"], fontweight="bold")
    box(axis, 0.12, 0.42, 0.21, 0.14, "L1", face="#dceeff", edge=COLORS["blue"])
    box(axis, 0.39, 0.42, 0.21, 0.14, "NRx", face="#e9e4ff", edge=COLORS["purple"])
    arrow(axis, (0.32, 0.49), (0.40, 0.49), color=COLORS["red"], label="같은 방의 경합", label_xy=(0.36, 0.36), mutation_scale=11)
    box(axis, 0.69, 0.23, 0.23, 0.48, "3g MIG\n\nQwen /\nBackground", face="#fff0d9", edge=COLORS["orange"], fontsize=8.5)
    axis.plot([0.665, 0.665], [0.20, 0.74], color="#111827", linewidth=4.0, zorder=5)
    axis.text(0.50, 0.17, "굵은 선 = MIG 하드웨어 격리벽", ha="center", fontsize=8.5, color="#111827", fontweight="bold")

    # (c) MIG+MPS: quota control inside the same 4g, without a new wall.
    axis = axes[0, 2]
    setup(axis, "(c) MIG+MPS: 같은 4g 안의 몫 조절")
    gpu_shell(axis)
    box(axis, 0.08, 0.23, 0.56, 0.48, "", face="#f7f3ff", edge=COLORS["purple"], linewidth=1.8)
    axis.text(0.11, 0.66, "4g MIG + MPS", fontsize=8.4, color=COLORS["purple"], fontweight="bold")
    box(axis, 0.12, 0.42, 0.21, 0.14, "L1 client", face="#dceeff", edge=COLORS["blue"], linestyle="--")
    box(axis, 0.39, 0.42, 0.21, 0.14, "NRx client", face="#e9e4ff", edge=COLORS["purple"], linestyle="--")
    axis.text(0.36, 0.33, "30:70 · 50:50 · 70:30", ha="center", fontsize=8.1, color=COLORS["purple"], fontweight="bold")
    axis.text(0.36, 0.27, "점선은 몫 제어이지 격리벽이 아님", ha="center", fontsize=8.0, color=COLORS["red"])
    box(axis, 0.69, 0.23, 0.23, 0.48, "3g MIG\n\nQwen /\nBackground", face="#fff0d9", edge=COLORS["orange"], fontsize=8.5)
    axis.plot([0.665, 0.665], [0.20, 0.74], color="#111827", linewidth=4.0, zorder=5)
    axis.text(0.50, 0.17, "Sibling은 격리 · L1/NRx 경합은 남음", ha="center", fontsize=8.5, color=COLORS["red"], fontweight="bold")

    # (d) Direct P2P: separate MIG compute domains, but only on a supported peer path.
    axis = axes[1, 0]
    setup(axis, "(d) MIG+P2P*: 분리된 GPU 공간을 직접 연결")
    gpu_shell(axis, y=0.20, height=0.65)
    box(axis, 0.08, 0.30, 0.25, 0.39, "2g MIG\n\nL1\nGPU buffer", face="#dceeff", edge=COLORS["blue"], fontsize=8.6)
    box(axis, 0.38, 0.30, 0.25, 0.39, "2g MIG\n\nNRx\nGPU buffer", face="#e9e4ff", edge=COLORS["purple"], fontsize=8.6)
    box(axis, 0.68, 0.30, 0.24, 0.39, "3g MIG\n\nBackground", face="#fff0d9", edge=COLORS["orange"], fontsize=8.6)
    axis.plot([0.355, 0.355], [0.27, 0.73], color="#111827", linewidth=3.5, zorder=5)
    axis.plot([0.655, 0.655], [0.27, 0.73], color="#111827", linewidth=3.5, zorder=5)
    arrow(axis, (0.28, 0.46), (0.43, 0.46), color=COLORS["cyan"], label="GPU P2P", label_xy=(0.355, 0.57), linewidth=2.8)
    axis.text(0.50, 0.13, "* 이번 gate: 한 process가 두 MIG CUDA context를 소유", ha="center", fontsize=8.2, color=COLORS["navy"], fontweight="bold")
    axis.text(0.50, 0.075, "peer access가 실제로 열리는 topology에서만 사용", ha="center", fontsize=7.9, color="#4b5563")

    # (e) GPUDirect RDMA: payload traverses the NIC without CPU-DRAM staging.
    axis = axes[1, 1]
    setup(axis, "(e) MIG+GDR: NIC loopback으로 GPU 메모리 연결")
    gpu_shell(axis, y=0.31, height=0.54)
    box(axis, 0.08, 0.39, 0.28, 0.32, "2g MIG\nL1 process\nGPU MR", face="#dceeff", edge=COLORS["blue"], fontsize=8.5)
    box(axis, 0.40, 0.39, 0.28, 0.32, "2g MIG\nNRx process\nGPU MR", face="#e9e4ff", edge=COLORS["purple"], fontsize=8.5)
    box(axis, 0.72, 0.39, 0.20, 0.32, "3g MIG\nBackground", face="#fff0d9", edge=COLORS["orange"], fontsize=8.2)
    axis.plot([0.38, 0.38], [0.36, 0.74], color="#111827", linewidth=3.5, zorder=5)
    axis.plot([0.70, 0.70], [0.36, 0.74], color="#111827", linewidth=3.5, zorder=5)
    box(axis, 0.20, 0.09, 0.42, 0.13, "ConnectX-6 Dx NIC\ninternal loopback", face="#e4f6e9", edge=COLORS["green"], fontsize=8.5)
    arrow(axis, (0.22, 0.40), (0.31, 0.21), color=COLORS["green"], connectionstyle="arc3,rad=0.10", linewidth=2.6)
    arrow(axis, (0.51, 0.21), (0.54, 0.40), color=COLORS["green"], connectionstyle="arc3,rad=0.10", linewidth=2.6)
    box(axis, 0.72, 0.09, 0.20, 0.13, "CPU\ncontrol only", face="#edf0f4", edge=COLORS["gray"], fontsize=8.0)
    axis.plot([0.71, 0.61], [0.155, 0.155], color=COLORS["gray"], linewidth=1.5, linestyle="--")
    axis.text(0.43, 0.265, "payload: GPU → NIC → GPU", ha="center", fontsize=8.2, color=COLORS["green"], fontweight="bold")
    axis.text(0.43, 0.035, "CPU DRAM을 payload가 통과하지 않음", ha="center", fontsize=8.0, color=COLORS["green"], fontweight="bold")

    # Shared legend and the exact request unit used later in the rate sweeps.
    axis = axes[1, 2]
    setup(axis, "(f) 그림을 읽는 기준")
    axis.add_patch(Rectangle((0.07, 0.76), 0.10, 0.07, facecolor="#dceeff", edgecolor=COLORS["blue"], linewidth=1.5))
    axis.text(0.20, 0.795, "L1: 반드시 끝나야 하는 PHY 경로", va="center", fontsize=8.7)
    axis.add_patch(Rectangle((0.07, 0.65), 0.10, 0.07, facecolor="#e9e4ff", edgecolor=COLORS["purple"], linewidth=1.5))
    axis.text(0.20, 0.685, "NRx: 선택적으로 호출하는 AI 수신기", va="center", fontsize=8.7)
    axis.add_patch(Rectangle((0.07, 0.54), 0.10, 0.07, facecolor="#fff0d9", edgecolor=COLORS["orange"], linewidth=1.5))
    axis.text(0.20, 0.575, "Background: 남는 자원에서 도는 AI", va="center", fontsize=8.7)
    axis.plot([0.07, 0.17], [0.465, 0.465], color="#111827", linewidth=4.0)
    axis.text(0.20, 0.465, "MIG hardware wall", va="center", fontsize=8.7)
    axis.plot([0.07, 0.17], [0.375, 0.375], color=COLORS["cyan"], linewidth=3.0)
    axis.text(0.20, 0.375, "지원되는 GPU peer data path", va="center", fontsize=8.7)
    axis.plot([0.07, 0.17], [0.285, 0.285], color=COLORS["green"], linewidth=3.0)
    axis.text(0.20, 0.285, "NIC GPUDirect data path", va="center", fontsize=8.7)
    box(axis, 0.08, 0.07, 0.84, 0.13, "NRx 요청 1개 = CE가 끝난 뒤\nL1 측 scheduler가 보낸 cell-slot 추론 1개", face="#f7f9fc", edge=COLORS["navy"], fontsize=8.7)

    fig.suptitle(
        "다섯 배치의 실제 차이: L1–NRx 사이의 벽과 데이터 이동 경로",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "배치 설명도(성능 그래프 아님): MPS와 local MIG/MPS는 L1·NRx가 같은 실행 공간, P2P/GDR는 둘을 분리",
        ha="center",
        fontsize=9.3,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), h_pad=1.8, w_pad=1.0)
    save(fig, "00_architecture_map.png")


def figure_00a_gdr_evolution():
    """Separate the three measured GDR topologies from the intended integrated design."""

    # Preserve the author's presentation-quality stage diagram in both report
    # variants instead of replacing it with the older generated draft.
    if GDR_EVOLUTION_SOURCE.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(GDR_EVOLUTION_SOURCE, OUT / "00a_gdr_evolution.png")
        return

    fig, axes = plt.subplots(2, 2, figsize=(17.6, 10.4))

    def setup(axis, title):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.set_title(title, fontsize=12.2, fontweight="bold", pad=8)

    def box(axis, x, y, width, height, label, face, edge, *, fontsize=8.5, linewidth=1.6, linestyle="-", zorder=2):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="#172033",
            zorder=zorder + 1,
            clip_on=True,
        )
        return patch

    def line(axis, start, end, *, color=COLORS["green"], linewidth=2.2, linestyle="-", arrowstyle="<->", zorder=5):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=arrowstyle,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=12,
            zorder=zorder,
        )
        axis.add_patch(patch)
        return patch

    def gpu(axis, x, y, width, height, label):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor="#f7f9fc",
            edgecolor=COLORS["navy"],
            linewidth=1.9,
            zorder=0,
        )
        axis.add_patch(patch)
        axis.text(x + 0.02, y + height - 0.035, label, fontsize=8.2, color=COLORS["navy"], fontweight="bold")
        return patch

    def badge(axis, text, color):
        axis.text(
            0.50,
            0.075,
            text,
            ha="center",
            va="center",
            fontsize=8.7,
            color=color,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "linewidth": 1.2},
        )

    # Stage 1: one physical GPU, one remote NRx, NIC loopback.
    axis = axes[0, 0]
    setup(axis, "1단계 · 단일 GPU cross-MIG GDR baseline")
    gpu(axis, 0.06, 0.25, 0.88, 0.58, "Physical GPU 0")
    box(axis, 0.10, 0.36, 0.24, 0.34, "2g MIG\nL1", "#dceeff", COLORS["blue"])
    box(axis, 0.39, 0.36, 0.24, 0.34, "2g MIG\nNRx 0", "#e9e4ff", COLORS["purple"])
    box(axis, 0.68, 0.36, 0.22, 0.34, "3g MIG\nQwen", "#fff0d9", COLORS["orange"])
    axis.plot([0.365, 0.365], [0.33, 0.73], color="#111827", linewidth=3.5)
    axis.plot([0.655, 0.655], [0.33, 0.73], color="#111827", linewidth=3.5)
    box(axis, 0.30, 0.14, 0.40, 0.10, "NIC physical loopback", "#e4f6e9", COLORS["green"], fontsize=8.2)
    line(axis, (0.22, 0.37), (0.40, 0.22))
    line(axis, (0.60, 0.37), (0.60, 0.22))
    axis.text(0.50, 0.865, "물리 GPU 1개 · NRx replica 1개", ha="center", fontsize=9.1, color=COLORS["navy"], fontweight="bold")
    badge(axis, "증명: 격리된 두 MIG 사이 GPU-memory GDR data path", COLORS["green"])

    # Stage 2: the actual fixed-MIG, three-endpoint pool across physical GPUs 0/1/2.
    axis = axes[0, 1]
    setup(axis, "2단계 · Fixed-MIG NRx 3-replica GDR pool")
    gpu_x = [0.035, 0.345, 0.655]
    gpu_labels = ["Physical GPU 0", "Physical GPU 1", "Physical GPU 2"]
    top_labels = ["4g source /\nL1-side GPU MR\n(radio 미실행)", "4g unused", "4g unused"]
    bottom_labels = ["3g NRx 0", "3g NRx 1", "3g NRx 2"]
    for index, x in enumerate(gpu_x):
        gpu(axis, x, 0.31, 0.285, 0.52, gpu_labels[index])
        top_face = "#dceeff" if index == 0 else "#edf0f4"
        top_edge = COLORS["blue"] if index == 0 else COLORS["gray"]
        box(axis, x + 0.025, 0.55, 0.235, 0.19, top_labels[index], top_face, top_edge, fontsize=7.5, linestyle="--" if index else "-")
        box(axis, x + 0.025, 0.35, 0.235, 0.15, bottom_labels[index], "#e9e4ff", COLORS["purple"], fontsize=8.1)
    box(axis, 0.22, 0.17, 0.56, 0.09, "ConnectX-6 Dx NIC · RC QP per endpoint", "#e4f6e9", COLORS["green"], fontsize=8.0)
    line(axis, (0.14, 0.55), (0.35, 0.25), linewidth=1.8)
    for x in (0.178, 0.488, 0.798):
        line(axis, (x, 0.35), (0.50, 0.25), linewidth=1.8)
    axis.text(0.50, 0.865, "물리 GPU 3개 · 실제 resident NRx replica 3개 · GPU3 unused", ha="center", fontsize=9.0, color=COLORS["navy"], fontweight="bold")
    badge(axis, "증명: full-size request/result와 request-level scale-out", COLORS["green"])

    # Stage 3: actual-radio correctness integration used full GPUs for the NRx workers.
    axis = axes[1, 0]
    setup(axis, "3단계 · Actual-radio 3-endpoint correctness gate")
    gpu(axis, 0.03, 0.31, 0.22, 0.50, "GPU 0")
    box(axis, 0.055, 0.40, 0.17, 0.28, "4g MIG\ncuPHY L1\n+ conventional\n+ LDPC/CRC", "#dceeff", COLORS["blue"], fontsize=7.7)
    for index, x in enumerate((0.30, 0.53, 0.76), start=1):
        gpu(axis, x, 0.31, 0.20, 0.50, f"GPU {index}")
        box(axis, x + 0.025, 0.40, 0.15, 0.28, f"Full GPU\nNRx {index - 1}", "#e9e4ff", COLORS["purple"], fontsize=8.0)
    box(axis, 0.25, 0.16, 0.50, 0.09, "NIC GDR request / result", "#e4f6e9", COLORS["green"], fontsize=8.2)
    line(axis, (0.14, 0.40), (0.33, 0.24), linewidth=1.8)
    for x in (0.375, 0.605, 0.835):
        line(axis, (0.50, 0.24), (x, 0.40), linewidth=1.8)
    axis.text(0.50, 0.865, "물리 GPU 4개 관여 · NRx는 GPU1/2/3 full GPU", ha="center", fontsize=9.0, color=COLORS["navy"], fontweight="bold")
    badge(axis, "증명: CE→NRx→LDPC/CRC·utility·epoch/expiry correctness", COLORS["green"])
    axis.text(0.50, 0.115, "주의: MIG 자원 효율 비교나 concurrent replica-capacity 실험이 아님", ha="center", fontsize=7.9, color=COLORS["red"], fontweight="bold")

    # Stage 4: intended fixed-topology integration; explicitly label it as pending.
    axis = axes[1, 1]
    setup(axis, "4단계 · 최종 목표: 고정 MIG 위의 resident NRx service pool")
    box(axis, 0.035, 0.37, 0.23, 0.36, "Protected 4g\nactual L1\n\nconventional\nfallback", "#dceeff", COLORS["blue"], fontsize=8.1, linewidth=2.0)
    box(axis, 0.35, 0.63, 0.17, 0.14, "3g NRx 0", "#e9e4ff", COLORS["purple"], fontsize=8.0)
    box(axis, 0.56, 0.63, 0.17, 0.14, "3g NRx 1", "#e9e4ff", COLORS["purple"], fontsize=8.0)
    box(axis, 0.77, 0.63, 0.17, 0.14, "3g NRx 2", "#e9e4ff", COLORS["purple"], fontsize=8.0)
    box(axis, 0.35, 0.34, 0.17, 0.14, "4g\nBackground", "#fff0d9", COLORS["orange"], fontsize=7.8)
    box(axis, 0.56, 0.34, 0.17, 0.14, "4g\nBackground", "#fff0d9", COLORS["orange"], fontsize=7.8)
    box(axis, 0.77, 0.34, 0.17, 0.14, "GPU3\nBackground", "#fff0d9", COLORS["orange"], fontsize=7.8)
    box(axis, 0.34, 0.13, 0.61, 0.10, "DART-Rx: utility · deadline · queue 상태로 endpoint 선택", "#e4f6e9", COLORS["green"], fontsize=8.1)
    for x in (0.435, 0.645, 0.855):
        line(axis, (0.26, 0.52), (x, 0.63), color=COLORS["green"], linewidth=1.8)
    line(axis, (0.46, 0.24), (0.18, 0.38), color=COLORS["green"], linewidth=1.8)
    axis.text(0.50, 0.865, "MIG를 합치는 것이 아니라, slot 요청을 상주 replica 사이에 분산", ha="center", fontsize=9.0, color=COLORS["navy"], fontweight="bold")
    badge(axis, "남은 gate: actual radio + concurrent burst + MIG NRx pool + background", COLORS["red"])
    axis.text(0.50, 0.285, "임의의 빈 GPU를 즉시 NRx로 바꾸는 구조가 아님 · model/context는 미리 resident", ha="center", fontsize=7.7, color="#4b5563")

    fig.suptitle(
        "GDR 실험을 단계별로 구분해야 하는 이유: 같은 이름, 다른 물리 topology와 claim",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "1–3단계는 실제 실행한 topology, 4단계는 아직 하나의 동시 실험으로 닫지 못한 목표 구조",
        ha="center",
        fontsize=9.2,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), h_pad=2.2, w_pad=1.4)
    save(fig, "00a_gdr_evolution.png")


def figure_00d_diagnostic_architecture():
    """Show the scheduler-visible queue state and the separate payload data plane."""

    fig, axis = plt.subplots(figsize=(18.2, 10.2))
    axis.set_xlim(0, 18.2)
    axis.set_ylim(0, 10.2)
    axis.axis("off")

    def box(
        x,
        y,
        width,
        height,
        label,
        *,
        face="#ffffff",
        edge=COLORS["navy"],
        fontsize=9.0,
        linewidth=1.8,
        linestyle="-",
        zorder=2,
        align="center",
    ):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.10",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        x_text = x + width / 2 if align == "center" else x + 0.14
        axis.text(
            x_text,
            y + height / 2,
            label,
            ha=align,
            va="center",
            fontsize=fontsize,
            color="#172033",
            fontweight="bold",
            zorder=zorder + 1,
        )
        return patch

    def arrow(
        start,
        end,
        *,
        color=COLORS["navy"],
        linewidth=2.0,
        linestyle="-",
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        zorder=8,
    ):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=arrowstyle,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            mutation_scale=14,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        return patch

    # Plane backgrounds make it explicit which state is host-resident and which
    # bytes travel directly between registered GPU buffers.
    box(
        0.25,
        2.75,
        17.70,
        6.50,
        "",
        face="#f7f9fc",
        edge="#c8d1dc",
        linewidth=1.4,
        zorder=0,
    )
    box(
        0.25,
        0.40,
        17.70,
        1.95,
        "",
        face="#eef8f0",
        edge="#9bc7a7",
        linewidth=1.4,
        zorder=0,
    )
    axis.text(
        0.50,
        9.03,
        "CONTROL PLANE · 작은 descriptor와 completion만 CPU가 관리",
        fontsize=10.0,
        color=COLORS["navy"],
        fontweight="bold",
    )
    axis.text(
        0.50,
        2.08,
        "DATA PLANE · 큰 tensor payload는 CPU DRAM을 거치지 않고 GPU → P2P/NIC → GPU",
        fontsize=10.0,
        color=COLORS["green"],
        fontweight="bold",
    )

    # Protected L1 domain.
    box(0.55, 3.10, 3.35, 5.48, "", face="#e5f1fb", edge=COLORS["blue"], linewidth=2.2)
    axis.text(2.225, 8.24, "보호된 L1 · 4g MIG", ha="center", fontsize=12.0, fontweight="bold", color=COLORS["blue"])
    box(0.88, 6.92, 2.69, 0.72, "cuPHY CE / front-end", face="#d2e9fb", edge=COLORS["blue"], fontsize=9.3)
    box(0.88, 5.63, 2.69, 0.88, "기존 수신기\n항상 실행 가능한 fallback", face="#ffffff", edge=COLORS["blue"], fontsize=8.8)
    box(0.88, 4.20, 2.69, 0.92, "요청 descriptor\ncell · slot · epoch · expiry · utility", face="#ffffff", edge=COLORS["navy"], fontsize=8.35)
    box(0.88, 3.28, 2.69, 0.60, "L1 GPU registered buffer", face="#dff4e5", edge=COLORS["green"], fontsize=8.8)

    # Scheduler and its locally maintained shadow ledger.
    box(4.35, 3.10, 6.55, 5.48, "", face="#fffdf7", edge="#b07d2b", linewidth=2.2)
    axis.text(
        7.625,
        8.24,
        "L1-side DART-Rx scheduler · host control plane",
        ha="center",
        fontsize=11.8,
        fontweight="bold",
        color="#8a5b14",
    )
    box(4.70, 6.92, 2.40, 0.76, "1 · Admission\nradio utility + deadline", face="#fff1da", edge=COLORS["orange"], fontsize=8.8)
    box(7.38, 6.92, 3.17, 0.76, "4 · 단일 commit\nslot · epoch · expiry · CRC", face="#ffe6e8", edge=COLORS["red"], fontsize=8.7)

    box(
        4.70,
        4.63,
        5.85,
        1.85,
        "",
        face="#ffffff",
        edge=COLORS["navy"],
        linewidth=1.7,
    )
    axis.text(7.625, 6.23, "2 · Endpoint shadow state", ha="center", fontsize=10.0, fontweight="bold", color=COLORS["navy"])
    axis.text(7.625, 5.96, "GPU queue를 원격으로 스캔하지 않고 completion으로 갱신 · 아래 값은 동작 예시", ha="center", fontsize=8.0, color="#4b5563")

    rows = [
        ("NRx 0", "pending=3", "예약 tail=늦음", "healthy", False),
        ("NRx 1", "pending=0", "예약 tail=현재", "healthy", True),
        ("NRx 2", "pending=1", "예약 tail=곧", "healthy", False),
    ]
    row_y = [5.60, 5.25, 4.90]
    for (name, pending, tail, health, selected), y in zip(rows, row_y):
        face = "#e4f6e9" if selected else "#f6f8fb"
        edge = COLORS["green"] if selected else "#c8d1dc"
        suffix = "  ← 선택" if selected else ""
        box(
            4.92,
            y,
            5.41,
            0.27,
            f"{name}  |  {pending}  |  {tail}  |  {health}{suffix}",
            face=face,
            edge=edge,
            fontsize=7.75,
            linewidth=1.15,
        )

    axis.text(
        7.625,
        4.44,
        "예측 완료[e] = max(현재시각, 예약 tail[e]) + 보수적 service bound[e]",
        ha="center",
        fontsize=8.5,
        color=COLORS["purple"],
        fontweight="bold",
    )
    box(
        4.70,
        3.38,
        5.85,
        0.78,
        "3 · Queue credit 예약\nsubmit: pending++ · tail 예약     |     completion: pending-- · bound 갱신",
        face="#f0edff",
        edge=COLORS["purple"],
        fontsize=8.35,
    )

    # Fixed resident endpoint fabric.  The queue glyphs intentionally mirror
    # the shadow-state example rather than implying hardware queue inspection.
    box(11.35, 3.10, 6.28, 5.48, "", face="#f5f1ff", edge=COLORS["purple"], linewidth=2.2)
    axis.text(14.49, 8.24, "고정 MIG 위의 resident NRx service fabric", ha="center", fontsize=11.6, fontweight="bold", color=COLORS["purple"])
    axis.text(14.49, 7.94, "모델 · TensorRT context · CUDA Graph · GPU MR은 미리 상주", ha="center", fontsize=8.15, color="#4b5563")

    endpoint_specs = [
        (6.38, "NRx 0", 3, False),
        (5.13, "NRx 1", 0, True),
        (3.88, "NRx 2", 1, False),
    ]
    for y, name, depth, selected in endpoint_specs:
        edge = COLORS["green"] if selected else COLORS["purple"]
        face = "#e5f6e9" if selected else "#ebe7ff"
        queue_label = "빈 queue" if depth == 0 else "대기 " + "● " * depth
        box(11.72, y, 1.45, 0.84, queue_label.strip(), face="#ffffff", edge=edge, fontsize=8.0)
        box(13.42, y, 3.82, 0.84, f"3g {name}\nTRT + CUDA Graph + GPU MR", face=face, edge=edge, fontsize=8.8)
        if selected:
            axis.text(17.00, y + 0.42, "예약됨", ha="center", va="center", fontsize=8.3, color=COLORS["green"], fontweight="bold")
    box(
        11.72,
        3.27,
        5.52,
        0.36,
        "Sibling 4g background lease · Qwen / BERT / Whisper / vision · work-unit 경계에서 양보",
        face="#fff0d9",
        edge=COLORS["orange"],
        fontsize=7.45,
        linewidth=1.35,
    )

    # Control messages and completion feedback.  There is no per-decision
    # remote query; the completion path updates local accounting.
    arrow((3.58, 4.66), (4.68, 7.27), color=COLORS["navy"], connectionstyle="arc3,rad=-0.18")
    axis.text(4.05, 5.86, "도착", fontsize=8.0, color=COLORS["navy"], fontweight="bold", rotation=55)
    arrow((10.56, 5.03), (11.70, 5.55), color=COLORS["green"], linewidth=2.4)
    axis.text(11.08, 5.59, "credit 예약", ha="center", fontsize=8.0, color=COLORS["green"], fontweight="bold")
    arrow((13.45, 6.27), (10.55, 6.00), color=COLORS["purple"], linewidth=1.7, linestyle="--", connectionstyle="arc3,rad=-0.18")
    axis.text(12.02, 6.48, "started / completed / error", ha="center", fontsize=7.7, color=COLORS["purple"], fontweight="bold")
    arrow((9.10, 6.49), (9.10, 6.90), color=COLORS["red"], linewidth=1.6)
    axis.text(9.25, 6.68, "valid result", fontsize=7.2, color=COLORS["red"], fontweight="bold")
    arrow((7.10, 7.30), (7.37, 7.30), color=COLORS["red"], linewidth=1.5)
    arrow((3.58, 6.06), (7.39, 7.02), color=COLORS["red"], connectionstyle="arc3,rad=-0.15", linewidth=1.8)

    # GPU payload path.
    box(0.78, 0.83, 2.72, 0.84, "L1 GPU registered buffer", face="#dff4e5", edge=COLORS["green"], fontsize=8.9)
    box(4.45, 1.32, 3.20, 0.53, "P2P · topology가 지원할 때", face="#dff7f5", edge=COLORS["cyan"], fontsize=8.7)
    box(4.45, 0.62, 3.20, 0.53, "NIC · GPUDirect RDMA", face="#dff4e5", edge=COLORS["green"], fontsize=8.7)
    axis.text(8.05, 1.25, "OR", ha="center", va="center", fontsize=9.2, color="#4b5563", fontweight="bold")
    box(9.08, 0.83, 4.95, 0.84, "선택된 endpoint의 GPU MR\nrequest / result payload", face="#e5f6e9", edge=COLORS["green"], fontsize=8.9)
    arrow((3.52, 1.25), (4.42, 1.58), color=COLORS["cyan"], linewidth=2.7)
    arrow((3.52, 1.25), (4.42, 0.88), color=COLORS["green"], linewidth=2.7)
    arrow((7.68, 1.58), (9.05, 1.36), color=COLORS["cyan"], linewidth=2.7)
    arrow((7.68, 0.88), (9.05, 1.12), color=COLORS["green"], linewidth=2.7)
    arrow((2.22, 3.27), (2.22, 1.69), color=COLORS["green"], linewidth=2.5)

    fig.suptitle(
        "DART-Rx 전체 구조: queue 상태를 추적해 가장 빨리 끝날 NRx를 선택",
        fontsize=16.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.008,
        "핵심: '줄이 짧은 곳'은 GPU 내부를 매번 읽어서 찾는 것이 아니라, L1 측 scheduler가 submit/completion으로 유지하는 shadow queue state로 판단",
        ha="center",
        fontsize=9.6,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0.01, 0.045, 0.99, 0.955))
    save(fig, "00d_dart_rx_overall_architecture.png")


def figure_00d_dart_rx_overall_architecture():
    """Presentation architecture in the panel-and-badge style of the supplied reference."""

    if DART_RX_ARCHITECTURE_SOURCE.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            DART_RX_ARCHITECTURE_SOURCE,
            OUT / "00d_dart_rx_overall_architecture.png",
        )
        return

    fig, axis = plt.subplots(figsize=(18.2, 10.3))
    axis.set_xlim(0, 18.2)
    axis.set_ylim(0, 10.3)
    axis.axis("off")

    black = "#111111"
    blue_panel = "#dfeaf8"
    blue_card = "#edf5ff"
    beige_panel = "#f5ead2"
    beige_card = "#fff8ea"
    green_panel = "#e1f1d9"
    green_card = "#f1faed"
    orange_card = "#fff0d9"
    selected_green = "#d4f0d2"

    def rounded(
        x,
        y,
        width,
        height,
        label="",
        *,
        face="white",
        edge=black,
        linewidth=2.0,
        fontsize=9.0,
        fontweight="bold",
        radius=0.14,
        linestyle="-",
        zorder=2,
        color=black,
    ):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0.018,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        if label:
            axis.text(
                x + width / 2,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=color,
                fontweight=fontweight,
                zorder=zorder + 1,
            )
        return patch

    def badge(x, y, number):
        circle = plt.Circle((x, y), 0.23, facecolor=black, edgecolor=black, linewidth=1.5, zorder=20)
        axis.add_patch(circle)
        axis.text(x, y - 0.005, str(number), ha="center", va="center", fontsize=12.0, color="white", fontweight="bold", zorder=21)

    def arrow(
        start,
        end,
        *,
        color=black,
        linewidth=2.5,
        linestyle="-",
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        zorder=15,
    ):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=arrowstyle,
            mutation_scale=18,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
        axis.add_patch(patch)
        return patch

    def panel_title(x, y, text):
        axis.text(x, y, text, ha="center", va="center", fontsize=14.2, color=black, fontweight="bold", zorder=10)

    # Four large regions use the same visual grammar as the supplied reference:
    # pastel rounded panels, thick black outlines, numbered black badges, and
    # one explicit reading order.
    rounded(0.35, 5.72, 7.30, 3.82, face=blue_panel, linewidth=2.25, radius=0.24, zorder=0)
    rounded(7.92, 5.72, 9.92, 3.82, face=beige_panel, linewidth=2.25, radius=0.24, zorder=0)
    rounded(0.35, 0.92, 7.30, 4.48, face="#e7eef8", linewidth=2.25, radius=0.24, zorder=0)
    rounded(7.92, 0.92, 9.92, 4.48, face=green_panel, linewidth=2.25, radius=0.24, zorder=0)

    panel_title(4.00, 9.22, "보호된 L1과 요청 생성 (Blue)")
    panel_title(12.88, 9.22, "Deadline 기반 endpoint 선택 (Beige)")
    panel_title(4.00, 5.08, "유효기간을 지키는 결과 commit (Blue)")
    panel_title(12.88, 5.08, "상주 NRx service pool (Green)")

    # Region A: protected L1 and the information that is actually sent to the
    # scheduler.  Big symbol cards replace the previous dense state-table view.
    l1_cards = [
        (0.72, "CE", "cuPHY CE /\nfront-end"),
        (2.70, "RX", "기존 수신기\n항상 준비"),
        (4.68, "MR", "L1 등록\nGPU buffer"),
    ]
    for x, symbol, label in l1_cards:
        rounded(x, 7.08, 1.66, 1.42, face=blue_card, linewidth=1.8, radius=0.12)
        circle = plt.Circle((x + 0.83, 8.05), 0.28, facecolor="white", edgecolor=black, linewidth=2.0, zorder=4)
        axis.add_patch(circle)
        axis.text(x + 0.83, 8.05, symbol, ha="center", va="center", fontsize=11.0, fontweight="bold", zorder=5)
        axis.text(x + 0.83, 7.48, label, ha="center", va="center", fontsize=8.8, fontweight="bold", zorder=5)

    rounded(
        1.33,
        6.04,
        5.25,
        0.68,
        "Slot 요청 descriptor · cell · slot · epoch · expiry · radio utility",
        face="white",
        linewidth=1.9,
        fontsize=8.45,
        radius=0.10,
    )
    badge(1.34, 6.72, 1)
    for x in (1.55, 3.53, 5.51):
        arrow((x, 7.08), (3.94, 6.72), linewidth=1.9)
    axis.text(0.72, 8.78, "전용 4g MIG · 반드시 끝나야 하는 L1 경로", fontsize=9.0, fontweight="bold", color=COLORS["blue"])

    # Region B: admission, shadow-state accounting, and reservation are shown
    # as three explicit steps.  The state values are illustrative, not data.
    rounded(8.28, 7.19, 2.38, 1.43, "Radio utility + expiry\nadmission", face=beige_card, linewidth=1.9, fontsize=9.2)
    badge(8.34, 8.64, 2)
    rounded(10.88, 6.58, 4.42, 2.04, face="white", linewidth=1.9)
    badge(10.94, 8.64, 3)
    axis.text(13.09, 8.33, "Endpoint shadow queue state", ha="center", fontsize=9.9, fontweight="bold")
    axis.text(13.09, 8.04, "submit +1 · completion -1 · 원격 GPU queue scan 없음", ha="center", fontsize=7.8, color="#444444")
    state_rows = [
        (7.63, "NRx 0   pending 3   ·   끝날 시각 늦음", False),
        (7.25, "NRx 1   pending 0   ·   지금 가능   ← 선택", True),
        (6.87, "NRx 2   pending 1   ·   곧 가능", False),
    ]
    for y, label, selected in state_rows:
        rounded(
            11.16,
            y,
            3.86,
            0.30,
            label,
            face=selected_green if selected else "#f3f5f7",
            edge=COLORS["green"] if selected else "#aab2bd",
            linewidth=1.25,
            fontsize=7.65,
            radius=0.06,
        )
    axis.text(13.09, 6.69, "예측 완료 = max(현재시각, 예약 tail) + service bound", ha="center", fontsize=7.65, color=COLORS["purple"], fontweight="bold")

    rounded(15.54, 7.19, 1.92, 1.43, "Queue credit 예약\n및 dispatch", face=beige_card, linewidth=1.9, fontsize=9.0)
    badge(15.60, 8.64, 4)
    arrow((10.66, 7.91), (10.86, 7.91), linewidth=2.2)
    arrow((15.31, 7.91), (15.52, 7.91), linewidth=2.2)
    axis.text(12.88, 5.94, "HOST CONTROL PLANE · counter · health · deadline 판단", ha="center", fontsize=8.3, fontweight="bold", color="#6c4f1e")

    # Region C: conventional and neural candidates join at one versioned commit.
    rounded(0.75, 3.38, 2.05, 0.90, "기존 수신기\n결과", face=blue_card, linewidth=1.8, fontsize=9.0)
    rounded(0.75, 1.77, 2.05, 0.90, "NRx completion\n결과 + epoch", face="#edf7e9", linewidth=1.8, fontsize=8.8)
    rounded(3.25, 2.25, 2.15, 1.42, "검증\nslot · epoch · expiry\nhealth · CRC", face="white", linewidth=2.0, fontsize=8.8)
    badge(3.30, 3.72, 6)
    rounded(5.75, 2.25, 1.48, 1.42, "정확히 하나의\nLDPC/CRC\n결과", face="#fff0f1", edge=COLORS["red"], linewidth=2.0, fontsize=8.6)
    arrow((2.82, 3.83), (3.23, 3.30), color=COLORS["blue"], linewidth=2.2)
    arrow((2.82, 2.22), (3.23, 2.62), color=COLORS["green"], linewidth=2.2)
    arrow((5.42, 2.96), (5.72, 2.96), color=COLORS["red"], linewidth=2.4)
    axis.text(3.98, 1.34, "늦거나 · 오래됐거나 · 고장났거나 · 불가능한 NRx → 기존 수신 결과 사용", ha="center", fontsize=8.2, fontweight="bold", color=COLORS["red"])

    # Region D: three fixed resident endpoints, visible queue depths, transport
    # alternatives, and bounded background work on sibling MIG capacity.
    endpoint_x = [8.35, 11.34, 14.33]
    endpoint_info = [
        ("대기 ● ● ●", "3g MIG · NRx 0", False),
        ("빈 queue", "3g MIG · NRx 1", True),
        ("대기 ●", "3g MIG · NRx 2", False),
    ]
    for x, (queue_label, worker_label, selected) in zip(endpoint_x, endpoint_info):
        rounded(
            x,
            3.49,
            2.55,
            0.47,
            queue_label,
            face="white",
            edge=COLORS["green"] if selected else black,
            linewidth=1.7,
            fontsize=8.2,
            radius=0.09,
        )
        rounded(
            x,
            2.37,
            2.55,
            0.96,
            worker_label + "\nTensorRT · CUDA Graph · GPU MR",
            face=selected_green if selected else green_card,
            edge=COLORS["green"] if selected else black,
            linewidth=2.2 if selected else 1.7,
            fontsize=8.2,
            radius=0.10,
        )
        rounded(
            x,
            1.44,
            2.55,
            0.48,
            "Sibling 4g background lease",
            face=orange_card,
            edge=COLORS["orange"],
            linewidth=1.5,
            fontsize=7.5,
            radius=0.08,
        )
    badge(8.40, 4.05, 5)
    rounded(9.12, 4.12, 7.35, 0.58, "GPU payload fabric · 지원될 때 P2P  |  ConnectX-6 Dx GPUDirect RDMA", face="white", linewidth=1.9, fontsize=8.55)
    for x in (9.62, 12.61, 15.60):
        arrow((x, 4.10), (x, 3.98), color=COLORS["green"], linewidth=2.0)
    axis.text(12.88, 1.13, "고정 MIG 격리벽 · 상주 model · fast path에서 MIG 재구성 없음", ha="center", fontsize=8.3, color="#315f31", fontweight="bold")

    # Inter-region arrows establish the single reading order.  Black is small
    # control metadata, green is GPU payload/result, and dotted is completion
    # feedback used to maintain the shadow state.
    arrow((6.59, 6.39), (8.26, 7.89), color=black, linewidth=2.8, connectionstyle="arc3,rad=-0.15")
    axis.text(7.27, 7.30, "작은 descriptor", fontsize=7.9, fontweight="bold", rotation=42)
    arrow((16.50, 7.17), (16.50, 4.72), color=black, linewidth=3.0)
    axis.text(16.66, 5.90, "endpoint id\n+ queue credit", fontsize=7.9, fontweight="bold")
    arrow((5.51, 7.07), (9.10, 4.42), color=COLORS["green"], linewidth=3.2, connectionstyle="arc3,rad=0.08")
    axis.text(7.05, 5.62, "GPU tensor payload", fontsize=8.1, color=COLORS["green"], fontweight="bold", rotation=-31)
    arrow((9.10, 4.10), (2.82, 2.24), color=COLORS["green"], linewidth=3.0, connectionstyle="arc3,rad=-0.10")
    axis.text(6.05, 3.02, "NRx 결과", fontsize=8.1, color=COLORS["green"], fontweight="bold", rotation=15)
    arrow((2.70, 7.06), (1.78, 4.30), color=COLORS["blue"], linewidth=2.7, connectionstyle="arc3,rad=0.10")
    arrow((16.98, 3.64), (17.50, 6.55), color=black, linewidth=2.0, linestyle=":", connectionstyle="arc3,rad=-0.18")
    arrow((17.50, 6.55), (15.30, 7.05), color=black, linewidth=2.0, linestyle=":", connectionstyle="arc3,rad=-0.10")
    axis.text(17.38, 5.50, "completion\nfeedback", ha="center", fontsize=7.8, fontweight="bold")

    # Legend and caption follow the reference's publication-figure treatment.
    axis.plot([0.72, 1.25], [0.50, 0.50], color=black, linewidth=2.6)
    axis.text(1.38, 0.50, "작은 control metadata", va="center", fontsize=8.2)
    axis.plot([3.05, 3.58], [0.50, 0.50], color=COLORS["green"], linewidth=3.0)
    axis.text(3.71, 0.50, "GPU payload / 결과", va="center", fontsize=8.2)
    axis.plot([5.72, 6.25], [0.50, 0.50], color=black, linewidth=2.0, linestyle=":")
    axis.text(6.38, 0.50, "Completion feedback", va="center", fontsize=8.2)
    axis.text(10.42, 0.50, "Queue 길이 = dispatcher가 추적하는 미완료 요청 수 · 원격 CUDA queue를 읽는 값이 아님", va="center", fontsize=8.6, fontweight="bold")

    fig.suptitle(
        "DART-Rx 전체 구조: L1을 보호하면서 queue 상태로 NRx endpoint를 선택",
        fontsize=17.0,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "그림: L1은 기존 수신 fallback을 유지하고, DART-Rx는 고정 MIG 경계를 넘어 제시간에 끝날 상주 NRx를 예약한다.",
        ha="center",
        fontsize=10.8,
        color=black,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.006, 0.045, 0.994, 0.958))
    save(fig, "00d_dart_rx_overall_architecture.png")


def figure_00c_mps_multi_nrx_breakdown():
    """Show the multi-client MPS knee hidden by the single-NRx placement summary."""

    counts = [1, 2, 3, 4, 6, 8]

    def p99_medians(config):
        medians = []
        for count in counts:
            values = [
                float(read_json(path)["p99_ms"])
                for path in sorted(
                    MPS_MULTI_NRX.glob(
                        f"realL1_cfg{config}_A_nrxN{count}_MPSon_t*.json"
                    )
                )
            ]
            assert len(values) == 3, (config, count, values)
            medians.append(float(np.median(values)))
        return medians

    mig4_p99 = p99_medians("A")
    full_p99 = p99_medians("B")
    gaps = read_json(MPS_KERNEL_GAPS)
    gap_median = [float(gaps[f"N{count}_MPSon"]["gap_median"]) / 1000 for count in counts]
    gap_p95 = [float(gaps[f"N{count}_MPSon"]["gap_p95"]) / 1000 for count in counts]
    duty = [float(gaps[f"N{count}_MPSon"]["duty_cycle"]) * 100 for count in counts]
    kernel_duration = [float(gaps[f"N{count}_MPSon"]["dur_median"]) / 1000 for count in counts]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))

    axes[0].plot(counts, full_p99, marker="o", linewidth=2.2, color=COLORS["blue"], label="Full A100 + MPS")
    axes[0].plot(counts, mig4_p99, marker="D", linewidth=2.2, color=COLORS["red"], label="4g MIG 안의 MPS")
    axes[0].axvspan(5.5, 8.4, color="#f6d98f", alpha=0.34, label="NRx client 증가 시 붕괴 구간")
    axes[0].set_xticks(counts)
    axes[0].set_xlabel("동시에 실행한 독립 NRx process 수")
    axes[0].set_ylabel("20-cell L1 p99(ms)")
    axes[0].set_title("(a) Full GPU도 NRx client가 늘면 무너짐")
    axes[0].legend(frameon=False, fontsize=8.0, loc="upper left")
    axes[0].annotate(
        f"{full_p99[-1] / full_p99[0]:.1f}×",
        (counts[-1], full_p99[-1]),
        xytext=(-17, 8),
        textcoords="offset points",
        color=COLORS["blue"],
        fontweight="bold",
    )
    axes[0].annotate(
        f"{mig4_p99[-1] / mig4_p99[0]:.1f}×",
        (counts[-1], mig4_p99[-1]),
        xytext=(-22, 8),
        textcoords="offset points",
        color=COLORS["red"],
        fontweight="bold",
    )
    style_axes(axes[0])

    axes[1].plot(counts, gap_median, marker="o", linewidth=2.2, color=COLORS["red"], label="중앙값")
    axes[1].plot(counts, gap_p95, marker="D", linewidth=2.0, color=COLORS["purple"], label="p95")
    axes[1].set_yscale("log")
    axes[1].set_xticks(counts)
    axes[1].set_xlabel("동시에 실행한 독립 NRx process 수")
    axes[1].set_ylabel("L1 kernel 사이 대기시간(us, 로그)")
    axes[1].set_title("(b) N=6부터 kernel 사이 빈 시간이 급증")
    axes[1].legend(frameon=False, fontsize=8.2)
    axes[1].annotate(
        "1.1 → 119.7 → 379.1 us",
        (6, gap_median[4]),
        xytext=(-63, -31),
        textcoords="offset points",
        fontsize=8.4,
        color=COLORS["red"],
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["red"], "linewidth": 1.1},
    )
    style_axes(axes[1])

    axes[2].plot(counts, duty, marker="o", linewidth=2.2, color=COLORS["green"], label="L1 GPU duty cycle")
    axes[2].set_xticks(counts)
    axes[2].set_ylim(0, 38)
    axes[2].set_xlabel("동시에 실행한 독립 NRx process 수")
    axes[2].set_ylabel("L1 GPU duty cycle(%)", color=COLORS["green"])
    axes[2].tick_params(axis="y", colors=COLORS["green"])
    twin = axes[2].twinx()
    twin.plot(counts, kernel_duration, marker="s", linestyle="--", linewidth=1.9, color=COLORS["orange"], label="L1 kernel 길이")
    twin.set_ylim(0, 18)
    twin.set_ylabel("L1 kernel 중앙값(us)", color=COLORS["orange"])
    twin.tick_params(axis="y", colors=COLORS["orange"])
    axes[2].set_title("(c) L1이 GPU를 쓰는 비율도 절반 이하로 감소")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8.0, loc="lower left")
    axes[2].annotate(
        "31.6% → 13.8%",
        (8, duty[-1]),
        xytext=(-78, -2),
        textcoords="offset points",
        fontsize=8.6,
        color=COLORS["green"],
        fontweight="bold",
    )
    style_axes(axes[2])
    twin.spines["top"].set_visible(False)

    fig.suptitle(
        "MPS의 숨은 scaling 문제: 독립 NRx client가 늘면 L1 scheduling이 급격히 붕괴",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 과거 20-cell real-cuPHY causal campaign, max-rate NRx process 1/2/3/4/6/8개, MPS on, L1 p99는 3회 중앙값. 현재 최적화 chain과 절대시간 직접 비교 금지",
        ha="center",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.91))
    save(fig, "00c_mps_multi_nrx_breakdown.png")


def figure_00_three_local_baselines():
    placement = _placement_rows()
    mps_names = [f"Full MPS Qwen {share}%" for share in (30, 50, 70, 100)]
    mps_share = [30, 50, 70, 100]
    mps_mean = [float(placement[name]["e2e_ms"]) for name in mps_names]
    mps_p99 = [float(placement[name]["e2e_p99_ms"]) for name in mps_names]
    qwen = [float(placement[name]["qwen_itps"]) for name in mps_names]

    isolation = DRAIN / "fixed_mig_sibling_isolation"
    alone = read_json(isolation / "nrx_4g_alone.json")["configurations"][0]
    sibling = read_json(isolation / "nrx_4g_qwen_3g.json")["configurations"][0]
    capacity_ratio = (
        float(sibling["closed_loop"]["throughput_slots_per_s"])
        / float(alone["closed_loop"]["throughput_slots_per_s"])
    )
    l1_slowdown = float(placement["MIG same"]["l1_slowdown"])
    quota = _mig_mps_quota_medians()

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2))

    axes[0].plot(mps_share, mps_mean, marker="o", linewidth=2.2, color=COLORS["blue"], label="평균 처리시간")
    axes[0].plot(mps_share, mps_p99, marker="D", linewidth=2.0, color=COLORS["navy"], label="느린 1% 경계(p99)")
    axes[0].set_xlabel("Qwen에 허용한 GPU 몫(%)")
    axes[0].set_ylabel("요청 전체 처리시간(ms)")
    axes[0].set_title("(a) MPS: Qwen을 더 쓰면 무선 처리가 느려짐")
    axes[0].legend(frameon=False, fontsize=8.3, loc="upper left")
    twin = axes[0].twinx()
    twin.plot(mps_share, qwen, marker="s", linestyle="--", linewidth=1.8, color=COLORS["orange"], label="Qwen 처리량")
    twin.set_ylabel("Qwen 처리량(iter/s)", color=COLORS["orange"])
    twin.tick_params(axis="y", colors=COLORS["orange"])
    axes[0].annotate("5.86 ms / 7.92 it/s", (30, mps_mean[0]), xytext=(8, 10), textcoords="offset points", fontsize=8.2)
    axes[0].annotate("8.57 ms / 21.11 it/s", (100, mps_mean[-1]), xytext=(-94, 15), textcoords="offset points", fontsize=8.2)
    style_axes(axes[0])
    twin.spines["top"].set_visible(False)

    x = np.arange(2)
    base_bars = axes[1].bar(x - 0.18, [1.0, 1.0], width=0.36, color=COLORS["gray"], label="단독 실행")
    corun_bars = axes[1].bar(
        x + 0.18,
        [capacity_ratio, l1_slowdown],
        width=0.36,
        color=[COLORS["green"], COLORS["red"]],
        label="다른 작업과 동시 실행",
    )
    axes[1].set_xticks(x, ["4g NRx 처리량\n+ 옆 3g에 Qwen", "L1 실행시간\n+ 같은 4g에 NRx"])
    axes[1].set_ylim(0, 1.82)
    axes[1].set_ylabel("단독 실행 대비 배율")
    axes[1].set_title("(b) MIG: 옆 파티션만 격리됨")
    axes[1].legend(frameon=False, fontsize=8.0, loc="upper left")
    for bar, value in zip(corun_bars, [capacity_ratio, l1_slowdown]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.3f}x", ha="center", fontsize=9, fontweight="bold")
    axes[1].text(0, 1.16, "옆 파티션 간섭은 차단", ha="center", fontsize=8.3, color=COLORS["green"])
    axes[1].text(1, 1.72, "같은 파티션의 경합은 남음", ha="center", fontsize=8.3, color=COLORS["red"])
    style_axes(axes[1])

    quota_labels = [f"L1 {l1}%\nNRx {nrx}%" for l1, nrx, _, _ in quota]
    quota_mean = [mean for _, _, mean, _ in quota]
    quota_p99 = [p99 for _, _, _, p99 in quota]
    qx = np.arange(len(quota))
    bars = axes[2].bar(qx, quota_mean, color=[COLORS["green"], COLORS["purple"], COLORS["red"]], width=0.6)
    axes[2].scatter(qx, quota_p99, marker="D", s=42, color=COLORS["navy"], label="느린 1% 경계(p99)", zorder=3)
    axes[2].set_xticks(qx, quota_labels)
    axes[2].set_ylim(0, 7.6)
    axes[2].set_ylabel("요청 전체 처리시간(ms)")
    axes[2].set_title("(c) MIG+MPS: 같은 파티션의 몫만 재분배")
    axes[2].legend(frameon=False, fontsize=8.2)
    annotate_bars(axes[2], bars, "{:.2f}")
    style_axes(axes[2])

    fig.suptitle(
        "출발점: 세 가지 내부 배치 모두 '강한 격리'와 '필요할 때 NRx 확장'을 함께 제공하지 못함",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.004,
        "서로 다른 하드웨어 실험 요약: MPS 몫 변화, MIG 인접 파티션 격리, 같은 MIG 안의 L1·NRx 몫 변화(각 3회 중앙값)",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "00_three_local_baselines.png")


def figure_00b_why_cross_endpoints():
    placement = _placement_rows()
    depth1 = {row["config"]: row for row in read_csv(PLACEMENT / "DEPTH1_TRANSPORT_COMPARISON.csv")}
    multi = read_csv(MIG_CAUSAL / "07_multicell_workloads" / "analysis" / "MULTICELL_HARDWARE_MEDIANS.csv")
    one_cell = {
        row["policy"]: row
        for row in multi
        if row["scenario"] == "single_periodic"
        and row["cells"] == "1"
        and row["slot_us"] == "1000.0"
        and row["nrx_probability_requested"] == "1.0"
        and row["policy"] in {"static_one", "predicted_finish"}
    }
    assert set(one_cell) == {"static_one", "predicted_finish"}

    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.0))
    labels = ["MIG 안에\nL1+NRx", "MIG+MPS 안에\nL1+NRx", "L1/NRx 분리\n+ P2P"]
    slowdown = [
        float(placement["MIG same"]["l1_slowdown"]),
        float(placement["MIG+MPS same"]["l1_slowdown"]),
        float(placement["Cross P2P"]["l1_slowdown"]),
    ]
    bars = axes[0].bar(labels, slowdown, color=[COLORS["blue"], COLORS["purple"], COLORS["cyan"]], width=0.62)
    axes[0].axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="L1 단독 실행")
    axes[0].set_ylim(0, 1.9)
    axes[0].set_ylabel("L1 실행시간 증가 배율")
    axes[0].set_title("(a) NRx를 분리하자 L1 성능이 회복됨")
    annotate_bars(axes[0], bars, "{:.3f}x")
    axes[0].legend(frameon=False, fontsize=8.2)
    style_axes(axes[0])

    transport_labels = ["GPU 사이 직접 전송\n(P2P)", "NIC를 거친 GPU 직접 전송\n(GDR)"]
    transport_mean = [
        float(depth1["Cross P2P depth 1"]["e2e_mean_ms"]),
        float(depth1["Cross NIC GDR depth 1"]["e2e_mean_ms"]),
    ]
    transport_p99 = [
        float(depth1["Cross P2P depth 1"]["e2e_p99_ms"]),
        float(depth1["Cross NIC GDR depth 1"]["e2e_p99_ms"]),
    ]
    tx = np.arange(2)
    bars = axes[1].bar(tx, transport_mean, color=[COLORS["cyan"], COLORS["green"]], width=0.62)
    axes[1].scatter(tx, transport_p99, marker="D", s=42, color=COLORS["navy"], label="느린 1% 경계(p99)", zorder=3)
    axes[1].set_xticks(tx, transport_labels)
    axes[1].set_ylim(0, 7.6)
    axes[1].set_ylabel("요청 전체 처리시간(ms)")
    axes[1].set_title("(b) NIC로 범위를 넓힌 비용은 평균 0.438 ms")
    axes[1].legend(frameon=False, fontsize=8.2)
    annotate_bars(axes[1], bars, "{:.3f}")
    style_axes(axes[1])

    pool_labels = ["NRx 한 곳에\n계속 고정", "NRx 3곳 중\n빨리 끝날 곳 선택"]
    pool_p99 = [float(one_cell[p]["median_p99_ms"]) for p in ("static_one", "predicted_finish")]
    bars = axes[2].bar(pool_labels, pool_p99, color=[COLORS["red"], COLORS["blue"]], width=0.62)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("NRx 응답의 느린 1% 경계(ms, 로그)")
    axes[2].set_title("(c) 고정 배치는 GPU가 놀아도 한 줄만 폭증")
    annotate_bars(axes[2], bars, "{:.2f}")
    axes[2].text(
        0.55,
        0.58,
        "NRx 처리기 3개 중 2개 이상이\n놀고 있는데도 대기 폭증",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=8.5,
        color=COLORS["red"],
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )
    style_axes(axes[2])

    fig.suptitle(
        "P2P/GDR를 고려한 이유: L1을 NRx와 분리한 채, 떨어진 NRx 처리기까지 사용하기 위해",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.004,
        "서로 다른 3개 실험의 연결: 배치별 L1 격리, 같은 대기열 깊이의 전송 비교, 매 1 ms 요청에서 NRx 3개 선택",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "00b_why_p2p_gdr.png")


def figure_01_isolation_and_queue_cliff():
    base = DRAIN / "fixed_mig_sibling_isolation"
    alone = read_json(base / "nrx_4g_alone.json")["configurations"][0]
    sibling = read_json(base / "nrx_4g_qwen_3g.json")["configurations"][0]

    cap = [
        alone["closed_loop"]["throughput_slots_per_s"],
        sibling["closed_loop"]["throughput_slots_per_s"],
    ]
    delta = 100 * (cap[1] / cap[0] - 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    bars = axes[0].bar(
        ["4g NRx만 실행", "4g NRx +\n옆 3g에 Qwen"],
        cap,
        color=[COLORS["blue"], COLORS["green"]],
        width=0.62,
    )
    axes[0].set_ylim(0, max(cap) * 1.18)
    axes[0].set_ylabel("초당 지속 처리 가능한 NRx 요청 수")
    axes[0].set_title("(a) 옆 MIG의 Qwen은 NRx 처리량에 거의 영향 없음")
    annotate_bars(axes[0], bars, "{:.1f}")
    axes[0].text(
        0.5,
        max(cap) * 1.09,
        f"처리량 변화: {delta:+.2f}%",
        ha="center",
        fontsize=10,
        color=COLORS["navy"],
        fontweight="bold",
    )
    style_axes(axes[0])

    for data, label, color, marker in [
        (alone, "4g NRx만", COLORS["blue"], "o"),
        (sibling, "4g NRx + 옆 3g Qwen", COLORS["green"], "s"),
    ]:
        rates = [x["arrival_rate_slots_per_s"] for x in data["open_loop"]]
        p99 = [x["latency_ms"]["p99"] for x in data["open_loop"]]
        axes[1].plot(rates, p99, marker=marker, linewidth=2.2, label=label, color=color)
    axes[1].axhline(5, linestyle="--", linewidth=1.2, color=COLORS["red"], label="실험 기준선 5 ms")
    axes[1].axvline(cap[0], linestyle=":", linewidth=1.4, color=COLORS["gray"])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("초당 들어오는 NRx 요청 수")
    axes[1].set_ylabel("대기시간의 느린 1% 경계(ms, 로그)")
    axes[1].set_title("(b) 처리 한계를 넘으면 격리돼도 대기시간은 폭증")
    axes[1].legend(frameon=False, fontsize=8.5)
    style_axes(axes[1])

    fig.suptitle(
        "MIG가 해결하는 것과 못 하는 것: 옆 작업의 간섭은 막지만, NRx 한 대의 처리 한계는 남음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "실험 범위: A100의 4g MIG 한 개에서 최적화한 TensorRT NRx 실행, 초당 요청 수를 단계적으로 증가",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "01_mig_isolation_queue_cliff.png")


def figure_01b_nrx_wrapper_optimization():
    base = PLACEMENT / "raw" / "nrx_deep_profile"
    wrapper = read_json(base / "nrx_deep_profile_4g.json")
    direct = read_json(base / "nrx_trt_direct_4g.json")
    graph = read_json(base / "nrx_trt_direct_graph_4g.json")
    compare = read_json(base / "nrx_trt_direct_compare_4g.json")

    labels = ["기존 Python\n실행 경로", "TensorRT 엔진\n직접 호출", "직접 호출\n+ CUDA Graph"]
    gpu_ms = [
        float(wrapper["metrics"]["raw_pycuphy"]["gpu_ms"]["mean"]),
        float(direct["direct"]["gpu_ms"]["mean"]),
        float(graph["direct"]["gpu_ms"]["mean"]),
    ]
    enqueue_us = [
        np.nan,
        float(direct["direct"]["enqueue_us"]["mean"]),
        float(graph["direct"]["enqueue_us"]["mean"]),
    ]
    equality = all(
        float(value["max_abs_difference"]) == 0.0
        for value in compare["pycuphy_comparison"].values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    bars = axes[0].bar(labels, gpu_ms, color=[COLORS["red"], COLORS["blue"], COLORS["green"]], width=0.62)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NRx GPU 실행시간(ms, 로그)")
    axes[0].set_title("(a) 105 ms의 대부분은 AI 계산 자체가 아니었음")
    annotate_bars(axes[0], bars, "{:.3f}")
    style_axes(axes[0])

    x = np.arange(2)
    bars = axes[1].bar(x, enqueue_us[1:], color=[COLORS["blue"], COLORS["green"]], width=0.58)
    axes[1].set_xticks(x, labels[1:])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("CPU가 GPU에 작업을 넣는 시간(us, 로그)")
    axes[1].set_title("(b) CUDA Graph로 반복 제출 비용까지 제거")
    annotate_bars(axes[1], bars, "{:.1f}")
    style_axes(axes[1])
    axes[1].text(
        0.5,
        0.78,
        f"기존 실행 경로와 결과 완전 일치: 최대 차이 0 ({'통과' if equality else '실패'})",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
        color=COLORS["navy"],
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )

    fig.suptitle(
        "구현 경로를 바로잡자 NRx가 105.15 ms에서 1.34 ms로 단축(약 74배)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: A100 4g MIG 한 개, 기존 Python 경로 30회, 직접 호출과 CUDA Graph는 준비 실행 뒤 1,000회",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "01b_nrx_wrapper_optimization.png")


def figure_02_fragmentation():
    rows = read_csv(
        MIG_CAUSAL / "07_multicell_workloads" / "analysis" / "MULTICELL_HARDWARE_MEDIANS.csv"
    )
    cases = [
        ("single_periodic", "1", "1000.0", "1.0", "셀 1개\n매 1 ms, 모두 NRx"),
        ("selective_bursty", "4", "1000.0", "0.1", "셀 4개\n매 1 ms, 몰림 10%"),
        ("selective_bursty", "4", "500.0", "0.1", "셀 4개\n매 0.5 ms, 몰림 10%"),
    ]
    policies = ["static_one", "predicted_finish"]
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for scenario, cells, slot_us, probability, label in cases:
        for row in rows:
            key = (row["scenario"], row["cells"], row["slot_us"], row["nrx_probability_requested"])
            if key == (scenario, cells, slot_us, probability) and row["policy"] in policies:
                selected[(label, row["policy"])] = row
    assert len(selected) == len(cases) * len(policies)

    labels = [case[-1] for case in cases]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for index, (policy, display, color) in enumerate(
        [
            ("static_one", "한 NRx에 고정", COLORS["red"]),
            ("predicted_finish", "빨리 끝날 NRx 선택", COLORS["blue"]),
        ]
    ):
        p99 = [float(selected[(label, policy)]["median_p99_ms"]) for label in labels]
        no_timely = [
            float(selected[(label, policy)]["median_no_timely_nrx_ratio"]) for label in labels
        ]
        idle = [
            float(selected[(label, policy)]["median_idle_endpoint_fraction"]) for label in labels
        ]
        offsets = x + (index - 0.5) * width
        axes[0].bar(offsets, p99, width, label=display, color=color)
        axes[1].bar(offsets, no_timely, width, label=display, color=color)
        axes[2].bar(offsets, idle, width, label=display, color=color)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("NRx 응답의 느린 1% 경계(ms, 로그)")
    axes[0].set_title("(a) 느린 요청의 대기시간")
    axes[1].set_ylabel("5 ms 안에 쓸 NRx 결과가 없는 비율")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("(b) 제시간 결과를 얻지 못한 요청")
    axes[2].set_ylabel("놀고 있는 NRx 처리기 비율")
    axes[2].set_ylim(0, 0.9)
    axes[2].set_title("(c) 다른 NRx 처리기는 얼마나 놀았나")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.5)
        style_axes(axis)
    axes[0].legend(frameon=False, fontsize=8.5)

    fig.suptitle(
        "문제의 직접 증거: 한 NRx 대기열은 무너지는데 다른 NRx 처리기는 동시에 놀고 있음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 독립적으로 상주한 TensorRT NRx 3개, 3회 중앙값, 5 ms는 비교용 실험 기준선",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "02_fixed_placement_fragmentation.png")


def figure_03_placement_and_transport():
    rows = read_csv(PLACEMENT / "PLACEMENT_SUMMARY.csv")
    desired = [
        "MIG same",
        "MIG+MPS same",
        "Cross P2P",
        "Cross NIC GDR",
        "Full MPS Qwen 30%",
        "Full MPS Qwen 50%",
        "Full MPS Qwen 70%",
        "Full MPS Qwen 100%",
    ]
    by_name = {row["config"]: row for row in rows}
    assert all(name in by_name for name in desired)
    labels = [
        "MIG 안에\nL1+NRx",
        "MIG+MPS 안에\nL1+NRx",
        "L1·NRx 분리\nGPU 직접(P2P)",
        "L1·NRx 분리\nNIC 직접(GDR)",
        "MPS\nQwen 30%",
        "MPS\nQwen 50%",
        "MPS\nQwen 70%",
        "MPS\nQwen 100%",
    ]
    colors = [
        COLORS["blue"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["green"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
        COLORS["orange"],
    ]
    mean = np.array([float(by_name[name]["e2e_ms"]) for name in desired])
    p99 = np.array([float(by_name[name]["e2e_p99_ms"]) for name in desired])
    qwen = np.array([float(by_name[name]["qwen_itps"]) for name in desired])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(labels))
    bars = axes[0].bar(x, mean, color=colors, width=0.68, label="평균")
    axes[0].scatter(x, p99, marker="D", s=38, color=COLORS["navy"], label="느린 1% 경계(p99)", zorder=3)
    axes[0].set_xticks(x, labels, rotation=24, ha="right")
    axes[0].set_ylabel("요청 전체 처리시간(ms)")
    axes[0].set_title("(a) 배치와 전송 방식별 처리시간")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars, "{:.2f}")
    style_axes(axes[0])

    family_markers = {
        "MIG": "o",
        "MIG+MPS": "P",
        "P2P": "s",
        "NIC GDR": "^",
        "MPS": "D",
    }
    label_offsets = {
        "MIG 안에\nL1+NRx": (-64, 0),
        "MIG+MPS 안에\nL1+NRx": (-78, 34),
        "L1·NRx 분리\nGPU 직접(P2P)": (24, 19),
        "L1·NRx 분리\nNIC 직접(GDR)": (24, 43),
        "MPS\nQwen 30%": (-10, -18),
        "MPS\nQwen 50%": (12, -14),
    }
    for name, label, color in zip(desired, labels, colors):
        row = by_name[name]
        axes[1].scatter(
            float(row["qwen_itps"]),
            float(row["e2e_ms"]),
            marker=family_markers[row["family"]],
            s=72,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axes[1].annotate(
            label,
            (float(row["qwen_itps"]), float(row["e2e_ms"])),
            xytext=label_offsets.get(label, (6, 4)),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "linewidth": 0.7}
            if label in label_offsets
            else None,
        )
    axes[1].set_xlabel("Qwen 처리량(iter/s)")
    axes[1].set_ylabel("요청 평균 처리시간(ms)")
    axes[1].set_title("(b) 무선 처리시간과 남는 GPU로 돌린 Qwen 처리량")
    style_axes(axes[1])

    fig.suptitle(
        "배치 방식 비교: P2P/GDR로 분리는 가능하지만 전체 시간은 NRx 계산과 대기열이 좌우",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "실험 범위: 최적화한 TensorRT 경로. GDR 대기열 깊이는 1, 주 P2P 결과는 2이므로 처리량 수치는 직접 비교하지 않음",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    save(fig, "03_placement_transport_baselines.png")


def figure_03e_stage1_equal_depth():
    """Stage-1 controlled comparison with the same request/response depth."""

    rows = read_csv(PLACEMENT / "DEPTH1_TRANSPORT_COMPARISON.csv")
    by_name = {row["config"]: row for row in rows}
    desired = [
        "MIG same depth 1",
        "Cross P2P depth 1",
        "Cross NIC GDR depth 1",
    ]
    assert all(name in by_name for name in desired)
    labels = [
        "같은 4g\nL1+NRx",
        "분리된 2g+2g\nGPU P2P",
        "분리된 2g+2g\nNIC GDR",
    ]
    colors = [COLORS["blue"], COLORS["cyan"], COLORS["green"]]
    means = [float(by_name[name]["e2e_mean_ms"]) for name in desired]
    p99s = [float(by_name[name]["e2e_p99_ms"]) for name in desired]
    throughput = [
        float(by_name[name]["completion_throughput_slots_s"])
        for name in desired
    ]
    trials = [int(by_name[name]["n_trials"]) for name in desired]

    x = np.arange(len(desired))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
    bars = axes[0].bar(x, means, color=colors, width=0.62, label="평균")
    axes[0].scatter(
        x,
        p99s,
        color=COLORS["navy"],
        marker="D",
        s=44,
        zorder=3,
        label="느린 1% 경계(p99)",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("요청 전체 처리시간(ms)")
    axes[0].set_title("(a) 같은 queue depth=1에서 transport 비교")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars, "{:.3f}")
    style_axes(axes[0])

    bars = axes[1].bar(x, throughput, color=colors, width=0.62)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("완료한 slot/s")
    axes[1].set_title("(b) 직렬 dependency path의 완료 처리량")
    annotate_bars(axes[1], bars, "{:.1f}")
    for index, count in enumerate(trials):
        axes[1].text(
            index,
            throughput[index] * 0.54,
            f"{count}회",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    style_axes(axes[1])

    fig.suptitle(
        "Stage 1: NIC GDR는 cross-MIG GPU-memory 경로를 열지만 P2P보다 평균 0.438 ms 추가",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: optimized direct TensorRT, request 1,415,232 B / result 314,496 B, Qwen은 별도 3g에서 약 10.22~10.24 iter/s; 같은 4g와 2g+2g는 동일 SM 비교가 아님",
        ha="center",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "03e_stage1_equal_depth.png")


def _fiveway_median(variant: str, rate: int, metric: str) -> float:
    values = []
    for trial_dir in sorted((FIVEWAY / variant / f"load_{rate}").glob("trial_*")):
        paths = list(trial_dir.glob("*.json"))
        assert len(paths) == 1, (variant, rate, trial_dir, paths)
        data = read_json(paths[0])
        if variant == "gdr":
            value = data[f"{metric}_p99_ms"] if metric == "sojourn" else data[metric]
        else:
            metric_key = f"{metric}_ms" if metric == "sojourn" else metric
            value = data["metrics"][metric_key]["p99"]
        values.append(float(value))
    assert len(values) == 3, (variant, rate, values)
    return float(np.median(values))


def figure_03b_fiveway_absolute_rate():
    rates = [50, 100, 140, 160, 180, 250, 300, 350]
    variants = [
        ("mps", "전체 GPU 공유(MPS)", COLORS["orange"], "D"),
        ("mig_local", "MIG 안에 L1+NRx", COLORS["blue"], "o"),
        ("mig_mps", "MIG+MPS 안에 L1+NRx", COLORS["purple"], "P"),
        ("p2p", "L1·NRx 분리(GPU 직접/P2P)", COLORS["cyan"], "s"),
        ("gdr", "L1·NRx 분리(NIC 직접/GDR)", COLORS["green"], "^"),
    ]
    series = {
        variant: [_fiveway_median(variant, rate, "sojourn") for rate in rates]
        for variant, _, _, _ in variants
    }
    last_before_cliff = {
        variant: max(rate for rate, p99 in zip(rates, series[variant]) if p99 < 100.0)
        for variant, _, _, _ in variants
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.9), gridspec_kw={"width_ratios": [1.55, 1]})
    for variant, label, color, marker in variants:
        axes[0].plot(
            rates,
            series[variant],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            label=label,
        )
    axes[0].axhline(100, color=COLORS["red"], linestyle="--", linewidth=1.2, label="100 ms 이상: 대기열 붕괴 표시")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("모든 방식에 똑같이 넣은 초당 요청 수")
    axes[0].set_ylabel("도착부터 완료까지 느린 1% 시간(ms, 로그)")
    axes[0].set_title("(a) 배치마다 대기열이 무너지는 지점이 다름")
    axes[0].legend(frameon=False, fontsize=8.2, ncol=2)
    style_axes(axes[0])

    labels = [label for _, label, _, _ in variants]
    stable = [last_before_cliff[variant] for variant, _, _, _ in variants]
    colors = [color for _, _, color, _ in variants]
    bars = axes[1].barh(np.arange(len(labels)), stable, color=colors, height=0.62)
    axes[1].set_yticks(np.arange(len(labels)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 390)
    axes[1].set_xlabel("100 ms 이상으로 폭증하기 전 마지막 측정 요청률")
    axes[1].set_title("(b) 이번 실험에서 안정적으로 처리한 범위")
    for bar, value, label in zip(bars, stable, labels):
        suffix = "+" if label == "전체 GPU 공유(MPS)" and value == max(rates) else ""
        axes[1].text(value + 7, bar.get_y() + bar.get_height() / 2, f"{value}{suffix}/s", va="center", fontsize=9)
    style_axes(axes[1])

    fig.suptitle(
        "같은 요청률 비교: 격리 방식마다 한계는 다르지만 고정된 NRx 경로는 모두 처리 상한이 있음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 다른 AI 작업 없이 초당 50~350개 요청, 10초 x 3회 x 5개 배치 = 120회. 100 ms는 대기열 붕괴 확인선",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "03b_fiveway_absolute_rate.png")


def figure_03c_mig_mps_quota():
    configs = [
        ("l1_30_nrx_70", 30, 70),
        ("l1_50_nrx_50", 50, 50),
        ("l1_70_nrx_30", 70, 30),
    ]
    means = []
    p99s = []
    for path, _, _ in configs:
        trials = []
        for json_path in sorted((MIG_MPS_QUOTA / path).glob("trial*/*.json")):
            data = read_json(json_path)
            trials.append((float(data["mean_ms"]), float(data["p99_ms"])))
        assert len(trials) == 3, (path, trials)
        means.append(float(np.median([trial[0] for trial in trials])))
        p99s.append(float(np.median([trial[1] for trial in trials])))

    x = np.arange(len(configs))
    labels = [f"L1 {l1}%\nNRx {nrx}%" for _, l1, nrx in configs]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    bars = axes[0].bar(x, means, color=[COLORS["green"], COLORS["purple"], COLORS["red"]], width=0.6)
    axes[0].scatter(x, p99s, color=COLORS["navy"], marker="D", s=45, zorder=3, label="느린 1% 경계(p99)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 7.6)
    axes[0].set_ylabel("요청 전체 처리시간(ms)")
    axes[0].set_title("(a) 같은 4g MIG 안의 L1/NRx 몫을 변경")
    axes[0].legend(frameon=False)
    annotate_bars(axes[0], bars, "{:.2f}")
    style_axes(axes[0])

    nrx_share = [nrx for _, _, nrx in configs]
    axes[1].plot(nrx_share, means, color=COLORS["purple"], marker="o", linewidth=2.2, label="평균")
    axes[1].plot(nrx_share, p99s, color=COLORS["navy"], marker="D", linewidth=2.0, label="느린 1% 경계(p99)")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("NRx에 허용한 실행 몫(%)")
    axes[1].set_ylabel("요청 전체 처리시간(ms)")
    axes[1].set_title("(b) L1 몫을 늘려도 전체 처리는 더 느려질 수 있음")
    axes[1].legend(frameon=False)
    axes[1].text(
        0.5,
        0.12,
        "옆 3g의 Qwen은 계속 약 10.21~10.22 iter/s",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=9,
        color=COLORS["navy"],
    )
    style_axes(axes[1])

    fig.suptitle(
        "MIG+MPS의 한계: 실행 몫은 한 파티션 안에서 나뉠 뿐, 새 격리 공간은 생기지 않음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 고정 4g MIG 안에서 L1과 NRx를 별도 MPS 작업으로 실행, 각 몫마다 슬롯 1,000개 x 3회, 두 프로세스는 GDR로 연결",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "03c_mig_mps_quota.png")


TRACKED_CUDA_APIS = [
    ("cudaFree_v3020", "메모리 해제(cudaFree)"),
    ("cudaFreeAsync_v11020", "비동기 해제(cudaFreeAsync)"),
    ("cudaMallocFromPoolAsync_v11020", "메모리 풀 할당"),
    ("cudaMalloc_v3020", "메모리 할당(cudaMalloc)"),
    ("cudaMemcpyAsync_v3020", "비동기 메모리 복사"),
    ("cudaStreamSynchronize_v3020", "Stream 완료 기다리기"),
]


def _sqlite_runtime_totals(path: Path) -> dict[str, tuple[float, int]]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT s.value, SUM(r.end-r.start)/1e6, COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId
            GROUP BY s.value
            """
        ).fetchall()
    return {str(name): (float(total_ms), int(count)) for name, total_ms, count in rows}


def figure_03d_cuda_host_blocking():
    cells = [4, 10, 40, 60]
    variants = [
        ("A_c{c}_alone", "L1만 실행", COLORS["green"]),
        ("A_c{c}_nrx_baseline", "L1+NRx 동시", COLORS["red"]),
        ("A_c{c}_nrx_freeasync", "비동기 free로 변경", COLORS["orange"]),
        ("A_c{c}_nrx_pool", "CUDA 메모리 풀", COLORS["purple"]),
    ]
    totals: dict[str, list[float]] = {}
    breakdown: dict[str, list[float]] = {}
    for pattern, label, _ in variants:
        totals[label] = []
        for cell in cells:
            api = _sqlite_runtime_totals(HOST_BLOCKING / f"{pattern.format(c=cell)}_L1.sqlite")
            totals[label].append(sum(api.get(name, (0.0, 0))[0] for name, _ in TRACKED_CUDA_APIS))
        api = _sqlite_runtime_totals(HOST_BLOCKING / f"{pattern.format(c=40)}_L1.sqlite")
        breakdown[label] = [api.get(name, (0.0, 0))[0] for name, _ in TRACKED_CUDA_APIS]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    api_colors = [COLORS["red"], "#ec4899", "#8b5cf6", COLORS["blue"], COLORS["orange"], COLORS["green"]]
    bottom = np.zeros(len(variants))
    labels = [label for _, label, _ in variants]
    for index, (_, api_label) in enumerate(TRACKED_CUDA_APIS):
        values = np.array([breakdown[label][index] for label in labels])
        if np.any(values > 0):
            axes[0].bar(labels, values, bottom=bottom, color=api_colors[index], label=api_label, width=0.66)
            bottom += values
    for index, total in enumerate(bottom):
        axes[0].text(index, total + 350, f"{total/1000:.2f} s", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, max(bottom) * 1.17)
    axes[0].set_ylabel("CPU가 CUDA API 안에서 기다린 누적시간(ms/30초)")
    axes[0].set_title("(a) 셀 40개: API를 바꿔도 CPU 대기 위치만 이동")
    axes[0].legend(frameon=False, fontsize=7.8, ncol=2)
    style_axes(axes[0])

    for _, label, color in variants:
        axes[1].plot(cells, totals[label], marker="o", linewidth=2.1, label=label, color=color)
    ratio = totals["L1+NRx 동시"][2] / totals["L1만 실행"][2]
    axes[1].annotate(
        f"셀 40개: CPU 대기 {ratio:.1f}배 증가",
        (40, totals["L1+NRx 동시"][2]),
        xytext=(-110, -42),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": COLORS["navy"]},
        fontsize=9,
        fontweight="bold",
    )
    axes[1].set_xlabel("동시에 처리하도록 설정한 셀 수")
    axes[1].set_ylabel("CPU가 CUDA API 안에서 기다린 누적시간(ms/30초)")
    axes[1].set_title("(b) 비동기 메모리 API도 쌓인 GPU 작업은 없애지 못함")
    axes[1].legend(frameon=False, fontsize=8.2)
    style_axes(axes[1])

    fig.suptitle(
        "Nsight 원인 분석: 같은 MIG의 NRx가 L1의 CPU 실행 흐름을 CUDA 호출 안에서 오래 막음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 같은 MIG에 cuPHY L1+NRx 배치. 30초 Nsight 구간의 주요 CUDA API 6개 누적값이며 슬롯 하나의 지연시간이 아님",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "03d_cuda_host_blocking.png")


def _background_metrics(workload: str, policy: str):
    base = MIG_CAUSAL / "06_background_contention" / workload / policy
    nrx = read_json(base / "nrx_timeline.json")
    background = read_json(base / "background_timeline.json")
    burst = next(item for item in nrx["phase_results"] if item["kind"] == "burst")
    activation = 0.0
    for transition in nrx.get("transitions", []):
        if transition.get("event") == "reclaim_request":
            activation = float(transition["activation_ms"])
    work_items = background.get("units", background.get("iterations", []))
    work_latency = background.get("unit_latency_ms", background.get("iteration_latency_ms"))
    assert work_items is not None and work_latency is not None
    return {
        "p99": float(burst["latency_ms"]["p99"]),
        "miss5": float(burst["deadline_miss_ratio"]["5ms"]),
        "max_outstanding": int(nrx["max_outstanding"]),
        "units": len(work_items),
        "unit_p99": float(work_latency["p99"]),
        "activation": activation,
    }


def figure_04_background_reclaim():
    workloads = [
        ("resnet50", "ResNet-50"),
        ("bert_base", "BERT-base"),
        ("whisper_base", "Whisper-base"),
        ("qwen_decode", "Qwen-7B\n생성"),
    ]
    metrics = {
        (display, policy): _background_metrics(path, policy)
        for path, display in workloads
        for policy in ("naive_share", "adaptive_reclaim")
    }
    labels = [display for _, display in workloads]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

    naive = [metrics[(label, "naive_share")]["p99"] for label in labels]
    adaptive = [metrics[(label, "adaptive_reclaim")]["p99"] for label in labels]
    axes[0].bar(x - width / 2, naive, width, color=COLORS["red"], label="계속 같이 실행")
    axes[0].bar(x + width / 2, adaptive, width, color=COLORS["green"], label="요청이 몰리면 NRx에 양보")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("요청 몰림 중 NRx의 느린 1% 시간(ms, 로그)")
    axes[0].set_title("(a) 다른 AI 작업을 그대로 두면 NRx 대기 폭증")
    axes[0].legend(frameon=False, fontsize=8.5)

    naive_miss = [metrics[(label, "naive_share")]["miss5"] for label in labels]
    adaptive_miss = [metrics[(label, "adaptive_reclaim")]["miss5"] for label in labels]
    axes[1].bar(x - width / 2, naive_miss, width, color=COLORS["red"])
    axes[1].bar(x + width / 2, adaptive_miss, width, color=COLORS["green"])
    axes[1].set_ylim(0, 0.78)
    axes[1].set_ylabel("5 ms를 넘긴 NRx 요청 비율")
    axes[1].set_title("(b) 실험 기준시간을 넘긴 요청")

    retained = [
        100 * metrics[(label, "adaptive_reclaim")]["units"] / metrics[(label, "naive_share")]["units"]
        for label in labels
    ]
    activation = [metrics[(label, "adaptive_reclaim")]["activation"] for label in labels]
    bars = axes[2].bar(x, retained, width=0.58, color=COLORS["blue"], label="계속 처리한 다른 AI 작업")
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("계속 처리한 다른 AI 작업 비율(%)")
    axes[2].set_title("(c) 다른 AI 작업 대부분을 유지하면서 NRx에 양보")
    twin = axes[2].twinx()
    twin.plot(x, activation, color=COLORS["orange"], marker="D", linewidth=2, label="양보 시작까지 걸린 시간")
    twin.set_ylabel("NRx에 양보하기까지 걸린 시간(ms)", color=COLORS["orange"])
    twin.tick_params(axis="y", colors=COLORS["orange"])
    annotate_bars(axes[2], bars, "{:.1f}%")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.tick_params(axis="x", labelsize=8.5)
        style_axes(axis)
    twin.spines["top"].set_visible(False)

    fig.suptitle(
        "다른 AI 작업을 짧게 나누면 처리량 대부분을 유지하며 NRx 요청 몰림을 흡수할 수 있음",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 최적화한 TensorRT NRx 대기열과 상주 AI 모델만 사용. 이 실험에는 cuPHY와 전송 경로가 없음",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "04_background_reclaim.png")


def figure_05_gdr_pool_policy():
    gaps = read_csv(GDR_POOL / "POLICY_GAPS.csv")
    comparisons = read_csv(GDR_POOL / "POLICY_COMPARISONS.csv")
    load_bands = ["at_or_below_1000", "1000_to_1500", "above_1500"]
    load_labels = ["낮은 부하\n<=1,000/s", "중간 부하\n1,000~1,500/s", "높은 부하\n>1,500/s"]
    policies = ["static_one", "static_cell", "predicted_finish", "tail_aware"]
    display = [
        "한 NRx에 계속 고정",
        "셀마다 지정한 NRx에 고정",
        "예상 완료가 가장 빠른 NRx",
        "완료 예측 + 느린 경우 억제",
    ]
    policy_display = dict(zip(policies, display))
    colors = [COLORS["red"], COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    lookup = {(row["load_band"], row["policy"]): row for row in gaps}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(load_bands))
    width = 0.19
    for i, (policy, label, color) in enumerate(zip(policies, display, colors)):
        values = [float(lookup[(band, policy)]["no_timely_ratio_median"]) for band in load_bands]
        axes[0].bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    axes[0].set_xticks(x, load_labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("5 ms 안에 쓸 NRx 결과가 없는 비율")
    axes[0].set_title("(a) 실제 크기 GPU 데이터를 처리하는 GDR NRx 3개")
    axes[0].legend(frameon=False, fontsize=8.2, ncol=2)
    style_axes(axes[0])

    names = [f"{policy_display[row['candidate']]}\n대비 {policy_display[row['baseline']]}" for row in comparisons]
    improvements = [100 * float(row["no_timely_improvement_median"]) for row in comparisons]
    better = [f"{row['candidate_better']}/{row['paired_traces']}" for row in comparisons]
    bars = axes[1].barh(np.arange(len(names)), improvements, color=[COLORS["blue"], COLORS["blue"], COLORS["purple"], COLORS["purple"]])
    axes[1].set_yticks(np.arange(len(names)), names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("제시간 결과가 없는 비율 감소(%p, 클수록 좋음)")
    axes[1].set_title("(b) 똑같은 요청 흐름으로 정책끼리 직접 비교")
    for bar, count in zip(bars, better):
        axes[1].text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"더 좋음 {count}", va="center", fontsize=8.2)
    style_axes(axes[1])

    fig.suptitle(
        "NRx 선택 정책 결과: 완료시간을 예측하면 실패할 전송은 줄지만, 요청 수락 판단은 아직 보수적",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 요청 패턴 29개 x 3회 x 정책 4개 = 348회, 5 ms 비교선. '결과 없음'에는 처음부터 기존 수신기를 고른 경우도 포함",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "05_gdr_pool_policy.png")


def figure_05b_gdr_replica_sweep():
    """Show how one, two, and three real GDR endpoints change timely capacity."""

    rows = read_csv(GDR_POOL / "MEDIANS.csv")
    scenarios = [
        (
            "single_periodic",
            "(a) 셀 1개 · 1 ms마다 NRx",
            "1,000 requests/s",
        ),
        (
            "multicell_synchronized",
            "(b) 셀 2개 · 같은 시각에 NRx",
            "2,000 requests/s",
        ),
        (
            "selective_bursty",
            "(c) 셀 4개 · 10% burst 선택",
            "평균 385 requests/s",
        ),
    ]
    policies = ["round_robin", "predicted_finish", "tail_aware"]
    displays = [
        "순서대로 분배",
        "예상 완료가 가장 빠른 곳",
        "완료 예측 + tail guard",
    ]
    colors = [COLORS["orange"], COLORS["blue"], COLORS["purple"]]
    markers = ["o", "s", "D"]
    lookup = {
        (
            int(row["stage"].rsplit("r", 1)[1]),
            row["scenario"],
            row["policy"],
        ): float(row["no_timely_ratio_median"])
        for row in rows
        if row["stage"].startswith("02_replica_sweep/r")
    }

    endpoints = np.array([1, 2, 3])
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), sharey=True)
    for axis, (scenario, title, rate) in zip(axes, scenarios):
        for policy_index, (policy, display, color, marker) in enumerate(zip(
            policies, displays, colors, markers
        )
        ):
            values = [lookup[(endpoint, scenario, policy)] for endpoint in endpoints]
            axis.plot(
                endpoints,
                values,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=6,
                label=display,
            )
            for endpoint, value in zip(endpoints, values):
                x_offset = (-0.065, 0.0, 0.065)[policy_index]
                y_offset = (0.040, -0.050, 0.065)[policy_index]
                axis.text(
                    endpoint + x_offset,
                    min(1.035, max(0.018, value + y_offset)),
                    f"{100 * value:.1f}%",
                    ha="center",
                    fontsize=7.4,
                    color=color,
                )
        axis.set_xticks(endpoints)
        axis.set_xlabel("동시에 상주한 NRx replica 수")
        axis.set_title(f"{title}\n{rate}")
        axis.set_ylim(0, 1.08)
        style_axes(axis)
    axes[0].set_ylabel("5 ms 안에 쓸 NRx 결과가 없는 비율")
    axes[0].legend(frameon=False, fontsize=8.1, loc="lower left")

    fig.suptitle(
        "Stage 2 replica sweep: NRx를 늘리면 capacity는 늘지만, 부하와 정책에 따라 효과가 달라짐",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 실제 1/2/3개의 resident 3g-MIG GDR endpoint, 각 점은 같은 representative trace 1회. 낮을수록 좋으며 full-matrix 통계는 별도 정책 그림에 제시",
        ha="center",
        fontsize=8.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    save(fig, "05b_gdr_replica_sweep.png")


def figure_06_radio_utility():
    rows = read_csv(RADIO / "SUMMARY.csv")
    rows3 = {row["mode"]: row for row in rows if row["endpoint_count"] == "3"}
    assert set(rows3) >= {"none", "all", "utility"}
    modes = ["none", "all", "utility"]
    labels = ["기존 수신기만", "모든 슬롯에 NRx", "어려운 슬롯만 NRx"]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["green"]]

    correct = [float(rows3[mode]["correct_ratio_median"]) for mode in modes]
    requests = [float(rows3[mode]["nrx_requests_median"]) for mode in modes]
    p50 = [float(rows3[mode]["decision_p50_ms_median"]) for mode in modes]
    p99 = [float(rows3[mode]["decision_p99_ms_median"]) for mode in modes]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    bars = axes[0].bar(labels, correct, color=colors, width=0.62)
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("정상적으로 복호한 전송 블록 비율")
    axes[0].set_title("(a) 실제 무선 복호 성공률")
    annotate_bars(axes[0], bars, "{:.2f}")

    bars = axes[1].bar(labels, requests, color=colors, width=0.62)
    axes[1].set_ylim(0, 112)
    axes[1].set_ylabel("슬롯 100개 중 NRx를 호출한 수")
    axes[1].set_title("(b) 실제로 실행한 AI 작업량")
    annotate_bars(axes[1], bars, "{:.0f}")

    x = np.arange(len(labels))
    width = 0.34
    b1 = axes[2].bar(x - width / 2, p50, width, color=COLORS["cyan"], label="중간값")
    b2 = axes[2].bar(x + width / 2, p99, width, color=COLORS["navy"], label="느린 1% 경계(p99)")
    axes[2].axhline(12, linestyle="--", color=COLORS["red"], linewidth=1.2, label="12 ms 뒤에는 결과 폐기")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 13.2)
    axes[2].set_ylabel("최종 결과를 선택하기까지 걸린 시간(ms)")
    axes[2].set_title("(c) 실제 슬롯 처리시간")
    axes[2].legend(frameon=False, fontsize=8.5)
    annotate_bars(axes[2], b1, "{:.2f}")
    annotate_bars(axes[2], b2, "{:.2f}")

    for axis in axes:
        axis.tick_params(axis="x", labelsize=8.4, rotation=8)
        style_axes(axis)

    fig.suptitle(
        "실제 무선 결과: 어려운 슬롯에만 NRx를 써도 성공률은 같고 AI 호출은 25% 감소",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 실제 cuPHY CE -> GDR NRx -> LDPC/CRC, NRx 3개, 요청 100개 x 3회, 결과 유효시간 12 ms",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "06_actual_radio_utility.png")


def figure_06b_radio_cuda_calls():
    api = _sqlite_runtime_totals(RADIO_NSYS)
    selected = [
        ("cudaStreamSynchronize_v3020", "GPU 작업 흐름\n완료 기다리기", COLORS["red"]),
        ("cudaMemcpyAsync_v3020", "비동기\n메모리 복사", COLORS["orange"]),
        ("cudaFree_v3020", "GPU 메모리\n해제", COLORS["purple"]),
        ("cudaMalloc_v3020", "GPU 메모리\n할당", COLORS["blue"]),
        ("cudaDeviceSynchronize_v3020", "GPU 전체 완료\n기다리기", COLORS["gray"]),
        ("cudaDeviceFlushGPUDirectRDMAWrites_v11030", "GDR 쓰기\n확인", COLORS["green"]),
    ]
    values = [api[name][0] for name, _, _ in selected]
    counts = [api[name][1] for name, _, _ in selected]

    with sqlite3.connect(RADIO_NSYS) as connection:
        kernel_rows = connection.execute(
            """
            SELECT s.value, SUM(k.end-k.start)/1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.demangledName
            GROUP BY s.value
            """
        ).fetchall()
    kernel_total = sum(float(total) for _, total in kernel_rows)
    conversion = sum(float(total) for name, total in kernel_rows if "convert_kernel" in str(name))
    ldpc_split = sum(float(total) for name, total in kernel_rows if "ldpc2_BG1_split" in str(name))
    other = kernel_total - conversion - ldpc_split

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    x = np.arange(len(selected))
    bars = axes[0].bar(x, values, color=[color for _, _, color in selected], width=0.65)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, [label for _, label, _ in selected])
    axes[0].set_ylabel("CPU가 CUDA API 안에서 보낸 누적시간(ms, 로그)")
    axes[0].set_title("(a) 실제 무선 처리 중 CPU가 기다린 위치")
    for bar, value, count in zip(bars, values, counts):
        axes[0].annotate(
            f"{value:.3f} ms\n{count}회",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    style_axes(axes[0])

    kernel_values = [conversion, ldpc_split, other]
    kernel_labels = ["FP32/FP16\n데이터 형식 변환", "주요 LDPC\n복호 계산", "기타 cuPHY /\nTensorFlow / 복사"]
    kernel_colors = [COLORS["orange"], COLORS["blue"], COLORS["gray"]]
    wedges, _, autotexts = axes[1].pie(
        kernel_values,
        labels=kernel_labels,
        colors=kernel_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8.5},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    for text_item in autotexts:
        text_item.set_fontweight("bold")
    axes[1].set_title("(b) GPU에서는 전송보다 데이터 형식 변환 비중이 큼")

    fig.suptitle(
        "실제 cuPHY-GDR-NRx 경로: GDR 확인보다 동기화와 데이터 형식 변환 비용이 더 큼",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "실험 범위: 실제 CE -> GDR NRx 3개 -> LDPC/CRC 경로에서 요청 12개를 Nsight로 추적. 모든 배치 조건을 대표하지는 않음",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    save(fig, "06b_actual_radio_cuda_calls.png")


def main():
    font_manager.fontManager.addfont(str(KOREAN_FONT))
    korean_family = font_manager.FontProperties(fname=KOREAN_FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": korean_family,
            "font.size": 10,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "axes.unicode_minus": False,
        }
    )
    figure_00_architecture_map()
    figure_00a_gdr_evolution()
    figure_00d_dart_rx_overall_architecture()
    figure_00_three_local_baselines()
    figure_00c_mps_multi_nrx_breakdown()
    figure_00b_why_cross_endpoints()
    figure_01_isolation_and_queue_cliff()
    figure_01b_nrx_wrapper_optimization()
    figure_02_fragmentation()
    figure_03_placement_and_transport()
    figure_03e_stage1_equal_depth()
    figure_03b_fiveway_absolute_rate()
    figure_03c_mig_mps_quota()
    figure_03d_cuda_host_blocking()
    figure_04_background_reclaim()
    figure_05_gdr_pool_policy()
    figure_05b_gdr_replica_sweep()
    figure_06_radio_utility()
    figure_06b_radio_cuda_calls()
    print(f"wrote 19 figures to {OUT}")


if __name__ == "__main__":
    main()
