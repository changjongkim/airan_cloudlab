#!/usr/bin/env python3
"""Build 14-page A4-landscape progress slide deck PDF.

Output: BigdataLab_progress_ppt_26.06.20_KCJ.pdf
"""
from __future__ import annotations

import os
import sys
import warnings as _warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/changjongkim/New_research/cloudlab_results/results/visual_evidence")
FIG_DIR = BASE_DIR / "figures"
OUTPUT_PDF = BASE_DIR / "BigdataLab_progress_ppt_26.06.20_KCJ.pdf"

PAGE_W, PAGE_H = 11.69, 8.27  # A4 landscape inches
NAVY = "#1e40af"
GRAY_DARK = "#374151"
GRAY_MID = "#6b7280"
GRAY_LIGHT = "#d1d5db"

# Korean-friendly font fallback
plt.rcParams["font.family"] = [
    "Apple SD Gothic Neo",
    "AppleGothic",
    "Nanum Gothic",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False
_warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

WARNINGS: list[str] = []
LOG: list[str] = []


# ---------------------------------------------------------------------------
# Slide content
# ---------------------------------------------------------------------------
SLIDES = [
    {
        "slide_num": 0,
        "title": "Progress",
        "subtitle": "AI-RAN GPU Isolation — Mechanism Deep-Dive",
        "bullets": [
            "Changjong Kim",
            "Bigdata and HPC Lab",
            "Department of Computer Science and Engineering",
            "Seoul National University of Science and Technology",
            "2026-06-20",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "Progress Report 2026-06-20",
        "transition": "Recap previous results, then drill into the mechanism.",
    },
    {
        "slide_num": 1,
        "title": "Recap & Today's Question",
        "subtitle": "From overhead measurement to mechanism identification",
        "bullets": [
            "Previous (5/29): MIG single-tenant overhead **+13–32ms** (Exp1–6)",
            "Today: AI co-tenant 환경에서 비용은 **어디서, 왜** 발생하는가?",
            "Scale: **n=500–1000/condition** (vs n=20 last time) — statistical power 25×",
            "Platforms: CloudLab d8545 driver 550 + **Perlmutter cross-validation**",
            "Goal: identify the **hardware resource** behind the overhead",
        ],
        "figure_main": "fig01_partition_baseline.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "Recap of 5/29 deck Exp1–6",
        "transition": "Cross-partition은 봤는데, same-partition은 어떻게 되나?",
    },
    {
        "slide_num": 2,
        "title": "Exp7) Same-Partition Coloc = Catastrophic Collapse",
        "subtitle": "Cross-partition stable, same-partition collapses 8×",
        "bullets": [
            "Cross-partition stays **~45ms**, same-partition coloc collapses to **~357ms** (**8×**)",
            "**12+ workloads** (chanpred, NeuralRx, qwen, xapp, ResNet) all collapse to ~357ms",
            "Workload-invariant → **systemic mechanism**, not a workload-specific trick",
            "n=500/condition, std=9ms (stable, reproducible)",
            "First evidence: cost is from **placement**, not the AI's nature",
        ],
        "figure_main": "fig04_g_coloc_explosion.png",
        "figure_aux": "",
        "table_md": (
            "| Workload | Cross-part (ms) | Same-part coloc (ms) |\n"
            "|---|---|---|\n"
            "| chanpred | 45 | 357 |\n"
            "| NeuralRx | — | 360 |\n"
            "| qwen | — | 357 |\n"
            "| xapp | — | 358 |\n"
            "| ResNet | — | 358 |"
        ),
        "footer_ref": "PART A §4 (CloudLab 6/1 G_coloc, n=500/condition)",
        "transition": "큰 partition으로 SM/HBM을 더 주면 해결되나?",
    },
    {
        "slide_num": 3,
        "title": "Exp8) Larger Partition = MORE Catastrophic",
        "subtitle": "Counter-intuitive partition size paradox",
        "bullets": [
            "Larger slice does **NOT** help; **4g coloc (371ms) WORSE than 2g (369ms)**",
            "Resource bandwidth share is decisive — but **NOT via partition allocation**",
            "Rejects the obvious 'give more SMs' mitigation hypothesis",
            "Bottleneck is **outside** the partition boundary",
            "First hint: the costly resource is **chip-global**, not slice-local",
        ],
        "figure_main": "fig_supp_03_g_coloc_partition_paradox.png",
        "figure_aux": "",
        "table_md": (
            "| Partition | Coloc latency (ms) |\n"
            "|---|---|\n"
            "| 2g.10gb | 369 |\n"
            "| 3g.20gb | 361 |\n"
            "| 4g.20gb | **371** |"
        ),
        "footer_ref": "PART A §4.1 (CloudLab 6/1 G_coloc 2g/3g/4g comparison)",
        "transition": "Placement는 binary인가, gradient인가?",
    },
    {
        "slide_num": 4,
        "title": "Exp9) Placement = Binary Decision, No Soft Transition",
        "subtitle": "One config bit decides 8× collapse",
        "bullets": [
            "Same workload (chanpred), only placement differs: **cross 44ms ↔ coloc 357ms**",
            "**No middle ground**, no gradient between the two regimes",
            "One config mistake = **8× collapse** — operational fragility",
            "Implication: there exists a **single shared resource** flipping state",
            "Now we must find WHICH resource — switch to NSYS profiling",
        ],
        "figure_main": "fig05_h_dual_sanity.png",
        "figure_aux": "",
        "table_md": (
            "| Condition | Placement | Latency (ms) |\n"
            "|---|---|---|\n"
            "| H_F1 | cross-partition | **44** |\n"
            "| H_G1 | same-partition coloc | **357** |"
        ),
        "footer_ref": "PART A §5 (CloudLab 6/1 H_dual, 9 conditions)",
        "transition": "같은 hardware에서 왜 8× 차이? 무엇이 변하는가?",
    },
    {
        "slide_num": 5,
        "title": "NSYS Layer 1 — Gap is at Memcpy/Memset Boundary",
        "subtitle": "Cost is NOT inside the kernel — it is between kernels",
        "bullets": [
            "L1 **kernel duration is unchanged** under coloc",
            "Inter-kernel **'gap' grows** by exactly the missing latency",
            "Gap is NOT random idle — concentrated at **memcpy/memset boundaries**",
            "Aggregated across **30 conditions** → robust, not cherry-picked",
            "The bottleneck is a **memory-transfer wait**, not compute",
        ],
        "figure_main": "fig12_nsys_kernel_vs_activity_gap.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "PART B §12 (CloudLab 5/31 NSYS deep, 30-condition aggregate)",
        "transition": "Gap이 늘었다면 — op count가 늘었나, op duration이 늘었나?",
    },
    {
        "slide_num": 6,
        "title": "NSYS Layer 2 — Same Ops, Just Slower",
        "subtitle": "Rejects 'more work' hypothesis — count invariant, duration inflates",
        "bullets": [
            "memcpy call count: **5,778 → 5,778 (0% change)**",
            "memset count: unchanged",
            "memcpy avg duration: **4.2 → 9.8μs (+133%)**",
            "AI is NOT giving L1 more work — **same work, blocked longer**",
            "Confirms: cost is **waiting**, not extra activity",
        ],
        "figure_main": "fig13_nsys_memory_activity_breakdown.png",
        "figure_aux": "",
        "table_md": (
            "| Metric | Alone | Coloc | Change |\n"
            "|---|---|---|---|\n"
            "| memcpy count | 5,778 | 5,778 | **0%** |\n"
            "| memcpy avg μs | 4.2 | 9.8 | **+133%** |"
        ),
        "footer_ref": "PART C §15.1, §15.1.5",
        "transition": "왜 같은 op가 더 오래 걸리나? 무엇이 op를 막는가?",
    },
    {
        "slide_num": 7,
        "title": "NSYS Layer 3 — Bimodal Signature = Direct Queue Evidence",
        "subtitle": "Per-call duration splits into two discrete modes",
        "bullets": [
            "Per-call **60KB memcpy** duration splits into **TWO discrete modes**",
            "Alone: single gaussian peak at **4.2μs**",
            "Coloc: **bimodal 4.2μs ↔ 14.3μs** (fast=queue empty, slow=queue waiting)",
            "Discrete state, not gaussian noise → **arbitration signature**",
            "This is the smoking gun for a **shared serialized queue**",
        ],
        "figure_main": "fig_slide7_60kb_bimodal_histogram.png",
        "figure_aux": "",
        "table_md": (
            "| Mode | Alone (μs) | Coloc (μs) |\n"
            "|---|---|---|\n"
            "| Fast peak | 4.2 | 4.2 |\n"
            "| Slow peak | — | **14.3** |\n"
            "| Distribution | unimodal | **bimodal** |"
        ),
        "footer_ref": "PART C §16, §16.1 (CloudLab 5/31 NSYS SQLite per-call extraction)",
        "transition": "어떤 queue? 위치를 알아야 격리 가능성을 판단한다.",
    },
    {
        "slide_num": 8,
        "title": "NSYS Layer 4 — Chip-Wide PCIe/DMA Copy Engine",
        "subtitle": "CLIMAX — the shared resource is identified",
        "bullets": [
            "Direction breakdown: **H2D 17,334 (89.8%) / D2H 1,920 (9.9%) / D2D 24 (0.12%)**",
            "**D2D ≈ 0** → queue is NOT at HBM controller (HBM carries D2D)",
            "**100% host-path** → queue is at **chip-wide PCIe/DMA copy engine**",
            "This single resource is **shared across the WHOLE chip** — partition cannot isolate it",
            "Explains: Slide 3 paradox, Slide 4 binary, Slide 7 bimodal — **all consistent**",
        ],
        "figure_main": "fig_slide8_memcpy_direction_breakdown.png",
        "figure_aux": "",
        "table_md": (
            "| Direction | Count | Share |\n"
            "|---|---|---|\n"
            "| H2D | 17,334 | **89.8%** |\n"
            "| D2H | 1,920 | 9.9% |\n"
            "| D2D | 24 | **0.12%** |\n"
            "| Other | 21 | 0.11% |"
        ),
        "footer_ref": "PART D §20.1, §23 (NSYS direction breakdown, 4 conditions)",
        "transition": "이건 MIG 결함인가, 아니면 NVIDIA stack 전체의 문제인가?",
    },
    {
        "slide_num": 9,
        "title": "Cross-Platform Validation — Mechanism is Universal",
        "subtitle": "Same signature on Perlmutter (no-MIG, different driver, different cluster)",
        "bullets": [
            "(A) Bimodal signature **reproduced**: **4.2 ↔ 16.8μs** on Perlmutter",
            "(B) cudaFree blow-up: alone **246μs** → NeuralRx **3,752μs (15×)** → sat_hbm MPS **115,506μs (115ms!)**",
            "(C) **correlationId causal proof**: **60–82% of GPU gap** overlaps EXACTLY with cudaFree host-blocking",
            "Conclusion: PCIe/DMA queue is an **NVIDIA-stack-wide fundamental property**",
            "Not a MIG bug — **disabling MIG does not escape it**",
        ],
        "figure_main": "figF15_memcpy_bimodal.png",
        "figure_aux": "figF14_cudafree_host_blocking.png",
        "table_md": (
            "| Metric | Alone | NeuralRx default | sat_hbm MPS |\n"
            "|---|---|---|---|\n"
            "| cudaFree avg | 246μs | **3,752μs (15×)** | **115,506μs (115ms)** |\n"
            "| Bimodal peaks | 4.2μs | 4.2 ↔ 16.8μs | — |\n"
            "| Gap↔cudaFree overlap | — | **60–82%** | — |"
        ),
        "footer_ref": "PART F §10.5–10.9 (Perlmutter no-MIG SQLite deep analysis)",
        "transition": "그럼 MPS는 cudaFree wait을 우회할 수 있나?",
    },
    {
        "slide_num": 10,
        "title": "MPS — Recovers Compute AI, Catastrophic for Memory AI",
        "subtitle": "Same trap, different trigger",
        "bullets": [
            "Compute AI: MPS **RECOVERS** — NeuralRx **389→40ms**, forecaster **381→42ms**, qwen **185→43ms**",
            "Memory AI: MPS **CATASTROPHIC** — sat_hbm **426 → 6,985ms** (**bistable 7-sec freeze**)",
            "Why recover? True concurrent exec → compute parallelizes → cudaFree returns fast",
            "Why collapse? Memory hog **starves DRAM** → device truly stuck → cudaFree blocks 115ms → 7s",
            "**No universal escape** — same mechanism, different trigger",
        ],
        "figure_main": "figF8_mps_vs_default_vs_mig.png",
        "figure_aux": "figF9_mps_sat_hbm_bistable.png",
        "table_md": (
            "| Workload | Default | MPS | MIG cross-part |\n"
            "|---|---|---|---|\n"
            "| NeuralRx | 389ms | **40ms** | 197ms |\n"
            "| forecaster | 381ms | **42ms** | — |\n"
            "| qwen | 185ms | **43ms** | — |\n"
            "| sat_hbm | 426ms | **6,985ms** | — |"
        ),
        "footer_ref": "PART F §7–§8 (Perlmutter MPS 5-workload comparison)",
        "transition": "이 모든 발견을 어떻게 하나의 framework로 종합하나?",
    },
    {
        "slide_num": 11,
        "title": "Cost Decomposition — 3-Platform Unified Framework",
        "subtitle": "Two hardware-level costs, not one — the paper contribution",
        "bullets": [
            "MIG overhead = **TWO hardware-level costs**, not one monolithic 'overhead'",
            "**Structural (memset)** = partition HBM bandwidth share — **controllable** by partition choice",
            "**Contention (memcpy)** = chip-wide PCIe/DMA queue — **NOT controllable**",
            "3 platforms (MIG / no-MIG / MPS) each **avoid one cost but fail at the other**",
            "**No universal isolation** exists in current NVIDIA stack",
        ],
        "figure_main": "fig08_tradeoff_summary.png",
        "figure_aux": "",
        "table_md": (
            "| Cost type | Hardware | MIG isolates? |\n"
            "|---|---|---|\n"
            "| Structural | HBM bandwidth | **Yes** |\n"
            "| Contention | PCIe/DMA copy engine | **NO** |"
        ),
        "footer_ref": "PART A §15.2, §15.3, §23 (mechanism decomposition)",
        "transition": "이 framework의 운영 함의(design rules)는?",
    },
    {
        "slide_num": 12,
        "title": "Design Rules & Next Steps",
        "subtitle": "Measurement-validated operational guidance",
        "bullets": [
            "**Rule 1**: L1과 AI는 반드시 **다른 MIG partition** (single mistake → **8× collapse**)",
            "**Rule 2**: Partition 크기는 **L1 SM/HBM headroom으로만** 결정 (coloc 회피용 무의미)",
            "**Rule 3**: AI workload **classification 필요** (memory-bound AI는 MPS에도 위험)",
            "**Rule 4**: **MIG isolation ≠ chip-wide resource isolation** (paper main contribution)",
            "Next: CloudLab Phase 4 (MIG-off + MPS cross-check), N-AI scaling law, driver 550 baseline",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": (
            "| Next Step | Target |\n"
            "|---|---|\n"
            "| CloudLab Phase 4 | MIG off + MPS — cross-check Perlmutter on same HW |\n"
            "| N-AI scaling law | L1 + N AI same-partition, direct measurement |\n"
            "| Driver 550 baseline | Reproduce on CloudLab d8545 |"
        ),
        "footer_ref": "Synthesis across PART A–G + Perlmutter PART F",
        "transition": "End.",
    },
    {
        "slide_num": 13,
        "title": "Thank You",
        "subtitle": "Questions & Discussion",
        "bullets": [
            "Changjong Kim",
            "Bigdata and HPC Lab, SeoulTech",
            "changjong5238@gmail.com",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "Progress Report 2026-06-20",
        "transition": "",
    },
]

TOTAL_PAGES = len(SLIDES)  # 14
LAST_INDEX = TOTAL_PAGES - 1  # 13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_md_bold(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip **bold** markers; return clean text plus bold spans for matplotlib."""
    out = []
    spans: list[tuple[int, int]] = []
    i = 0
    bold_start: int | None = None
    while i < len(text):
        if text[i:i + 2] == "**":
            if bold_start is None:
                bold_start = len(out)
            else:
                spans.append((bold_start, len(out)))
                bold_start = None
            i += 2
            continue
        out.append(text[i])
        i += 1
    if bold_start is not None:
        # Unclosed — treat as plain text
        pass
    return "".join(out), spans


def _draw_header(fig, title: str) -> None:
    """Draw bold title with navy underline at the top of the slide."""
    fig.text(
        0.06, 0.92, title,
        fontsize=24, fontweight="bold",
        color=GRAY_DARK, ha="left", va="center",
    )
    # Navy underline
    line_ax = fig.add_axes((0.06, 0.885, 0.88, 0.004))
    line_ax.set_facecolor(NAVY)
    line_ax.set_xticks([]); line_ax.set_yticks([])
    for spine in line_ax.spines.values():
        spine.set_visible(False)


def _draw_subtitle(fig, subtitle: str) -> None:
    if not subtitle:
        return
    fig.text(
        0.06, 0.86, subtitle,
        fontsize=12, fontstyle="italic",
        color=GRAY_MID, ha="left", va="center",
    )


def _draw_footer(fig, page_idx: int, footer_ref: str, transition: str) -> None:
    """Bottom-left lab tag, bottom-right page number, bottom-center source ref, transition note."""
    fig.text(
        0.04, 0.025, "▎BigData and HPC Lab",
        fontsize=9, color=GRAY_MID, ha="left", va="bottom",
    )
    fig.text(
        0.96, 0.025, f"{page_idx} / {LAST_INDEX}",
        fontsize=9, color=GRAY_MID, ha="right", va="bottom",
    )
    if footer_ref:
        fig.text(
            0.5, 0.025, footer_ref,
            fontsize=8, fontstyle="italic", color=GRAY_MID,
            ha="center", va="bottom",
        )
    if transition:
        fig.text(
            0.5, 0.055, transition,
            fontsize=9, fontstyle="italic", color=GRAY_LIGHT,
            ha="center", va="bottom",
        )


def _render_bullets(fig, bullets: list[str], top: float, bottom: float, left: float = 0.06, right: float = 0.94) -> None:
    """Render bullet list inside the given vertical range."""
    if not bullets:
        return
    n = len(bullets)
    # Vertical positions (top-down)
    y_top = top - 0.015
    y_bottom = bottom + 0.01
    if n == 1:
        ys = [(y_top + y_bottom) / 2]
    else:
        step = (y_top - y_bottom) / max(n - 1, 1)
        ys = [y_top - i * step for i in range(n)]

    for bullet, y in zip(bullets, ys):
        clean, bold_spans = _strip_md_bold(bullet)
        sub = clean.startswith("  ") or clean.startswith("\t")
        marker_x = left + (0.02 if sub else 0.0)
        text_x = marker_x + 0.018
        fig.text(marker_x, y, "▪", fontsize=14, color=NAVY, ha="left", va="center")

        # Build colored text: bold parts in dark navy + bold weight
        # Simple approach: render plain text; overlay bold segments
        fig.text(text_x, y, clean.strip(), fontsize=13.5, color=GRAY_DARK,
                 ha="left", va="center")
        # Overlay bold: place bold parts as separate texts after measuring offsets.
        # Easier: render the full string with bold spans using a simple post-processing pass.


def _render_bullets_richtext(fig, bullets: list[str], top: float, bottom: float,
                             left: float = 0.06) -> None:
    """Render bullets with **bold** support using sequential text segments."""
    if not bullets:
        return
    n = len(bullets)
    y_top = top - 0.02
    y_bottom = bottom + 0.01
    if n == 1:
        ys = [(y_top + y_bottom) / 2]
    else:
        step = (y_top - y_bottom) / max(n - 1, 1)
        ys = [y_top - i * step for i in range(n)]

    for bullet, y in zip(bullets, ys):
        clean, spans = _strip_md_bold(bullet)
        sub = bullet.startswith("  ") or bullet.startswith("\t")
        marker_x = left + (0.02 if sub else 0.0)
        text_x = marker_x + 0.018

        fig.text(marker_x, y, "▪", fontsize=14, color=NAVY, ha="left", va="center",
                 fontweight="bold")

        # Segment the text into (text, bold) tokens
        segments: list[tuple[str, bool]] = []
        cursor = 0
        for s, e in spans:
            if cursor < s:
                segments.append((clean[cursor:s], False))
            segments.append((clean[s:e], True))
            cursor = e
        if cursor < len(clean):
            segments.append((clean[cursor:], False))

        # Convert axes coords to display coords for sequential layout
        renderer = fig.canvas.get_renderer()
        cur_x = text_x
        fig_w_inches = fig.get_size_inches()[0]
        dpi = fig.dpi
        for seg, is_bold in segments:
            if not seg:
                continue
            weight = "bold" if is_bold else "normal"
            color = NAVY if is_bold else GRAY_DARK
            t = fig.text(cur_x, y, seg, fontsize=13.5, color=color,
                         ha="left", va="center", fontweight=weight)
            bbox = t.get_window_extent(renderer=renderer)
            width_inches = bbox.width / dpi
            cur_x += width_inches / fig_w_inches


def _place_image(fig, rel_path: str, rect: tuple[float, float, float, float]) -> bool:
    """Place an image inside the given (x, y, w, h) rect, preserving aspect."""
    path = FIG_DIR / rel_path
    if not path.exists():
        msg = f"Missing figure: {rel_path}"
        WARNINGS.append(msg)
        print(f"[WARN] {msg}", file=sys.stderr)
        return False

    try:
        img = Image.open(path)
    except Exception as exc:
        WARNINGS.append(f"Failed to open {rel_path}: {exc}")
        return False

    x, y, w, h = rect
    # Compute aspect-preserving fit inside rect
    fig_w_in, fig_h_in = fig.get_size_inches()
    rect_w_in = w * fig_w_in
    rect_h_in = h * fig_h_in
    img_w, img_h = img.size
    img_aspect = img_w / img_h
    rect_aspect = rect_w_in / rect_h_in

    if img_aspect > rect_aspect:
        # image is wider — fit by width
        new_w = w
        new_h = (rect_w_in / img_aspect) / fig_h_in
    else:
        # image is taller — fit by height
        new_h = h
        new_w = (rect_h_in * img_aspect) / fig_w_in

    new_x = x + (w - new_w) / 2
    new_y = y + (h - new_h) / 2

    ax = fig.add_axes((new_x, new_y, new_w, new_h))
    ax.imshow(img)
    ax.axis("off")
    return True


def _parse_table(md: str) -> tuple[list[str], list[list[str]]]:
    """Parse a small GitHub-style markdown table; return (headers, rows)."""
    lines = [ln.strip() for ln in md.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], []

    def split_row(line: str) -> list[str]:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        return parts

    headers = split_row(lines[0])
    rows: list[list[str]] = []
    # Skip the separator line (---)
    data_lines = lines[1:]
    if data_lines and set(data_lines[0].replace("|", "").replace(" ", "")) <= set("-:"):
        data_lines = data_lines[1:]
    for ln in data_lines:
        rows.append(split_row(ln))
    return headers, rows


def _render_table(fig, table_md: str, rect: tuple[float, float, float, float]) -> bool:
    """Render a small table inside the given rect."""
    headers, rows = _parse_table(table_md)
    if not headers:
        return False

    x, y, w, h = rect
    ax = fig.add_axes((x, y, w, h))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    n_cols = len(headers)
    n_rows = len(rows) + 1  # incl header
    if n_rows == 0 or n_cols == 0:
        return False

    col_w = 1.0 / n_cols
    row_h = 1.0 / n_rows
    fontsize = max(7.5, min(10.5, 11 - 0.4 * max(n_rows - 4, 0)))

    # Header background (navy)
    ax.add_patch(plt.Rectangle((0, 1 - row_h), 1.0, row_h,
                               facecolor=NAVY, edgecolor="white", linewidth=0.8))
    for ci, h_text in enumerate(headers):
        clean, _ = _strip_md_bold(h_text)
        ax.text(
            ci * col_w + col_w / 2, 1 - row_h / 2,
            clean, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold",
        )

    # Data rows
    for ri, row in enumerate(rows):
        y_top = 1 - (ri + 2) * row_h
        bg = "#f3f4f6" if ri % 2 == 0 else "white"
        ax.add_patch(plt.Rectangle((0, y_top), 1.0, row_h,
                                   facecolor=bg, edgecolor=GRAY_LIGHT, linewidth=0.5))
        for ci, cell in enumerate(row):
            if ci >= n_cols:
                continue
            clean, spans = _strip_md_bold(cell)
            is_bold = bool(spans) and spans[0] == (0, len(clean))
            weight = "bold" if is_bold else "normal"
            color = NAVY if is_bold else GRAY_DARK
            ax.text(
                ci * col_w + col_w / 2, y_top + row_h / 2,
                clean, ha="center", va="center",
                color=color, fontsize=fontsize, fontweight=weight,
            )
    return True


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def build_title_slide(slide: dict) -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")

    # Top decorative bar
    bar = fig.add_axes((0.0, 0.96, 1.0, 0.04))
    bar.set_facecolor(NAVY); bar.set_xticks([]); bar.set_yticks([])
    for sp in bar.spines.values():
        sp.set_visible(False)

    # Centered "Progress" big text
    fig.text(0.5, 0.66, slide["title"], fontsize=54, fontweight="bold",
             color=NAVY, ha="center", va="center")
    if slide.get("subtitle"):
        fig.text(0.5, 0.56, slide["subtitle"], fontsize=18, fontstyle="italic",
                 color=GRAY_DARK, ha="center", va="center")

    # Underline below title
    ul = fig.add_axes((0.30, 0.535, 0.40, 0.003))
    ul.set_facecolor(NAVY); ul.set_xticks([]); ul.set_yticks([])
    for sp in ul.spines.values():
        sp.set_visible(False)

    # Author info stacked
    info_lines = slide["bullets"]
    n = len(info_lines)
    y_start = 0.42
    for i, line in enumerate(info_lines):
        y = y_start - i * 0.05
        fontsize = 16 if i == 0 else 13
        weight = "bold" if i == 0 else "normal"
        color = GRAY_DARK if i == 0 else GRAY_MID
        fig.text(0.5, y, line, fontsize=fontsize, fontweight=weight,
                 color=color, ha="center", va="center")

    # Footer
    _draw_footer(fig, slide["slide_num"], slide.get("footer_ref", ""),
                 slide.get("transition", ""))
    return fig


def build_thanks_slide(slide: dict) -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")

    # Top decorative bar
    bar = fig.add_axes((0.0, 0.96, 1.0, 0.04))
    bar.set_facecolor(NAVY); bar.set_xticks([]); bar.set_yticks([])
    for sp in bar.spines.values():
        sp.set_visible(False)

    fig.text(0.5, 0.62, slide["title"], fontsize=64, fontweight="bold",
             color=NAVY, ha="center", va="center")
    if slide.get("subtitle"):
        fig.text(0.5, 0.50, slide["subtitle"], fontsize=18, fontstyle="italic",
                 color=GRAY_MID, ha="center", va="center")

    # Underline
    ul = fig.add_axes((0.32, 0.475, 0.36, 0.003))
    ul.set_facecolor(NAVY); ul.set_xticks([]); ul.set_yticks([])
    for sp in ul.spines.values():
        sp.set_visible(False)

    # Contact info
    for i, line in enumerate(slide["bullets"]):
        y = 0.38 - i * 0.05
        fig.text(0.5, y, line, fontsize=14, color=GRAY_DARK,
                 ha="center", va="center")

    _draw_footer(fig, slide["slide_num"], slide.get("footer_ref", ""),
                 slide.get("transition", ""))
    return fig


def build_content_slide(slide: dict) -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")

    _draw_header(fig, slide["title"])
    _draw_subtitle(fig, slide.get("subtitle", ""))

    # Bullets: top region (subtitle below underline)
    bullet_top = 0.83
    bullet_bottom = 0.55
    _render_bullets_richtext(fig, slide["bullets"], bullet_top, bullet_bottom)

    has_main = bool(slide.get("figure_main"))
    has_aux = bool(slide.get("figure_aux"))
    has_table = bool(slide.get("table_md"))

    # Lower region for figure + table
    lower_top = 0.50
    lower_bottom = 0.10
    lower_h = lower_top - lower_bottom

    if has_main and has_aux and has_table:
        # main | aux on left half, table on right
        # Actually: per spec — main + aux side-by-side, table beside or below.
        # Layout: left 60% figures (main, aux), right 36% table
        fig_w = 0.55
        each_w = fig_w / 2
        _place_image(fig, slide["figure_main"],
                     (0.06, lower_bottom, each_w - 0.01, lower_h))
        _place_image(fig, slide["figure_aux"],
                     (0.06 + each_w, lower_bottom, each_w - 0.01, lower_h))
        _render_table(fig, slide["table_md"],
                      (0.66, lower_bottom + 0.05, 0.30, lower_h - 0.10))
    elif has_main and has_aux:
        # main + aux side by side, full width
        each_w = 0.43
        _place_image(fig, slide["figure_main"],
                     (0.06, lower_bottom, each_w, lower_h))
        _place_image(fig, slide["figure_aux"],
                     (0.06 + each_w + 0.02, lower_bottom, each_w, lower_h))
    elif has_main and has_table:
        # figure on left, table on right
        _place_image(fig, slide["figure_main"],
                     (0.06, lower_bottom, 0.55, lower_h))
        _render_table(fig, slide["table_md"],
                      (0.66, lower_bottom + 0.04, 0.30, lower_h - 0.08))
    elif has_main:
        # figure centered, wide
        _place_image(fig, slide["figure_main"],
                     (0.18, lower_bottom, 0.64, lower_h))
    elif has_table:
        # large table centered
        _render_table(fig, slide["table_md"],
                      (0.20, lower_bottom + 0.04, 0.60, lower_h - 0.08))
    else:
        # nothing; expand bullet region to use the rest
        # (Bullets already rendered above; just leave blank)
        pass

    _draw_footer(fig, slide["slide_num"], slide.get("footer_ref", ""),
                 slide.get("transition", ""))
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not FIG_DIR.exists():
        print(f"[ERROR] Figures dir not found: {FIG_DIR}", file=sys.stderr)
        return 1

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    LOG.append(f"Building {len(SLIDES)} slides -> {OUTPUT_PDF.name}")

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        for slide in SLIDES:
            idx = slide["slide_num"]
            if idx == 0:
                fig = build_title_slide(slide)
                LOG.append(f"  slide {idx}: title")
            elif idx == LAST_INDEX:
                fig = build_thanks_slide(slide)
                LOG.append(f"  slide {idx}: thank-you")
            else:
                fig = build_content_slide(slide)
                LOG.append(f"  slide {idx}: content")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

        meta = pdf.infodict()
        meta["Title"] = "Progress — AI-RAN GPU Isolation Mechanism Deep-Dive"
        meta["Author"] = "Changjong Kim"
        meta["Subject"] = "MIG / PCIe-DMA queue mechanism — 2026-06-20"
        meta["Keywords"] = "MIG, AI-RAN, PCIe, DMA, NSYS, Perlmutter"

    LOG.append(f"Wrote: {OUTPUT_PDF}  ({OUTPUT_PDF.stat().st_size/1024:.1f} KB)")
    print("\n".join(LOG))
    if WARNINGS:
        print("\nWARNINGS:")
        for w in WARNINGS:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
