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
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 1,
        "title": "Today",
        "subtitle": "",
        "bullets": [
            "**Continuation of Exp1–6 (5/29):** MIG single-tenant overhead measured "
            "at +13–32 ms.",
            "**Goal:** Identify the hardware resource responsible for the L1 latency "
            "cost under AI co-tenancy.",
            "**Setup:** n=500–1000 per condition (25× the prior n=20), CloudLab "
            "d8545 (driver 550) + Perlmutter cross-validation.",
        ],
        "figure_main": "fig01_partition_baseline.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 2,
        "title": "Placement Determines an 8× Collapse",
        "subtitle": "",
        "bullets": [
            "**Cross-Partition is Stable:** Five distinct AI workloads (chanpred, "
            "NeuralRx, xApp, ResNet, forecaster) all keep L1 p99 at ~45 ms when "
            "placed on a separate MIG partition.",
            "**Same-Partition Collapses 8×:** Identical workloads co-located in the "
            "same partition as L1 jump to ~358 ms.",
            "**The Cost Driver is Placement, Not Workload:** The contrast holds "
            "across all five workloads. The cost is set by where AI is placed.",
        ],
        "figure_main": "fig_slide_8x_contrast.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 3,
        "title": "NeuralRx is Uniquely Dangerous Even Cross-Partition",
        "subtitle": "",
        "bullets": [
            "**PHY-AI Breaks Cross-Partition Isolation:** NeuralRx in a separate "
            "partition inflates L1 p99 by +376% (41 → 197 ms). ChanPred, xApp, and "
            "Qwen on the same setup only inflate L1 by +60–72%.",
            "**Why NeuralRx Specifically:** NeuralRx (TensorRT inline NN) shares the "
            "cuPHY-style copy/convert traffic pattern with L1, producing contention "
            "even across MIG partition boundaries.",
            "**Implication:** The MIG cross-partition isolation that holds for "
            "generic AI does not hold for PHY-AI co-tenants.",
        ],
        "figure_main": "fig02_phase4_neuralrx_risk.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 4,
        "title": "Larger Partition = MORE Catastrophic",
        "subtitle": "",
        "bullets": [
            "**Counter-Intuitive Outcome:** 4g co-location (371 ms) is worse than "
            "2g co-location (369 ms). Adding SMs and HBM does not reduce the "
            "collapse.",
            "**Resource Allocation is Not the Lever:** Partition sizing controls "
            "the structural HBM share but not the contention path.",
            "**Bottleneck is Chip-Global:** The contended resource lies outside the "
            "per-partition abstraction MIG provides.",
        ],
        "figure_main": "fig_supp_03_g_coloc_partition_paradox.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 5,
        "title": "Mechanism — Gaps Grow at Memory-Op Boundaries with Same Op Count",
        "subtitle": "",
        "bullets": [
            "**Kernel Duration Unchanged:** Per-kernel execution time is identical "
            "under alone and co-location; only inter-kernel gaps grow.",
            "**Gaps Are Not Idle:** Gaps coincide with memcpy and memset boundaries "
            "— filled with memory ops, not idle time (confirmed across 30 NSYS "
            "captures).",
            "**Op Count is Invariant, Duration Inflates:** memcpy and memset call "
            "counts are identical (5,778 each); memcpy average duration rises from "
            "4.2 µs to 9.8 µs (+133%). Same work, longer wait per operation.",
        ],
        "figure_main": "fig12_nsys_kernel_vs_activity_gap.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 6,
        "title": "Mechanism — Arithmetic Proves Queue Wait, Not Throughput",
        "subtitle": "",
        "bullets": [
            "**Actual Transfer is Negligible:** 60 KB / 640 GB/s = 0.09 µs. "
            "Bandwidth saturation cannot account for the inflation.",
            "**Baseline Decomposition:** 4.2 µs alone = launch + runtime overhead; "
            "transfer contributes essentially nothing.",
            "**Contended Decomposition:** 14.3 µs = 4.2 µs baseline + ~10 µs queue "
            "wait. The slow mode is queue wait by construction.",
        ],
        "figure_main": "fig_supp_12_time_decomposition.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 7,
        "title": "Mechanism — Queue at Chip-Wide PCIe/DMA + cudaFree Manifestation",
        "subtitle": "",
        "bullets": [
            "**Queue Location:** memcpy direction breakdown is 89.8% H2D, 9.9% D2H, "
            "0.12% D2D. D2D ≈ 0 rules out the HBM controller; the queue is at the "
            "chip-wide PCIe/DMA copy engine.",
            "**Host-Side Manifestation:** cudaFree average grows 246 µs (alone) → "
            "3,752 µs (NeuralRx, 15×) → 115,506 µs (sat_hbm MPS, 470×). The same "
            "call recovers to 279 µs under MPS + NeuralRx.",
            "**Causal Proof via correlationId:** 60–82% of GPU idle gap time aligns "
            "temporally with cudaFree blocking — the host-side block is the direct "
            "cause of the GPU-side gap.",
        ],
        "figure_main": "fig_slide7_cudafree_direct_evidence.png",
        "figure_aux": "fig_slide8_memcpy_direction_breakdown.png",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 8,
        "title": "Cross-Platform Mechanism Validation",
        "subtitle": "",
        "bullets": [
            "**Bimodal Signature Reproduced:** Perlmutter no-MIG (different cluster, "
            "different driver) shows the same 60 KB memcpy bimodal split "
            "(4.2 ↔ 16.8 µs).",
            "**cudaFree Scaling Matches:** cudaFree average grows 15× under NeuralRx "
            "default and 469× under MPS + sat_hbm, mirroring CloudLab.",
            "**Conclusion:** The mechanism is NVIDIA-stack-wide. It is not specific "
            "to MIG or to the CloudLab driver.",
        ],
        "figure_main": "figF14_cudafree_host_blocking.png",
        "figure_aux": "figF15_memcpy_bimodal.png",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 9,
        "title": "MPS Recovers Compute AI, Catastrophic for Memory AI",
        "subtitle": "",
        "bullets": [
            "**Compute AI Recovers:** NeuralRx 389 → 40 ms under MPS (below MIG "
            "cross-partition 197 ms); forecaster 381 → 42 ms; Qwen 185 → 43 ms.",
            "**Memory AI Becomes Worse:** sat_hbm 426 → 6,985 ms under MPS "
            "(bistable across runs).",
            "**Same Mechanism, Different Trigger:** MPS routes through the same "
            "cudaFree path; memory-bound workloads starve the device and re-engage "
            "the failure mode.",
        ],
        "figure_main": "figF8_mps_vs_default_vs_mig.png",
        "figure_aux": "figF9_mps_sat_hbm_bistable.png",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 10,
        "title": "Operational Scaling Law — Each AI Adds ~330 ms",
        "subtitle": "",
        "bullets": [
            "**Linear Cost Model:** L1 = 37.5 + N_AI × 330.6 ms when AI processes "
            "share the same MIG partition.",
            "**A2 Validates Slope:** L1 + 2 AI measures 698 ms (n=900, std=0.42 ms) "
            "— exactly 2× the single-AI co-location (354 ms).",
            "**Multi-Tenant Mistakes Compound:** Predicted 1,029 ms at 3 AI and "
            "1,360 ms at 4 AI. Operational error scales linearly, not as a single "
            "step.",
        ],
        "figure_main": "fig_slide_n_ai_scaling.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 11,
        "title": "Cost Decomposition — Two Hardware-Level Costs",
        "subtitle": "",
        "bullets": [
            "**Structural Cost (memset):** Tied to partition HBM bandwidth share "
            "(2/7, 3/7, 4/7); selectable through partition sizing.",
            "**Contention Cost (memcpy):** Tied to the chip-wide PCIe/DMA copy "
            "engine; not isolable by any partition choice.",
            "**Unified Across Three Platforms:** MIG, no-MIG, and MPS each avoid "
            "one cost while remaining exposed to the other.",
        ],
        "figure_main": "fig08_tradeoff_summary.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 12,
        "title": "Design Rules",
        "subtitle": "",
        "bullets": [
            "**Placement Enforcement is Mandatory:** L1 and AI must reside in "
            "different MIG partitions; one misplacement produces the full 8× "
            "collapse.",
            "**Partition Sizing Serves Only L1 Budget:** Sizing has no mitigating "
            "effect under co-location; choose for L1 SM/HBM headroom only.",
            "**AI Workload Classification is Required:** Memory-bound AI cannot be "
            "safely co-located even under MPS.",
            "**MIG ≠ Chip-Wide Isolation:** MIG isolates per-partition SM/HBM but "
            "not the chip-global PCIe/DMA copy engine.",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 13,
        "title": "Thank You",
        "subtitle": "",
        "bullets": [],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
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
    """Draw bold title with navy underline. Auto-shrink to fit when long."""
    # Auto-shrink long titles so they fit between left margin 0.06 and right 0.94
    fontsize = 24
    if len(title) > 50:
        fontsize = 20
    if len(title) > 65:
        fontsize = 17
    if len(title) > 80:
        fontsize = 15
    fig.text(
        0.06, 0.92, title,
        fontsize=fontsize, fontweight="bold",
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
    """Render bullets with **bold** prefix support and automatic word-wrap.

    Each bullet is formatted as `**Headline:** body`. Long bodies wrap onto
    additional lines aligned with the first text line.
    """
    import textwrap
    if not bullets:
        return

    fig_w_in, _ = fig.get_size_inches()
    dpi = fig.dpi
    renderer = fig.canvas.get_renderer()

    # Estimate characters per line based on a sample average glyph width
    sample = fig.text(0, 0, "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ",
                      fontsize=13.5, alpha=0)
    sample_w_inches = sample.get_window_extent(renderer=renderer).width / dpi
    sample.remove()
    avg_char_inch = sample_w_inches / 46.0

    right_margin = 0.07
    marker_inset = 0.022
    text_width_frac = 1.0 - left - marker_inset - right_margin
    # Bold prefix glyphs and special chars (em-dash, ×, ≈) widen the actual line
    # vs the sample average; apply a safety factor.
    safety = 1.18
    max_chars = max(40, int((text_width_frac * fig_w_in) / (avg_char_inch * safety)))

    # Pre-wrap each bullet; remember bold prefix span for the first line.
    wrapped: list[dict] = []
    for bullet in bullets:
        clean, spans = _strip_md_bold(bullet)
        bold_prefix = ""
        if spans and spans[0][0] == 0:
            bold_prefix = clean[:spans[0][1]]
        lines = textwrap.wrap(clean, width=max_chars) or [clean]
        wrapped.append({"lines": lines, "bold_prefix": bold_prefix})

    # Vertical layout: lines + gap between bullets
    avail = top - bottom
    total_lines = sum(len(b["lines"]) for b in wrapped)
    n_bullets = len(wrapped)
    inter_bullet = 0.6  # in line-height units
    units = total_lines + max(n_bullets - 1, 0) * inter_bullet
    unit_h = avail / max(units, 1)

    y_cursor = top
    for wb in wrapped:
        marker_x = left
        text_x = marker_x + marker_inset
        first_y = y_cursor - unit_h / 2.0

        # Bullet marker on first line
        fig.text(marker_x, first_y, "▪", fontsize=14, color=NAVY,
                 ha="left", va="center", fontweight="bold")

        for i, line in enumerate(wb["lines"]):
            y_line = y_cursor - unit_h * (i + 0.5)
            if i == 0 and wb["bold_prefix"] and line.startswith(wb["bold_prefix"]):
                # Bold prefix + rest on same line
                bp = wb["bold_prefix"]
                t = fig.text(text_x, y_line, bp, fontsize=13.5,
                             color=NAVY, ha="left", va="center", fontweight="bold")
                bp_w_inch = t.get_window_extent(renderer=renderer).width / dpi
                rest_x = text_x + bp_w_inch / fig_w_in
                rest = line[len(bp):]
                if rest:
                    fig.text(rest_x, y_line, rest, fontsize=13.5,
                             color=GRAY_DARK, ha="left", va="center")
            else:
                fig.text(text_x, y_line, line, fontsize=13.5,
                         color=GRAY_DARK, ha="left", va="center")

        y_cursor -= unit_h * len(wb["lines"]) + unit_h * inter_bullet


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
    bullet_bottom = 0.57  # bullets occupy top ~30% of slide
    _render_bullets_richtext(fig, slide["bullets"], bullet_top, bullet_bottom)

    has_main = bool(slide.get("figure_main"))
    has_aux = bool(slide.get("figure_aux"))
    has_table = bool(slide.get("table_md"))

    # Lower region: figure occupies bottom ~45% (matches original PPT proportions)
    lower_top = 0.53
    lower_bottom = 0.07
    lower_h = lower_top - lower_bottom

    if has_main and has_aux and has_table:
        # main + aux on left, narrower table on right — give figures more room
        fig_w = 0.64
        each_w = fig_w / 2
        _place_image(fig, slide["figure_main"],
                     (0.04, lower_bottom, each_w - 0.01, lower_h))
        _place_image(fig, slide["figure_aux"],
                     (0.04 + each_w, lower_bottom, each_w - 0.01, lower_h))
        _render_table(fig, slide["table_md"],
                      (0.72, lower_bottom + 0.04, 0.25, lower_h - 0.08))
    elif has_main and has_aux:
        # main + aux side by side, near full width
        each_w = 0.45
        _place_image(fig, slide["figure_main"],
                     (0.03, lower_bottom, each_w, lower_h))
        _place_image(fig, slide["figure_aux"],
                     (0.03 + each_w + 0.02, lower_bottom, each_w, lower_h))
    elif has_main and has_table:
        # figure on left, table on right — widen figure for wide-aspect images
        _place_image(fig, slide["figure_main"],
                     (0.04, lower_bottom, 0.62, lower_h))
        _render_table(fig, slide["table_md"],
                      (0.69, lower_bottom + 0.04, 0.28, lower_h - 0.08))
    elif has_main:
        # figure centered, near full width (no table, give image maximum space)
        _place_image(fig, slide["figure_main"],
                     (0.10, lower_bottom, 0.80, lower_h))
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
