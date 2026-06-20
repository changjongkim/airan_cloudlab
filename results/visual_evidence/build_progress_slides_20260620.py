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
        "subtitle": "AI-RAN GPU Isolation — NSYS Time Decomposition Deep-Dive",
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
            "**Continuation of Exp1–6 (5/29):** MIG single-tenant overhead "
            "+13–32 ms measured.",
            "**Today's approach:** Profile L1 + AI co-tenancy with NSYS "
            "time-decomposition analysis.",
            "**Flow:** cuPHY L1 background → NSYS decomposition (GPU/host/pipeline) "
            "→ queue mechanism → verification experiments → Perlmutter default "
            "→ MPS introduces a paradox → cudaFree unifies the whole story.",
        ],
        "figure_main": "fig01_partition_baseline.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 2,
        "title": "cuPHY L1 — What We Measure",
        "subtitle": "",
        "bullets": [
            "**Pipeline:** PUSCH RX = pre-ChEst → ChEst dispatch → Noise/Intf "
            "estimation → MMSE equalizer → LDPC decode, with `convert_kernel` "
            "between stages.",
            "**Per-frame setup:** 20 cells serial, MCS=2, 273 PRB, 4×4 antennas. "
            "n=500–1000 per condition.",
            "**Decomposition target:** L1 frame latency p99 + NSYS GPU/host trace "
            "broken into kernel / memcpy / memset / sync / idle gap and CUDA "
            "runtime API per call.",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 3,
        "title": "NSYS §17.1 — GPU Activity Decomposition",
        "subtitle": "",
        "bullets": [
            "**Wall-clock varies 1.4×:** 7g 2,027 ms, 3g alone 1,831 ms, 2g alone "
            "2,518 ms — same cuPHY work, different total time.",
            "**GPU kernel time is invariant:** ~400–500 ms in every condition. AI "
            "co-location does not slow L1 compute itself.",
            "**Idle gap dominates 40–60% of wall-clock:** The cost is in waiting "
            "between operations, not in computing.",
        ],
        "figure_main": "fig_supp_14_gpu_activity_decomposition.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 4,
        "title": "NSYS §17.2 — cuPHY Pipeline Stage Breakdown",
        "subtitle": "",
        "bullets": [
            "**Per-stage time is small:** ChEst, Noise/Intf, Equalizer, LDPC all "
            "5–10 ms each.",
            "**convert_kernel dominates 357 ms:** A boundary op between pipeline "
            "stages, called ~4× per cell iteration (2,576 calls / 640 iter).",
            "**Stage timing is partition/AI-invariant:** L1 kernel work is "
            "isolated as expected; the cost lives at the stage boundary, not in "
            "the stages.",
        ],
        "figure_main": "fig_supp_15_pipeline_stages.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 5,
        "title": "NSYS §17.3 — Host CPU Runtime API",
        "subtitle": "",
        "bullets": [
            "**cudaFree is the largest host cost: 598 ms (1st).** cuPHY allocates "
            "per-frame and calls cudaFree ~1,983 times per measurement.",
            "**cuLaunchKernel is invariant ~80 ms:** Launch queue is not contended; "
            "the host-side overhead is in memory management, not in kernel launch.",
            "**Small partition compounds the cost:** cudaFree on 2g = 1,380 ms vs "
            "7g = 880 ms (1.6×). Driver's page-table management scales with "
            "partition size.",
        ],
        "figure_main": "fig_supp_16_runtime_api.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 6,
        "title": "NSYS §17.4 — Wall-Clock Normalized Decomposition",
        "subtitle": "",
        "bullets": [
            "**GPU kernel: ~20% in every condition.** L1 compute is a small slice "
            "of wall-clock; the rest is overhead.",
            "**memset grows with smaller partition:** 7g 10% → 3g 19% → 2g 36%. "
            "Structural HBM bandwidth share scaling, visible as memset time.",
            "**idle gap shrinks with smaller partition (63% → 42%) but wall-clock "
            "still grows:** Memset displaces idle as the bottleneck, so smaller "
            "partition still loses on total latency.",
        ],
        "figure_main": "fig_supp_17_normalized_wallclock.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 7,
        "title": "Mechanism — Arithmetic Proves Queue Wait, Not Throughput",
        "subtitle": "",
        "bullets": [
            "**Actual transfer is negligible:** 60 KB / 640 GB/s = 0.09 µs. "
            "Bandwidth saturation cannot account for the inflation.",
            "**Baseline:** 4.2 µs alone = launch + runtime overhead; transfer "
            "contributes essentially nothing.",
            "**Contended:** 14.3 µs = 4.2 µs baseline + ~10 µs queue wait. The "
            "slow mode is queue wait by construction.",
        ],
        "figure_main": "fig_supp_12_time_decomposition.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 8,
        "title": "Mechanism — Queue Located at Chip-Wide PCIe/DMA Copy Engine",
        "subtitle": "",
        "bullets": [
            "**Direction breakdown:** 89.8% H2D, 9.9% D2H, 0.12% D2D across all "
            "measured memcpy calls.",
            "**D2D ≈ 0 rules out the HBM controller:** Which would carry D2D "
            "traffic.",
            "**Single shared resource:** The PCIe/DMA copy engine is one physical "
            "unit serving the entire chip; partitioning cannot isolate it.",
        ],
        "figure_main": "fig_slide8_memcpy_direction_breakdown.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 9,
        "title": "Verification 1 — Placement Determines an 8× Collapse",
        "subtitle": "",
        "bullets": [
            "**Cross-partition is stable:** Five distinct AI workloads (chanpred, "
            "NeuralRx, xApp, ResNet, forecaster) all keep L1 p99 at ~45 ms on a "
            "separate MIG partition.",
            "**Same-partition collapses 8×:** Identical workloads co-located in "
            "the same partition jump to ~358 ms.",
            "**Confirms the queue hypothesis:** The contended resource is "
            "placement-sensitive but workload-invariant, as predicted by a single "
            "chip-wide queue.",
        ],
        "figure_main": "fig_slide_8x_contrast.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 10,
        "title": "Verification 2 — NeuralRx PHY-AI Uniqueness",
        "subtitle": "",
        "bullets": [
            "**PHY-AI breaks cross-partition isolation:** NeuralRx in a separate "
            "partition still inflates L1 p99 by +376% (41 → 197 ms); ChanPred, "
            "xApp, Qwen on the same setup only inflate L1 by +60–72%.",
            "**Why NeuralRx specifically:** NeuralRx (TensorRT inline NN) shares "
            "the cuPHY-style copy/convert traffic pattern with L1, producing "
            "contention even across MIG partition boundaries.",
            "**Implication:** Even cross-partition placement does not isolate "
            "L1 from PHY-AI co-tenants.",
        ],
        "figure_main": "fig02_phase4_neuralrx_risk.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 11,
        "title": "Verification 3 — N-AI Operational Scaling Law",
        "subtitle": "",
        "bullets": [
            "**Linear cost model:** L1 = 37.5 + N_AI × 330.6 ms when AI processes "
            "share the same MIG partition.",
            "**A2 validates the slope:** L1 + 2 AI measures 698 ms (n=900, "
            "std=0.42 ms) — exactly 2× the single-AI co-location (354 ms).",
            "**Multi-tenant mistakes compound:** Predicted 1,029 ms at 3 AI and "
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
        "slide_num": 12,
        "title": "Cross-Platform Validation on Perlmutter (Default Mode)",
        "subtitle": "",
        "bullets": [
            "**Bimodal signature reproduced:** Perlmutter no-MIG (different "
            "cluster, different driver) shows the same 60 KB memcpy bimodal "
            "split (4.2 ↔ 16.8 µs) under L1 + AI co-tenancy.",
            "**cudaFree growth matches CloudLab:** under NeuralRx default the "
            "Perlmutter cudaFree average grows 15× — identical fold-change to "
            "the CloudLab MIG measurement.",
            "**Conclusion:** the queue + cudaFree mechanism is NVIDIA-stack-wide, "
            "not specific to MIG or the CloudLab driver — same default-mode "
            "signature on independent hardware.",
        ],
        "figure_main": "figF15_memcpy_bimodal.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 13,
        "title": "MPS Recovers Compute AI, Catastrophic for Memory AI",
        "subtitle": "",
        "bullets": [
            "**Compute AI recovers:** NeuralRx 389 → 40 ms under MPS (below MIG "
            "cross-partition 197 ms); forecaster 381 → 42 ms; Qwen 185 → 43 ms.",
            "**Memory AI becomes worse:** sat_hbm 426 → 6,985 ms under MPS "
            "(bistable across runs).",
            "**Open question:** why does MPS help compute-bound AI but break "
            "memory-bound AI? The next slide answers via the cudaFree mechanism.",
        ],
        "figure_main": "figF8_mps_vs_default_vs_mig.png",
        "figure_aux": "figF9_mps_sat_hbm_bistable.png",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 14,
        "title": "Mechanism — cudaFree Host-Blocking Explains Both MIG and MPS",
        "subtitle": "",
        "bullets": [
            "**Default-mode mechanism:** alone 246 µs → L1 + NeuralRx default "
            "3,752 µs (15× alone). The host blocks on cudaFree; correlationId "
            "shows 60–82% of the GPU idle gap overlaps with cudaFree on the host.",
            "**MPS recovery explained:** L1 + NeuralRx under MPS drops cudaFree "
            "back to 279 µs (≈ alone), so the GPU stops idling — recovery is the "
            "cudaFree path being un-blocked, not a new isolation mechanism.",
            "**MPS catastrophic explained:** L1 + sat_hbm under MPS pushes "
            "cudaFree to 115,506 µs (470× alone). Memory-bound co-tenants starve "
            "the device of free-able memory, so cudaFree blocks longer — same "
            "mechanism, opposite outcome.",
            "**Unifying conclusion:** kernel-invariant + gap-dominated + cudaFree-"
            "dominated decomposition from §17 is the host-blocking story all the "
            "way through — MIG, MPS recover, MPS catastrophic are three points on "
            "the same curve.",
        ],
        "figure_main": "fig_slide7_cudafree_direct_evidence.png",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 15,
        "title": "Design Rules",
        "subtitle": "",
        "bullets": [
            "**Placement enforcement is mandatory:** L1 and AI must reside in "
            "different MIG partitions; one misplacement produces the full 8× "
            "collapse.",
            "**Partition sizing serves only L1 budget:** Sizing has no mitigating "
            "effect under co-location; choose for L1 SM/HBM headroom only.",
            "**AI workload classification is required:** Memory-bound AI cannot "
            "be safely co-located even under MPS.",
            "**MIG ≠ chip-wide isolation:** MIG isolates per-partition SM/HBM but "
            "not the chip-global PCIe/DMA copy engine.",
        ],
        "figure_main": "",
        "figure_aux": "",
        "table_md": "",
        "footer_ref": "",
        "transition": "",
    },
    {
        "slide_num": 16,
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
