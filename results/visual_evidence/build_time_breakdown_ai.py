#!/usr/bin/env python3
"""
§18 Time-breakdown figures — AI-process perspective (mirrors §17 for cuPHY L1).

Input: SQLite files produced by s18_dual_capture.sh, located under
       results/$DATE_DIR/s18_ai_nsys/   (rsync from CloudLab)

Outputs (figures/):
  fig_s18_01_ai_gpu_activity_decomposition.png   — kernel/memcpy/memset/sync/idle per AI condition
  fig_s18_02_ai_kernel_stages.png                — model-internal kernel-class breakdown
  fig_s18_03_ai_runtime_api.png                  — host-side cuLaunchKernel/cudaFree/cudaMemcpy
  fig_s18_04_ai_normalized_wallclock.png         — same as 01 but normalized to 100%

Conditions (must match labels in s18_dual_capture.sh):
  X1 NeuralRx alone (3g)
  X2 NeuralRx + L1 cross-partition (NeuralRx 2g, L1 3g)
  X3 NeuralRx + L1 same-partition coloc (3g)
  X4 chanpred alone (3g)
  X5 chanpred + L1 cross-partition (chanpred 2g, L1 3g)
  X6 chanpred + L1 same-partition coloc (3g)

For the §18 narrative we expect:
  - NeuralRx alone vs X2 (cross-part): GPU idle gap should already grow (PHY-AI is
    cross-part-sensitive), and cudaFree on the AI host should inflate.
  - NeuralRx alone vs X3 (coloc): catastrophic AI-side inflation, mirroring L1.
  - chanpred alone vs X5 (cross-part): minor or no change (queue not symmetric for chanpred).
  - chanpred alone vs X6 (coloc): partial inflation (placement-driven, workload-invariant).

The same-AI / different-placement pair is the key comparison; cross-AI comparisons
are secondary because absolute kernel time differs by model.
"""
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11, "figure.figsize": (13, 7),
    "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "font.family": ["DejaVu Sans"],
})

ROOT = Path(__file__).parent.parent
# DATE_DIR can be overridden via env var; default to today's date
DATE_DIR = os.environ.get("S18_DATE_DIR", "20260622")
SQLITE = ROOT / DATE_DIR / "s18_ai_nsys"
OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# (label, sqlite filename, AI workload tag)
CONDS = [
    ("NeuralRx alone (3g)",            "X1_neuralrx_alone_3g_AI.sqlite",          "NeuralRx"),
    ("NeuralRx + L1 cross-part",       "X2_neuralrx_2g_L1_3g_crosspart_AI.sqlite","NeuralRx"),
    ("NeuralRx + L1 coloc (3g)",       "X3_neuralrx_L1_coloc_3g_AI.sqlite",       "NeuralRx"),
    ("chanpred alone (3g)",            "X4_chanpred_alone_3g_AI.sqlite",          "chanpred"),
    ("chanpred + L1 cross-part",       "X5_chanpred_2g_L1_3g_crosspart_AI.sqlite","chanpred"),
    ("chanpred + L1 coloc (3g)",       "X6_chanpred_L1_coloc_3g_AI.sqlite",       "chanpred"),
]


# ----------------------------------------------------------------------------
# Schema-identical primitives (copied from build_time_breakdown.py)
# ----------------------------------------------------------------------------
def get_activity_breakdown(db):
    if not os.path.exists(db):
        return None
    con = sqlite3.connect(db); cur = con.cursor()
    out = {}
    for tab, key in [("CUPTI_ACTIVITY_KIND_KERNEL", "kernel_ms"),
                     ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy_ms"),
                     ("CUPTI_ACTIVITY_KIND_MEMSET", "memset_ms"),
                     ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "sync_ms")]:
        try:
            cur.execute(f"SELECT SUM(end-start)/1e6 FROM {tab}")
            out[key] = cur.fetchone()[0] or 0
        except sqlite3.OperationalError:
            out[key] = 0
    try:
        cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        ks, ke = cur.fetchone()
        out["wall_ms"] = (ke - ks) / 1e6 if (ks and ke) else 0
        gpu_busy = out["kernel_ms"] + out["memcpy_ms"] + out["memset_ms"]
        out["idle_ms"] = max(0, out["wall_ms"] - gpu_busy)
    except sqlite3.OperationalError:
        out["wall_ms"] = 0; out["idle_ms"] = 0
    con.close()
    return out


def get_runtime_api_breakdown(db):
    if not os.path.exists(db): return None
    con = sqlite3.connect(db); cur = con.cursor()
    out = {}
    try:
        cur.execute("""SELECT sn.value, COUNT(*), SUM(r.end-r.start)/1e6
                       FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                       LEFT JOIN StringIds sn ON sn.id=r.nameId GROUP BY sn.value""")
        for name, n, ms in cur.fetchall():
            out[name or "?"] = ms or 0
    except sqlite3.OperationalError:
        pass
    con.close()
    return out


# ----------------------------------------------------------------------------
# AI-aware kernel-stage classifier (model-internal categories, not cuPHY stages)
# ----------------------------------------------------------------------------
def classify_ai_kernel(name):
    n = (name or "?").lower()
    # TensorRT inline engine layers (NeuralRx)
    if n.startswith("trt_") or "tensorrt" in n or "trtopt" in n:
        return "TRT engine"
    # Conv / GEMM (NN compute)
    if any(k in n for k in ("conv", "gemm", "cudnn", "cutlass", "sgemm", "hgemm",
                             "fused_conv", "implicit_gemm")):
        return "Conv / GEMM"
    # Normalisation + activation
    if any(k in n for k in ("batchnorm", "layernorm", "softmax", "relu", "gelu",
                             "sigmoid", "tanh", "activation", "instance_norm")):
        return "Norm / Activation"
    # Reduction / pooling / argmax
    if any(k in n for k in ("reduce", "pool", "argmax", "argmin", "topk",
                             "scatter", "gather")):
        return "Reduce / Pool"
    # Copy / convert / cast (data movement on device)
    if any(k in n for k in ("cupy_copy", "copy_", "convert", "cast", "transpose",
                             "permute", "memcpy")):
        return "Copy / Convert"
    # Elementwise / binary ops
    if any(k in n for k in ("elementwise", "add_", "mul_", "div_", "binary",
                             "unary", "fill_", "scale_")):
        return "Elementwise"
    return "Other"


def get_ai_kernel_stage_breakdown(db):
    if not os.path.exists(db): return None
    con = sqlite3.connect(db); cur = con.cursor()
    try:
        cur.execute("""SELECT sn.value, SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k
                       LEFT JOIN StringIds sn ON sn.id=k.shortName GROUP BY sn.value""")
        raw = cur.fetchall()
    except sqlite3.OperationalError:
        raw = []
    con.close()
    stages = defaultdict(float)
    for name, ms in raw:
        stages[classify_ai_kernel(name)] += (ms or 0)
    return dict(stages)


# ----------------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------------
def fig_s18_01_ai_activity():
    data = []
    for lbl, db, _ in CONDS:
        b = get_activity_breakdown(SQLITE / db)
        if b: data.append((lbl, b))
    if not data:
        print("  ✗ s18_01: no input sqlite files found"); return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(data))
    components = [("kernel_ms", "GPU kernel",   "#3498DB"),
                  ("memcpy_ms", "memcpy",       "#9B59B6"),
                  ("memset_ms", "memset",       "#E74C3C"),
                  ("sync_ms",   "sync",         "#F39C12"),
                  ("idle_ms",   "idle (gap)",   "#95A5A6")]
    bottom = np.zeros(len(data))
    for key, lbl, col in components:
        vals = [d[1][key] for d in data]
        ax.bar(x, vals, bottom=bottom, label=lbl, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=20, ha="right")
    ax.set_ylabel("Time during 30s NSYS window (ms)")
    ax.set_title("§18.1 — AI process GPU activity decomposition\n"
                 "Does the AI side show the same idle-gap-dominated pattern as L1?")
    ax.legend(loc="upper left")
    for i, (lbl, b) in enumerate(data):
        ax.text(i, b["wall_ms"] + 30, f"{b['wall_ms']:.0f}ms",
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_s18_01_ai_gpu_activity_decomposition.png")
    plt.close(fig)
    print(f"  ✓ s18_01 AI GPU activity ({len(data)} conditions)")


def fig_s18_02_ai_kernel_stages():
    rows = []
    for lbl, db, _ in CONDS:
        s = get_ai_kernel_stage_breakdown(SQLITE / db)
        if s: rows.append((lbl, s))
    if not rows:
        print("  ✗ s18_02: no input sqlite files found"); return

    all_stages = ["TRT engine", "Conv / GEMM", "Norm / Activation",
                  "Reduce / Pool", "Copy / Convert", "Elementwise", "Other"]
    colors = ["#16A085", "#3498DB", "#F39C12",
              "#9B59B6", "#E74C3C", "#E67E22", "#95A5A6"]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for stage, col in zip(all_stages, colors):
        vals = [r[1].get(stage, 0) for r in rows]
        ax.bar(x, vals, bottom=bottom, label=stage, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Kernel time (ms)")
    ax.set_title("§18.2 — AI model kernel-class breakdown\n"
                 "Which model component dominates and how does it shift under contention")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    for i, (lbl, s) in enumerate(rows):
        total = sum(s.values())
        ax.text(i, total + 5, f"{total:.0f}ms",
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_s18_02_ai_kernel_stages.png")
    plt.close(fig)
    print(f"  ✓ s18_02 AI kernel stages ({len(rows)} conditions)")


def fig_s18_03_ai_runtime_api():
    data = []
    for lbl, db, _ in CONDS:
        b = get_runtime_api_breakdown(SQLITE / db)
        if b: data.append((lbl, b))
    if not data:
        print("  ✗ s18_03: no input sqlite files found"); return

    keep = ["cudaLaunchKernel", "cuLaunchKernel", "cudaFree", "cudaMalloc",
            "cudaMemcpy", "cudaMemcpyAsync", "cudaStreamSynchronize",
            "cudaDeviceSynchronize", "cudaMemsetAsync"]
    colors = ["#3498DB", "#2980B9", "#E67E22", "#F39C12",
              "#9B59B6", "#8E44AD", "#27AE60", "#16A085", "#E74C3C"]
    other = "Other"

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    for k, col in zip(keep, colors):
        vals = [d[1].get(k, 0) for d in data]
        ax.bar(x, vals, bottom=bottom, label=k, color=col)
        bottom += vals
    # Other = everything not in keep
    other_vals = []
    for _, b in data:
        other_vals.append(sum(v for k, v in b.items() if k not in keep))
    ax.bar(x, other_vals, bottom=bottom, label=other, color="#95A5A6")
    bottom += other_vals
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=20, ha="right")
    ax.set_ylabel("Host CPU time in CUDA runtime API (ms)")
    ax.set_title("§18.3 — AI process host runtime API decomposition\n"
                 "Does AI-side cudaFree also inflate under contention?")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    for i, total in enumerate(bottom):
        ax.text(i, total + 30, f"{total:.0f}ms",
                ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_s18_03_ai_runtime_api.png")
    plt.close(fig)
    print(f"  ✓ s18_03 AI runtime API ({len(data)} conditions)")


def fig_s18_04_ai_normalized_wallclock():
    data = []
    for lbl, db, _ in CONDS:
        b = get_activity_breakdown(SQLITE / db)
        if b and b["wall_ms"] > 0: data.append((lbl, b))
    if not data:
        print("  ✗ s18_04: no input sqlite files found"); return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(data))
    components = [("kernel_ms", "GPU kernel",   "#3498DB"),
                  ("memcpy_ms", "memcpy",       "#9B59B6"),
                  ("memset_ms", "memset",       "#E74C3C"),
                  ("sync_ms",   "sync",         "#F39C12"),
                  ("idle_ms",   "idle (gap)",   "#95A5A6")]
    bottom = np.zeros(len(data))
    for key, lbl, col in components:
        pcts = [100.0 * d[1][key] / d[1]["wall_ms"] for d in data]
        ax.bar(x, pcts, bottom=bottom, label=lbl, color=col)
        for i, p in enumerate(pcts):
            if p >= 4:
                ax.text(i, bottom[i] + p / 2, f"{p:.0f}%",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
        bottom += pcts
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=20, ha="right")
    ax.set_ylabel("% of AI-process wall-clock window")
    ax.set_ylim(0, 105)
    ax.set_title("§18.4 — AI process wall-clock normalized decomposition\n"
                 "Same 30s window, different time ratios across placements")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_s18_04_ai_normalized_wallclock.png")
    plt.close(fig)
    print(f"  ✓ s18_04 AI normalized wall-clock ({len(data)} conditions)")


def main():
    print(f"§18 AI-side time-breakdown — inputs: {SQLITE}")
    if not SQLITE.exists():
        print(f"  WARNING: {SQLITE} does not exist yet. After CloudLab run, rsync the")
        print(f"  s18_ai_nsys/ directory under results/$DATE_DIR/, then re-run.")
        print(f"  Or set S18_DATE_DIR env var to point at your actual date dir.")
    fig_s18_01_ai_activity()
    fig_s18_02_ai_kernel_stages()
    fig_s18_03_ai_runtime_api()
    fig_s18_04_ai_normalized_wallclock()
    print("\nDone.")


if __name__ == "__main__":
    main()
