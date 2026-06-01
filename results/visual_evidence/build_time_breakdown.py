#!/usr/bin/env python3
"""
Time-breakdown figures from NSYS sqlite — component-by-component decomposition.

supp_14: GPU activity time decomposition (stacked bar per condition)
         Components: kernel / memcpy / memset / sync / idle (gap)
         Shows what GPU is "doing" in each condition.

supp_15: cuPHY pipeline stage decomposition (kernel breakdown)
         Stages: pre-ChEst / ChEst dispatch / Noise est / Equalizer /
                 LDPC / Convert (boundary) / Copy ops / Other
         Shows which stages get slower under AI/partition variation.

supp_16: Runtime API time decomposition (host CPU side)
         Components: cuLaunchKernel / cudaMemcpyAsync / cudaMemsetAsync /
                     cudaMalloc/Free / cudaStreamSync / Other
         Shows host CPU's CUDA driver overhead.

supp_17: Wall-clock decomposition normalized (% of total time)
         Shows what fraction of total run time each component took.
"""
import sqlite3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

mpl.rcParams.update({
    'font.size': 10,
    'figure.figsize': (12, 7),
    'savefig.dpi': 140,
    'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo', 'NanumGothic', 'sans-serif'],
})

ROOT = Path(__file__).parent.parent
SQLITE = ROOT / "20260531" / "nsys_sqlite_v2"
OUT = Path(__file__).parent / "figures"

# Condition matrix
CONDS = [
    ("7g full",         "S2_7g_mig_run1.sqlite",          "no AI"),
    ("4g alone",        "S15_4g_sat_compute_run1.sqlite", "sat_compute"),  # closest to 4g alone-like
    ("4g + NeuralRx",   "S18_4g_neuralrx_run1.sqlite",    "NeuralRx"),
    ("4g + ResNet",     "S34_4g_resnet_run1.sqlite",      "ResNet"),
    ("3g alone",        "S5_3g_alone_run1.sqlite",        "no AI"),
    ("3g + sat_compute","S13_3g_sat_compute_run1.sqlite", "sat_compute"),
    ("3g + Qwen",       "S6_3g_qwen_run1.sqlite",         "Qwen"),
    ("3g + NeuralRx",   "S7_3g_neuralrx_run1.sqlite",     "NeuralRx"),
    ("3g + ResNet",     "S28_3g_resnet_run1.sqlite",      "ResNet"),
    ("3g + chanpred",   "S27_3g_chanpred_run1.sqlite",    "chanpred"),
    ("2g alone",        "S10_2g_alone_run1.sqlite",       "no AI"),
    ("2g + NeuralRx",   "S22_2g_neuralrx_run1.sqlite",    "NeuralRx"),
    ("2g + chanpred",   "S35_2g_chanpred_run1.sqlite",    "chanpred"),
]


def get_activity_breakdown(db):
    """Return GPU activity time per category + wall clock + idle."""
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
    # Wall clock span (from first event to last)
    try:
        cur.execute("SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        ks, ke = cur.fetchone()
        out["wall_ms"] = (ke - ks) / 1e6 if (ks and ke) else 0
        # Idle = wall - kernel - memcpy - memset (these can overlap, so this is approximate)
        gpu_busy = out["kernel_ms"] + out["memcpy_ms"] + out["memset_ms"]
        out["idle_ms"] = max(0, out["wall_ms"] - gpu_busy)
    except sqlite3.OperationalError:
        out["wall_ms"] = 0; out["idle_ms"] = 0
    con.close()
    return out


def get_runtime_api_breakdown(db):
    """Return runtime API time per call type."""
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


def get_kernel_stage_breakdown(db):
    """Group kernel times by cuPHY pipeline stage."""
    if not os.path.exists(db): return None
    con = sqlite3.connect(db); cur = con.cursor()
    try:
        cur.execute("""SELECT sn.value, SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k
                       LEFT JOIN StringIds sn ON sn.id=k.shortName GROUP BY sn.value""")
        raw = cur.fetchall()
    except sqlite3.OperationalError:
        raw = []
    con.close()
    # Classify
    stages = defaultdict(float)
    for name, ms in raw:
        n = name or "?"
        ms = ms or 0
        nl = n.lower()
        if "windowedchest" in nl or "preNoDft" in n:
            stages["pre-ChEst"] += ms
        elif "chestfilter" in nl or "chest" in nl and "dispatch" in nl:
            stages["ChEst dispatch"] += ms
        elif "noiseintf" in nl or "noise" in nl:
            stages["Noise/Intf est"] += ms
        elif "eqmmse" in nl:
            stages["Equalizer (MMSE)"] += ms
        elif "ldpc" in nl or "deratematch" in nl or "decode" in nl:
            stages["LDPC"] += ms
        elif "convert" in nl:
            stages["Convert (boundary)"] += ms
        elif "cupy_copy" in nl or "copy_" in nl:
            stages["Copy ops"] += ms
        elif "crc" in nl:
            stages["CRC check"] += ms
        else:
            stages["Other"] += ms
    return dict(stages)


def fig_supp_14_activity_decomposition():
    """Stacked bar: kernel / memcpy / memset / sync / idle per condition."""
    data = []
    for lbl, db, _ in CONDS:
        b = get_activity_breakdown(SQLITE / db)
        if b: data.append((lbl, b))
    if not data: return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(data))
    components = [("kernel_ms", "GPU kernel", "#3498DB"),
                  ("memcpy_ms", "memcpy", "#9B59B6"),
                  ("memset_ms", "memset", "#E74C3C"),
                  ("sync_ms", "sync", "#F39C12"),
                  ("idle_ms", "idle (gap)", "#95A5A6")]
    bottom = np.zeros(len(data))
    for key, lbl, col in components:
        vals = [d[1][key] for d in data]
        ax.bar(x, vals, bottom=bottom, label=lbl, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=25, ha='right')
    ax.set_ylabel("Time (ms)")
    ax.set_title("Supp 14 — GPU Activity Time Decomposition\n"
                 "각 condition에서 GPU가 무엇으로 시간을 보냈는가 (kernel / memcpy / memset / sync / idle)")
    ax.legend(loc='upper left')
    for i, (lbl, b) in enumerate(data):
        ax.text(i, b["wall_ms"] + 30, f"{b['wall_ms']:.0f}ms", ha='center', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_14_gpu_activity_decomposition.png")
    plt.close(fig)
    print(f"  ✓ supp_14 GPU activity decomposition ({len(data)} conditions)")


def fig_supp_15_pipeline_stages():
    """Stacked: cuPHY pipeline stages per condition."""
    rows = []
    for lbl, db, _ in CONDS:
        s = get_kernel_stage_breakdown(SQLITE / db)
        if s: rows.append((lbl, s))
    if not rows: return

    all_stages = ["pre-ChEst", "ChEst dispatch", "Noise/Intf est", "Equalizer (MMSE)",
                  "LDPC", "Convert (boundary)", "Copy ops", "CRC check", "Other"]
    colors = ["#16A085", "#27AE60", "#F39C12", "#3498DB",
              "#9B59B6", "#E74C3C", "#E67E22", "#95A5A6", "#666666"]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for stage, col in zip(all_stages, colors):
        vals = [r[1].get(stage, 0) for r in rows]
        ax.bar(x, vals, bottom=bottom, label=stage, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=25, ha='right')
    ax.set_ylabel("Kernel time (ms)")
    ax.set_title("Supp 15 — cuPHY pipeline 별 kernel 시간 분해\n"
                 "PUSCH RX의 어느 stage가 partition/AI에 의해 늘어나는가")
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    for i, (lbl, s) in enumerate(rows):
        total = sum(s.values())
        ax.text(i, total + 5, f"{total:.0f}ms", ha='center', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_15_pipeline_stages.png")
    plt.close(fig)
    print(f"  ✓ supp_15 cuPHY pipeline stage decomposition ({len(rows)} conditions)")


def fig_supp_16_runtime_api():
    """Runtime API (host CPU CUDA driver call) time decomposition."""
    rows = []
    for lbl, db, _ in CONDS:
        b = get_runtime_api_breakdown(SQLITE / db)
        if b: rows.append((lbl, b))
    if not rows: return

    # Pick top categories
    cats_show = ["cuLaunchKernel", "cudaLaunchKernel_v7000", "cudaMemcpyAsync_v3020",
                 "cudaMemsetAsync_v3020", "cudaMalloc_v3020", "cudaFree_v3020",
                 "cudaStreamSynchronize_v3020", "cudaEventSynchronize_v3020",
                 "cudaStreamCreate_v3020", "cuLibraryLoadData"]
    nice_names = {"cuLaunchKernel": "cuLaunchKernel (low-level)",
                  "cudaLaunchKernel_v7000": "cudaLaunchKernel",
                  "cudaMemcpyAsync_v3020": "cudaMemcpyAsync",
                  "cudaMemsetAsync_v3020": "cudaMemsetAsync",
                  "cudaMalloc_v3020": "cudaMalloc",
                  "cudaFree_v3020": "cudaFree",
                  "cudaStreamSynchronize_v3020": "cudaStreamSync",
                  "cudaEventSynchronize_v3020": "cudaEventSync",
                  "cudaStreamCreate_v3020": "cudaStreamCreate (init)",
                  "cuLibraryLoadData": "cuLibraryLoadData (init)"}
    colors = ["#3498DB", "#5DADE2", "#9B59B6", "#E74C3C",
              "#F39C12", "#E67E22", "#27AE60", "#16A085",
              "#95A5A6", "#7F8C8D"]
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for cat, col in zip(cats_show, colors):
        vals = [r[1].get(cat, 0) for r in rows]
        ax.bar(x, vals, bottom=bottom, label=nice_names.get(cat, cat), color=col)
        bottom += vals
    # Add "other"
    others = []
    for r in rows:
        total = sum(r[1].values())
        accounted = sum(r[1].get(c, 0) for c in cats_show)
        others.append(max(0, total - accounted))
    ax.bar(x, others, bottom=bottom, label='Other', color='#BDC3C7')

    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=25, ha='right')
    ax.set_ylabel("Runtime API total time (ms)")
    ax.set_title("Supp 16 — CUDA Runtime API host-side time decomposition\n"
                 "Host CPU가 CUDA driver call에 쓴 시간")
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_16_runtime_api.png")
    plt.close(fig)
    print(f"  ✓ supp_16 runtime API breakdown ({len(rows)} conditions)")


def fig_supp_17_normalized_wallclock():
    """100%-normalized wall clock decomposition."""
    data = []
    for lbl, db, _ in CONDS:
        b = get_activity_breakdown(SQLITE / db)
        if b and b["wall_ms"] > 0:
            data.append((lbl, b))
    if not data: return

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(data))
    components = [("kernel_ms", "GPU kernel", "#3498DB"),
                  ("memcpy_ms", "memcpy", "#9B59B6"),
                  ("memset_ms", "memset", "#E74C3C"),
                  ("sync_ms", "sync", "#F39C12"),
                  ("idle_ms", "idle (gap)", "#95A5A6")]
    bottom = np.zeros(len(data))
    for key, lbl, col in components:
        vals = [d[1][key] / d[1]["wall_ms"] * 100 for d in data]
        ax.bar(x, vals, bottom=bottom, label=lbl, color=col)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data], rotation=25, ha='right')
    ax.set_ylabel("% of wall-clock")
    ax.set_ylim(0, 110)
    ax.set_title("Supp 17 — Wall-clock 정규화 분해: 각 condition에서 GPU 시간 비율\n"
                 "동일한 cuPHY L1 work인데 시간 비율이 어떻게 다른지 (작은 partition이 idle 비율 증가)")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / "fig_supp_17_normalized_wallclock.png")
    plt.close(fig)
    print(f"  ✓ supp_17 normalized wall-clock decomposition")


def main():
    print(f"Generating time-breakdown figures → {OUT}")
    fig_supp_14_activity_decomposition()
    fig_supp_15_pipeline_stages()
    fig_supp_16_runtime_api()
    fig_supp_17_normalized_wallclock()
    print("\nDone.")


if __name__ == "__main__":
    main()
