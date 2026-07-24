#!/usr/bin/env python3
"""Chain 13 figures — SP MPS breakdown gradient + CP isolation."""
import json, os, glob, sqlite3, statistics, subprocess
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
CHAIN = os.path.join(BASE, "chain13")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

# =============================================================================
# 1. Convert nsys-rep → sqlite (once) using local nsys if available
# =============================================================================
def get_stats(sq_path):
    con = sqlite3.connect(sq_path); cur = con.cursor()
    try:
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFree_v____'")
        cfs = [x[0] for x in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMemcpyAsync%'")
        mcs = [x[0] for x in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cuLaunchKernel'")
        lks = [x[0] for x in cur.fetchall()]
        def sumn(ids):
            if not ids: return 0,0
            ph = ','.join(['?']*len(ids))
            cur.execute(f'SELECT SUM(end-start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph})', ids)
            r = cur.fetchone(); return (r[0] or 0), (r[1] or 0)
        cf_ns, cf_n = sumn(cfs)
        mc_ns, mc_n = sumn(mcs)
        lk_ns, lk_n = sumn(lks)
        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        tot = cur.fetchone()[0] or 0
    finally:
        con.close()
    return {"cf_ms": cf_ns/1e6, "cf_n": cf_n, "mc_ms": mc_ns/1e6, "lk_n": lk_n, "tot_ms": tot/1e6}

# already-generated sqlite from remote analysis stored per-nsys-rep on node,
# here we re-parse whatever sqlite files exist locally.
r = defaultdict(list)
for sq in sorted(glob.glob(f"{CHAIN}/*.sqlite")):
    try:
        s = get_stats(sq)
        base = os.path.basename(sq).replace(".sqlite","")
        for suf in ["_t1","_t2","_t3"]:
            if base.endswith(suf): base = base[:-3]; break
        r[base].append(s)
    except Exception as e:
        print(f"skip {sq}: {e}")

if not r:
    print("ERROR: no sqlite files locally; need to convert nsys-rep first.")
    print("Run on node: docker run --rm -v $DIR:/data -w /data airan:25-3-final \\")
    print("  bash -c 'for f in *.nsys-rep; do nsys export -t sqlite -o \"${f%.nsys-rep}.sqlite\" --force-overwrite=true \"$f\"; done'")
    exit(1)

summary = {}
for k, ts in r.items():
    def avg(f): return statistics.mean(x[f] for x in ts)
    summary[k] = {
        "n": len(ts),
        "cf_ms": avg("cf_ms"),
        "cf_n": avg("cf_n"),
        "mc_ms": avg("mc_ms"),
        "lk_n": avg("lk_n"),
        "tot_ms": avg("tot_ms"),
    }
with open(f"{BASE}/chain13_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print("Wrote chain13_summary.json")

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

# =============================================================================
# fig1 — CP isolation (all 3g workloads leave 4g L1 flat)
# =============================================================================
cp_labels = ["idle","NRx","ChanPred","Qwen-LLM","Qwen-RAG","Qwen-VL","Whisper"]
cp_keys   = ["CP_idle","CP_nrx","CP_chanpred","CP_qwen_llm","CP_qwen_rag","CP_qwen_vl","CP_whisper"]
cp_vals   = [summary[k]["cf_ms"] for k in cp_keys]

fig, ax = plt.subplots(figsize=(10, 5.2))
xs = np.arange(len(cp_labels))
bars = ax.bar(xs, cp_vals, color="#10b981", edgecolor="#065f46", linewidth=1)
for x, v in zip(xs, cp_vals):
    ax.text(x, v+80, f"{v:,.0f}", ha="center", fontweight="bold", fontsize=10)
ax.axhline(summary["CP_idle"]["cf_ms"], color="#065f46", linestyle="--", linewidth=1, alpha=0.5, label=f"CP-idle: {summary['CP_idle']['cf_ms']:.0f} ms")
ax.set_xticks(xs); ax.set_xticklabels(cp_labels, fontsize=11)
ax.set_ylabel("L1 cudaFree total (ms, 30s window)")
ax.set_title("Figure 1. Chain 13 CP — MIG cross-partition = perfect isolation\n"
             "L1 in 4g stays at ~1,800 ms regardless of what runs in 3g",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, max(cp_vals)*1.20)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_chain13_cp_isolation.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved fig1_chain13_cp_isolation.png")

# =============================================================================
# fig2 — SP MPS off vs on for each workload
# =============================================================================
sp_workloads = [("NRx","compute","nrx"), ("ChanPred","compute","chanpred"),
                ("Whisper","memory","whisper"), ("Qwen-VL","memory","qwen_vl"),
                ("Qwen-RAG","memory (L1 OOM)","qwen_rag")]

baseline = summary["SP0_baseline"]["cf_ms"]
fig, ax = plt.subplots(figsize=(11, 6))
xs = np.arange(len(sp_workloads))
w = 0.4
offs = [summary[f"SP_{k}_MPSoff"]["cf_ms"] for _,_,k in sp_workloads]
ons  = [summary[f"SP_{k}_MPSon"]["cf_ms"]  for _,_,k in sp_workloads]

bars_off = ax.bar(xs-w/2, offs, w, color="#dc2626", label="MPS off (temporal)", edgecolor="#7f1d1d")
bars_on  = ax.bar(xs+w/2, ons, w, color="#10b981", label="MPS on (spatial)", edgecolor="#065f46")

for x, (off, on) in enumerate(zip(offs, ons)):
    ax.text(x-w/2, off+400, f"{off:,.0f}\n({off/baseline:.1f}×)",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(x+w/2, on+400, f"{on:,.0f}\n({on/baseline:.2f}×)",
            ha="center", fontsize=9, fontweight="bold")

ax.axhline(baseline, color="#111", linestyle=":", linewidth=1.2, alpha=0.7,
           label=f"SP0 baseline ({baseline:.0f} ms)")
ax.set_xticks(xs); ax.set_xticklabels([f"{n}\n({c})" for n,c,_ in sp_workloads], fontsize=10)
ax.set_ylabel("L1 cudaFree total (ms, 30s window)")
ax.set_title("Figure 2. Chain 13 SP — MPS effect gradient by workload class\n"
             "Compute: perfect recovery.  Memory: partial (Whisper 1.15×, VL 1.27×).  Qwen-RAG: L1 OOM (7B fp16 + L1 > 20GB 4g)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, max(offs)*1.20)
ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_chain13_sp_mps_gradient.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved fig2_chain13_sp_mps_gradient.png")

# =============================================================================
# fig3 — MPS-on vs baseline residual sync (compute vs memory)
# =============================================================================
labels = ["NRx", "ChanPred", "Whisper", "Qwen-VL"]
classes = ["compute", "compute", "memory (batch=1)", "memory (batch=1)"]
mps_on_penalty = [summary[f"SP_{k}_MPSon"]["cf_ms"] / baseline
                  for k in ["nrx","chanpred","whisper","qwen_vl"]]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#3b82f6", "#3b82f6", "#f59e0b", "#f59e0b"]
xs = np.arange(len(labels))
bars = ax.bar(xs, mps_on_penalty, color=colors, edgecolor="#111", linewidth=1)
for x, v in zip(xs, mps_on_penalty):
    ax.text(x, v+0.02, f"{v:.2f}×", ha="center", fontweight="bold", fontsize=11)

ax.axhline(1.0, color="#059669", linestyle="--", linewidth=1.5, label="Perfect isolation (1.0×)")
ax.axhline(1.15, color="#eab308", linestyle=":", linewidth=1, alpha=0.7, label="Breakdown threshold (empirical)")

ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n({c})" for l,c in zip(labels,classes)], fontsize=10)
ax.set_ylabel("L1 cudaFree penalty (MPS on / baseline)")
ax.set_title("Figure 3. MPS residual sync — compute vs memory-bound gradient\n"
             "MPS fully rescues compute; memory-bound shows partial break — real sanction of 20260708 hypothesis",
             fontsize=11, fontweight="bold", pad=12)
ax.set_ylim(0.8, 1.5); ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_chain13_mps_residual.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved fig3_chain13_mps_residual.png")

print("\nAll figures saved.")
