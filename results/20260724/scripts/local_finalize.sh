#!/usr/bin/env bash
# Local-side finalizer: waits for node's AUTO_PIPELINE_DONE, then syncs
# everything, generates figures, writes REPORT, commits + pushes.
#
# Runs on THIS mac while user sleeps. Node runs auto_pipeline.sh.
set -uo pipefail

NODE=sgkim@d8545-10s10505.wisc.cloudlab.us
DATE_DIR=${DATE_DIR:-20260724}
LOCAL_RESULTS=/Users/changjongkim/New_research/cloudlab_results/results/$DATE_DIR
LOCAL_SCRIPTS=/Users/changjongkim/New_research/cloudlab_aerial
LOCAL_REPO=/Users/changjongkim/New_research/cloudlab_results

LOG=/tmp/local_finalize.log

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=========================================================================="
log "local_finalize START — waiting for node AUTO_PIPELINE_DONE"
log "=========================================================================="

# ─── 1. Poll for done marker (max 24 hours) ───────────────────────
DEADLINE=$(($(date +%s) + 86400))
while true; do
  if ssh -o ConnectTimeout=10 -o BatchMode=yes $NODE "test -f /users/sgkim/AUTO_PIPELINE_DONE" 2>/dev/null; then
    log "node auto_pipeline complete!"
    break
  fi
  now=$(date +%s)
  if [ $now -gt $DEADLINE ]; then
    log "24h timeout waiting for AUTO_PIPELINE_DONE — aborting"
    exit 1
  fi
  sleep 300  # poll every 5 min
done

# ─── 2. Sync results to local ─────────────────────────────────────
log "syncing /mydata/results/$DATE_DIR to local"
mkdir -p "$LOCAL_RESULTS"
rsync -a --info=stats2 "$NODE:/mydata/results/$DATE_DIR/" "$LOCAL_RESULTS/" >>"$LOG" 2>&1
log "results synced ($(du -sh "$LOCAL_RESULTS" | cut -f1))"

# ─── 3. Sync all scripts + logs ───────────────────────────────────
log "syncing scripts + logs"
scp $NODE:/users/sgkim/cloudlab_aerial/*.sh $NODE:/users/sgkim/cloudlab_aerial/*.py \
  "$LOCAL_SCRIPTS/" >>"$LOG" 2>&1
scp $NODE:/users/sgkim/auto_pipeline.log $NODE:/users/sgkim/chain14.log $NODE:/users/sgkim/chain15.log \
  "$LOCAL_RESULTS/" >>"$LOG" 2>&1 || true
log "scripts + logs synced"

# ─── 4. Generate figures (chain14 + chain15) ──────────────────────
log "generating figures"
cd "$LOCAL_RESULTS/.." && python3 <<'PYEOF' 2>&1 | tee -a $LOG
import json, os, glob
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

for chain in ("chain14", "chain15"):
    sp = os.path.join(BASE, f"{chain}_summary.json")
    if not os.path.exists(sp):
        print(f"skip {chain} — no summary")
        continue
    with open(sp) as f: d = json.load(f)
    # SP MPS off vs on scatter
    fig, ax = plt.subplots(figsize=(12, 6))
    off_ks = [k for k in d if "SP_" in k and "MPSoff" in k]
    for k in sorted(off_ks):
        on_k = k.replace("MPSoff","MPSon")
        if on_k not in d: continue
        cf_off = d[k].get("cudaFree_ms",0); cf_on = d[on_k].get("cudaFree_ms",0)
        wl = k.split("_SP_")[-1].replace("_MPSoff","")
        ax.scatter([cf_off], [cf_on], s=100, label=wl)
    ax.set_xlabel("MPS off  cudaFree (ms)"); ax.set_ylabel("MPS on cudaFree (ms)")
    ax.set_title(f"{chain} — SP MPS on vs off (each point = workload)")
    ax.plot([0, 30000], [0, 30000], 'k--', alpha=0.3, label='y=x')
    ax.legend(fontsize=7, loc='best'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/{chain}_sp_mps_scatter.png", dpi=140, bbox_inches='tight')
    plt.close()
    print(f"saved {chain}_sp_mps_scatter.png")
PYEOF

# ─── 5. Auto-generate REPORT stub ─────────────────────────────────
log "writing REPORT_20260724.md stub"
python3 <<'PYEOF' >>"$LOG" 2>&1
import json, os
BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"

def load(p):
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

c14 = load(f"{BASE}/chain14_summary.json")
c15 = load(f"{BASE}/chain15_summary.json")

md = ["# 20260724 Session — Chain 14 + Chain 15 auto-generated report\n\n"]
md += [f"**Auto-generated at pipeline completion.**\n\n"]
md += [f"Chain 14: {len(c14)} conditions\n"]
md += [f"Chain 15: {len(c15)} conditions (batch sweep)\n\n"]

for chain, d in (("Chain 14", c14), ("Chain 15", c15)):
    if not d: continue
    md += [f"## {chain}\n\n"]
    md += ["| condition | n | cudaFree_ms | l1_mean_ms | ai_metric |\n|---|---:|---:|---:|---:|\n"]
    for k in sorted(d.keys()):
        s = d[k]
        ai = ""
        for f in ("ai_tok_s","ai_rtf","ai_imgs_per_s","ai_hbm_bw_gbps","ai_rate_per_s","ai_fwd_per_s","ai_effective_bw_gbps"):
            if f in s: ai = f"{f}={s[f]:.1f}"; break
        md += [f"| {k} | {s.get('n','')} | {s.get('cudaFree_ms',0):.0f} | {s.get('l1_mean_ms',0):.1f} | {ai} |\n"]
    md += ["\n"]

with open(f"{BASE}/REPORT_20260724_auto.md","w") as f: f.writelines(md)
print("wrote REPORT_20260724_auto.md")
PYEOF

# ─── 6. Git commit + push ─────────────────────────────────────────
log "committing to airan_cloudlab repo"
cd "$LOCAL_REPO"
git add "results/$DATE_DIR/" 2>>"$LOG"
git commit -m "$(cat <<CMSG
Chain 14 + Chain 15 auto-collected results ($DATE_DIR)

Chain 14: 3 configs × 11 workloads × MPS on/off × 3 trials
  Includes original 6 workloads + 5 new realistic memory-bound:
  qwen_chat_b1, whisper_stream_b1, bert_b1, embed_lookup, memcpy_loop

Chain 15: 3 configs × 17 batch-swept workloads × MPS on/off × 3 trials
  Batch sweep: qwen_chat {1,2,4,8,16,32}, bert {1,4,16,64},
  whisper {1,2,4,8}, qwen_vl {1,2,4}

Auto-generated by local_finalize.sh after auto_pipeline.sh on d8545 node.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
CMSG
)" 2>>"$LOG" || log "commit failed (maybe nothing to commit)"
git push 2>>"$LOG" && log "pushed to origin" || log "push failed"

log "=========================================================================="
log "local_finalize DONE at $(ts)"
log "=========================================================================="
osascript -e 'display notification "Chain 14+15 done, results pushed" with title "AI-RAN experiment"' 2>/dev/null || true
