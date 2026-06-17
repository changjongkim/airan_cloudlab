#!/usr/bin/env python3
"""no-MIG nsys supplementary figures in the EXACT style of the MIG doc's
build_time_breakdown.py / build_queue_evidence.py / build_nsys_extra_breakdown.py,
so they compare 1:1 with fig_supp_11/14/15/16/24. Conditions = default vs MPS."""
import sqlite3, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl

# ---- EXACT reference style ----
mpl.rcParams.update({
    'font.size': 10, 'figure.figsize': (12, 7), 'savefig.dpi': 140, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
    'font.family': ['DejaVu Sans', 'Apple SD Gothic Neo', 'NanumGothic', 'sans-serif'],
})
BASE = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"
OUT = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/figures"

# default vs MPS per condition (directly readable)
CONDS = [
    ("alone",        f"{BASE}/nsys_nomig/nsys_alone.sqlite"),
    ("chanpred def", f"{BASE}/nsys_nomig/nsys_chanpred.sqlite"),
    ("chanpred MPS", f"{BASE}/MPS_nsys/MPSnsys_chanpred.sqlite"),
    ("neuralrx def", f"{BASE}/nsys_nomig/nsys_neuralrx.sqlite"),
    ("neuralrx MPS", f"{BASE}/MPS_nsys/MPSnsys_neuralrx.sqlite"),
    ("resnet def",   f"{BASE}/nsys_nomig/nsys_resnet.sqlite"),
    ("resnet MPS",   f"{BASE}/MPS_nsys/MPSnsys_resnet.sqlite"),
    ("sat_hbm def",  f"{BASE}/nsys_nomig/nsys_sat_hbm.sqlite"),
    ("sat_hbm MPS",  f"{BASE}/MPS_nsys/MPSnsys_sat_hbm.sqlite"),
]

def te(c,t): return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",(t,)).fetchone() is not None

def get_activity_breakdown(db):
    if not os.path.exists(db): return None
    con=sqlite3.connect(db); out={"kernel_ms":0,"memcpy_ms":0,"memset_ms":0,"sync_ms":0}
    spans=[]
    for tbl,key in [("CUPTI_ACTIVITY_KIND_KERNEL","kernel_ms"),("CUPTI_ACTIVITY_KIND_MEMCPY","memcpy_ms"),
                    ("CUPTI_ACTIVITY_KIND_MEMSET","memset_ms")]:
        if te(con,tbl):
            for s,e in con.execute(f"SELECT start,end FROM {tbl}"): out[key]+=(e-s)/1e6; spans.append((s,e))
    if te(con,"CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        r=con.execute("SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION").fetchone()
        out["sync_ms"]=r[0] or 0
    con.close()
    if not spans: return None
    starts=np.array([s for s,_ in spans]); ends=np.array([e for _,e in spans])
    out["wall_ms"]=(ends.max()-starts.min())/1e6
    out["idle_ms"]=max(0,out["wall_ms"]-out["kernel_ms"]-out["memcpy_ms"]-out["memset_ms"])
    return out

def get_kernel_stage_breakdown(db):
    if not os.path.exists(db): return None
    con=sqlite3.connect(db)
    raw=con.execute("""SELECT sn.value, SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k
                       LEFT JOIN StringIds sn ON sn.id=k.shortName GROUP BY sn.value""").fetchall()
    con.close()
    stages=defaultdict(float)
    for name,ms in raw:
        n=name or "?"; ms=ms or 0; nl=n.lower()
        if "windowedchest" in nl or "preNoDft" in n: stages["pre-ChEst"]+=ms
        elif "chestfilter" in nl or ("chest" in nl and "dispatch" in nl): stages["ChEst dispatch"]+=ms
        elif "noiseintf" in nl or "noise" in nl: stages["Noise/Intf est"]+=ms
        elif "eqmmse" in nl: stages["Equalizer (MMSE)"]+=ms
        elif "ldpc" in nl or "deratematch" in nl or "decode" in nl: stages["LDPC"]+=ms
        elif "convert" in nl: stages["Convert (boundary)"]+=ms
        elif "cupy_copy" in nl or "copy_" in nl: stages["Copy ops"]+=ms
        elif "crc" in nl: stages["CRC check"]+=ms
        else: stages["Other"]+=ms
    return dict(stages)

def per_call_60k(db):
    if not os.path.exists(db): return None
    c=sqlite3.connect(db)
    d=np.array([e-s for s,e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=61440")])
    c.close(); return d/1e3 if len(d) else None

# ===== pmF_supp_14: GPU activity decomposition (style of supp_14) =====
data=[(l,get_activity_breakdown(db)) for l,db in CONDS]; data=[d for d in data if d[1]]
fig,ax=plt.subplots(figsize=(13,7)); x=np.arange(len(data))
comps=[("kernel_ms","GPU kernel","#3498DB"),("memcpy_ms","memcpy","#9B59B6"),("memset_ms","memset","#E74C3C"),
       ("sync_ms","sync","#F39C12"),("idle_ms","idle (gap)","#95A5A6")]
bottom=np.zeros(len(data))
for key,lbl,col in comps:
    vals=[d[1][key] for d in data]; ax.bar(x,vals,bottom=bottom,label=lbl,color=col); bottom+=vals
ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data],rotation=25,ha='right'); ax.set_ylabel("Time (ms)")
ax.set_title("no-MIG — GPU Activity Time Decomposition (kernel/memcpy/memset/sync/idle)\n비교용: MIG_AIRAN_VISUAL_EVIDENCE Supp 14 와 동일 스타일")
ax.legend(loc='upper left')
for i,(l,b) in enumerate(data): ax.text(i,b["wall_ms"]+30,f"{b['wall_ms']:.0f}ms",ha='center',fontsize=9,fontweight='bold')
fig.tight_layout(); fig.savefig(f"{OUT}/pmF_supp_14_gpu_activity.png"); plt.close(fig)

# ===== pmF_supp_15: cuPHY pipeline stages (§17.2) — #2 =====
rows=[(l,get_kernel_stage_breakdown(db)) for l,db in CONDS]; rows=[r for r in rows if r[1]]
all_stages=["pre-ChEst","ChEst dispatch","Noise/Intf est","Equalizer (MMSE)","LDPC","Convert (boundary)","Copy ops","CRC check","Other"]
colors=["#16A085","#27AE60","#F39C12","#3498DB","#9B59B6","#E74C3C","#E67E22","#95A5A6","#666666"]
fig,ax=plt.subplots(figsize=(13,7)); x=np.arange(len(rows)); bottom=np.zeros(len(rows))
for stage,col in zip(all_stages,colors):
    vals=[r[1].get(stage,0) for r in rows]; ax.bar(x,vals,bottom=bottom,label=stage,color=col); bottom+=vals
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],rotation=25,ha='right'); ax.set_ylabel("Kernel time (ms)")
ax.set_title("no-MIG — cuPHY pipeline stage별 kernel 시간 분해 (§17.2)\n비교용: MIG Supp 15 와 동일 스타일 — 어느 stage가 AI/MPS로 늘어나는가")
ax.legend(loc='upper left',ncol=2,fontsize=9)
for i,(l,s) in enumerate(rows): ax.text(i,sum(s.values())+5,f"{sum(s.values()):.0f}ms",ha='center',fontsize=9,fontweight='bold')
fig.tight_layout(); fig.savefig(f"{OUT}/pmF_supp_15_pipeline_stages.png"); plt.close(fig)

# ===== pmF_supp_11: per-call 60KB memcpy median+p99 (§16.1, style of supp_11) =====
rows2=[(l,per_call_60k(db)) for l,db in CONDS]; rows2=[r for r in rows2 if r[1] is not None and len(r[1])]
fig,ax=plt.subplots(figsize=(13,7)); x=np.arange(len(rows2)); w=0.38
meds=[np.percentile(r[1],50) for r in rows2]; p99s=[np.percentile(r[1],99) for r in rows2]
cols=['#C0392B' if p>8 else '#3498DB' for p in p99s]
ax.bar(x-w/2,meds,w,color=cols,label='median per-call')
ax.bar(x+w/2,p99s,w,color=cols,alpha=0.5,edgecolor='black',label='p99 per-call')
ax.axhline(4.2,ls='--',color='#666666',alpha=0.6); ax.text(0.0,4.5,"baseline launch overhead (4.2us)",fontsize=9,color='#666666')
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows2],rotation=25,ha='right'); ax.set_ylabel("60KB memcpy per-call duration (us)")
ax.set_title("no-MIG — Per-call 60KB memcpy duration: bimodal split = queue 직접 증거 (§16.1)\n비교용: MIG Supp 11 과 동일 스타일")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/pmF_supp_11_percall_queue.png"); plt.close(fig)

print("wrote pmF_supp_14/15/11.")
for l,s in rows:
    conv=s.get("Convert (boundary)",0); tot=sum(s.values())
    print(f"  {l:14s} total_kernel={tot:7.0f}ms  convert={conv:7.0f}ms ({100*conv/tot:.0f}%)")
