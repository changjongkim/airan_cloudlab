"""Aggregate chain sqlite → summary JSON with L1 cudaFree/latency + AI throughput.

Usage:
  python3 aggregate_summary.py --chain-dir /mydata/results/YYYYMMDD/chainNN --output summary.json
"""
import argparse, os, glob, json, sqlite3, statistics, re
from collections import defaultdict

def get_l1_stats(sq):
    con = sqlite3.connect(sq); cur = con.cursor()
    try:
        def like(pat):
            cur.execute("SELECT id FROM StringIds WHERE value LIKE ?", (pat,))
            return [x[0] for x in cur.fetchall()]
        cf   = like('cudaFree_v____')
        mca  = like('cudaMemcpyAsync%')
        cma  = like('cudaMalloc_v____')
        clk  = like('cuLaunchKernel')
        cdlk = like('cudaLaunchKernel_v____')
        def sumns(ids):
            if not ids: return 0, 0
            ph = ",".join(["?"]*len(ids))
            cur.execute(f"SELECT SUM(end-start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph})", ids)
            r = cur.fetchone(); return (r[0] or 0), (r[1] or 0)
        cf_ns, cf_n   = sumns(cf)
        mca_ns, _     = sumns(mca)
        cma_ns, _     = sumns(cma)
        clk_ns, clk_n = sumns(clk)
        cdlk_ns, cdlk_n = sumns(cdlk)
        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        tot_ns = cur.fetchone()[0] or 0
    finally:
        con.close()
    return {
        "cudaFree_ms":    cf_ns/1e6, "cudaFree_n": cf_n,
        "memcpyAsync_ms": mca_ns/1e6,
        "cudaMalloc_ms":  cma_ns/1e6,
        "launches_n":     clk_n + cdlk_n,
        "total_host_ms":  tot_ns/1e6,
    }

def parse_l1_json(jf):
    with open(jf) as f: d = json.load(f)
    return {"mean_ms": d.get("mean_ms"), "p50_ms": d.get("p50_ms"),
            "p95_ms": d.get("p95_ms"),  "p99_ms": d.get("p99_ms"),
            "miss_1ms": d.get("miss_1ms")}

def parse_ai_throughput(log_path):
    """Extract throughput from AI-side log. Returns dict of measured throughputs."""
    if not os.path.exists(log_path): return {}
    with open(log_path, errors='ignore') as f: text = f.read()
    out = {}
    for m in re.finditer(r'tok/s=([\d.]+)', text): out["tok_s"] = float(m.group(1))
    for m in re.finditer(r'tok_per_s=([\d.]+)', text): out["tok_s"] = float(m.group(1))
    for m in re.finditer(r'audio_sec_per_wall=([\d.]+)', text): out["audio_sec_per_wall"] = float(m.group(1))
    for m in re.finditer(r'rtf=([\d.]+)', text): out["rtf"] = float(m.group(1))
    for m in re.finditer(r'imgs_per_s=([\d.]+)', text): out["imgs_per_s"] = float(m.group(1))
    for m in re.finditer(r'HBM_BW=([\d.]+)', text): out["hbm_bw_gbps"] = float(m.group(1))
    for m in re.finditer(r'rate=(\d+)/s', text): out["rate_per_s"] = int(m.group(1))
    for m in re.finditer(r'fwd_per_s=([\d.]+)', text): out["fwd_per_s"] = float(m.group(1))
    for m in re.finditer(r'effective_BW=([\d.]+)', text): out["effective_bw_gbps"] = float(m.group(1))
    return out

def strip_trial(base):
    for suf in ["_t1","_t2","_t3"]:
        if base.endswith(suf): return base[:-3]
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    r = defaultdict(list)
    for sq in sorted(glob.glob(f"{args.chain_dir}/*.sqlite")):
        if "_ai." in sq: continue
        base = os.path.basename(sq).replace(".sqlite","")
        label = strip_trial(base)
        try:
            l1 = get_l1_stats(sq)
        except Exception:
            continue

        entry = dict(l1)
        # attach L1 latency JSON
        jf_list = glob.glob(f"{args.chain_dir}/realL1_{base}_*.json")
        if jf_list:
            try: entry.update({f"l1_{k}": v for k, v in parse_l1_json(jf_list[0]).items()})
            except Exception: pass

        # attach AI throughput (search for co-tenant log)
        for suffix in ["_same_t1", "_same_t2", "_same_t3", "_t1", "_t2", "_t3"]:
            if base.endswith(suffix.replace("_same","")):
                # look for matching AI log
                ai_log_candidates = glob.glob(f"{args.chain_dir}/{base}_same_*.log") + \
                                    glob.glob(f"{args.chain_dir}/{base}.log")
                for c in ai_log_candidates:
                    tp = parse_ai_throughput(c)
                    if tp:
                        entry.update({f"ai_{k}": v for k, v in tp.items()})
                        break
                break

        r[label].append(entry)

    summary = {}
    for k, ts in r.items():
        agg = {"n": len(ts)}
        for field in ("cudaFree_ms","memcpyAsync_ms","cudaMalloc_ms","launches_n","total_host_ms",
                      "l1_mean_ms","l1_p95_ms","l1_p99_ms","l1_miss_1ms",
                      "ai_tok_s","ai_audio_sec_per_wall","ai_rtf","ai_imgs_per_s",
                      "ai_hbm_bw_gbps","ai_rate_per_s","ai_fwd_per_s","ai_effective_bw_gbps"):
            vals = [t.get(field) for t in ts if t.get(field) is not None]
            if vals: agg[field] = statistics.mean(vals)
        summary[k] = agg

    with open(args.output, "w") as f: json.dump(summary, f, indent=2)
    print(f"wrote {args.output} with {len(summary)} labels")

if __name__ == "__main__":
    main()
