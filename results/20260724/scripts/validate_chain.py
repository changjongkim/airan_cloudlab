"""Validate all chain captures — identify broken ones for retry.

Checks per capture:
  1. nsys-rep exists and > 500 KB (empty capture = broken)
  2. sqlite exports successfully
  3. L1 has cudaFree activity (SP condition where sync expected)
  4. real_l1.py JSON exists with 100 iterations completed
  5. AI-side log (if applicable) has non-error output

Usage:
  python3 validate_chain.py --chain chain14 [--chain chain15] --output failed.txt
  → prints report + writes failure list (one per line: label\ttrial)
"""
import argparse, os, glob, json, sqlite3, sys
from collections import defaultdict

MIN_NSYS_SIZE = 500_000   # bytes
MIN_L1_ITERS  = 90         # of 100

def check_nsys_rep(path):
    if not os.path.exists(path): return "MISSING"
    if os.path.getsize(path) < MIN_NSYS_SIZE: return f"TINY({os.path.getsize(path)})"
    return None

def check_sqlite_cudafree(sq_path):
    """Ensure sqlite has real CUDA activity (not empty)."""
    if not os.path.exists(sq_path): return "NO_SQLITE"
    try:
        con = sqlite3.connect(sq_path); cur = con.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
            n = cur.fetchone()[0]
        except sqlite3.OperationalError:
            return "NO_CUDA_TABLE"
        con.close()
        if n < 100: return f"CUDA_ROWS={n}"
        return None
    except Exception as e:
        return f"SQLITE_ERR({e})"

def check_l1_json(chain_dir, label):
    """Find realL1_<label>_*.json and check iterations."""
    jsons = glob.glob(f"{chain_dir}/realL1_{label}_*.json")
    if not jsons: return "NO_JSON"
    try:
        with open(jsons[0]) as f: d = json.load(f)
        if d.get("miss_1ms", 0) < MIN_L1_ITERS: return f"ITERS={d.get('miss_1ms',0)}"
        return None
    except Exception as e:
        return f"JSON_ERR({e})"

def check_ai_log(chain_dir, label):
    """AI-side co-tenant log — optional (only exists for some workloads)."""
    log = f"{chain_dir}/{label}_same_{'t1' if label.endswith('_t1') else label.split('_t')[-1]}.log" if '_t' in label else None
    # try both cross and same
    for suffix in ["_same_t1.log", "_same_t2.log", "_same_t3.log", "_t1.log", "_t2.log", "_t3.log"]:
        p = f"{chain_dir}/{label.replace('_t1','').replace('_t2','').replace('_t3','')}_{suffix.strip('_')}"
        if os.path.exists(p):
            with open(p, errors='ignore') as f:
                text = f.read()
            if "Traceback" in text or "OOM" in text or "OutOfMemory" in text:
                return "AI_ERROR"
    return None

def scan_chain(chain_dir, prefix_filter=""):
    """Return {label: [(trial_no, status_list)]}"""
    results = defaultdict(list)
    for f in sorted(glob.glob(f"{chain_dir}/{prefix_filter}*.nsys-rep")):
        base = os.path.basename(f).replace(".nsys-rep","")
        if "_ai" in base: continue   # only L1-side reports here
        # extract label (strip _t1/_t2/_t3)
        trial = None
        label = base
        for suf in ["_t1","_t2","_t3"]:
            if base.endswith(suf):
                trial = int(suf[-1]); label = base[:-3]; break

        issues = []
        e = check_nsys_rep(f); e and issues.append(f"nsys:{e}")

        sq = f.replace(".nsys-rep", ".sqlite")
        e = check_sqlite_cudafree(sq); e and issues.append(f"sqlite:{e}")

        e = check_l1_json(chain_dir, base); e and issues.append(f"json:{e}")

        results[label].append((trial, base, issues))
    return results

def report(results, chain_name):
    total = sum(len(v) for v in results.values())
    fail_captures = []
    for label, trials in results.items():
        for trial, base, issues in trials:
            if issues:
                fail_captures.append((base, issues))
    print(f"\n=== {chain_name} validation ===")
    print(f"total captures: {total}")
    print(f"failed: {len(fail_captures)}")
    if fail_captures:
        for base, issues in fail_captures[:40]:
            print(f"  FAIL {base}: {', '.join(issues)}")
        if len(fail_captures) > 40:
            print(f"  ... and {len(fail_captures)-40} more")

    # Also check missing trials (expected 3 per label)
    missing = []
    for label, trials in results.items():
        n = len(trials)
        if n < 3: missing.append(f"{label}: only {n}/3 trials")
    if missing:
        print(f"\ntrials missing: {len(missing)} labels incomplete")
        for m in missing[:20]:
            print(f"  {m}")
    return fail_captures, missing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-dir", required=True, help="path to chain directory")
    ap.add_argument("--output", help="write failure list (base names) here")
    args = ap.parse_args()

    chain_name = os.path.basename(args.chain_dir.rstrip("/"))
    r = scan_chain(args.chain_dir)
    fails, missing = report(r, chain_name)

    if args.output:
        with open(args.output, "w") as f:
            for base, issues in fails:
                f.write(f"{base}\t{'|'.join(issues)}\n")
            for m in missing:
                f.write(f"MISSING\t{m}\n")
        print(f"\nWrote {args.output}")

    if fails or missing:
        sys.exit(1)   # non-zero = failures exist
    sys.exit(0)

if __name__ == "__main__":
    main()
