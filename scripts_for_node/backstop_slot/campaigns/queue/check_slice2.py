"""Check of the smoke test of v22 (script copy scripts_slice2): the rule without the option gives what it gave before,
the lanes of the off-slice policies ran on 80 SMs, the option put pieces on the slice, and the baselines serve AI."""
import json, sys
from pathlib import Path
RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
def run(pol):
    paths = sorted(RAW.glob(f"ltm1{pol}_c16_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit(f"FAIL no result of ltm1{pol}")
    path = paths[-1]
    work = Path(str(path)[:-5] + "_work")
    ai = [json.loads(p.read_text())["summary"] for p in sorted(work.glob("ai*.json"))]
    lane_files = [json.loads(p.read_text()) for p in sorted(work.glob("lane*.json"))]
    lanes = [r for d in lane_files for r in d["records"]]
    run_ms = sorted((r[3] - r[2]) / 1e6 for r in lanes)
    pick = lambda q: run_ms[min(len(run_ms) - 1, int(len(run_ms) * q))] if run_ms else None
    return {"ai_tokens_per_s": sum(a["tokens_within_slo_per_s"] for a in ai),
            "slice_pieces": sum(a.get("slice_pieces", 0) for a in ai),
            "slice_wall_ms": sum(a.get("slice_wall_ms", 0.0) for a in ai),
            "lane_sms": sorted({d.get("sms", 0) for d in lane_files}),
            "nrx_runs": len(lanes), "nrx_run_ms_p50_p90_p99": [pick(0.5), pick(0.9), pick(0.99)],
            "nrx_crc_ok": sum(r[4] for r in lanes)}
ok = True
none, rule, wd, gd, gn = run("n"), run("r32wm"), run("r32wd28"), run("r32gd28"), run("r32gn28")
for name, r in (("no AI", none), ("rule", rule), ("rule + slice 28, receivers off the slice", wd),
                ("always on 28 SMs, receivers off the slice", gd), ("always on 28 SMs at normal priority", gn)):
    print(name, r)
if rule["slice_pieces"] != 0 or rule["lane_sms"] != [0] or not 7000 < rule["ai_tokens_per_s"] < 16000:
    ok = False; print("FAIL the rule without the option changed (expected about 11k tokens/s, no slice pieces, lanes on the whole GPU)")
if wd["lane_sms"] != [80] or gd["lane_sms"] != [80] or gn["lane_sms"] != [0]:
    ok = False; print("FAIL the lanes did not run on the SMs expected (80, 80, whole GPU)")
if wd["slice_pieces"] < 100 or wd["ai_tokens_per_s"] < 0.9 * rule["ai_tokens_per_s"]:
    ok = False; print("FAIL the option put no pieces on the slice or served less AI than the rule")
if min(gd["ai_tokens_per_s"], gn["ai_tokens_per_s"]) < 3000:
    ok = False; print("FAIL a baseline on a slice served no AI")
if min(wd["nrx_runs"], gd["nrx_runs"], gn["nrx_runs"]) < 0.5 * none["nrx_runs"]:
    ok = False; print("FAIL few neural receiver runs")
if min(wd["nrx_crc_ok"] / max(1, wd["nrx_runs"]), gd["nrx_crc_ok"] / max(1, gd["nrx_runs"])) < 0.5 * none["nrx_crc_ok"] / max(1, none["nrx_runs"]):
    ok = False; print("FAIL the neural receiver on 80 SMs decodes far fewer TBs than on the whole GPU")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
