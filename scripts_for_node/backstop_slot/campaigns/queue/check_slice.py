"""Check of the smoke test of the SM slice: the rule without the option gives what it gave before, the option puts
pieces on the slice, and the baseline on a slice serves AI."""
import json, sys
from pathlib import Path
RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
def run(pol):
    paths = sorted(RAW.glob(f"lsm1{pol}_c16_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit(f"FAIL no result of lsm1{pol}")
    path = paths[-1]
    work = Path(str(path)[:-5] + "_work")
    ai = [json.loads(p.read_text())["summary"] for p in sorted(work.glob("ai*.json"))]
    served = sum(a["tokens_within_slo_per_s"] for a in ai)
    lanes = [r for p in sorted(work.glob("lane*.json")) for r in json.loads(p.read_text())["records"]]
    run_ms = sorted((r[3] - r[2]) / 1e6 for r in lanes)
    return {"ai_tokens_per_s": served, "slice_pieces": sum(a.get("slice_pieces", 0) for a in ai),
            "slice_wall_ms": sum(a.get("slice_wall_ms", 0.0) for a in ai), "slice_sms": [a.get("slice_sms") for a in ai],
            "nrx_runs": len(lanes), "nrx_run_p50_ms": run_ms[len(run_ms) // 2] if run_ms else None,
            "nrx_crc_ok": sum(r[4] for r in lanes)}
ok = True
none, rule, sl, green = run("n"), run("r32wm"), run("r32ws28"), run("r32g28")
for name, r in (("no AI", none), ("rule", rule), ("rule + slice 28", sl), ("always on 28 SMs", green)):
    print(name, r)
if rule["slice_pieces"] != 0 or not 7000 < rule["ai_tokens_per_s"] < 16000:
    ok = False; print("FAIL the rule without the option changed (expected about 11.5k tokens/s and no slice pieces)")
if sl["slice_pieces"] < 100 or sl["ai_tokens_per_s"] < 0.9 * rule["ai_tokens_per_s"]:
    ok = False; print("FAIL the option put no pieces on the slice or served less AI than the rule")
if green["ai_tokens_per_s"] < 3000 or green["slice_pieces"] < 100:
    ok = False; print("FAIL the baseline on a slice served no AI")
if min(sl["nrx_runs"], green["nrx_runs"]) < 0.5 * none["nrx_runs"]:
    ok = False; print("FAIL few neural receiver runs")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
