"""Check of the smoke test of the channel states: lsr (one state) and lss (two states, the same dataset twice) must
give the same radio results, and lss must use the rings of the second state."""
import json, sys
from pathlib import Path
import numpy as np
RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
WARM = 800

def stats(tag, pol):
    paths = sorted(RAW.glob(f"{tag}1{pol}_c16_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit(f"FAIL no result of {tag}1{pol}")
    path = paths[-1]
    config = json.loads(path.read_text())["config"]
    la = config["la"]; m = len(la["levels"])
    work = Path(str(path)[:-5] + "_work")
    lanes = {}
    for lane in sorted(work.glob("lane*.json")):
        for r in json.loads(lane.read_text())["records"]:
            lanes[(r[0], r[1])] = r
    levels, fails, tbs, recovered, passes = [], 0, 0, 0, 0
    for cell in config["cells"]:
        if cell.get("nrx_gpu") is None:
            continue
        for r in json.loads((work / f"conv{cell['cell']}.json").read_text())["records"]:
            if r[0] < WARM:
                continue
            levels.append(r[10]); tbs += 2; fails += 2 - bin(r[9]).count("1")
            ran = lanes.get((cell["cell"], r[0]))
            if ran is not None:
                passes += 1
                recovered += bin(ran[7] & ~r[9] & 3).count("1")
    levels = np.asarray(levels)
    return {"runs": passes, "mean_mcs": float(np.mean(np.asarray(la["mcs"])[levels % m])), "max_level": int(levels.max()),
            "second_state_share": float(np.mean(levels >= m)), "fail_pct": 100.0 * fails / tbs, "recovered": recovered,
            "states": len(la.get("datasets", [])) or 1}

ok = True
for pol in ("n", "r32wm"):
    a, b = stats("lsr", pol), stats("lss", pol)
    print(pol, "one state:", a); print(pol, "two states:", b)
    if a["states"] != 1 or b["states"] != 2:
        ok = False; print("FAIL number of states")
    if not 0.2 < b["second_state_share"] < 0.8:
        ok = False; print("FAIL the second state is not used")
    if abs(a["mean_mcs"] - b["mean_mcs"]) > 0.25 or abs(a["fail_pct"] - b["fail_pct"]) > 3.0:
        ok = False; print("FAIL the radio results differ")
    if a["recovered"] < 20 or abs(a["recovered"] - b["recovered"]) > 0.25 * a["recovered"] + 10:
        ok = False; print("FAIL the recoveries differ")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
