"""Slots made by added noise against slots generated at the same Es/No: TBs decoded by each receiver, per MCS."""
import json, math, sys
from pathlib import Path
RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
pairs = [tuple(a.split(":")) for a in sys.argv[1:]] or [("syn16", "high16"), ("syn14", "high14")]
ok = True
for made, generated in pairs:
    print(f"## {made} (added noise) against {generated} (generated)")
    print("| MCS | conventional: made / generated | neural: made / generated | |")
    for mcs in range(10, 17):
        a = json.loads((RAW / f"lacl_{made}_m{mcs}.json").read_text())
        b = json.loads((RAW / f"lacl_{generated}_m{mcs}.json").read_text())
        flags = []
        cells = []
        for key in ("conventional", "neural"):
            pa, pb = a[key] / a["tbs"], b[key] / b["tbs"]
            p = (a[key] + b[key]) / (a["tbs"] + b["tbs"])
            sigma = math.sqrt(max(p * (1 - p), 1e-4) * (1 / a["tbs"] + 1 / b["tbs"]))
            if abs(pa - pb) > 3 * sigma + 0.015:
                flags.append(key); ok = False
            cells.append(f"{100 * pa:.1f}% / {100 * pb:.1f}%")
        print(f"| {mcs} | {cells[0]} | {cells[1]} | {'DIFFERENT: ' + ', '.join(flags) if flags else 'same'} |")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
