#!/usr/bin/env python3
"""Does the assumed success of a retransmission change the comparison of the policies under rate control?

usage: analyze_retx_assumption.py OUT.json "LABEL=LA_CLOSED.json" ...      (outputs of analyze_la_closed.py)
The goodput of analyze_la_closed.py counts a retransmission as one uplink slot of the user that succeeds with
probability 0.95.  Here it is recomputed per seed with other values of that probability and with no gain from
combining (a retransmission succeeds like a first transmission of the run, 1 - its share of TBs that need one).
Printed: goodput of every policy against the run without AI of the same seed, for each assumption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ASSUMED = 0.95
CASES = (0.80, 0.90, 0.95, 1.00, None)
NAMES = {"wm": "Antiphase", "s10": "fixed 10%", "p30": "low priority + 30%", "p70": "low priority + 70%",
         "p100": "low priority, no cap"}


def main() -> None:
    out_path, out = Path(sys.argv[1]), {}
    print("| condition | policy | TBs that need a retransmission | goodput against no AI, success 0.80 | 0.90 | 0.95 | 1.00 | no combining |")
    print("|---|---|---|---|---|---|---|---|")
    for item in sys.argv[2:]:
        label, path = item.rsplit("=", 1)
        pol = json.loads(Path(path).read_text())["policies"]
        ref = {s: (g, r / 100.0) for s, g, r in zip(pol["n"]["seeds"], pol["n"]["goodput_by_seed"], pol["n"]["retx_by_seed"])}
        out[label] = {}
        for code in NAMES:
            if code not in pol:
                continue
            row = []
            for ok in CASES:
                vals = []
                for s, g, r in zip(pol[code]["seeds"], pol[code]["goodput_by_seed"], pol[code]["retx_by_seed"]):
                    if s not in ref:
                        continue
                    both = []
                    for goodput, retx in ((g, r / 100.0), ref[s]):
                        per_tb = goodput * (1.0 + retx / ASSUMED)             # bits per TB before the retransmission slots
                        both.append(per_tb / (1.0 + retx / ((1.0 - retx) if ok is None else ok)))
                    vals.append(100.0 * (both[0] / both[1] - 1.0))
                row.append(float(np.mean(vals)))
            out[label][code] = {"retx_pct": pol[code]["retx_pct"], "vs_no_ai_pct": row}
            print(f"| {label} | {NAMES[code]} | {pol[code]['retx_pct']:.2f}% | " + " | ".join(f"{v:+.2f}%" for v in row) + " |")
    out_path.write_text(json.dumps({"cases": ["0.80", "0.90", "0.95", "1.00", "no combining"], "conditions": out}, indent=1))


if __name__ == "__main__":
    main()
