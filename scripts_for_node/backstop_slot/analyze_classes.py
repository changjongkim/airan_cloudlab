#!/usr/bin/env python3
"""Step 5 table: per policy and mix, radio outcome and per-class AI (mean over seeds)."""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_runs import run_row  # noqa: E402

ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot")
JOB = sys.argv[1]
RUNS = sys.argv[2] if len(sys.argv) > 2 else "w"      # tag letter of the AI runs (no-AI references: w)
LABEL = {"u": "Antiphase (urgent class first)", "f": "Antiphase (one line, arrival order)",
         "s50": "fixed 50% share", "s70": "fixed 70% share"}


def main() -> None:
    refs, groups = {}, {}
    paths = sorted(glob.glob(str(ROOT / "raw" / f"{RUNS}[0-9][nm]_*_j{JOB}.json"))
                   + glob.glob(str(ROOT / "raw" / f"{RUNS}[0-9]r*_j{JOB}.json")))
    for path in paths:
        m = re.match(r"[a-z](\d)(n|m|r(\d+)(cs|csb)(u|f|s\d+))_c\d+_", Path(path).name)
        if not m:
            continue
        row = run_row(path)
        if m.group(2) in ("n", "m"):
            refs.setdefault(int(m.group(1)), []).append(row["rescues"])
        else:
            groups.setdefault((int(m.group(3)), m.group(4), m.group(5)), []).append((int(m.group(1)), row))
    ref = {s: float(np.mean(v)) for s, v in refs.items()}
    out = []
    print("| rate per class | classes | policy | rescues (vs no AI) | L1 late | chat in SLO | small in SLO | interactive tokens in SLO /s | batch tokens/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (rate, mix, pol), items in sorted(groups.items()):
        ratio = np.mean([r["rescues"] / ref[s] for s, r in items if s in ref])
        late = np.mean([r["l1_late_pct"] for _, r in items])

        def cls(name, field):
            return float(np.mean([r["classes"].get(name, {}).get(field, 0) for _, r in items]))

        def share(name):
            a = sum(r["classes"].get(name, {}).get("arrived", 0) for _, r in items)
            w = sum(r["classes"].get(name, {}).get("within_slo", 0) for _, r in items)
            return 100 * w / max(1, a)

        rec = {"rate": rate, "mix": mix, "policy": LABEL[pol], "rescue_ratio": float(ratio), "l1_late_pct": float(late),
               "chat_in_slo_pct": share("chat"), "small_in_slo_pct": share("small"),
               "interactive_slo_tokens": cls("chat", "slo_tokens") + cls("small", "slo_tokens"),
               "batch_tokens": cls("batch", "tokens"), "seeds": len(items)}
        out.append(rec)
        print(f"| {rate} | {'chat+small+batch' if mix == 'csb' else 'chat+small'} | {rec['policy']} | {100*ratio:.1f}% | "
              f"{late:.3f}% | {rec['chat_in_slo_pct']:.0f}% | {rec['small_in_slo_pct']:.0f}% | "
              f"{rec['interactive_slo_tokens']:.0f} | {rec['batch_tokens']:.0f} |")
    (ROOT / f"v3_classes_{RUNS}_j{JOB}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
