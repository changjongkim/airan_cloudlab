#!/usr/bin/env python3
"""Tables of candidates per slot with errors, for the value of the information that the clairvoyant controller uses.

usage: oracle_noise.py OUT_DIR TAG CELLS JOB SEED [SEED ...]      -> OUT_DIR/<TAG><seed>_<variant>.json
The true table is that of oracle_table.py (a run without AI of the same seed).  Variants:
  rNN   misses: each true candidate is kept with probability NN% (r00: the table is empty)
  fNN   false alarms: each (two-user cell, slot) without a candidate becomes one with probability NN%
  m     both: r75 and f10
The controller that reads a table with misses lets AI run next to neural receivers that are needed; with false
alarms it stops AI more often than it has to (f100 would be the rule of Antiphase).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from loss_trace import RAW, trace

VARIANTS = {"r90": (0.90, 0.0), "r75": (0.75, 0.0), "r50": (0.50, 0.0), "r25": (0.25, 0.0), "r00": (0.0, 0.0),
            "f05": (1.0, 0.05), "f10": (1.0, 0.10), "f25": (1.0, 0.25), "f50": (1.0, 0.50), "m": (0.75, 0.10)}


def main() -> None:
    out_dir, tag, cells, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in sys.argv[5:]:
        matches = sorted(RAW.glob(f"{tag}{seed}n_c{cells}_*_j{job}.json"))
        if len(matches) != 1:
            raise SystemExit(f"{tag}{seed}n_c{cells} j{job}: {len(matches)} files")
        t = trace(matches[0].stem)
        periods, weak = int(t["periods"]), int(t["weak_cells"])
        truth = np.zeros((weak, periods), dtype=bool)
        truth[np.asarray(t["cell"]), np.asarray(t["period"])] = True
        for i, (name, (keep, alarm)) in enumerate(VARIANTS.items()):
            rng = np.random.default_rng(1000 * int(seed) + i)
            shown = (truth & (rng.random(truth.shape) < keep)) | (~truth & (rng.random(truth.shape) < alarm))
            (out_dir / f"{tag}{seed}_{name}.json").write_text(json.dumps(shown.sum(axis=0).tolist()))
            hit = (shown & truth).sum() / max(1, truth.sum())
            precision = (shown & truth).sum() / max(1, shown.sum())
            print(f"{tag}{seed} {name}: candidates shown {int(shown.sum())} of {int(truth.sum())} true; "
                  f"recall {100 * hit:.0f}%, precision {100 * precision:.0f}%")


if __name__ == "__main__":
    main()
