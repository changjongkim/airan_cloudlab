#!/usr/bin/env python3
"""Candidates per slot of a run without AI: the table of the clairvoyant baseline (controller5.py, oracle_candidates).

usage: oracle_table.py OUT_DIR TAG CELLS JOB SEED [SEED ...]      -> OUT_DIR/<TAG><seed>.json
Which slots of a two-user cell are candidates is fixed by the data set, the cell and the seed, so the table of a
run without AI holds for every run of the same condition and seed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from loss_trace import RAW, trace


def main() -> None:
    out_dir, tag, cells, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in sys.argv[5:]:
        matches = sorted(RAW.glob(f"{tag}{seed}n_c{cells}_*_j{job}.json"))
        if len(matches) != 1:
            raise SystemExit(f"{tag}{seed}n_c{cells} j{job}: {len(matches)} files")
        t = trace(matches[0].stem)
        counts = np.bincount(np.asarray(t["period"]), minlength=int(t["periods"]))[:int(t["periods"])]
        (out_dir / f"{tag}{seed}.json").write_text(json.dumps(counts.tolist()))
        print(f"{tag}{seed}: {int(counts.sum())} candidates in {len(counts)} slots, at most {int(counts.max())} in a slot")


if __name__ == "__main__":
    main()
