#!/usr/bin/env python3
"""Recompute a run summary from its saved worker records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from summarize_slot import summarize

for name in sys.argv[1:]:
    path = Path(name)
    old = json.loads(path.read_text())
    work = path.parent / (path.stem + "_work")
    new = summarize(old["config"], work, 0)
    for key in ("host", "slurm_job_id", "config"):
        new[key] = old[key]
    path.write_text(json.dumps(new, indent=2))
    print(path.name, json.dumps({k: new["headline"][k] for k in (
        "tbs", "ul_indication_on_time", "late_tbs", "decoded_on_time",
        "nrx_rescues_on_time", "nrx_runs")}))
