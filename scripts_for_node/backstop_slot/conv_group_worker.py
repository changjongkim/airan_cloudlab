#!/usr/bin/env python3
"""Conventional receivers of several cells in one process (one cuPHY cell-group call).

Each period the worker decodes the cells of its group that carry a TB (see ``activity.py``)
with a single pipeline call; every decoded cell gets the call's start and completion time.
It writes the same per-cell records and state words as ``conv_worker.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import cupy as cp
import numpy as np

from activity import activity_mask
from cell_ring import CellRing
from cellgroup_radio import CellGroupRx
from slot_radio import pusch_configs
from slot_state import SlotState, now_ns, spin_until


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cells", required=True, help="comma-separated cell ids of this group")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    ids = [int(c) for c in args.cells.split(",")]
    by_id = {int(c["cell"]): c for c in config["cells"]}
    cells = [by_id[i] for i in ids]
    cp.cuda.Device(args.gpu).use()
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    size = int(config.get("ring", 128))
    rings = [CellRing(Path(config["dataset"]), c["profile"], size, int(c.get("offset", 97 * int(c["cell"]))))
             for c in cells]
    configs = [pusch_configs(r.profile, r.tb_bytes)[0] for r in rings]
    stream = cp.cuda.Stream(non_blocking=True)
    rx = CellGroupRx(len(cells), stream)
    use_nrx = config["nrx_policy"] != "off"
    for cell, ring in zip(cells, rings):
        if use_nrx and cell.get("nrx_gpu") is not None:
            ipc_file = args.work / f"cell{cell['cell']}.ipc"
            temporary = ipc_file.with_suffix(".tmp")
            temporary.write_text(json.dumps(ring.ipc_record()), encoding="utf-8")
            temporary.replace(ipc_file)
    mask = activity_mask(config)[ids]            # (group cells, periods)

    def decode(index: int, members: list[int]):
        return rx.run([rings[m].views[index] for m in members], [rings[m].slots[index] for m in members],
                      [configs[m] for m in members], members)

    everyone = list(range(len(cells)))
    warm_ok = 0
    for step in range(args.warmup):
        # Warm every subset size so no period pays a first-use cost.
        members = everyone[:1 + step % len(cells)] if step % 2 else everyone
        ok, _ = decode(step % size, members)
        warm_ok += sum(ok)

    records = {cell: [] for cell in ids}
    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    for period in range(args.periods):
        members = [m for m in everyone if mask[m, period]]
        if not members:
            continue
        release = epoch + period * args.period_ns
        spin_until(release)
        index = period % size
        start = now_ns()
        ok, cb_fail = decode(index, members)
        done = now_ns()
        for i, m in enumerate(members):
            state.set_conv(ids[m], period, start, done, ok[i], cb_fail[i])
        for i, m in enumerate(members):
            ring = rings[m]
            payload_ok = bool(ok[i] and np.array_equal(rx.payload(i, ring.tb_bytes), ring.blocks[index]))
            records[ids[m]].append([period, index, release, start, done, int(ok[i]), int(payload_ok), cb_fail[i]])
    gc.enable()

    for m, cell in enumerate(cells):
        ring = rings[m]
        (args.work / f"conv{cell['cell']}.json").write_text(json.dumps({
            "schema": "backstop-slot-conv-worker-v1",
            "cell": int(cell["cell"]), "gpu": args.gpu, "profile": cell["profile"], "pid": os.getpid(),
            "runtime": "cell_group", "group_cells": ids,
            "warmup": args.warmup, "warmup_crc_pass": warm_ok,
            "dataset_indices": ring.indices, "esno_db": ring.esno_db,
            "columns": ["period", "ring_index", "release_ns", "start_ns", "done_ns",
                        "crc_pass", "payload_ok", "cb_fail"],
            "records": records[int(cell["cell"])],
        }), encoding="utf-8")
    for ring in rings:
        ring.close()


if __name__ == "__main__":
    main()
