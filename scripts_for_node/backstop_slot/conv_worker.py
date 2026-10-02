#!/usr/bin/env python3
"""One cell's conventional cuPHY PUSCH receiver, released every UL period."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import conv_path, result_masks
from slot_state import SlotState, now_ns, spin_until


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ring", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--ipc-file", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cp.cuda.Device(args.gpu).use()
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    ring = CellRing(args.dataset, args.profile, args.ring, args.offset)
    stream = cp.cuda.Stream(non_blocking=True)
    run_config = json.loads(args.config.read_text()) if args.config else {}
    conv = conv_path(ring.profile, ring.tb_bytes, stream, int(run_config.get("conv_ldpc_iterations", 0)))
    every_ue = (1 << ring.profile.num_ue) - 1
    if args.ipc_file:
        temporary = args.ipc_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(ring.ipc_record()), encoding="utf-8")
        temporary.replace(args.ipc_file)

    warm_ok = 0
    for index in range(args.warmup):
        ok, _ = conv.run(ring.views[index % ring.ring], ring.slots[index % ring.ring])
        warm_ok += int(ok)

    active = None
    if args.config:
        from activity import activity_mask
        active = activity_mask(run_config)[args.cell]
    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    records = []
    for period in range(args.periods):
        if active is not None and not active[period]:
            continue
        release = epoch + period * args.period_ns
        spin_until(release)
        index = period % ring.ring
        start = now_ns()
        ok, payload = conv.run(ring.views[index], ring.slots[index])
        done = now_ns()
        state.set_conv(args.cell, period, start, done, ok, conv.last_cb_fail)
        crc_mask, good = result_masks(conv, ok, payload, ring.blocks[index], ring.tb_bytes)
        records.append([period, index, release, start, done, int(ok), int(good == every_ue),
                        conv.last_cb_fail, crc_mask, good])
    gc.enable()

    result = {
        "schema": "backstop-slot-conv-worker-v1",
        "cell": args.cell,
        "gpu": args.gpu,
        "profile": args.profile,
        "pid": os.getpid(),
        "warmup": args.warmup,
        "warmup_crc_pass": warm_ok,
        "dataset_indices": ring.indices,
        "esno_db": ring.esno_db,
        "columns": ["period", "ring_index", "release_ns", "start_ns", "done_ns",
                    "crc_pass", "payload_ok", "cb_fail", "ue_crc_mask", "ue_good_mask"],
        "records": records,
    }
    args.output.write_text(json.dumps(result), encoding="utf-8")
    ring.close()


if __name__ == "__main__":
    main()
