#!/usr/bin/env python3
"""One cell's conventional cuPHY PUSCH receiver, released every UL period.

With a ``la`` section in the run configuration, a two-user cell keeps one slot ring per MCS
level and an outer-loop link adaptation picks the level of every slot: a pointer over the
levels goes up by ``down * target / (1 - target)`` for every TB that needs no retransmission
(decoded here, or recovered by the neural receiver before the recovery deadline) and down by
``down`` for every other TB, ``delay`` periods after the slot.
"""

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
from slot_state import C_LEVEL, C_NRX_DONE, C_NRX_MASK, C_NRX_STATUS, FAIL, PASS, SlotState, now_ns, spin_until


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
    stream = cp.cuda.Stream(non_blocking=True)
    run_config = json.loads(args.config.read_text()) if args.config else {}
    la = run_config.get("la")
    if la and args.profile not in la["levels"]:
        la = None
    names = la["levels"] if la else [args.profile]              # lowest MCS first
    rings = [CellRing(args.dataset, name, args.ring, args.offset) for name in names]
    ring = rings[-1]
    conv = conv_path(ring.profile, ring.tb_bytes, stream, int(run_config.get("conv_ldpc_iterations", 0)))
    for other in rings[:-1]:
        conv.add_level(other.profile, other.tb_bytes)
    every_ue = (1 << ring.profile.num_ue) - 1
    if args.ipc_file:
        temporary = args.ipc_file.with_suffix(".tmp")
        record = {"levels": [r.ipc_record() for r in rings]} if la else ring.ipc_record()
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(args.ipc_file)

    warm_ok = 0
    for index in range(args.warmup):                            # the highest MCS first: it sizes the HARQ buffers
        ok, _ = conv.run(ring.views[index % ring.ring], ring.slots[index % ring.ring])
        warm_ok += int(ok)
    for other in (rings[:-1] if la else []):
        conv.use(other.profile.name)
        for index in range(8):
            conv.run(other.views[index], other.slots[index])
    if la:
        top = len(names) - 1
        down = float(la.get("down", 0.25))
        up = down * float(la["target"]) / (1.0 - float(la["target"]))
        delay = int(la.get("delay", 5))
        pointer = float(la.get("start", top / 2.0))
        rescue_ns = int(float(run_config.get("rescue_deadline_ms", run_config["deadline_ms"])) * 1e6)
        sent: dict[int, tuple[int, int]] = {}                   # period -> (release, UEs decoded here)

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
        level = 0
        if la:
            past = sent.pop(period - delay, None)
            if past is not None:                                # outcome of the slot sent `delay` periods ago
                row = state.cells[args.cell, period - delay]
                recovered = 0
                if int(row[C_NRX_STATUS]) in (PASS, FAIL) and int(row[C_NRX_DONE]) - past[0] <= rescue_ns:
                    recovered = int(row[C_NRX_MASK])
                final = past[1] | recovered
                for ue in range(ring.profile.num_ue):
                    pointer += up if (final >> ue) & 1 else -down
                pointer = min(top + 0.999, max(0.0, pointer))
            level = int(pointer)
            ring = rings[level]
            conv.use(names[level])
            state.cells[args.cell, period, C_LEVEL] = level
        spin_until(release)
        index = period % ring.ring
        start = now_ns()
        ok, payload = conv.run(ring.views[index], ring.slots[index])
        done = now_ns()
        state.set_conv(args.cell, period, start, done, ok, conv.last_cb_fail)
        crc_mask, good = result_masks(conv, ok, payload, ring.blocks[index], ring.tb_bytes)
        if la:
            sent[period] = (release, crc_mask)
        records.append([period, index, release, start, done, int(ok), int(good == every_ue),
                        conv.last_cb_fail, crc_mask, good, level])
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
                    "crc_pass", "payload_ok", "cb_fail", "ue_crc_mask", "ue_good_mask", "mcs_level"],
        "records": records,
    }
    if la:
        result["la"] = {"levels": names, "tb_bytes": [r.tb_bytes for r in rings], "pointer_end": pointer}
    args.output.write_text(json.dumps(result), encoding="utf-8")
    for r in rings:
        r.close()


if __name__ == "__main__":
    main()
