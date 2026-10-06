#!/usr/bin/env python3
"""One NeuralRx lane per GPU; the controller assigns it one TB at a time.

The lane opens every weak cell's slot ring over CUDA IPC, so it can decode a
TB of any cell: the received slot is copied peer-to-peer from the GPU that
holds it, then the NeuralRx chain runs on this GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import SLOT_BYTES
from slot_radio import fortran_slot_views, nrx_path, result_masks
from slot_state import (
    C_LEVEL, C_NRX_LANE, C_NRX_MASK, FAIL, H_ABORT, L_ASSIGN_SEQ, L_CELL, L_DONE_SEQ, L_PERIOD, PASS,
    SlotState, now_ns,
)
from ul_profiles import PROFILES

MEMCPY_DEFAULT = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    dataset = Path(config["dataset"])
    cp.cuda.Device(args.gpu).use()
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    lane = state.lanes[args.lane]

    weak = [c for c in config["cells"] if PROFILES[c["profile"]].neural_eligible]
    la = config.get("la")                       # link adaptation: one ring per MCS level and cell
    names = la["levels"] if la else [weak[0]["profile"]]
    profile = PROFILES[names[-1]]
    # one ring per (channel state, MCS level), state-major, as the cell workers export them (la_states.py)
    datasets = [Path(d) for d in la.get("datasets", [])] if la else []
    datasets = datasets or [dataset]
    metas = [json.loads((d / f"{name}_meta.json").read_text()) for d in datasets for name in names]
    blocks = [np.load(d / f"{name}_tb.npy") for d in datasets for name in names]
    tb_sizes = [int(meta["tb_bytes"]) for meta in metas]
    tb_bytes = tb_sizes[-1]
    rings = {}
    for cell in weak:
        ipc_file = args.work / f"cell{cell['cell']}.ipc"
        deadline = time.monotonic() + 600
        while not ipc_file.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(f"no IPC record at {ipc_file}")
            time.sleep(0.01)
        record = json.loads(ipc_file.read_text())
        levels = []
        for level, entry in enumerate(record["levels"] if la else [record]):
            pointer = cp.cuda.runtime.ipcOpenMemHandle(
                bytes.fromhex(entry["handle"]), cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
            )
            levels.append({
                "ptr": pointer, "ring": int(entry["ring"]),
                "slots": [int(metas[level]["slots"][i]) for i in entry["indices"]],
                "refs": [blocks[level][i] for i in entry["indices"]],
            })
        rings[int(cell["cell"])] = levels

    local_ptr = cp.cuda.runtime.malloc(SLOT_BYTES)
    local = fortran_slot_views(local_ptr, rings, 1)[0]
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = nrx_path(config, profile, tb_bytes, stream)
    for name, size in (list(zip(names, tb_sizes))[:-1] if la else []):
        nrx.add_level(PROFILES[name], size)
    every_ue = (1 << profile.num_ue) - 1

    def fetch(cell: int, period: int, level: int = -1) -> tuple[dict, int]:
        ring = rings[cell][level]
        index = period % ring["ring"]
        cp.cuda.runtime.memcpyAsync(
            local_ptr, ring["ptr"] + index * SLOT_BYTES, SLOT_BYTES, MEMCPY_DEFAULT, stream.ptr
        )
        # pyAerial's channel estimator reads the slot outside this stream's
        # order, so the copy must be complete before the chain starts.
        stream.synchronize()
        return ring, index

    for step in range(args.warmup):
        cell = weak[step % len(weak)]["cell"]
        ring, index = fetch(cell, step)
        nrx.run(local, ring["slots"][index])
    for level in (range(len(names) - 1) if la else []):
        nrx.use(names[level])
        for step in range(4):
            ring, index = fetch(weak[step % len(weak)]["cell"], step, level)
            nrx.run(local, ring["slots"][index])

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    state.wait_epoch()
    records = []
    diag = []
    done_seq = 0
    while True:
        seq = int(lane[L_ASSIGN_SEQ])
        if seq == done_seq:
            if state.header[H_ABORT]:
                break
            continue
        cell = int(lane[L_CELL])
        period = int(lane[L_PERIOD])
        if cell < 0:            # controller's stop signal
            break
        start = now_ns()
        state.cells[cell, period, C_NRX_LANE] = args.gpu
        state.set_nrx_running(cell, period, start)
        level = int(state.cells[cell, period, C_LEVEL]) if la else 0
        if la:
            nrx.use(names[level % len(names)])
        ring, index = fetch(cell, period, level)
        ok, payload = nrx.run(local, ring["slots"][index])
        done = now_ns()
        state.cells[cell, period, C_NRX_MASK] = getattr(nrx, "last_crc_mask", int(ok))
        state.set_nrx(cell, period, PASS if ok else FAIL, done)
        crc_mask, good = result_masks(nrx, ok, payload, ring["refs"][index], tb_sizes[level])
        payload_ok = good == every_ue
        records.append([cell, period, start, done, int(ok), int(payload_ok), crc_mask, good, level])
        if ok and not payload_ok and profile.num_ue == 1:
            # Diagnostic: which TB did we actually decode?
            match = None
            for other_cell, other_levels in rings.items():
                for j, ref in enumerate(other_levels[-1]["refs"]):
                    if np.array_equal(payload[:tb_bytes], ref):
                        match = (other_cell, j)
                        break
                if match:
                    break
            diag.append({"cell": cell, "period": period, "index": index, "match": match,
                         "differing_bytes": int(np.count_nonzero(payload[:tb_bytes] != ring["refs"][index]))})
        done_seq = seq
        lane[L_DONE_SEQ] = seq
    gc.enable()
    args.output.write_text(json.dumps({
        "schema": "backstop-slot-nrx-lane-v1",
        "gpu": args.gpu,
        "lane": args.lane,
        "pid": os.getpid(),
        "columns": ["cell", "period", "start_ns", "done_ns", "crc_pass", "payload_ok",
                    "ue_crc_mask", "ue_good_mask", "mcs_level"],
        "records": records,
        "false_pass_diagnostics": diag,
    }), encoding="utf-8")
    for levels in rings.values():
        for ring in levels:
            cp.cuda.runtime.ipcCloseMemHandle(ring["ptr"])
    cp.cuda.runtime.free(local_ptr)


if __name__ == "__main__":
    main()
