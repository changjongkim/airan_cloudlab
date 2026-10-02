#!/usr/bin/env python3
"""NeuralRx for one weak cell on a partner GPU.

The received slot stays in the conventional worker's GPU memory and is copied
peer-to-peer when NeuralRx starts.  Start rules:

* ``parallel``: start at data arrival, alongside the conventional receiver.
* ``rescue``: start at the earlier of the conventional CRC failure and the
  NeuralRx latest start (deadline minus the NeuralRx bound).  If the
  conventional receiver passes first, NeuralRx is skipped for that TB.
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
from slot_radio import NrxPath, fortran_slot_views
from slot_state import FAIL, PASS, SKIPPED, SlotState, now_ns, spin_until
from ul_profiles import PROFILES

MEMCPY_DEFAULT = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--ipc-file", type=Path, required=True)
    parser.add_argument("--policy", choices=("parallel", "rescue"), required=True)
    parser.add_argument("--deadline-ns", type=int, required=True)
    parser.add_argument("--nrx-bound-ns", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--periods", type=int, required=True)
    parser.add_argument("--period-ns", type=int, required=True)
    parser.add_argument("--ready-slot", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cp.cuda.Device(args.gpu).use()
    state = SlotState(args.state, args.num_cells, args.periods, args.num_gpus)
    deadline = time.monotonic() + 300
    while not args.ipc_file.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no IPC record at {args.ipc_file}")
        time.sleep(0.01)
    record = json.loads(args.ipc_file.read_text(encoding="utf-8"))
    remote = cp.cuda.runtime.ipcOpenMemHandle(
        bytes.fromhex(record["handle"]), cp.cuda.runtime.cudaIpcMemLazyEnablePeerAccess
    )
    ring = int(record["ring"])
    meta = json.loads((args.dataset / f"{args.profile}_meta.json").read_text())
    blocks = np.load(args.dataset / f"{args.profile}_tb.npy")
    slots = [int(meta["slots"][i]) for i in record["indices"]]
    references = [blocks[i] for i in record["indices"]]
    tb_bytes = int(meta["tb_bytes"])

    local_ptr = cp.cuda.runtime.malloc(SLOT_BYTES)
    local = fortran_slot_views(local_ptr, record, 1)[0]
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = NrxPath(args.engine, PROFILES[args.profile], tb_bytes, stream)

    def fetch(index: int) -> None:
        cp.cuda.runtime.memcpyAsync(
            local_ptr, remote + index * SLOT_BYTES, SLOT_BYTES, MEMCPY_DEFAULT, stream.ptr
        )
        # pyAerial's channel estimator reads the slot outside this stream's
        # order, so the copy must be complete before the chain starts.
        stream.synchronize()

    warm_ok = 0
    for index in range(args.warmup):
        fetch(index % ring)
        ok, _ = nrx.run(local, slots[index % ring])
        warm_ok += int(ok)

    gc.collect()
    gc.disable()
    state.mark_ready(args.ready_slot)
    epoch = state.wait_epoch()
    slack = args.deadline_ns - args.nrx_bound_ns
    records = []
    for period in range(args.periods):
        release = epoch + period * args.period_ns
        latest = release + slack
        spin_until(release)
        reason = "arrival"
        if args.policy == "rescue":
            while True:
                status = state.conv_status(args.cell, period)
                if status == PASS:
                    reason = "skipped"
                    break
                if status == FAIL:
                    reason = "conv_fail"
                    break
                if now_ns() >= latest:
                    reason = "latest_start"
                    break
        index = period % ring
        if reason == "skipped":
            done = now_ns()
            state.set_nrx(args.cell, period, SKIPPED, done)
            records.append([period, index, release, 0, done, -1, -1, reason])
            continue
        start = now_ns()
        state.set_nrx_running(args.cell, period, start)
        fetch(index)
        ok, payload = nrx.run(local, slots[index])
        done = now_ns()
        state.set_nrx(args.cell, period, PASS if ok else FAIL, done)
        payload_ok = bool(ok and np.array_equal(payload[:tb_bytes], references[index]))
        records.append([period, index, release, start, done, int(ok), int(payload_ok), reason])
    gc.enable()

    result = {
        "schema": "backstop-slot-nrx-worker-v1",
        "cell": args.cell,
        "gpu": args.gpu,
        "profile": args.profile,
        "policy": args.policy,
        "pid": os.getpid(),
        "deadline_ns": args.deadline_ns,
        "nrx_bound_ns": args.nrx_bound_ns,
        "warmup": args.warmup,
        "warmup_crc_pass": warm_ok,
        "columns": ["period", "ring_index", "release_ns", "start_ns", "done_ns",
                    "crc_pass", "payload_ok", "start_reason"],
        "records": records,
    }
    args.output.write_text(json.dumps(result), encoding="utf-8")
    cp.cuda.runtime.ipcCloseMemHandle(remote)
    cp.cuda.runtime.free(local_ptr)


if __name__ == "__main__":
    main()
