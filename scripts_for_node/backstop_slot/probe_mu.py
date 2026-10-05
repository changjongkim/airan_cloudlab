#!/usr/bin/env python3
"""Run-time receivers of the MU-MIMO profile on its whole slot pool, one GPU, nothing else running.

Reports per UE which receiver recovered the TB, how the run-time neural chain (TensorRT +
pyAerial LS, rate recovery, LDPC, CRC) compares with NVlabs' own TensorFlow evaluation of
the same slots, the rescue probability by failed code blocks, and both receivers' times.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import conv_path, nrx_path, result_masks


def spread(values) -> dict:
    values = np.asarray(values) * 1e3
    return {"p50": float(np.percentile(values, 50)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", default="nv_mu2")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--ldpc-iterations", type=int, default=10)
    parser.add_argument("--conv-iterations", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-slot", type=Path,
                        help="also save the per-slot outcomes (.npz: conv_good, nrx_good, cb_fail per UE, esno_db)")
    args = parser.parse_args()

    count = json.loads((args.dataset / f"{args.profile}_meta.json").read_text())["count"]
    ring = CellRing(args.dataset, args.profile, count, 0)
    num_ue = ring.profile.num_ue
    stream = cp.cuda.Stream(non_blocking=True)
    conv = conv_path(ring.profile, ring.tb_bytes, stream, args.conv_iterations)
    nrx = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": args.ldpc_iterations},
                   ring.profile, ring.tb_bytes, stream)

    conv_good = np.zeros((count, num_ue), dtype=bool)
    nrx_good = np.zeros((count, num_ue), dtype=bool)
    cb_fail = np.zeros((count, num_ue), dtype=int)
    conv_s, nrx_s = [], []
    for attempt in range(args.passes):
        for i in range(count):
            start = time.perf_counter()
            ok, payload = conv.run(ring.views[i], ring.slots[i])
            elapsed = time.perf_counter() - start
            if attempt:
                conv_s.append(elapsed)
            _, good = result_masks(conv, ok, payload, ring.blocks[i], ring.tb_bytes)
            conv_good[i] = [(good >> u) & 1 for u in range(num_ue)]
            cb_fail[i] = getattr(conv, "last_cb_fail_by_ue", [conv.last_cb_fail])
        for i in range(count):
            start = time.perf_counter()
            ok, payload = nrx.run(ring.views[i], ring.slots[i])
            elapsed = time.perf_counter() - start
            if attempt:
                nrx_s.append(elapsed)
            _, good = result_masks(nrx, ok, payload, ring.blocks[i], ring.tb_bytes)
            nrx_good[i] = [(good >> u) & 1 for u in range(num_ue)]

    report = {"dataset": str(args.dataset), "profile": args.profile, "slots": count, "ues": num_ue,
              "engine": args.engine, "ldpc_iterations": args.ldpc_iterations,
              "conv_iterations": args.conv_iterations,
              "tbs": int(conv_good.size), "conventional": int(conv_good.sum()), "neural": int(nrx_good.sum()),
              "both": int((conv_good & nrx_good).sum()), "conventional_only": int((conv_good & ~nrx_good).sum()),
              "neural_only": int((~conv_good & nrx_good).sum()), "neither": int((~conv_good & ~nrx_good).sum()),
              "conv_ms": spread(conv_s), "nrx_ms": spread(nrx_s)}
    reference = args.dataset / f"{args.profile}_nvlabs_tf_ok.npy"
    if reference.is_file():
        tf_ok = np.load(reference)[ring.indices]
        report["nvlabs_tensorflow"] = int(tf_ok.sum())
        report["runtime_vs_tensorflow"] = {"same": int((tf_ok == nrx_good).sum()),
                                           "runtime_only": int((nrx_good & ~tf_ok).sum()),
                                           "tensorflow_only": int((~nrx_good & tf_ok).sum())}
    # Rescue probability of a failed TB by its own failed code blocks, and of a slot by the
    # failed code blocks of all its UEs (the number the controller sees).
    by_tb = {}
    for k in sorted(set(cb_fail[~conv_good].tolist())):
        chosen = (~conv_good) & (cb_fail == k)
        by_tb[str(k)] = [int(nrx_good[chosen].sum()), int(chosen.sum())]
    by_slot = {}
    total = cb_fail.sum(axis=1)
    failed = (~conv_good).any(axis=1)
    for k in sorted(set(total[failed].tolist())):
        chosen = failed & (total == k)
        by_slot[str(k)] = {"slots": int(chosen.sum()), "failed_tbs": int((~conv_good[chosen]).sum()),
                           "rescued_tbs": int((~conv_good[chosen] & nrx_good[chosen]).sum())}
    report["rescue_by_failed_code_blocks_of_tb"] = by_tb
    report["rescue_by_failed_code_blocks_of_slot"] = by_slot
    by_snr = {}
    for e in sorted(set(ring.esno_db)):
        chosen = np.asarray(ring.esno_db) == e
        by_snr[str(e)] = {"tbs": int(conv_good[chosen].size), "conventional": int(conv_good[chosen].sum()),
                          "neural": int(nrx_good[chosen].sum()),
                          "neural_only": int((~conv_good[chosen] & nrx_good[chosen]).sum()),
                          "conventional_only": int((conv_good[chosen] & ~nrx_good[chosen]).sum())}
    report["by_ebno_db"] = by_snr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.per_slot:
        np.savez(args.per_slot, conv_good=conv_good, nrx_good=nrx_good, cb_fail=cb_fail,
                 esno_db=np.asarray(ring.esno_db), tb_bytes=ring.tb_bytes)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
