#!/usr/bin/env python3
"""Where the MU-MIMO neural path spends its time, stage by stage (each stage synchronized),
and whether the LS estimate can be replaced by one multiply (LS = pilot RE x fixed factor)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import nrx_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    ring = CellRing(args.dataset, "nv_mu2", args.count, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": args.iterations}, ring.profile, ring.tb_bytes, stream)
    stages = {k: [] for k in ("ls", "inputs", "engine", "take", "derate", "ldpc", "crc", "host", "total")}

    def tick(name, start):
        stream.synchronize()
        now = time.perf_counter()
        stages[name].append((now - start) * 1e3)
        return now

    factor = None
    worst = 0.0
    for attempt in range(2):
        for i in range(args.count):
            rx_slot, slot = ring.views[i], ring.slots[i]
            begin = t = time.perf_counter()
            channel = nrx.estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=nrx.configs)
            t = tick("ls", t) if attempt else time.perf_counter()
            with stream:
                est = cp.asarray(channel[0])                      # (pilot, UE, antenna, DMRS symbol)
                estimate = cp.transpose(est, (3, 0, 1, 2)) * cp.float32(0.70710678)
                estimate = estimate.reshape(-1, estimate.shape[2], estimate.shape[3])[None, ...]
                window = rx_slot[None, ...]
                inputs = nrx.engine.inputs
                cp.copyto(inputs["rx_slot_real"], window.real)
                cp.copyto(inputs["rx_slot_imag"], window.imag)
                cp.copyto(inputs["h_hat_real"], estimate.real)
                cp.copyto(inputs["h_hat_imag"], estimate.imag)
            t = tick("inputs", t) if attempt else time.perf_counter()
            with stream:
                outputs = nrx.engine.launch(use_graph=True)
            t = tick("engine", t) if attempt else time.perf_counter()
            with stream:
                llrs = cp.take(outputs["output_1"][0, ...], nrx.data_symbols, axis=3)
                split = [cp.ascontiguousarray(llrs[:, u:u + 1]) for u in range(nrx.num_ue)]
            t = tick("take", t) if attempt else time.perf_counter()
            with stream:
                coded = nrx.derate.derate_match(input_llrs=split, pusch_configs=nrx.per_ue)
            t = tick("derate", t) if attempt else time.perf_counter()
            with stream:
                blocks = nrx.decoder.decode(input_llrs=coded, pusch_configs=nrx.per_ue)
            t = tick("ldpc", t) if attempt else time.perf_counter()
            with stream:
                tbs, crcs = nrx.crc.check_crc(input_bits=blocks, pusch_configs=nrx.per_ue)
            t = tick("crc", t) if attempt else time.perf_counter()
            payload = [cp.asnumpy(tbs[u]) if isinstance(tbs[u], cp.ndarray) else np.asarray(tbs[u]) for u in range(nrx.num_ue)]
            _ = [cp.asnumpy(crcs[u]) if isinstance(crcs[u], cp.ndarray) else np.asarray(crcs[u]) for u in range(nrx.num_ue)]
            t = tick("host", t) if attempt else time.perf_counter()
            if attempt:
                stages["total"].append((t - begin) * 1e3)
            # LS as one multiply: pilots of UE u sit on its own comb of the DMRS symbols.
            if attempt == 0:
                sub = np.asarray(nrx.engine.meta["dmrs_subcarrier_pos"])
                syms = list(ring.profile.dmrs_positions)
                picked = cp.stack([cp.stack([rx_slot[cp.asarray(np.concatenate([12 * p + sub[u] for p in range(273)]))][:, s, :]
                                             for s in syms], axis=-1) for u in range(nrx.num_ue)], axis=1)
                picked = cp.asarray(picked)                         # (pilot, UE, antenna, DMRS symbol)
                if factor is None:
                    ratio = est / picked
                    factor = ratio[:, :, :1, :].copy()              # same for every antenna if LS is per RE
                    spread = float(cp.abs(ratio - factor).max())
                    print("factor spread across antennas", spread, "abs", float(cp.abs(factor).min()), float(cp.abs(factor).max()))
                else:
                    err = float(cp.linalg.norm(picked * factor - est) / cp.linalg.norm(est))
                    worst = max(worst, err)
    report = {name: {"p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99))} for name, v in stages.items()}
    report["fast_ls_worst_relative_error"] = worst
    print(json.dumps(report))


if __name__ == "__main__":
    main()
