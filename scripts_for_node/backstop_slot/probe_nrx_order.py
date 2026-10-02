#!/usr/bin/env python3
"""Side check: does the pyAerial example NeuralRx decode more TBs when its LS channel
estimate input is ordered and scaled the way the NVlabs code produces it?

Variants: pilot order (per pilot across DMRS symbols, as in the pyAerial example, or per DMRS
symbol across pilots, as in NVlabs' Sionna-based export) and scale (1 or 1/sqrt(2)).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import ConvPath, NrxPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ring = CellRing(args.dataset, "weak_rank1", args.count, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    conv = ConvPath(ring.profile, ring.tb_bytes, stream)
    nrx = NrxPath(args.engine, ring.profile, ring.tb_bytes, stream)
    start = ring.profile.start_sym
    conv_ok = [conv.run(ring.views[i], ring.slots[i])[0] for i in range(args.count)]
    results = {"conventional": int(sum(conv_ok))}
    for order in ("pilot_major", "symbol_major"):
        for scale in (1.0, 2 ** -0.5):
            passes = only = 0
            for i in range(args.count):
                rx_slot = ring.views[i]
                channel = nrx.estimator.estimate(rx_slot=rx_slot, slot=ring.slots[i], pusch_configs=nrx.configs)
                with stream:
                    est = cp.asarray(channel[0])
                    perm = (0, 3, 1, 2) if order == "pilot_major" else (3, 0, 1, 2)
                    est = cp.transpose(est, perm).reshape(est.shape[0] * est.shape[3], est.shape[1], est.shape[2])[None]
                    est = est * scale
                    window = rx_slot[None, :, start:start + 12, :]
                    cp.copyto(nrx.engine.inputs["rx_slot_real"], window.real)
                    cp.copyto(nrx.engine.inputs["rx_slot_imag"], window.imag)
                    cp.copyto(nrx.engine.inputs["h_hat_real"], est.real.astype(cp.float32))
                    cp.copyto(nrx.engine.inputs["h_hat_imag"], est.imag.astype(cp.float32))
                    outputs = nrx.engine.launch(use_graph=True)
                    llrs = cp.take(outputs["output_1"][0, ...], nrx.data_symbols, axis=3)
                    coded = nrx.derate.derate_match(input_llrs=[llrs], pusch_configs=nrx.configs)
                    blocks = nrx.decoder.decode(input_llrs=coded, pusch_configs=nrx.configs)
                    tbs, crcs = nrx.crc.check_crc(input_bits=blocks, pusch_configs=nrx.configs)
                stream.synchronize()
                ok = int(cp.asnumpy(crcs[0]).reshape(-1)[0]) == 0 and np.array_equal(
                    cp.asnumpy(tbs[0])[:ring.tb_bytes], ring.blocks[i])
                passes += ok
                only += ok and not conv_ok[i]
            results[f"{order}_scale{scale:.3f}"] = {"nrx_pass": int(passes), "nrx_only": int(only)}
            print(order, scale, passes, only, flush=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
