#!/usr/bin/env python3
"""Does the conventional receiver's equalizer choice change the MU-MIMO comparison?

Decodes one slot pool with the cells' cuPHY pipeline under each equalizer (1: MMSE, the
setting used everywhere else; 2: MMSE-IRC; 0: ZF), LDPC iterations fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

import mu_radio
from cell_ring import CellRing
from slot_radio import conv_path, result_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", default="nv_mu2")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--algo", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = json.loads((args.dataset / f"{args.profile}_meta.json").read_text())["count"]
    ring = CellRing(args.dataset, args.profile, count, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    original = mu_radio.PersistentPuschRx
    report = {"dataset": str(args.dataset), "tbs": count * ring.profile.num_ue, "ldpc_iterations": args.iterations}
    names = {1: "mmse", 2: "mmse_irc", 0: "zf"}
    for algo, name in ((args.algo, names[args.algo]),):
        def build(**kwargs):
            kwargs["eq_coeff_algo"] = algo
            return original(**kwargs)
        mu_radio.PersistentPuschRx = build
        try:
            path = conv_path(ring.profile, ring.tb_bytes, stream, args.iterations)
        finally:
            mu_radio.PersistentPuschRx = original
        good = 0
        for i in range(count):
            ok, payload = path.run(ring.views[i], ring.slots[i])
            _, mask = result_masks(path, ok, payload, ring.blocks[i], ring.tb_bytes)
            good += bin(mask).count("1")
        report[name] = good
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
