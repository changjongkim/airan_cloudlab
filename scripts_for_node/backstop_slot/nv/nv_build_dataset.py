#!/usr/bin/env python3
"""Turn slots written by ``nv_dataset.py`` into a slot-scale dataset for the ``nv_mu2`` profile.

Keeps the slots of the chosen Eb/No points, in a fixed shuffled order so every window of the
ring mixes them.  Files follow ``build_ul_dataset.py``: ``<profile>_rx.npy`` (slot, subcarrier,
symbol, antenna), ``<profile>_tb.npy`` (slot, UE, TB bytes) and ``<profile>_meta.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--ebno", default="2,2.5,3")
    parser.add_argument("--profile", default="nv_mu2")
    parser.add_argument("--link", type=Path, help="dataset whose strong-cell files are linked in")
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keep = [float(e) for e in args.ebno.split(",")]
    rx, tb, ebno, nrx_ok, meta = [], [], [], [], None
    for path in args.input:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        chosen = np.isin(np.round(data["ebno"], 3), np.round(keep, 3))
        rx.append(data["rx"][chosen])
        tb.append(data["payload"][chosen])
        ebno.append(data["ebno"][chosen])
        nrx_ok.append(data["nrx_ok"][chosen])
    rx, tb, ebno, nrx_ok = (np.concatenate(v) for v in (rx, tb, ebno, nrx_ok))
    order = np.random.default_rng(args.seed).permutation(rx.shape[0])
    rx, tb, ebno, nrx_ok = rx[order], tb[order], ebno[order], nrx_ok[order]
    tb_bytes = meta["tb_size_bits"] // 8
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"{args.profile}_rx.npy", np.ascontiguousarray(rx.astype(np.complex64)))
    np.save(args.output / f"{args.profile}_tb.npy", np.ascontiguousarray(tb[..., :tb_bytes]))
    np.save(args.output / f"{args.profile}_nvlabs_tf_ok.npy", nrx_ok)
    (args.output / f"{args.profile}_meta.json").write_text(json.dumps({
        "schema": "backstop-slot-ul-dataset-v1", "profile": {"name": args.profile}, "source": meta,
        "count": int(rx.shape[0]), "tb_bytes": tb_bytes, "num_ue": int(tb.shape[1]),
        "slots": [int(meta["slot_number"])] * int(rx.shape[0]),
        "esno_db": [float(e) for e in ebno], "snr_axis": "Eb/No (dB), NVlabs link simulation",
        "inputs": [str(p) for p in args.input], "shuffle_seed": args.seed,
    }))
    if args.link:
        for path in sorted(args.link.glob("strong_rank2_*")):
            target = args.output / path.name
            if not target.exists():
                target.symlink_to(path.resolve())
    print("slots", rx.shape, "tb", tb.shape, "Eb/No", {float(e): int((ebno == e).sum()) for e in np.unique(ebno)})


if __name__ == "__main__":
    main()
