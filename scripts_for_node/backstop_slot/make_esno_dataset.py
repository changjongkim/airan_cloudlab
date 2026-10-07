#!/usr/bin/env python3
"""A dataset whose slots follow an Es/No trajectory, made from the slots of a dataset at a higher Es/No.

A slot of the base dataset is y = Hx + n.  The NVlabs link simulation draws n with the variance
N0 = K * 10^(-Es/No / 10) per resource element and antenna; K = 1.26709 for 273 PRBs and 14 symbols with two DMRS
symbols (Sionna ebnodb2no with the resource grid; the same for every MCS and channel).  Adding white Gaussian
noise of variance K * (10^(-e/10) - 10^(-base/10)) gives the same channel realization and the same TB at
Es/No = e <= base.  No channel is generated here.

Slot i of the output is base slot i mod (base count) with the Es/No of the trajectory at i.  Every MCS level gets the
same trajectory, so a cell that replays the output as a ring sees an Es/No that depends on the ring index alone;
the trajectories are periodic over the output length.

usage: make_esno_dataset.py --base DIR --base-esno 20 --output DIR --trajectory SPEC [--slots N] [--levels 10,..,16]
  SPEC  const:E                       every slot at E dB
        sine:LO:HI                    one period of a sine between LO and HI dB over the N slots
        walk:MEAN:SIGMA:TAU[:LO:HI]   Gaussian process with mean MEAN dB, standard deviation SIGMA dB and a
                                      correlation time of TAU slots (first-order spectrum), clipped to LO..HI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

K = 1.2670939788222313          # N0 * 10^(Es/No / 10) of the link simulation (n0_probe.py)


def trajectory(spec: str, slots: int, ceiling: float, seed: int) -> np.ndarray:
    kind, *values = spec.split(":")
    v = [float(x) for x in values]
    i = np.arange(slots)
    if kind == "const":
        e = np.full(slots, v[0])
    elif kind == "sine":
        e = (v[0] + v[1]) / 2 + (v[1] - v[0]) / 2 * np.sin(2 * np.pi * i / slots)
    elif kind == "walk":
        mean, sigma, tau = v[:3]
        lo, hi = (v[3], v[4]) if len(v) >= 5 else (mean - 3 * sigma, ceiling)
        rng = np.random.default_rng(seed)
        f = np.fft.rfftfreq(slots)
        z = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(1 + (2 * np.pi * f * tau) ** 2)
        z[0] = 0
        x = np.fft.irfft(z, n=slots)
        e = np.clip(mean + sigma * x / x.std(), lo, hi)
    else:
        raise SystemExit(f"unknown trajectory {spec}")
    if e.max() > ceiling + 1e-9:
        raise SystemExit(f"trajectory reaches {e.max():.2f} dB, above the base Es/No {ceiling} dB")
    return e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-esno", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--slots", type=int, default=0, help="output slots per level (default: the base count)")
    parser.add_argument("--levels", default="10,11,12,13,14,15,16")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    esno = None
    for level in args.levels.split(","):
        name = f"nv_mu2_m{level}"
        meta = json.loads((args.base / f"{name}_meta.json").read_text())
        base_rx = np.load(args.base / f"{name}_rx.npy", mmap_mode="r")
        base_tb = np.load(args.base / f"{name}_tb.npy")
        count = int(meta["count"])
        slots = args.slots or count
        if esno is None:
            esno = trajectory(args.trajectory, slots, args.base_esno, args.seed)
        index = np.arange(slots) % count
        added = K * (10 ** (-esno / 10) - 10 ** (-args.base_esno / 10))            # noise variance to add per slot
        out = np.lib.format.open_memmap(args.output / f"{name}_rx.npy", mode="w+", dtype=np.complex64,
                                        shape=(slots,) + base_rx.shape[1:])
        rng = np.random.default_rng(args.seed + 1000 * int(level))
        for start in range(0, slots, args.chunk):
            stop = min(slots, start + args.chunk)
            shape = (stop - start,) + base_rx.shape[1:]
            noise = rng.standard_normal(shape, dtype=np.float32) + 1j * rng.standard_normal(shape, dtype=np.float32)
            scale = np.sqrt(np.maximum(added[start:stop], 0.0) / 2).astype(np.float32)[:, None, None, None]
            out[start:stop] = np.asarray(base_rx[index[start:stop]]) + scale * noise
        out.flush()
        del out
        np.save(args.output / f"{name}_tb.npy", np.ascontiguousarray(base_tb[index]))
        shift = args.base_esno - esno                                              # dB below the base, per slot
        new = dict(meta)
        new.update(count=slots, slots=[int(meta["slots"][i]) for i in index],
                   esno_db=[float(meta["esno_db"][i] - s) for i, s in zip(index, shift)],
                   esno_true_db=[float(e) for e in esno],
                   synthesized={"base": str(args.base), "base_esno_db": args.base_esno, "trajectory": args.trajectory,
                                "noise_constant": K, "seed": args.seed, "base_index": [int(i) for i in index]})
        (args.output / f"{name}_meta.json").write_text(json.dumps(new))
        print(f"{name}: {slots} slots, Es/No {esno.min():.2f}..{esno.max():.2f} dB (mean {esno.mean():.2f})", flush=True)
    for path in sorted(args.base.glob("strong_rank2_*")):
        target = args.output / path.name
        if not target.exists():
            target.symlink_to(path.resolve())


if __name__ == "__main__":
    main()
