#!/usr/bin/env python3
"""Is a rescue the neural receiver, or just more LDPC iterations?

On one slot pool this decodes every TB with
  conv        the cuPHY PUSCH pipeline as the cells run it (its own LDPC iteration rule),
  conv_fixN   the same cuPHY pipeline with its LDPC iteration limit fixed to N,
  conv_itN    the conventional front end (channel estimate, MMSE equalizer, demapper) from the
              separable pyAerial components, then LDPC with N iterations,
  nrx_itN     the neural receiver, then the same LDPC with N iterations,
and reports, of the TBs ``conv`` failed, how many each alternative recovers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from aerial.phy5g.algorithms import ChannelEqualizer, ChannelEstimator, NoiseIntfEstimator
from aerial.phy5g.ldpc import CrcChecker, LdpcDecoder, LdpcDeRateMatch

from cell_ring import CellRing
from slot_radio import conv_path, nrx_path, pusch_configs, result_masks
from ul_profiles import NUM_RX_ANT


def fixed_iteration_conv(profile, tb_bytes: int, stream: cp.cuda.Stream, iterations: int):
    """The cells' cuPHY pipeline, built with a fixed LDPC iteration limit instead of its table."""
    import aerial.phy5g.pusch.pusch_rx as module
    from aerial.pycuphy import LdpcMaxItrAlgoType
    original = module.get_pusch_stat_prms

    def patched(*args, **kwargs):
        prms = original(*args, **kwargs)
        return prms._replace(ldpcMaxNumItrAlgo=LdpcMaxItrAlgoType.LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED,
                             fixedMaxNumLdpcItrs=np.uint8(iterations))

    module.get_pusch_stat_prms = patched
    try:
        return conv_path(profile, tb_bytes, stream)
    finally:
        module.get_pusch_stat_prms = original


class SeparableConv:
    """Conventional front end with a chosen LDPC iteration count, per UE."""

    def __init__(self, profile, tb_bytes: int, stream: cp.cuda.Stream, iterations: int) -> None:
        self.profile = profile
        self.stream = stream
        handle = int(stream.ptr)
        if profile.num_ue > 1:
            from mu_radio import mu_pusch_configs
            self.configs, self.per_ue = mu_pusch_configs(profile, tb_bytes)
        else:
            self.configs = pusch_configs(profile, tb_bytes)
            self.per_ue = self.configs
        self.estimator = ChannelEstimator(num_rx_ant=NUM_RX_ANT, cuda_stream=handle)
        self.noise = NoiseIntfEstimator(num_rx_ant=NUM_RX_ANT, eq_coeff_algo=1, cuda_stream=handle)
        self.equalizer = ChannelEqualizer(num_rx_ant=NUM_RX_ANT, enable_pusch_tdi=profile.enable_pusch_tdi,
                                          eq_coeff_algo=1, cuda_stream=handle)
        self.derate = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=handle)
        self.decoder = LdpcDecoder(num_iterations=iterations, cuda_stream=handle)
        self.crc = CrcChecker(cuda_stream=handle)
        self.last_crc_mask = 0

    def llrs(self, rx_slot: cp.ndarray, slot: int):
        estimate = self.estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=self.configs)
        lw_inv, noise_var = self.noise.estimate(rx_slot=rx_slot, channel_est=estimate, slot=slot,
                                                pusch_configs=self.configs)
        return self.equalizer.equalize(rx_slot=rx_slot, channel_est=estimate, lw_inv=lw_inv,
                                       noise_var_pre_eq=noise_var, pusch_configs=self.configs)[0]

    def decode(self, llrs) -> tuple[bool, list]:
        if self.profile.num_ue > 1:
            llrs = [cp.ascontiguousarray(llrs[0][:, u:u + 1]) for u in range(self.profile.num_ue)]
        coded = self.derate.derate_match(input_llrs=llrs, pusch_configs=self.per_ue)
        blocks = self.decoder.decode(input_llrs=coded, pusch_configs=self.per_ue)
        tbs, crcs = self.crc.check_crc(input_bits=blocks, pusch_configs=self.per_ue)
        self.stream.synchronize()
        mask, payloads = 0, []
        for u in range(self.profile.num_ue):
            payloads.append(cp.asnumpy(tbs[u]) if isinstance(tbs[u], cp.ndarray) else np.asarray(tbs[u]))
            crc = cp.asnumpy(crcs[u]) if isinstance(crcs[u], cp.ndarray) else np.asarray(crcs[u])
            if int(crc.reshape(-1)[0]) == 0:
                mask |= 1 << u
        self.last_crc_mask = mask
        every = mask == (1 << self.profile.num_ue) - 1
        return every, (payloads if self.profile.num_ue > 1 else payloads[0])

    def run(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, list]:
        return self.decode(self.llrs(rx_slot, slot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--iterations", default="10,20,40")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total = json.loads((args.dataset / f"{args.profile}_meta.json").read_text())["count"]
    count = min(args.count or total, total)
    ring = CellRing(args.dataset, args.profile, count, 0)
    profile, num_ue = ring.profile, ring.profile.num_ue
    stream = cp.cuda.Stream(non_blocking=True)
    iterations = [int(v) for v in args.iterations.split(",")]
    paths = {"conv": conv_path(profile, ring.tb_bytes, stream)}
    for n in iterations:
        paths[f"conv_fix{n}"] = fixed_iteration_conv(profile, ring.tb_bytes, stream, n)
        if num_ue == 1:
            paths[f"conv_it{n}"] = SeparableConv(profile, ring.tb_bytes, stream, n)
        neural = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": n}, profile, ring.tb_bytes, stream)
        neural.decoder = LdpcDecoder(num_iterations=n, cuda_stream=int(stream.ptr))
        paths[f"nrx_it{n}"] = neural

    good = {name: np.zeros((count, num_ue), dtype=bool) for name in paths}
    wall = {name: [] for name in paths}
    for name, path in paths.items():
        for attempt in range(2):
            for i in range(count):
                start = time.perf_counter()
                ok, payload = path.run(ring.views[i], ring.slots[i])
                if attempt:
                    wall[name].append(time.perf_counter() - start)
                _, mask = result_masks(path, ok, payload, ring.blocks[i], ring.tb_bytes)
                good[name][i] = [(mask >> u) & 1 for u in range(num_ue)]

    failed = ~good["conv"]
    report = {"dataset": str(args.dataset), "profile": args.profile, "slots": count, "tbs": int(failed.size),
              "conv_failed": int(failed.sum()), "variants": {}}
    for name in paths:
        ms = np.asarray(wall[name]) * 1e3
        report["variants"][name] = {
            "decoded": int(good[name].sum()),
            "recovers_of_conv_failures": int((failed & good[name]).sum()),
            "loses_of_conv_successes": int((~failed & ~good[name]).sum()),
            "wall_ms_p50": float(np.percentile(ms, 50)), "wall_ms_p99": float(np.percentile(ms, 99)),
        }
    for n in iterations:
        a, b = good[f"nrx_it{n}"], good[f"conv_fix{n}"]
        report["variants"][f"nrx_it{n}"]["only_neural_vs_conv_fix_same_iterations"] = int((a & ~b).sum())
        report["variants"][f"nrx_it{n}"]["only_conv_fix_vs_neural_same_iterations"] = int((~a & b).sum())
    # Decoded TBs per SNR bin (0.2 dB), every variant.
    snr = np.repeat(np.asarray(ring.esno_db)[:, None], num_ue, axis=1)
    edges = np.arange(np.floor(snr.min() * 5) / 5, snr.max() + 0.2, 0.2)
    report["by_snr"] = []
    for low in edges:
        chosen = (snr >= low) & (snr < low + 0.2)
        if chosen.any():
            report["by_snr"].append({"snr_db": round(float(low), 1), "tbs": int(chosen.sum()),
                                     **{name: int(good[name][chosen].sum()) for name in paths}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("| receiver | TBs decoded | of conventional failures recovered | wall ms p50 / p99 |")
    print("|---|---|---|---|")
    for name, row in report["variants"].items():
        print(f"| {name} | {row['decoded']} / {report['tbs']} | {row['recovers_of_conv_failures']} / "
              f"{report['conv_failed']} | {row['wall_ms_p50']:.2f} / {row['wall_ms_p99']:.2f} |")
    names = [n for n in paths if not n.startswith("conv_it")]
    print("| SNR dB | TBs | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for row in report["by_snr"]:
        print(f"| {row['snr_db']} | {row['tbs']} | " + " | ".join(str(row[n]) for n in names) + " |")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
