#!/usr/bin/env python3
"""Validate a TensorRT build of an NVlabs neural receiver against its TensorFlow test vector,
time it, and decode its LLRs with pyAerial (rate recovery, LDPC, CRC) for every user.

Also checks the input path we use at run time: the pyAerial LS channel estimate in place of
the Sionna LS estimate saved with the test vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import CrcChecker, LdpcDecoder, LdpcDeRateMatch, get_mcs

from nv_trt import NvNrx


def pusch_configs(meta: dict) -> list:
    mod_order, code_rate = get_mcs(meta["mcs_index"], 1)
    dmrs_syms = [0] * 14
    for position in meta["dmrs_ofdm_pos"][0]:
        dmrs_syms[position] = 1
    ports = [0 if row[0] == 0 else 2 for row in meta["dmrs_subcarrier_pos"]][:meta["num_tx"]]
    ues = [PuschUeConfig(scid=1, layers=1, dmrs_ports=1 << port, rnti=1, data_scid=1, mcs_table=0,
                         mcs_index=meta["mcs_index"], code_rate=int(code_rate * 10), mod_order=int(mod_order),
                         tb_size=meta["tb_size_bits"] // 8) for port in ports]
    def group(members: list, streams: int) -> PuschConfig:
        return PuschConfig(ue_configs=members, num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=1, start_prb=0,
                           num_prbs=meta["prbs"], prg_size=1, num_ul_streams=streams, dmrs_syms=dmrs_syms,
                           dmrs_max_len=1, dmrs_add_ln_pos=len(meta["dmrs_ofdm_pos"][0]) - 1, start_sym=0,
                           num_symbols=14)

    # One group with every user for the receivers that see the slot; one single-user group
    # per user for the bit-level chain after the neural receiver (its LLRs are per user).
    return [group(ues, meta["num_tx"])], [group([ue], 1) for ue in ues]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    stem = args.engine.rsplit(".", 1)[0]
    vec = np.load(stem + "_testvector.npz")
    meta = json.loads(str(vec["meta"]))
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = NvNrx(args.engine, stem + ".json", stream=stream)
    for name in ("rx_slot_real", "rx_slot_imag", "h_hat_real", "h_hat_imag"):
        nrx.inputs[name][...] = cp.asarray(vec[name])
    out = nrx.run_sync()
    llr = cp.asnumpy(out["output_1"])
    ref = vec["llr"]
    report = {"engine": args.engine, "llr_shape": list(llr.shape),
              "llr_max_abs_diff": float(np.abs(llr - ref).max()),
              "llr_sign_agreement": float((np.sign(llr) == np.sign(ref)).mean())}

    handle = int(stream.ptr)
    configs, per_user = pusch_configs(meta)
    derate = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=handle)
    decoder = LdpcDecoder(cuda_stream=handle)
    crc = CrcChecker(cuda_stream=handle)
    data_symbols = cp.asarray([s for s in range(14) if s not in meta["dmrs_ofdm_pos"][0]], dtype=cp.int64)
    tb_bytes = meta["tb_size_bits"] // 8

    def decode(outputs) -> list:
        with stream:
            llrs = cp.take(outputs["output_1"][0, ...], data_symbols, axis=3)
            split = [cp.ascontiguousarray(llrs[:, u:u + 1]) for u in range(meta["num_tx"])]
            coded = derate.derate_match(input_llrs=split, pusch_configs=per_user)
            blocks = decoder.decode(input_llrs=coded, pusch_configs=per_user)
            tbs, crcs = crc.check_crc(input_bits=blocks, pusch_configs=per_user)
        stream.synchronize()
        result = []
        for u in range(meta["num_tx"]):
            ok = int(cp.asnumpy(crcs[u]).reshape(-1)[0]) == 0
            same = bool(np.array_equal(cp.asnumpy(tbs[u])[:tb_bytes], vec["payload"][0, u, :tb_bytes]))
            result.append({"crc_pass": ok, "payload_ok": same})
        return result

    report["decode_with_sionna_ls"] = decode(out)

    # Run-time input path: LS estimate from pyAerial on the same slot.
    estimator = ChannelEstimator(num_rx_ant=meta["num_rx_ant"], ch_est_algo=3, cuda_stream=handle)
    rx = cp.asarray(np.asfortranarray((vec["rx_slot_real"][0] + 1j * vec["rx_slot_imag"][0]).astype(np.complex64)))
    channel = estimator.estimate(rx_slot=rx, slot=meta["slot_number"], pusch_configs=configs)
    stream.synchronize()
    est = cp.asarray(channel[0])
    report["pyaerial_ls_shape"] = list(est.shape)
    cp.save(stem + "_pyaerial_ls.npy", est)
    report["h_hat_input_shape"] = list(vec["h_hat_real"].shape)

    # Timing of the engine alone (CUDA graph replay).
    nrx.capture_graph()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(args.iterations):
        with stream:
            begin.record()
            nrx.launch(use_graph=True)
            end.record()
        end.synchronize()
        times.append(float(cp.cuda.get_elapsed_time(begin, end)))
    report["trt_gpu_ms"] = {"p50": float(np.percentile(times, 50)), "p99": float(np.percentile(times, 99))}
    Path(stem + "_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
