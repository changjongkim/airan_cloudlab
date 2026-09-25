#!/usr/bin/env python3
"""Test the exact NVIDIA transmitter contract and Aerial TDL execution modes."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import cupy as cp
import numpy as np

from aerial import pycuphy
from aerial.phy5g.chan_models.fading_chan import FadingChan

from dual_receiver_phy import PairedDualReceiver


ARMS = (
    {"name": "frequency_delay0", "proc_sig_freq": True, "delay_s": 0.0},
    {"name": "frequency_delay1us", "proc_sig_freq": True, "delay_s": 1e-6},
    {"name": "time_delay1us", "proc_sig_freq": False, "delay_s": 1e-6},
)


def reference_receiver(engine: str, seed: int) -> PairedDualReceiver:
    dmrs = [0] * 14
    for position in (0, 5, 10):
        dmrs[position] = 1
    return PairedDualReceiver(
        engine,
        seed=seed,
        mcs_index=7,
        start_sym=0,
        dmrs_syms=dmrs,
        enable_pusch_tdi=1,
        transmit_antennas=1,
        num_ul_streams=1,
    )


def make_channel(
    receiver: PairedDualReceiver,
    seed: int,
    *,
    proc_sig_freq: bool,
    delay_s: float,
) -> FadingChan:
    tx_grid = np.asarray(receiver.rx_slot)
    if tx_grid.shape != (3276, 14, 1):
        raise RuntimeError(f"reference transmitter shape mismatch: {tx_grid.shape}")
    freq_in = np.ascontiguousarray(tx_grid.transpose(2, 1, 0)[None, None]).reshape(-1)
    carrier = pycuphy.CuphyCarrierPrms()
    carrier.n_sc = 3276
    carrier.n_bs_layer = 4
    carrier.n_ue_layer = 1
    config = pycuphy.TdlConfig()
    config.delay_profile = "A"
    config.delay_spread = 30
    config.max_doppler_shift = 10
    config.cfo_hz = 0
    config.delay = delay_s
    config.n_bs_ant = 4
    config.n_ue_ant = 1
    config.n_sc = 3276
    config.run_mode = 2
    config.save_ant_pair_sample = False
    return FadingChan(
        cuphy_carrier_prms=carrier,
        tdl_cfg=config,
        fading_type=1,
        freq_in=freq_in,
        proc_sig_freq=proc_sig_freq,
        disable_noise=True,
        rand_seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--payload-seed", type=int, default=20357400)
    parser.add_argument("--tdl-seed", type=int, default=57401)
    args = parser.parse_args()

    arm_results = []
    for arm in ARMS:
        receiver = reference_receiver(args.engine, args.payload_seed)
        channel = make_channel(
            receiver,
            args.tdl_seed,
            proc_sig_freq=arm["proc_sig_freq"],
            delay_s=arm["delay_s"],
        )
        records = []
        for tti in range(args.iterations):
            output = channel.run(
                tti_idx=tti,
                snr_db=100.0,
                enable_swap_tx_rx=True,
                tx_column_major_ind=False,
            )
            receiver.rx_slot = cp.asfortranarray(
                cp.asarray(output[0, 0].transpose(2, 1, 0))
            )
            receiver.stream.synchronize()
            conventional = receiver.run_conventional()
            neural = receiver.run_neural()
            records.append({
                "tti": tti,
                "conventional_correct": bool(conventional[1]),
                "neural_correct": bool(neural[1]),
                "conventional_gpu_ms": conventional[0],
                "neural_gpu_ms": neural[0],
                "conventional_crc_failures": conventional[2],
                "neural_crc_failures": neural[2],
                "conventional_payload_mismatches": conventional[3],
                "neural_payload_mismatches": neural[3],
            })
        arm_results.append({
            **arm,
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "records": records,
        })

    result = {
        "schema": "softwall-aerial-tdl-tx-contract-development-v1",
        "analysis_role": (
            "Post-failure development screen of exact public-reference transmitter "
            "and Aerial channel execution modes; not a qualification outcome."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "channel": "Aerial TDL-A, 30 ns, 10 Hz, 1x4 UL, no AWGN",
        "radio_profile": "NVIDIA public example: MCS7, start0, DMRS 0/5/10, TDI1, one UE TX",
        "payload_seed": args.payload_seed,
        "tdl_seed": args.tdl_seed,
        "iterations_per_arm": args.iterations,
        "arms": arm_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        x["name"]: {
            "conventional_correct": x["conventional_correct"],
            "neural_correct": x["neural_correct"],
        }
        for x in arm_results
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
