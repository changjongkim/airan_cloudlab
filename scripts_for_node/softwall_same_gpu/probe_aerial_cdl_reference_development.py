#!/usr/bin/env python3
"""Screen the public NeuralRx profile on Aerial CDL-A before a frozen holdout."""

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
        direct_nrx_same_stream=True,
        num_ul_streams=1,
    )


def make_channel(
    receiver: PairedDualReceiver,
    seed: int,
    *,
    proc_sig_freq: bool,
    delay_s: float,
) -> tuple[FadingChan, dict]:
    tx_grid = np.asarray(receiver.rx_slot)
    if tx_grid.shape != (3276, 14, 1):
        raise RuntimeError(f"reference transmitter shape mismatch: {tx_grid.shape}")
    freq_in = np.ascontiguousarray(tx_grid.transpose(2, 1, 0)[None, None]).reshape(-1)
    carrier = pycuphy.CuphyCarrierPrms()
    carrier.n_sc = 3276
    carrier.n_bs_layer = 4
    carrier.n_ue_layer = 1
    config = pycuphy.CdlConfig()
    config.delay_profile = "A"
    config.delay_spread = 30
    config.max_doppler_shift = 10
    config.cfo_hz = 0
    config.delay = delay_s
    # Match the public example's one UE antenna and four gNB antennas.
    config.ue_ant_size = [1, 1, 1, 1, 1]
    config.ue_ant_pattern = 0
    config.ue_ant_polar_angles = [0, 90]
    if int(np.prod(config.bs_ant_size)) != 4:
        raise RuntimeError(f"unexpected default CDL BS array: {config.bs_ant_size}")
    config.n_sc = 3276
    config.run_mode = 2
    config.save_ant_pair_sample = False
    channel = FadingChan(
        cuphy_carrier_prms=carrier,
        cdl_cfg=config,
        fading_type=2,
        freq_in=freq_in,
        proc_sig_freq=proc_sig_freq,
        disable_noise=True,
        rand_seed=seed,
    )
    contract = {
        "delay_profile": "A",
        "delay_spread_ns": 30,
        "max_doppler_hz": 10,
        "bs_ant_size": list(config.bs_ant_size),
        "ue_ant_size": list(config.ue_ant_size),
    }
    return channel, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--payload-seed", type=int, default=20358500)
    parser.add_argument("--channel-seed", type=int, default=58501)
    args = parser.parse_args()

    arm_results = []
    for arm_index, arm in enumerate(ARMS):
        receiver = reference_receiver(args.engine, args.payload_seed + arm_index)
        channel, channel_contract = make_channel(
            receiver,
            args.channel_seed + arm_index,
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
            "channel_contract": channel_contract,
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "records": records,
        })

    result = {
        "schema": "softwall-aerial-cdl-reference-development-v1",
        "analysis_role": (
            "Development screen of the public NeuralRx reference profile on an "
            "advertised CDL family; this is not the frozen external-channel holdout."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "channel": "Aerial CDL-A, 30 ns, 10 Hz, 1x4 UL, no AWGN",
        "radio_profile": "NVIDIA public example: MCS7, start0, DMRS 0/5/10, TDI1, one UE TX",
        "payload_seed": args.payload_seed,
        "channel_seed": args.channel_seed,
        "iterations_per_arm": args.iterations,
        "arms": arm_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        x["name"]: {
            "conventional_correct": x["conventional_correct"],
            "neural_correct": x["neural_correct"],
        }
        for x in arm_results
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
