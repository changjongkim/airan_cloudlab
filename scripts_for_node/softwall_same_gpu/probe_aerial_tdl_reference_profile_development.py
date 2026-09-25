#!/usr/bin/env python3
"""Localize the Aerial TDL-A mismatch against NVIDIA's NeuralRx example.

This is a post-failure development screen.  The arms progressively replace
the historical SoftWall radio profile with the public reference profile.  A
successful arm still requires a disjoint-seed holdout before it can support a
compatibility claim.
"""

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


PROFILES = (
    {
        "name": "historical",
        "mcs_index": 2,
        "start_sym": 2,
        "dmrs_positions": [2, 7, 11],
        "enable_pusch_tdi": 0,
    },
    {
        "name": "reference_dmrs_spacing",
        "mcs_index": 2,
        "start_sym": 2,
        "dmrs_positions": [2, 7, 12],
        "enable_pusch_tdi": 0,
    },
    {
        "name": "reference_dmrs_and_mcs",
        "mcs_index": 7,
        "start_sym": 2,
        "dmrs_positions": [2, 7, 12],
        "enable_pusch_tdi": 0,
    },
    {
        "name": "nvidia_reference",
        "mcs_index": 7,
        "start_sym": 0,
        "dmrs_positions": [0, 5, 10],
        "enable_pusch_tdi": 1,
    },
)


def make_dmrs(positions: list[int]) -> list[int]:
    value = [0] * 14
    for position in positions:
        value[position] = 1
    return value


def make_channel(receiver: PairedDualReceiver, seed: int) -> tuple[FadingChan, float]:
    tx_grid = np.asarray(receiver.rx_slot)
    precoder = np.asarray([0.5, 0.5j, 0.5j, -0.5], dtype=np.complex64)
    ue_grid = np.ascontiguousarray(tx_grid[:, :, 0] / precoder[0])
    reconstructed = ue_grid[:, :, None] * precoder[None, None, :]
    error = float(np.max(np.abs(tx_grid - reconstructed)))
    if not np.allclose(tx_grid, reconstructed, rtol=1e-5, atol=2e-5):
        raise RuntimeError("paired PHY does not match its one-stream precoder")
    freq_in = np.ascontiguousarray(ue_grid.T[None, None, None, ...]).reshape(-1)

    carrier = pycuphy.CuphyCarrierPrms()
    carrier.n_sc = 3276
    carrier.n_bs_layer = 4
    carrier.n_ue_layer = 1
    tdl_cfg = pycuphy.TdlConfig()
    tdl_cfg.delay_profile = "A"
    tdl_cfg.delay_spread = 30
    tdl_cfg.max_doppler_shift = 10
    tdl_cfg.cfo_hz = 0
    tdl_cfg.delay = 0
    tdl_cfg.n_bs_ant = 4
    tdl_cfg.n_ue_ant = 1
    tdl_cfg.n_sc = 3276
    tdl_cfg.run_mode = 2
    tdl_cfg.save_ant_pair_sample = False
    return FadingChan(
        cuphy_carrier_prms=carrier,
        tdl_cfg=tdl_cfg,
        fading_type=1,
        freq_in=freq_in,
        proc_sig_freq=True,
        disable_noise=True,
        rand_seed=seed,
    ), error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--payload-seed", type=int, default=20357300)
    parser.add_argument("--tdl-seed", type=int, default=57301)
    args = parser.parse_args()

    results = []
    for profile in PROFILES:
        receiver = PairedDualReceiver(
            args.engine,
            seed=args.payload_seed,
            mcs_index=profile["mcs_index"],
            start_sym=profile["start_sym"],
            dmrs_syms=make_dmrs(profile["dmrs_positions"]),
            enable_pusch_tdi=profile["enable_pusch_tdi"],
        )
        channel, precoding_error = make_channel(receiver, args.tdl_seed)
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
        results.append({
            **profile,
            "precoding_max_abs_error": precoding_error,
            "mod_order": receiver.mod_order,
            "code_rate_x10": receiver.code_rate,
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "records": records,
        })

    result = {
        "schema": "softwall-aerial-tdl-reference-profile-development-v1",
        "analysis_role": (
            "Post-failure development localization; no channel qualification "
            "or paper outcome is opened by this artifact."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "channel": "Aerial TDL-A, 30 ns, 10 Hz, 1x4 UL, no AWGN",
        "payload_seed": args.payload_seed,
        "tdl_seed": args.tdl_seed,
        "iterations_per_profile": args.iterations,
        "profiles": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        x["name"]: {
            "conventional_correct": x["conventional_correct"],
            "neural_correct": x["neural_correct"],
        }
        for x in results
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
