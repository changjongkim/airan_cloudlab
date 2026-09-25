#!/usr/bin/env python3
"""Diagnose the failed Aerial TDL-A/NeuralRx interface contract.

This is a development screen after two preserved compatibility failures.  It
does not qualify a channel mode.  Each generated TTI is decoded under the raw
Aerial output and three deterministic scale conventions.  The CFR-RMS arm is
the direct analogue of the NeuralRx reference notebook's
``normalize_channel=True`` setting.
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


def power(value: np.ndarray) -> float:
    return float(np.mean(np.abs(value) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--payload-seed", type=int, default=20357100)
    parser.add_argument("--tdl-seed", type=int, default=57101)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if not 0 <= args.tdl_seed <= 0xFFFF:
        parser.error("Aerial TdlChan rand_seed is uint16")

    receiver = PairedDualReceiver(args.engine, seed=args.payload_seed)
    for _ in range(10):
        receiver.run_conventional()
        receiver.run_neural()

    tx_grid = np.asarray(receiver.rx_slot)
    precoder = np.asarray([0.5, 0.5j, 0.5j, -0.5], dtype=np.complex64)
    ue_grid = np.ascontiguousarray(tx_grid[:, :, 0] / precoder[0])
    reconstructed = ue_grid[:, :, None] * precoder[None, None, :]
    precoding_max_abs_error = float(np.max(np.abs(tx_grid - reconstructed)))
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
    channel = FadingChan(
        cuphy_carrier_prms=carrier,
        tdl_cfg=tdl_cfg,
        fading_type=1,
        freq_in=freq_in,
        proc_sig_freq=True,
        disable_noise=True,
        rand_seed=args.tdl_seed,
    )

    ue_power = power(ue_grid)
    component_power = power(tx_grid)
    records = []
    totals = {
        name: {"conventional_correct": 0, "neural_correct": 0}
        for name in ("raw", "cfr_rms", "match_ue_power", "match_component_power")
    }
    for tti in range(args.iterations):
        output = channel.run(
            tti_idx=tti,
            snr_db=100.0,
            enable_swap_tx_rx=True,
            tx_column_major_ind=False,
        )
        if output.shape != (1, 1, 4, 14, 3276):
            raise RuntimeError(f"unexpected channel output shape: {output.shape}")
        raw = np.asarray(output[0, 0].transpose(2, 1, 0), dtype=np.complex64)
        raw_power = power(raw)
        cfr = np.asarray(channel.cfr_sc[0, 0])
        cfr_power = power(cfr)
        if min(raw_power, cfr_power, ue_power, component_power) <= 0:
            raise RuntimeError("non-positive signal or channel power")
        variants = {
            "raw": raw,
            "cfr_rms": raw / np.sqrt(cfr_power),
            "match_ue_power": raw * np.sqrt(ue_power / raw_power),
            "match_component_power": raw * np.sqrt(component_power / raw_power),
        }
        row = {
            "tti": tti,
            "ue_grid_power": ue_power,
            "precoded_component_power": component_power,
            "raw_rx_power": raw_power,
            "cfr_power": cfr_power,
            "variants": {},
        }
        for name, value in variants.items():
            receiver.rx_slot = cp.asfortranarray(cp.asarray(value))
            receiver.stream.synchronize()
            conventional = receiver.run_conventional()
            neural = receiver.run_neural()
            item = {
                "scale": float(np.sqrt(power(value) / raw_power)),
                "rx_power": power(value),
                "conventional_correct": bool(conventional[1]),
                "neural_correct": bool(neural[1]),
                "conventional_crc_failures": conventional[2],
                "neural_crc_failures": neural[2],
                "conventional_payload_mismatches": conventional[3],
                "neural_payload_mismatches": neural[3],
            }
            row["variants"][name] = item
            totals[name]["conventional_correct"] += int(conventional[1])
            totals[name]["neural_correct"] += int(neural[1])
        records.append(row)

    result = {
        "schema": "softwall-aerial-tdl-normalization-development-v1",
        "analysis_role": (
            "Post-failure development hypothesis screen; no channel qualification "
            "or paper outcome is opened by this artifact."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "channel": "Aerial TDL-A, 30 ns, 10 Hz, 1x4 UL, no AWGN",
        "reference_difference": (
            "The NVIDIA NeuralRx notebook uses Sionna OFDMChannel with "
            "normalize_channel=True; the prior Aerial TDL canaries used raw output."
        ),
        "payload_seed": args.payload_seed,
        "tdl_seed": args.tdl_seed,
        "iterations": args.iterations,
        "precoding_max_abs_error": precoding_max_abs_error,
        "totals": totals,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"job": result["slurm_job_id"], "totals": totals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
