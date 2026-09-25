#!/usr/bin/env python3
"""Check Aerial TDL-A with the NeuralRx reference example's 1x4 UL geometry.

This is a workload-compatibility canary. It contains no MPS co-tenant,
admission policy, AI throughput, or deadline claim.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--payload-seed", type=int, default=20357000)
    parser.add_argument("--tdl-seed", type=int, default=57001)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if not 0 <= args.tdl_seed <= 0xFFFF:
        parser.error("Aerial TdlChan rand_seed is uint16")

    receiver = PairedDualReceiver(args.engine, seed=args.payload_seed)
    for _ in range(10):
        receiver.run_conventional()
        receiver.run_neural()

    # PdschTx emits [subcarrier, symbol, transmit antenna]. The paired PHY
    # uses one spatial layer with the fixed 1x4 precoder below. Recover the
    # one UE waveform before Aerial's 1-Tx/4-Rx UL channel, and verify all
    # four precoded columns against that waveform before running any radio.
    tx_grid = np.asarray(receiver.rx_slot)
    if tx_grid.shape != (3276, 14, 4):
        raise RuntimeError(f"unexpected transmit grid shape: {tx_grid.shape}")
    precoder = np.asarray([0.5, 0.5j, 0.5j, -0.5], dtype=np.complex64)
    ue_grid = np.ascontiguousarray(tx_grid[:, :, 0] / precoder[0])
    reconstructed = ue_grid[:, :, None] * precoder[None, None, :]
    precoding_max_abs_error = float(np.max(np.abs(tx_grid - reconstructed)))
    if not np.allclose(tx_grid, reconstructed, rtol=1e-5, atol=2e-5):
        raise RuntimeError(
            f"paired PHY is not the declared one-stream precoder: {precoding_max_abs_error}"
        )
    # The pybind TdlChan constructor accepts a flat complex64 host buffer.
    # Keep [cell, UE, antenna, symbol, subcarrier] C order.
    freq_in = np.ascontiguousarray(
        ue_grid.T[None, None, None, ...]
    ).reshape(-1)

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

    records = []
    for tti in range(args.iterations):
        channel_output = channel.run(
            tti_idx=tti,
            snr_db=100.0,
            enable_swap_tx_rx=True,
            tx_column_major_ind=False,
        )
        if channel_output.shape != (1, 1, 4, 14, 3276):
            raise RuntimeError(f"unexpected channel output shape: {channel_output.shape}")
        receiver.rx_slot = cp.asfortranarray(
            cp.asarray(channel_output[0, 0].transpose(2, 1, 0))
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

    result = {
        "schema": "softwall-aerial-tdl-1x4-compatibility-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "channel": "Aerial TDL-A, 30 ns delay spread, 10 Hz max Doppler, 1x4 UL, no AWGN",
        "input_logical_shape": [1, 1, 1, 14, 3276],
        "input_flat_elements": int(freq_in.size),
        "precoding_max_abs_error": precoding_max_abs_error,
        "payload_seed": args.payload_seed,
        "tdl_seed": args.tdl_seed,
        "iterations": args.iterations,
        "conventional_correct": sum(r["conventional_correct"] for r in records),
        "neural_correct": sum(r["neural_correct"] for r in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "slurm_job_id", "conventional_correct", "neural_correct", "iterations"
    )}), flush=True)
    if result["conventional_correct"] != args.iterations or result["neural_correct"] != args.iterations:
        raise SystemExit("TDL-A compatibility canary failed clean no-noise decoding")


if __name__ == "__main__":
    main()
