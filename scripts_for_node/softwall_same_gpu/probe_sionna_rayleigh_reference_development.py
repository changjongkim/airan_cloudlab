#!/usr/bin/env python3
"""Run the public default Sionna Rayleigh profile as a compatibility control."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import json
import platform
from pathlib import Path

import cupy as cp
import numpy as np
import sionna
import tensorflow as tf

from dual_receiver_phy import PairedDualReceiver
from probe_sionna_cdl_reference_development import apply_channel


def build_rayleigh_channel():
    resource_grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=4096,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=288,
        num_guard_carriers=(410, 410),
        dc_null=False,
        pilot_pattern=None,
        pilot_ofdm_symbol_indices=None,
    )
    mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
    remove_guards = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)
    model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1,
        num_rx_ant=4,
        num_tx=1,
        num_tx_ant=1,
    )
    channel = sionna.phy.channel.OFDMChannel(
        model,
        resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=False,
    )
    return mapper, remove_guards, channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--payload-seed", type=int, default=20358940)
    parser.add_argument("--channel-seed", type=int, default=58941)
    parser.add_argument("--esno-db", type=float, default=30.0)
    args = parser.parse_args()

    tf.random.set_seed(args.channel_seed)
    np.random.seed(args.payload_seed)
    dmrs = [0] * 14
    for position in (0, 5, 10):
        dmrs[position] = 1
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.payload_seed,
        mcs_index=7,
        start_sym=0,
        dmrs_syms=dmrs,
        enable_pusch_tdi=1,
        transmit_antennas=1,
        direct_nrx_same_stream=True,
        num_ul_streams=1,
    )
    mapper, remove_guards, channel = build_rayleigh_channel()
    tx_cpu = cp.asnumpy(receiver.clean_rx_slot)
    noise_variance = tf.constant(10.0 ** (-args.esno_db / 10.0), tf.float32)
    records = []
    for iteration in range(args.iterations):
        received = cp.asarray(apply_channel(
            tf.convert_to_tensor(tx_cpu, dtype=tf.complex64),
            noise_variance,
            mapper,
            remove_guards,
            channel,
        ).numpy())
        receiver.rx_slot = received
        receiver.stream.synchronize()
        features = receiver.observed_channel_features()
        conventional = receiver.run_conventional()
        neural = receiver._measure(receiver.neural_once_wrapper)
        records.append({
            "iteration": iteration,
            "features": features,
            "conventional_correct": bool(conventional[1]),
            "neural_correct": bool(neural[1]),
            "conventional_crc_failures": conventional[2],
            "neural_crc_failures": neural[2],
        })

    result = {
        "schema": "softwall-sionna-rayleigh-reference-development-v1",
        "analysis_role": "Public-default channel compatibility control; not a BLER holdout.",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": "CPU",
        "channel": "Sionna normalized 1x4 Rayleigh block fading",
        "radio_profile": "NVIDIA public NeuralRx notebook MCS7/start0/DMRS0,5,10/TDI1",
        "esno_db": args.esno_db,
        "iterations": args.iterations,
        "payload_seed": args.payload_seed,
        "channel_seed": args.channel_seed,
        "conventional_correct": sum(x["conventional_correct"] for x in records),
        "neural_correct": sum(x["neural_correct"] for x in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "channel": result["channel"],
        "esno_db": args.esno_db,
        "conventional_correct": result["conventional_correct"],
        "neural_correct": result["neural_correct"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
