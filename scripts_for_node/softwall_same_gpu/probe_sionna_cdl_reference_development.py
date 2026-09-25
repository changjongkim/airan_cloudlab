#!/usr/bin/env python3
"""Reproduce the public NeuralRx notebook's exact Sionna CDL-A channel path."""

from __future__ import annotations

import os

# These must precede TensorFlow import, matching the public notebook.
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


def build_channel():
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
    ue_array = sionna.phy.channel.tr38901.Antenna(
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=3.5e9,
    )
    gnb_array = sionna.phy.channel.tr38901.AntennaArray(
        num_rows=1,
        num_cols=2,
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=3.5e9,
    )
    model = sionna.phy.channel.tr38901.CDL(
        "A",
        100e-9,
        3.5e9,
        ue_array,
        gnb_array,
        "uplink",
        min_speed=0.8333,
    )
    channel = sionna.phy.channel.OFDMChannel(
        model,
        resource_grid,
        add_awgn=True,
        normalize_channel=True,
        return_channel=False,
    )
    return mapper, remove_guards, channel


def apply_channel(tx_tensor, noise_variance, mapper, remove_guards, channel):
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
    tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]
    tx_tensor = mapper(tx_tensor)
    rx_tensor = channel(tx_tensor, noise_variance)
    rx_tensor = remove_guards(rx_tensor)
    rx_tensor = rx_tensor[0, 0]
    return tf.transpose(rx_tensor, (2, 1, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--payload-seed", type=int, default=20358700)
    parser.add_argument("--channel-seed", type=int, default=58701)
    parser.add_argument("--esno-db", type=float, default=30.0)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    tensorflow_device = "GPU" if gpus else "CPU"
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
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
    mapper, remove_guards, channel = build_channel()
    noise_variance = tf.constant(10.0 ** (-args.esno_db / 10.0), tf.float32)
    records = []
    for iteration in range(args.iterations):
        if gpus:
            tx_tensor = tf.experimental.dlpack.from_dlpack(
                cp.asarray(receiver.clean_rx_slot).toDlpack()
            )
        else:
            tx_tensor = tf.convert_to_tensor(
                cp.asnumpy(receiver.clean_rx_slot), dtype=tf.complex64
            )
        rx_tensor = apply_channel(
            tx_tensor, noise_variance, mapper, remove_guards, channel
        )
        if gpus:
            received = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(rx_tensor))
        else:
            received = cp.asarray(rx_tensor.numpy())
        receiver.rx_slot = cp.asfortranarray(received)
        receiver.stream.synchronize()
        conventional = receiver.run_conventional()
        neural = receiver.run_neural()
        records.append({
            "iteration": iteration,
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
        "schema": "softwall-sionna-cdl-reference-development-v1",
        "analysis_role": (
            "Development reproduction of the public notebook's Sionna CDL path; "
            "not a frozen external-channel qualification."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": tensorflow_device,
        "channel": "Sionna CDL-A, 100 ns, 0.8333 m/s, normalized, 1x4 uplink",
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
        key: result[key] for key in (
            "tensorflow_version", "sionna_version", "channel", "esno_db",
            "tensorflow_channel_device", "iterations", "conventional_correct",
            "neural_correct",
        )
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
