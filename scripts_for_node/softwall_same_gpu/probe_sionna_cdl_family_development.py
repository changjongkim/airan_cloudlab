#!/usr/bin/env python3
"""Map the public NeuralRx compatibility boundary across CDL model and delay."""

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


def build_channel(model_name: str, delay_spread_ns: float):
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
        model_name,
        delay_spread_ns * 1e-9,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple("ABCDE"), required=True)
    parser.add_argument("--delay-spread-ns", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--payload-seed", type=int, default=20358960)
    parser.add_argument("--channel-seed", type=int, required=True)
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
    mapper, remove_guards, channel = build_channel(args.model, args.delay_spread_ns)
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
        "schema": "softwall-sionna-cdl-family-development-v1",
        "analysis_role": "Compatibility-boundary development screen; not a BLER holdout.",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": "CPU",
        "cdl_model": args.model,
        "delay_spread_ns": args.delay_spread_ns,
        "speed_mps": 0.8333,
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
        "cdl_model": args.model,
        "delay_spread_ns": args.delay_spread_ns,
        "conventional_correct": result["conventional_correct"],
        "neural_correct": result["neural_correct"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
