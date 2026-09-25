#!/usr/bin/env python3
"""Localize CDL compatibility at the TensorFlow-to-cuPHY layout boundary."""

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
from probe_sionna_cdl_reference_development import apply_channel, build_channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("native", "fortran"), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--payload-seed", type=int, default=20358920)
    parser.add_argument("--channel-seed", type=int, default=58921)
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
    mapper, remove_guards, channel = build_channel()
    tx_cpu = cp.asnumpy(receiver.clean_rx_slot)
    noise_variance = tf.constant(10.0 ** (-args.esno_db / 10.0), tf.float32)
    records = []
    layout_observation = None
    for iteration in range(args.iterations):
        rx_tensor = apply_channel(
            tf.convert_to_tensor(tx_cpu, dtype=tf.complex64),
            noise_variance,
            mapper,
            remove_guards,
            channel,
        )
        received = cp.asarray(rx_tensor.numpy())
        native_observation = {
            "shape": list(received.shape),
            "strides_bytes": list(received.strides),
            "c_contiguous": bool(received.flags.c_contiguous),
            "f_contiguous": bool(received.flags.f_contiguous),
        }
        receiver.rx_slot = (
            received if args.layout == "native" else cp.asfortranarray(received)
        )
        receiver.stream.synchronize()
        if layout_observation is None:
            layout_observation = {
                "native": native_observation,
                "receiver": {
                    "shape": list(receiver.rx_slot.shape),
                    "strides_bytes": list(receiver.rx_slot.strides),
                    "c_contiguous": bool(receiver.rx_slot.flags.c_contiguous),
                    "f_contiguous": bool(receiver.rx_slot.flags.f_contiguous),
                },
            }
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
        "schema": "softwall-sionna-nrx-layout-development-v1",
        "analysis_role": "Post-failure interface localization; not a BLER holdout.",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": "CPU",
        "channel": "Sionna CDL-A, 100 ns, 0.8333 m/s, normalized, 1x4 uplink",
        "layout": args.layout,
        "layout_observation": layout_observation,
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
        "layout": args.layout,
        "layout_observation": layout_observation,
        "conventional_correct": result["conventional_correct"],
        "neural_correct": result["neural_correct"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
