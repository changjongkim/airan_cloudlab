#!/usr/bin/env python3
"""Screen NeuralRx over the exact Es/No range declared by its public notebook."""

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


OFFICIAL_ESNO_GRID = (-4.0, -3.8, -3.6, -3.4, -3.2, -3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations-per-snr", type=int, default=100)
    parser.add_argument("--payload-seed", type=int, default=20358800)
    parser.add_argument("--channel-seed", type=int, default=58801)
    parser.add_argument("--neural-path", choices=("direct", "wrapper"), default="direct")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
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
    tx_cpu = cp.asnumpy(receiver.clean_rx_slot)

    grid = []
    for esno_db in OFFICIAL_ESNO_GRID:
        noise_variance = tf.constant(10.0 ** (-esno_db / 10.0), tf.float32)
        records = []
        for iteration in range(args.iterations_per_snr):
            tx_tensor = tf.convert_to_tensor(tx_cpu, dtype=tf.complex64)
            rx_tensor = apply_channel(
                tx_tensor, noise_variance, mapper, remove_guards, channel
            )
            receiver.rx_slot = cp.asfortranarray(cp.asarray(rx_tensor.numpy()))
            receiver.stream.synchronize()
            conventional = receiver.run_conventional()
            neural = (
                receiver.run_neural()
                if args.neural_path == "direct"
                else receiver._measure(receiver.neural_once_wrapper)
            )
            records.append({
                "iteration": iteration,
                "conventional_correct": bool(conventional[1]),
                "neural_correct": bool(neural[1]),
                "conventional_crc_failures": conventional[2],
                "neural_crc_failures": neural[2],
                "conventional_payload_mismatches": conventional[3],
                "neural_payload_mismatches": neural[3],
            })
        grid.append({
            "esno_db": esno_db,
            "iterations": args.iterations_per_snr,
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "records": records,
        })

    total = len(OFFICIAL_ESNO_GRID) * args.iterations_per_snr
    result = {
        "schema": "softwall-sionna-cdl-official-snr-development-v1",
        "analysis_role": (
            "Development screen over the public notebook's declared Es/No range, "
            "using a fixed payload and varying CDL realization; not a BLER holdout."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": "GPU" if gpus else "CPU",
        "channel": "Sionna CDL-A, 100 ns, 0.8333 m/s, normalized, 1x4 uplink",
        "radio_profile": "NVIDIA public NeuralRx notebook MCS7/start0/DMRS0,5,10/TDI1",
        "esno_grid_db": list(OFFICIAL_ESNO_GRID),
        "iterations_per_snr": args.iterations_per_snr,
        "payload_seed": args.payload_seed,
        "channel_seed": args.channel_seed,
        "neural_path": args.neural_path,
        "total_trials": total,
        "conventional_correct": sum(x["conventional_correct"] for x in grid),
        "neural_correct": sum(x["neural_correct"] for x in grid),
        "grid": grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "total_trials": total,
        "conventional_correct": result["conventional_correct"],
        "neural_correct": result["neural_correct"],
        "grid": [{key: row[key] for key in (
            "esno_db", "conventional_correct", "neural_correct"
        )} for row in grid],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
