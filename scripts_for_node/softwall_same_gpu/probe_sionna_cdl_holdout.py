#!/usr/bin/env python3
"""Paired, disjoint-seed BLER holdout for the screened CDL-D/E domain."""

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
from probe_sionna_cdl_family_development import build_channel
from probe_sionna_cdl_reference_development import apply_channel


ESNO_GRID_DB = (-4.0, -3.6, -3.2, -3.0, 10.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("D", "E"), required=True)
    parser.add_argument("--iterations-per-snr", type=int, default=50)
    parser.add_argument("--payload-seed", type=int, required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
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
    mapper, remove_guards, channel = build_channel(args.model, 100.0)
    strata = []
    global_index = 0
    for esno_db in ESNO_GRID_DB:
        noise_variance = tf.constant(10.0 ** (-esno_db / 10.0), tf.float32)
        records = []
        for iteration in range(args.iterations_per_snr):
            slot = global_index % 20
            receiver.regenerate_clean_slot(slot)
            tx_cpu = cp.asnumpy(receiver.clean_rx_slot)
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
                "global_index": global_index,
                "iteration": iteration,
                "slot": slot,
                "features": features,
                "conventional_correct": bool(conventional[1]),
                "neural_correct": bool(neural[1]),
                "conventional_crc_failures": conventional[2],
                "neural_crc_failures": neural[2],
                "conventional_payload_mismatches": conventional[3],
                "neural_payload_mismatches": neural[3],
            })
            global_index += 1
        strata.append({
            "esno_db": esno_db,
            "trials": len(records),
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "both_correct": sum(
                x["conventional_correct"] and x["neural_correct"] for x in records
            ),
            "neural_only_correct": sum(
                (not x["conventional_correct"]) and x["neural_correct"] for x in records
            ),
            "conventional_only_correct": sum(
                x["conventional_correct"] and (not x["neural_correct"]) for x in records
            ),
            "neither_correct": sum(
                (not x["conventional_correct"]) and (not x["neural_correct"]) for x in records
            ),
            "records": records,
        })

    total = len(ESNO_GRID_DB) * args.iterations_per_snr
    result = {
        "schema": "softwall-sionna-cdl-holdout-v1",
        "analysis_role": (
            "Prespecified disjoint-seed paired BLER holdout for the CDL-D/E domain "
            "selected by a separately preserved development screen."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "tensorflow_version": tf.__version__,
        "sionna_version": sionna.__version__,
        "tensorflow_channel_device": "CPU",
        "cdl_model": args.model,
        "delay_spread_ns": 100.0,
        "speed_mps": 0.8333,
        "radio_profile": "MCS7/start0/DMRS0,5,10/TDI1/one TX/four RX/FP32 engine",
        "esno_grid_db": list(ESNO_GRID_DB),
        "iterations_per_snr": args.iterations_per_snr,
        "payload_seed": args.payload_seed,
        "channel_seed": args.channel_seed,
        "payload_and_slot_vary_per_trial": True,
        "total_trials": total,
        "conventional_correct": sum(row["conventional_correct"] for row in strata),
        "neural_correct": sum(row["neural_correct"] for row in strata),
        "neural_only_correct": sum(row["neural_only_correct"] for row in strata),
        "conventional_only_correct": sum(row["conventional_only_correct"] for row in strata),
        "strata": strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "cdl_model": args.model,
        "total_trials": total,
        "conventional_correct": result["conventional_correct"],
        "neural_correct": result["neural_correct"],
        "neural_only_correct": result["neural_only_correct"],
        "strata": [{key: row[key] for key in (
            "esno_db", "conventional_correct", "neural_correct",
            "neural_only_correct", "conventional_only_correct"
        )} for row in strata],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
