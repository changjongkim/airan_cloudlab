#!/usr/bin/env python3
"""Localize NeuralRx compatibility before introducing any channel model."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver


PROFILES = (
    {
        "name": "historical_mcs2_stream4_tx4_direct",
        "mcs_index": 2, "start_sym": 2, "dmrs": (2, 7, 12),
        "num_ul_streams": 4, "transmit_antennas": 4, "path": "direct",
    },
    {
        "name": "reference_mcs7_stream1_tx1_direct",
        "mcs_index": 7, "start_sym": 0, "dmrs": (0, 5, 10),
        "num_ul_streams": 1, "transmit_antennas": 1, "path": "direct",
    },
    {
        "name": "reference_mcs7_stream1_tx1_wrapper",
        "mcs_index": 7, "start_sym": 0, "dmrs": (0, 5, 10),
        "num_ul_streams": 1, "transmit_antennas": 1, "path": "wrapper",
    },
    {
        "name": "reference_mcs7_stream4_tx1_direct",
        "mcs_index": 7, "start_sym": 0, "dmrs": (0, 5, 10),
        "num_ul_streams": 4, "transmit_antennas": 1, "path": "direct",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20358900)
    parser.add_argument("--profile-name", choices=tuple(x["name"] for x in PROFILES))
    args = parser.parse_args()

    profiles = []
    selected_profiles = tuple(
        profile for profile in PROFILES
        if args.profile_name is None or profile["name"] == args.profile_name
    )
    for index, profile in enumerate(selected_profiles):
        dmrs = [0] * 14
        for position in profile["dmrs"]:
            dmrs[position] = 1
        receiver = PairedDualReceiver(
            args.engine,
            seed=args.seed + index,
            mcs_index=profile["mcs_index"],
            start_sym=profile["start_sym"],
            dmrs_syms=dmrs,
            enable_pusch_tdi=1 if profile["mcs_index"] == 7 else 0,
            transmit_antennas=profile["transmit_antennas"],
            direct_nrx_same_stream=True,
            num_ul_streams=profile["num_ul_streams"],
        )
        if profile["transmit_antennas"] == 1:
            # No fading object is present in this matrix, so explicitly apply
            # a deterministic unit-gain 1x4 channel before the four-antenna
            # receiver.  Feeding the one-antenna Tx grid directly is invalid.
            receiver.clean_rx_slot = cp.asfortranarray(
                cp.repeat(receiver.clean_rx_slot, 4, axis=2)
            )
            receiver.rx_slot = receiver.clean_rx_slot
            receiver.stream.synchronize()
        records = []
        for iteration in range(args.iterations):
            receiver.restore_clean_slot()
            conventional = receiver.run_conventional()
            neural = (
                receiver.run_neural()
                if profile["path"] == "direct"
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
        profiles.append({
            **profile,
            "dmrs": list(profile["dmrs"]),
            "conventional_correct": sum(x["conventional_correct"] for x in records),
            "neural_correct": sum(x["neural_correct"] for x in records),
            "records": records,
        })

    result = {
        "schema": "softwall-nrx-reference-clean-matrix-v1",
        "analysis_role": "Post-failure model/interface localization with no channel or noise.",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations_per_profile": args.iterations,
        "seed": args.seed,
        "requested_profile": args.profile_name,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({row["name"]: {
        "conventional_correct": row["conventional_correct"],
        "neural_correct": row["neural_correct"],
    } for row in profiles}, indent=2), flush=True)


if __name__ == "__main__":
    main()
