#!/usr/bin/env python3
"""Export the exact C165 clean-PUSCH contract for a native C++ replay.

This fixture is a parity input, not a timing result.  It carries both the
current real/imag C-order P2P payload and a direct complex Fortran-layout slot
so the native implementation cannot silently change the tensor contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cupy as cp
import numpy as np

from dual_receiver_phy import PairedDualReceiver


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20359400)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with cp.cuda.Device(args.device):
        receiver = PairedDualReceiver(
            str(args.engine),
            seed=args.seed,
            device=args.device,
            direct_nrx_same_stream=True,
        )
        # Exercise every lazy receiver object before exporting the immutable
        # fixture, then require both branches to decode the same TB.
        conventional = receiver.run_conventional()
        neural = receiver.run_neural()
        if not conventional[1] or not neural[1]:
            raise RuntimeError(
                f"fixture parity decode failed conventional={conventional} neural={neural}"
            )
        rx_slot = np.asarray(cp.asnumpy(receiver.rx_slot), dtype=np.complex64)

    if rx_slot.shape != (3276, 14, 4):
        raise RuntimeError(f"unexpected slot shape {rx_slot.shape}")
    complex_path = args.output_dir / "rx_slot_complex64_fortran.bin"
    p2p_path = args.output_dir / "rx_slot_real_imag_float32_c.bin"
    tb_path = args.output_dir / "reference_tb_uint8.bin"

    np.asfortranarray(rx_slot).ravel(order="F").tofile(complex_path)
    np.concatenate(
        (
            np.asarray(rx_slot.real, dtype=np.float32).ravel(order="C"),
            np.asarray(rx_slot.imag, dtype=np.float32).ravel(order="C"),
        )
    ).tofile(p2p_path)
    np.asarray(receiver.reference_tb, dtype=np.uint8).tofile(tb_path)

    source = Path(__file__).resolve()
    dual_receiver = source.with_name("dual_receiver_phy.py")
    manifest = {
        "schema": "softwall-c166-native-fixture-v1",
        "role": (
            "Exact Python-to-native parity fixture for C165; no latency, "
            "deadline, WCET, or production qualification claim."
        ),
        "seed": args.seed,
        "device": args.device,
        "engine": {
            "path_at_export": str(args.engine),
            "sha256": sha256(args.engine),
        },
        "radio": {
            "slot": int(receiver.slot),
            "start_symbol": int(receiver.start_sym),
            "num_symbols": int(receiver.num_symbols),
            "dmrs_symbol_bitmap": [int(value) for value in receiver.dmrs_syms],
            "mcs_index": int(receiver.mcs_index),
            "mod_order": int(receiver.mod_order),
            "code_rate_x10": int(receiver.code_rate),
            "num_prbs": 273,
            "start_prb": 0,
            "num_layers": 1,
            "num_rx_antennas": 4,
            "num_uplink_streams": int(receiver.num_ul_streams),
            "num_dmrs_cdm_groups_without_data": 2,
            "dmrs_scrambling_id": 41,
            "scid": 0,
            "rnti": 1234,
            "data_scrambling_id": 0,
            "channel_estimation_algorithm": 3,
            "conventional_channel_estimation_algorithm": 1,
            "equalizer_coefficient_algorithm": 1,
        },
        "fixture_decode": {
            "conventional_correct": bool(conventional[1]),
            "neural_correct": bool(neural[1]),
            "reference_tb_bytes": int(np.asarray(receiver.reference_tb).size),
        },
        "buffers": {
            "complex_fortran": {
                "file": complex_path.name,
                "shape": list(rx_slot.shape),
                "dtype": "complex64",
                "order": "F",
                "bytes": complex_path.stat().st_size,
                "sha256": sha256(complex_path),
            },
            "p2p_real_imag": {
                "file": p2p_path.name,
                "logical_shape": list(rx_slot.shape),
                "dtype": "float32",
                "order_per_plane": "C",
                "plane_order": ["real", "imag"],
                "elements_per_plane": int(rx_slot.size),
                "bytes": p2p_path.stat().st_size,
                "sha256": sha256(p2p_path),
            },
            "reference_tb": {
                "file": tb_path.name,
                "dtype": "uint8",
                "bytes": tb_path.stat().st_size,
                "sha256": sha256(tb_path),
            },
        },
        "source_sha256": {
            source.name: sha256(source),
            dual_receiver.name: sha256(dual_receiver),
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
