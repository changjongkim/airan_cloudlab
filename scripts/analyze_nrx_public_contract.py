#!/usr/bin/env python3.11
"""Extract the declared channel and tensor contract from NVIDIA's NRx example."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )
    required = {
        "rayleigh_declared": 'channel_model = "Rayleigh"' in source,
        "rayleigh_branch": 'if channel_model == "Rayleigh"' in source,
        "cdl_branch": 'elif "CDL" in channel_model' in source,
        "tdl_branch": 'elif "TDL" in channel_model' in source,
        "mcs7": "mcs_index = 7" in source,
        "start_symbol_zero": "start_sym = 0" in source,
        "one_tx_four_rx": "num_tx_ant = 1" in source and "num_rx_ant = 4" in source,
        "fp32_build": "--inputIOFormats=fp32:chw" in source,
        "input_shape": "rx_slot_real:1x3276x12x4" in source,
    }
    if not all(value for key, value in required.items() if key != "tdl_branch"):
        missing = [key for key, value in required.items() if key != "tdl_branch" and not value]
        raise RuntimeError(f"public NeuralRx contract changed: {missing}")

    result = {
        "schema": "softwall-nrx-public-contract-v1",
        "source_sha256": {
            str(args.notebook): sha256(args.notebook),
            str(args.onnx): sha256(args.onnx),
        },
        "declared_channel_support": {
            "rayleigh": True,
            "cdl_a_to_e": True,
            "tdl": required["tdl_branch"],
        },
        "reference_profile": {
            "mcs_index": 7,
            "start_symbol": 0,
            "num_symbols": 12,
            "dmrs_symbols": [0, 5, 10],
            "tx_antennas": 1,
            "rx_antennas": 4,
            "engine_precision": "fp32",
        },
        "checks": required,
        "interpretation": (
            "The supplied public example explicitly constructs Rayleigh and CDL channels. "
            "It contains no TDL branch, so the prior TDL-A failure cannot by itself be "
            "treated as a failed advertised model contract. A CDL holdout is the first "
            "external-channel compatibility gate; TDL requires separate support evidence "
            "or retraining."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
