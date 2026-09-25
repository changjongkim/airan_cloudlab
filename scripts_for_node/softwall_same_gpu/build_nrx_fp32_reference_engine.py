#!/usr/bin/env python3
"""Build the public NeuralRx ONNX in FP32 using TensorRT's Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import tensorrt as trt


DYNAMIC_SHAPES = {
    "rx_slot_real": (1, 3276, 12, 4),
    "rx_slot_imag": (1, 3276, 12, 4),
    "h_hat_real": (1, 4914, 1, 4),
    "h_hat_imag": (1, 4914, 1, 4),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()

    cp.cuda.runtime.setDevice(0)
    cp.cuda.runtime.free(0)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser_impl = trt.OnnxParser(network, logger)
    if not parser_impl.parse(args.onnx.read_bytes()):
        errors = [str(parser_impl.get_error(i)) for i in range(parser_impl.num_errors)]
        raise RuntimeError("ONNX parse failed: " + " | ".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    profile = builder.create_optimization_profile()
    for name, shape in DYNAMIC_SHAPES.items():
        # TensorRT 10.12's Python binding mutates the profile and returns None.
        profile.set_shape(name, shape, shape, shape)
    if not profile:
        raise RuntimeError("invalid TensorRT optimization profile")
    config.add_optimization_profile(profile)
    # Intentionally do not set BuilderFlag.FP16. This matches the public
    # notebook's precision contract rather than the historical SoftWall build.
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no serialized engine")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f"wrote {args.engine} ({args.engine.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
