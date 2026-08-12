#!/usr/bin/env python3
"""Decompose pyAerial neural-receiver latency into wrapper and raw stages."""

from __future__ import annotations

import argparse
import json
import os
import time

import cupy as cp
import numpy as np
from cuda.bindings import runtime as cudart

from aerial import pycuphy
from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms
from aerial.util.cuda import get_cuda_stream


RX_SHAPE = (1, 3276, 12, 4)
H_SHAPE = (1, 4914, 1, 4)
OUTPUT_2_SHAPE = (1, 3276, 12, 8)


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def measure(function, stream, iterations):
    gpu_ms = []
    wall_ms = []
    last = None
    for _ in range(iterations):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        with stream:
            start_event.record()
            last = function()
            end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        gpu_ms.append(float(cp.cuda.get_elapsed_time(start_event, end_event)))
    return {"gpu_ms": summary(gpu_ms), "wall_ms": summary(wall_ms)}, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output-bits", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    cudart.cudaSetDevice(0)
    stream_handle = get_cuda_stream()
    stream = cp.cuda.ExternalStream(int(stream_handle))
    engine = TrtEngine(
        trt_model_file=args.engine,
        max_batch_size=1,
        cuda_stream=stream_handle,
        input_tensors=[
            TrtTensorPrms("rx_slot_real", (3276, 12, 4), np.float32),
            TrtTensorPrms("rx_slot_imag", (3276, 12, 4), np.float32),
            TrtTensorPrms("h_hat_real", (4914, 1, 4), np.float32),
            TrtTensorPrms("h_hat_imag", (4914, 1, 4), np.float32),
            TrtTensorPrms("active_dmrs_ports", (1,), np.float32),
            TrtTensorPrms("dmrs_ofdm_pos", (3,), np.int32),
            TrtTensorPrms("dmrs_subcarrier_pos", (6,), np.int32),
        ],
        output_tensors=[
            TrtTensorPrms(
                "output_1", (args.output_bits, 1, 3276, 12), np.float32
            ),
            TrtTensorPrms("output_2", OUTPUT_2_SHAPE, np.float32),
        ],
    )

    with stream:
        inputs_c = {
            "rx_slot_real": cp.random.standard_normal(RX_SHAPE, dtype=cp.float32),
            "rx_slot_imag": cp.random.standard_normal(RX_SHAPE, dtype=cp.float32),
            "h_hat_real": cp.random.standard_normal(H_SHAPE, dtype=cp.float32),
            "h_hat_imag": cp.random.standard_normal(H_SHAPE, dtype=cp.float32),
            "active_dmrs_ports": cp.ones((1, 1), dtype=cp.float32),
            "dmrs_ofdm_pos": cp.asarray([[2, 2, 2]], dtype=cp.int32),
            "dmrs_subcarrier_pos": cp.asarray(
                [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
            ),
        }
        inputs_f = {name: cp.asfortranarray(value) for name, value in inputs_c.items()}
    stream.synchronize()
    wrapped_f = {
        name: (
            pycuphy.CudaArrayFloat(value)
            if value.dtype == cp.float32
            else pycuphy.CudaArrayInt(value)
        )
        for name, value in inputs_f.items()
    }

    def pack_c_to_f():
        return {name: cp.array(value, order="F") for name, value in inputs_c.items()}

    def raw_run():
        return engine.trt_engine.run(wrapped_f)

    def raw_plus_output_conversion():
        output = engine.trt_engine.run(wrapped_f)
        return {
            name: cp.array(output[name]).astype(engine.output_data_types[index])
            for index, name in enumerate(engine.output_names)
        }

    def public_c():
        return engine.run(inputs_c)

    def public_f():
        return engine.run(inputs_f)

    for _ in range(args.warmup):
        raw_plus_output_conversion()
    stream.synchronize()

    metrics = {}
    outputs = {}
    for name, function in (
        ("pack_c_to_f", pack_c_to_f),
        ("raw_pycuphy", raw_run),
        ("raw_plus_output_conversion", raw_plus_output_conversion),
        ("public_wrapper_c_input", public_c),
        ("public_wrapper_f_input", public_f),
    ):
        cp.cuda.nvtx.RangePush(name)
        metrics[name], outputs[name] = measure(function, stream, args.iterations)
        cp.cuda.nvtx.RangePop()
        print(
            f"[NRX-PROFILE] {name} "
            f"gpu_mean={metrics[name]['gpu_ms']['mean']:.3f}ms "
            f"wall_mean={metrics[name]['wall_ms']['mean']:.3f}ms",
            flush=True,
        )

    raw_converted = outputs["raw_plus_output_conversion"]
    public_output = outputs["public_wrapper_f_input"]
    correctness = {}
    for name in engine.output_names:
        raw_array = raw_converted[name]
        public_array = public_output[name]
        if raw_array.shape != public_array.shape:
            raise RuntimeError(
                f"output shape mismatch {name}: {raw_array.shape} != {public_array.shape}"
            )
        difference = cp.max(cp.abs(raw_array - public_array)).item()
        correctness[name] = {
            "shape": list(raw_array.shape),
            "dtype": str(raw_array.dtype),
            "max_abs_difference": float(difference),
        }

    result = {
        "engine": args.engine,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "output_bits": args.output_bits,
        "cuda_device_count": int(cp.cuda.runtime.getDeviceCount()),
        "input_layout": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "c_contiguous": bool(value.flags.c_contiguous),
                "f_contiguous": bool(value.flags.f_contiguous),
            }
            for name, value in inputs_f.items()
        },
        "metrics": metrics,
        "correctness": correctness,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream_out:
        json.dump(result, stream_out, indent=2)
    print(f"[NRX-PROFILE] OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
