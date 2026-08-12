#!/usr/bin/env python3
"""Caller-owned TensorRT bindings for the neural receiver."""

from __future__ import annotations

import argparse
import json
import os
import time

import cupy as cp
import numpy as np
import tensorrt as trt


INPUT_VALUES = {
    "rx_slot_real": ((1, 3276, 12, 4), cp.float32),
    "rx_slot_imag": ((1, 3276, 12, 4), cp.float32),
    "h_hat_real": ((1, 4914, 1, 4), cp.float32),
    "h_hat_imag": ((1, 4914, 1, 4), cp.float32),
    "active_dmrs_ports": ((1, 1), cp.float32),
    "dmrs_ofdm_pos": ((1, 3), cp.int32),
    "dmrs_subcarrier_pos": ((1, 6), cp.int32),
}


def stats(values):
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


class DirectNrx:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as stream:
            self.engine = self.runtime.deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.graph = None
        self.inputs = {}
        self.outputs = {}

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = cp.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                expected_shape, expected_dtype = INPUT_VALUES[name]
                if shape != expected_shape or dtype != cp.dtype(expected_dtype):
                    raise RuntimeError(
                        f"input contract mismatch {name}: {shape}/{dtype}"
                    )
                if dtype == cp.float32:
                    value = cp.random.standard_normal(shape, dtype=dtype)
                else:
                    value = cp.zeros(shape, dtype=dtype)
                self.inputs[name] = value
            else:
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"dynamic output unsupported: {name} {shape}")
                self.outputs[name] = cp.empty(shape, dtype=dtype)

        self.inputs["active_dmrs_ports"].fill(1)
        self.inputs["dmrs_ofdm_pos"][:] = cp.asarray([[2, 2, 2]], dtype=cp.int32)
        self.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        )
        self.stream.synchronize()
        for name, value in {**self.inputs, **self.outputs}.items():
            if not self.context.set_tensor_address(name, int(value.data.ptr)):
                raise RuntimeError(f"failed to bind tensor {name}")

    def enqueue(self):
        if not self.context.execute_async_v3(int(self.stream.ptr)):
            raise RuntimeError("TensorRT enqueue failed")
        return self.outputs

    def bind_tensor(self, name, value):
        """Replace an I/O binding with a caller-owned contiguous GPU tensor."""
        tensors = self.inputs if name in self.inputs else self.outputs
        if name not in tensors:
            raise KeyError(f"unknown TensorRT I/O tensor: {name}")
        expected = tensors[name]
        if (
            value.shape != expected.shape
            or value.dtype != expected.dtype
            or not value.flags.c_contiguous
        ):
            raise RuntimeError(
                f"binding contract mismatch {name}: "
                f"{value.shape}/{value.dtype}/C={value.flags.c_contiguous}, "
                f"expected {expected.shape}/{expected.dtype}/C=True"
            )
        if self.graph is not None:
            raise RuntimeError("cannot rebind tensors after CUDA graph capture")
        if not self.context.set_tensor_address(name, int(value.data.ptr)):
            raise RuntimeError(f"failed to bind caller-owned tensor {name}")
        tensors[name] = value

    def run_sync(self):
        output = self.enqueue()
        self.stream.synchronize()
        return output

    def capture_graph(self):
        """Capture one persistent-buffer inference for low-overhead replay."""
        self.run_sync()
        with self.stream:
            self.stream.begin_capture()
            self.enqueue()
            self.graph = self.stream.end_capture()
        self.graph.launch(stream=self.stream)
        self.stream.synchronize()

    def launch(self, use_graph=False):
        if use_graph:
            if self.graph is None:
                raise RuntimeError("CUDA graph has not been captured")
            self.graph.launch(stream=self.stream)
            return self.outputs
        return self.enqueue()


def measure(runtime, iterations, use_graph=False):
    gpu_ms = []
    enqueue_us = []
    wall_ms = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        with runtime.stream:
            start.record()
            enqueue_start = time.perf_counter_ns()
            runtime.launch(use_graph=use_graph)
            enqueue_us.append((time.perf_counter_ns() - enqueue_start) / 1e3)
            end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        gpu_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {
        "gpu_ms": stats(gpu_ms),
        "enqueue_us": stats(enqueue_us),
        "wall_ms": stats(wall_ms),
    }


def compare_with_pycuphy(runtime, engine_path):
    """Check caller-owned TRT buffers against the public pyAerial wrapper."""
    from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms
    from aerial.util.cuda import get_cuda_stream

    stream_handle = get_cuda_stream()
    wrapper_stream = cp.cuda.ExternalStream(int(stream_handle))
    wrapper = TrtEngine(
        trt_model_file=engine_path,
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
            TrtTensorPrms("output_1", (2, 1, 3276, 12), np.float32),
            TrtTensorPrms("output_2", (1, 3276, 12, 8), np.float32),
        ],
    )
    direct_output = runtime.run_sync()
    wrapper_output = wrapper.run(runtime.inputs)
    wrapper_stream.synchronize()
    comparison = {}
    for name in direct_output:
        direct_value = direct_output[name]
        wrapper_value = wrapper_output[name]
        if direct_value.shape != wrapper_value.shape:
            raise RuntimeError(
                f"comparison shape mismatch {name}: "
                f"{direct_value.shape} != {wrapper_value.shape}"
            )
        difference = cp.abs(direct_value - wrapper_value)
        comparison[name] = {
            "max_abs_difference": float(cp.max(difference).item()),
            "mean_abs_difference": float(cp.mean(difference).item()),
            "allclose_rtol_1e-3_atol_1e-3": bool(
                cp.allclose(direct_value, wrapper_value, rtol=1e-3, atol=1e-3).item()
            ),
        }
    if not all(
        item["allclose_rtol_1e-3_atol_1e-3"] for item in comparison.values()
    ):
        raise RuntimeError(f"direct TensorRT and pycuphy outputs differ: {comparison}")
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare-wrapper", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")

    runtime = DirectNrx(args.engine)
    if args.cuda_graph:
        runtime.capture_graph()
    for _ in range(args.warmup):
        runtime.launch(use_graph=args.cuda_graph)
    runtime.stream.synchronize()
    result = {
        "engine": args.engine,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tensor_contract": {
            "inputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in runtime.inputs.items()
            },
            "outputs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in runtime.outputs.items()
            },
        },
        "cuda_graph": args.cuda_graph,
        "direct": measure(runtime, args.iterations, use_graph=args.cuda_graph),
        "output_integrity": {
            name: {
                "finite": bool(cp.all(cp.isfinite(value)).item()),
                "checksum": float(cp.sum(value, dtype=cp.float64).item()),
            }
            for name, value in runtime.outputs.items()
        },
    }
    if args.compare_wrapper:
        result["pycuphy_comparison"] = compare_with_pycuphy(runtime, args.engine)
    if not all(item["finite"] for item in result["output_integrity"].values()):
        raise RuntimeError("non-finite TensorRT output")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(
        f"[NRX-DIRECT] gpu_mean={result['direct']['gpu_ms']['mean']:.6f}ms "
        f"gpu_p99={result['direct']['gpu_ms']['p99']:.6f}ms "
        f"enqueue_mean={result['direct']['enqueue_us']['mean']:.3f}us",
        flush=True,
    )
    print(f"[NRX-DIRECT] OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
