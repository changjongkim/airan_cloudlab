#!/usr/bin/env python3
"""Regular NRx stack experiment with identical tensors and correctness gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import cupy as cp
import numpy as np

from aerial.phy5g.algorithms import TrtEngine, TrtTensorPrms
from aerial.util.cuda import get_cuda_stream
from nrx_trt_direct import DirectNrx


OUTPUT_BITS = 2
INPUT_SPECS = (
    ("rx_slot_real", (3276, 12, 4), np.float32),
    ("rx_slot_imag", (3276, 12, 4), np.float32),
    ("h_hat_real", (4914, 1, 4), np.float32),
    ("h_hat_imag", (4914, 1, 4), np.float32),
    ("active_dmrs_ports", (1,), np.float32),
    ("dmrs_ofdm_pos", (3,), np.int32),
    ("dmrs_subcarrier_pos", (6,), np.int32),
)


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def canonical_inputs(seed: int) -> dict[str, cp.ndarray]:
    rng = cp.random.RandomState(seed)
    values = {
        "rx_slot_real": rng.standard_normal((1, 3276, 12, 4), dtype=cp.float32),
        "rx_slot_imag": rng.standard_normal((1, 3276, 12, 4), dtype=cp.float32),
        "h_hat_real": rng.standard_normal((1, 4914, 1, 4), dtype=cp.float32),
        "h_hat_imag": rng.standard_normal((1, 4914, 1, 4), dtype=cp.float32),
        "active_dmrs_ports": cp.ones((1, 1), dtype=cp.float32),
        "dmrs_ofdm_pos": cp.asarray([[2, 2, 2]], dtype=cp.int32),
        "dmrs_subcarrier_pos": cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        ),
    }
    cp.cuda.get_current_stream().synchronize()
    for name, value in values.items():
        if not value.flags.c_contiguous:
            raise RuntimeError(f"canonical input is not C contiguous: {name}")
    return values


def build_wrapper(engine_path: str) -> tuple[TrtEngine, cp.cuda.ExternalStream]:
    stream_handle = get_cuda_stream()
    stream = cp.cuda.ExternalStream(int(stream_handle))
    engine = TrtEngine(
        trt_model_file=engine_path,
        max_batch_size=1,
        cuda_stream=stream_handle,
        input_tensors=[TrtTensorPrms(*spec) for spec in INPUT_SPECS],
        output_tensors=[
            TrtTensorPrms("output_1", (OUTPUT_BITS, 1, 3276, 12), np.float32),
            TrtTensorPrms("output_2", (1, 3276, 12, 8), np.float32),
        ],
    )
    return engine, stream


def measure_wrapper(
    engine: TrtEngine,
    stream: cp.cuda.ExternalStream,
    inputs: dict[str, cp.ndarray],
    warmup: int,
    iterations: int,
) -> tuple[dict, dict[str, cp.ndarray]]:
    last = None
    for _ in range(warmup):
        last = engine.run(inputs)
    stream.synchronize()
    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        with stream:
            start.record()
            last = engine.run(inputs)
            end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        gpu_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
    assert last is not None
    reference = {name: cp.array(value, copy=True) for name, value in last.items()}
    return {
        "gpu_ms": summarize(gpu_ms),
        "wall_ms": summarize(wall_ms),
        "raw_gpu_ms": gpu_ms,
        "raw_wall_ms": wall_ms,
    }, reference


def configure_direct(
    engine_path: str,
    inputs: dict[str, cp.ndarray],
    caller_input: bool,
    caller_output: bool,
) -> tuple[DirectNrx, dict[str, cp.ndarray]]:
    runtime = DirectNrx(engine_path)
    if caller_input:
        for name, value in inputs.items():
            runtime.bind_tensor(name, value)
    else:
        with runtime.stream:
            for name, value in inputs.items():
                cp.copyto(runtime.inputs[name], value)
        runtime.stream.synchronize()
    outputs: dict[str, cp.ndarray] = {}
    if caller_output:
        for name, value in runtime.outputs.items():
            outputs[name] = cp.empty_like(value)
            runtime.bind_tensor(name, outputs[name])
    return runtime, outputs


def compare_outputs(
    observed: dict[str, cp.ndarray], reference: dict[str, cp.ndarray]
) -> dict[str, dict[str, float | bool | list[int] | str]]:
    comparison = {}
    for name, expected in reference.items():
        actual = observed[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(
                f"output contract mismatch {name}: "
                f"{actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}"
            )
        difference = cp.abs(actual - expected)
        item = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "max_abs_difference": float(cp.max(difference).item()),
            "mean_abs_difference": float(cp.mean(difference).item()),
            "allclose_rtol_1e-3_atol_1e-3": bool(
                cp.allclose(actual, expected, rtol=1e-3, atol=1e-3).item()
            ),
        }
        if not item["allclose_rtol_1e-3_atol_1e-3"]:
            raise RuntimeError(f"output mismatch for {name}: {item}")
        comparison[name] = item
    return comparison


def measure_direct(
    runtime: DirectNrx, warmup: int, iterations: int, graph: bool
) -> dict:
    if graph:
        runtime.capture_graph()
    for _ in range(warmup):
        runtime.launch(use_graph=graph)
    runtime.stream.synchronize()
    gpu_ms: list[float] = []
    wall_ms: list[float] = []
    enqueue_us: list[float] = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_start = time.perf_counter_ns()
        with runtime.stream:
            start.record()
            enqueue_start = time.perf_counter_ns()
            runtime.launch(use_graph=graph)
            enqueue_us.append((time.perf_counter_ns() - enqueue_start) / 1e3)
            end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        gpu_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
    return {
        "gpu_ms": summarize(gpu_ms),
        "wall_ms": summarize(wall_ms),
        "enqueue_us": summarize(enqueue_us),
        "raw_gpu_ms": gpu_ms,
        "raw_wall_ms": wall_ms,
        "raw_enqueue_us": enqueue_us,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.trial <= 0:
        parser.error("iterations and trial must be positive; warmup non-negative")

    inputs = canonical_inputs(0xD47A + args.trial)
    result = {
        "schema_version": 1,
        "trial": args.trial,
        "engine": args.engine,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "input_pointers": {name: int(value.data.ptr) for name, value in inputs.items()},
        "variants": {},
    }

    wrapper, wrapper_stream = build_wrapper(args.engine)
    wrapper_pointers_before = {
        name: int(value.data.ptr) for name, value in inputs.items()
    }
    metrics, reference = measure_wrapper(
        wrapper, wrapper_stream, inputs, args.warmup, args.iterations
    )
    wrapper_pointers_after = {
        name: int(value.data.ptr) for name, value in inputs.items()
    }
    if wrapper_pointers_before != wrapper_pointers_after:
        raise RuntimeError("caller input pointers changed during wrapper trial")
    result["variants"]["N00_public_wrapper"] = {
        "metrics": metrics,
        "caller_input_pointer_stable": True,
        "reference": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(cp.all(cp.isfinite(value)).item()),
                "checksum": float(cp.sum(value, dtype=cp.float64).item()),
            }
            for name, value in reference.items()
        },
    }
    print(
        f"[NRX-FULL] trial={args.trial} N00_public_wrapper "
        f"gpu_mean={metrics['gpu_ms']['mean']:.6f}ms",
        flush=True,
    )
    del wrapper
    gc.collect()

    configurations = (
        ("N01_raw_tensorrt", False, False, False),
        ("N02_caller_input", True, False, False),
        ("N03_caller_io", True, True, False),
        ("N04_cuda_graph", True, True, True),
    )
    for name, caller_input, caller_output, graph in configurations:
        runtime, owned_outputs = configure_direct(
            args.engine, inputs, caller_input, caller_output
        )
        pointers_before = {
            "inputs": {key: int(value.data.ptr) for key, value in runtime.inputs.items()},
            "outputs": {key: int(value.data.ptr) for key, value in runtime.outputs.items()},
        }
        metrics = measure_direct(runtime, args.warmup, args.iterations, graph)
        pointers_after = {
            "inputs": {key: int(value.data.ptr) for key, value in runtime.inputs.items()},
            "outputs": {key: int(value.data.ptr) for key, value in runtime.outputs.items()},
        }
        if pointers_before != pointers_after:
            raise RuntimeError(f"TensorRT bindings changed during {name}")
        comparison = compare_outputs(runtime.outputs, reference)
        result["variants"][name] = {
            "metrics": metrics,
            "caller_input": caller_input,
            "caller_output": caller_output,
            "cuda_graph": graph,
            "binding_pointers_stable": True,
            "correctness": comparison,
        }
        print(
            f"[NRX-FULL] trial={args.trial} {name} "
            f"gpu_mean={metrics['gpu_ms']['mean']:.6f}ms "
            f"p99={metrics['gpu_ms']['p99']:.6f}ms",
            flush=True,
        )
        del owned_outputs, runtime
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    result["pass"] = True
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
    os.replace(temporary, args.output)
    print(f"[NRX-FULL] PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
