"""TensorRT runner for an exported NVlabs neural receiver (any bandwidth / user count).

Same caller-owned-buffer and CUDA-graph mechanics as ``nrx_trt_direct.DirectNrx``; the input
shapes come from the engine and the DMRS layout from the export's JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import cupy as cp
import tensorrt as trt

from nrx_trt_direct import DirectNrx


class NvNrx(DirectNrx):
    def __init__(self, engine_path: str, meta_path: str | None = None, *, stream=None) -> None:
        self.meta = json.loads(Path(meta_path or str(engine_path).rsplit(".", 1)[0] + ".json").read_text())
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = stream if stream is not None else cp.cuda.Stream(non_blocking=True)
        self.caller_owned_stream = stream is not None
        self.graph = None
        self.inputs, self.outputs = {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = cp.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"build the engine with fixed shapes: {name} {shape}")
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = cp.zeros(shape, dtype=dtype)
            else:
                self.outputs[name] = cp.empty(shape, dtype=dtype)
        self.inputs["active_dmrs_ports"].fill(1)
        self.inputs["dmrs_ofdm_pos"][:] = cp.asarray(self.meta["dmrs_ofdm_pos"], dtype=cp.int32)
        self.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(self.meta["dmrs_subcarrier_pos"], dtype=cp.int32)
        self.stream.synchronize()
        for name, value in {**self.inputs, **self.outputs}.items():
            if not self.context.set_tensor_address(name, int(value.data.ptr)):
                raise RuntimeError(f"failed to bind tensor {name}")
