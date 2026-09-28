#!/usr/bin/env python3
"""GPU-resident service padding for the C168 bound-realization diagnostic.

The kernel reads CUDA's device-wide nanosecond timer.  C168 uses it only to
realize an already-declared service bound in a diagnostic arm; it is not a
production delay mechanism and its samples cannot qualify a service bound.
"""

from __future__ import annotations

import time

import cupy as cp
import numpy as np


_SOURCE = r'''
extern "C" __global__ void c168_spin_ns(unsigned long long duration_ns) {
    unsigned long long begin;
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - begin < duration_ns);
}
'''


class GpuBoundPad:
    """Pad a same-host transaction to just below a declared upper bound."""

    def __init__(self, device: int = 0, margin_ms: float = 0.08) -> None:
        if margin_ms <= 0:
            raise ValueError("margin_ms must be positive")
        self.device = int(device)
        self.margin_ms = float(margin_ms)
        with cp.cuda.Device(self.device):
            self.kernel = cp.RawKernel(_SOURCE, "c168_spin_ns")
            # Compile and establish the launch path before any timed request.
            self.kernel((1,), (1,), (np.uint64(1_000),))
            cp.cuda.get_current_stream().synchronize()

    def pad_elapsed(self, started_ns: int, bound_ms: float) -> dict:
        """Keep the GPU occupied until close to ``started + bound``.

        A small frozen margin covers host launch and synchronization overhead.
        The caller records the complete path after this method returns and the
        analyzer checks both lower accuracy and the declared upper bound.
        """
        if bound_ms <= self.margin_ms:
            raise ValueError("bound_ms must exceed margin_ms")
        before_ns = time.perf_counter_ns()
        desired_ns = int(started_ns + (bound_ms - self.margin_ms) * 1e6)
        requested_ns = max(0, desired_ns - before_ns)
        gpu_ms = 0.0
        if requested_ns > 0:
            with cp.cuda.Device(self.device):
                begin = cp.cuda.Event()
                end = cp.cuda.Event()
                begin.record()
                self.kernel((1,), (1,), (np.uint64(requested_ns),))
                end.record()
                end.synchronize()
                gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
        returned_ns = time.perf_counter_ns()
        return {
            "bound_ms": float(bound_ms),
            "margin_ms": self.margin_ms,
            "pre_pad_elapsed_ms": (before_ns - started_ns) / 1e6,
            "requested_spin_ms": requested_ns / 1e6,
            "spin_gpu_ms": gpu_ms,
            "post_pad_elapsed_ms": (returned_ns - started_ns) / 1e6,
        }

