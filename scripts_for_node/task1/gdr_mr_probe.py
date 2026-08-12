#!/usr/bin/env python3
"""Minimal, non-transfer GPUDirect RDMA memory-registration probe.

The probe allocates CUDA device memory with CuPy, enables CUDA synchronous
memory-operation semantics, and registers that exact device virtual address as
an ibverbs MR.  It deliberately posts no work requests and never accesses the
GPU-backed MR through the CPU.

References:
  - rdma-core v57 ``MR(address=...)`` forwards the supplied address to
    ``ibv_reg_mr``.
  - NVIDIA's GPUDirect RDMA guide requires ``SYNC_MEMOPS`` for CUDA/RDMA memory
    ordering: https://docs.nvidia.com/cuda/gpudirect-rdma/

Run inside ``airan:25-3-rdma`` with one MIG device selected through Docker's
``--gpus`` option.  Example:

    python3 gdr_mr_probe.py             # 64 KiB
    python3 gdr_mr_probe.py 1048576     # 1 MiB

Environment:
    CUDA_DEVICE       CUDA ordinal inside the container (default: 0)
    RDMA_DEV          ibverbs device (default: mlx5_0)
    RDMA_IB_PORT      RDMA port (default: 1)
    RDMA_GID_INDEX    GID table index to query (default: 3)
"""

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import sys
import traceback

import cupy as cp
from cuda.bindings import driver as cuda_driver
import pyverbs.device as vdev
from pyverbs.enums import (
    IBV_ACCESS_LOCAL_WRITE,
    IBV_ACCESS_REMOTE_READ,
    IBV_ACCESS_REMOTE_WRITE,
)
from pyverbs.mr import MR
from pyverbs.pd import PD


DEFAULT_SIZE = 64 * 1024
PEERMEM_COUNTER_DIR = Path("/sys/kernel/mm/memory_peers/nv_mem")
PEERMEM_COUNTER_NAMES = (
    "version",
    "num_alloc_mrs",
    "num_dealloc_mrs",
    "num_reg_pages",
    "num_dereg_pages",
    "num_reg_bytes",
    "num_dereg_bytes",
    "num_free_callbacks",
)


def positive_int(value):
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("size must be a positive integer")
    return parsed


def read_text(path):
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError as exc:
        return f"unavailable:{exc.errno}"


def read_peermem_counters():
    return {
        name: read_text(PEERMEM_COUNTER_DIR / name)
        for name in PEERMEM_COUNTER_NAMES
    }


def print_counters(stage):
    counters = read_peermem_counters()
    print(
        f"PEERMEM_COUNTERS_{stage}="
        f"{json.dumps(counters, sort_keys=True)}",
        flush=True,
    )
    return counters


def check_iommu(rdma_dev):
    """Warn about translated IOMMU domains without changing host policy."""
    group = Path(f"/sys/class/infiniband/{rdma_dev}/device/iommu_group")
    iommu_type = read_text(group / "type") if group.exists() else "none"
    cmdline = read_text(Path("/proc/cmdline"))
    print(f"IOMMU_STATE type={iommu_type} cmdline={cmdline}", flush=True)

    # Linux reports translated domains as DMA or DMA-FQ; GPUDirect RDMA needs
    # identity/pass-through addressing.  This is a warning so registration can
    # still provide the definitive gate result on the installed stack.
    if iommu_type.upper().startswith("DMA"):
        print(
            "IOMMU_WARNING: RDMA device is in a translated "
            f"{iommu_type} domain; GPUDirect RDMA normally requires "
            "identity/pass-through mapping and MR registration may fail",
            flush=True,
        )


def cuda_check(result, operation):
    status = result[0]
    if status != cuda_driver.CUresult.CUDA_SUCCESS:
        name = getattr(status, "name", str(status))
        raise RuntimeError(f"{operation} failed: {name} ({int(status)})")
    return result[1:]


def set_sync_memops(ptr):
    attribute = cuda_driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    # cuda-bindings 12.9 converts this Python bool to the unsigned/bool storage
    # expected by cuPointerSetAttribute; passing a ctypes pointer is unnecessary.
    cuda_check(
        cuda_driver.cuPointerSetAttribute(True, attribute, ptr),
        "cuPointerSetAttribute(SYNC_MEMOPS)",
    )
    values = cuda_check(
        cuda_driver.cuPointerGetAttribute(attribute, ptr),
        "cuPointerGetAttribute(SYNC_MEMOPS)",
    )
    enabled = bool(values[0])
    if not enabled:
        raise RuntimeError("CUDA SYNC_MEMOPS did not remain enabled")
    print("CUDA_SYNC_MEMOPS enabled=1", flush=True)


def report_gdr_pointer_capability(ptr):
    name = "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE"
    if not hasattr(cuda_driver.CUpointer_attribute, name):
        print("CUDA_GDR_POINTER_CAPABLE=unavailable", flush=True)
        return
    attribute = getattr(cuda_driver.CUpointer_attribute, name)
    values = cuda_check(
        cuda_driver.cuPointerGetAttribute(attribute, ptr),
        f"cuPointerGetAttribute({name})",
    )
    print(f"CUDA_GDR_POINTER_CAPABLE={int(bool(values[0]))}", flush=True)


def close_resource(resource_obj, name, cleanup_errors):
    if resource_obj is None:
        return
    try:
        resource_obj.close()
        print(f"RESOURCE_CLOSED name={name}", flush=True)
    except Exception as exc:  # Preserve later cleanup while making failure fatal.
        cleanup_errors.append(f"{name}.close: {exc}")


def run_probe(size):
    rdma_dev = os.environ.get("RDMA_DEV", "mlx5_0")
    ib_port = int(os.environ.get("RDMA_IB_PORT", "1"))
    gid_index = int(os.environ.get("RDMA_GID_INDEX", "3"))
    cuda_device = int(os.environ.get("CUDA_DEVICE", "0"))
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    print(
        "GDR_PROBE_CONFIG "
        f"size={size} cuda_device={cuda_device} "
        f"cuda_visible_devices={cuda_visible} rdma_dev={rdma_dev} "
        f"ib_port={ib_port} gid_index={gid_index}",
        flush=True,
    )
    soft_memlock, hard_memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(
        f"MEMLOCK_LIMIT soft={soft_memlock} hard={hard_memlock}", flush=True
    )
    check_iommu(rdma_dev)
    print_counters("BEFORE")

    ctx = None
    pd = None
    mr = None
    gpu_memory = None
    cleanup_errors = []
    try:
        device = cp.cuda.Device(cuda_device)
        device.use()
        props = cp.cuda.runtime.getDeviceProperties(cuda_device)
        gpu_name = props.get("name", b"<unknown>")
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")
        print(f"CUDA_DEVICE_SELECTED name={gpu_name}", flush=True)

        # CuPy Memory is a direct RAII cudaMalloc allocation, not a NumPy or
        # host-pinned buffer.  Keep this object alive until after MR teardown.
        gpu_memory = cp.cuda.Memory(size)
        ptr = int(gpu_memory.ptr)
        cp.cuda.runtime.memset(ptr, 0, size)
        cp.cuda.runtime.deviceSynchronize()
        print(f"CUDA_ALLOCATION ptr=0x{ptr:x} size={size}", flush=True)

        set_sync_memops(ptr)
        report_gdr_pointer_capability(ptr)

        ctx = vdev.Context(name=rdma_dev)
        gid = ctx.query_gid(ib_port, gid_index)
        print(
            f"RDMA_CONTEXT dev={rdma_dev} gid={gid} "
            f"port={ib_port} gid_index={gid_index}",
            flush=True,
        )
        pd = PD(ctx)

        access = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        # pyverbs v57 casts address through uintptr_t and calls ibv_reg_mr.
        # Do not call MR.read()/write(): they use CPU memcpy on this GPU VA.
        mr = MR(pd, size, access, address=ptr)
        print(
            "GPU_MR_REGISTERED "
            f"addr=0x{mr.buf:x} length={mr.length} "
            f"lkey=0x{mr.lkey:x} rkey=0x{mr.rkey:x}",
            flush=True,
        )
        print_counters("DURING")

    finally:
        # Dependency order is intentional: deregister/unpin before releasing
        # the PD/context, and only then release the CUDA allocation.
        close_resource(mr, "mr", cleanup_errors)
        if mr is not None:
            mr = None
        print_counters("AFTER")
        close_resource(pd, "pd", cleanup_errors)
        pd = None
        close_resource(ctx, "ctx", cleanup_errors)
        ctx = None

        if gpu_memory is not None:
            gpu_memory = None
            gc.collect()
            print("CUDA_ALLOCATION_RELEASED after_mr_close=1", flush=True)

    if cleanup_errors:
        raise RuntimeError("cleanup failed: " + "; ".join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(
        description="Register a CuPy GPU allocation as a pyverbs MR"
    )
    parser.add_argument(
        "size",
        nargs="?",
        default=DEFAULT_SIZE,
        type=positive_int,
        help=f"allocation size in bytes (default: {DEFAULT_SIZE})",
    )
    args = parser.parse_args()

    try:
        run_probe(args.size)
    except BaseException as exc:
        traceback.print_exc()
        print(
            f"GDR_PROBE_FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print("GDR_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
