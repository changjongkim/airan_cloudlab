#!/usr/bin/env python3
"""GPUDirect RDMA WRITE channel with GPU payload and CPU control MRs.

Each endpoint owns one CuPy device allocation registered by virtual address and
one 8-byte host MR used only for a sequence marker.  QP/MR coordinates are
exchanged through session-scoped files in /dev/shm.  The consumer owns the
session epoch, so stale files from an interrupted prior run cannot be accepted.

GPU-backed MRs must never be accessed with pyverbs ``MR.read``/``MR.write``:
those helpers perform a CPU memcpy.  Only ``ctrl_mr`` uses those helpers here.
"""

import gc
import json
import os
from pathlib import Path
import random
import re
import struct
import sys
import time
import uuid

import cupy as cp
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cuda_runtime
import pyverbs.device as vdev
from pyverbs.addr import AHAttr
from pyverbs.cq import CQ
from pyverbs.enums import (
    IBV_ACCESS_LOCAL_WRITE,
    IBV_ACCESS_REMOTE_READ,
    IBV_ACCESS_REMOTE_WRITE,
    IBV_MTU_1024,
    IBV_QPS_INIT,
    IBV_QPS_RTR,
    IBV_QPS_RTS,
    IBV_QPT_RC,
    IBV_SEND_SIGNALED,
    IBV_WR_RDMA_WRITE,
)
from pyverbs.mr import MR
from pyverbs.pd import PD
from pyverbs.qp import QP, QPAttr, QPCap, QPInitAttr
from pyverbs.wr import SGE, SendWR


CTRL_SIZE = struct.calcsize("!Q")
QUEUE_DEPTH = 64
SAFE_TAG = re.compile(r"^[A-Za-z0-9_.-]+$")


def _cuda_check(result, operation):
    status = result[0]
    if status != cuda_driver.CUresult.CUDA_SUCCESS:
        name = getattr(status, "name", str(status))
        raise RuntimeError(f"{operation} failed: {name} ({int(status)})")
    return result[1:]


def _prepare_gpu_pointer(ptr):
    sync_attr = (
        cuda_driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    )
    # cuda-bindings 12.9 accepts a Python bool and owns the temporary C storage.
    _cuda_check(
        cuda_driver.cuPointerSetAttribute(True, sync_attr, ptr),
        "cuPointerSetAttribute(SYNC_MEMOPS)",
    )
    enabled = _cuda_check(
        cuda_driver.cuPointerGetAttribute(sync_attr, ptr),
        "cuPointerGetAttribute(SYNC_MEMOPS)",
    )[0]
    if not enabled:
        raise RuntimeError("CUDA SYNC_MEMOPS did not remain enabled")

    cap_name = "CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE"
    if hasattr(cuda_driver.CUpointer_attribute, cap_name):
        cap_attr = getattr(cuda_driver.CUpointer_attribute, cap_name)
        capable = _cuda_check(
            cuda_driver.cuPointerGetAttribute(cap_attr, ptr),
            f"cuPointerGetAttribute({cap_name})",
        )[0]
        if not capable:
            raise RuntimeError("CUDA allocation is not GPUDirect RDMA capable")


def flush_gpudirect_writes():
    """Make peer RDMA WRITEs visible to CUDA work on the current device.

    cuda-bindings releases expose the target/scope enums under slightly
    different generated Python types.  Use the CUDA 12.9 spellings when they
    are available, and retain a CPU-initiated CUDA synchronization fallback
    for stacks where the dedicated host flush is unsupported.
    """
    flush = getattr(cuda_runtime, "cudaDeviceFlushGPUDirectRDMAWrites", None)
    target_type = getattr(
        cuda_runtime, "cudaFlushGPUDirectRDMAWritesTarget", None)
    scope_type = getattr(
        cuda_runtime, "cudaFlushGPUDirectRDMAWritesScope", None)
    if flush is not None and target_type is not None and scope_type is not None:
        target = getattr(
            target_type, "cudaFlushGPUDirectRDMAWritesTargetCurrentDevice", None)
        scope = getattr(
            scope_type, "cudaFlushGPUDirectRDMAWritesToOwner", None)
        if target is not None and scope is not None:
            result = flush(target, scope)
            status = result[0]
            if int(status) == 0:
                return True
            error_type = getattr(cuda_runtime, "cudaError_t", None)
            not_supported = (
                getattr(error_type, "cudaErrorNotSupported", None)
                if error_type is not None else None
            )
            is_not_supported = (
                (not_supported is not None and status == not_supported)
                or getattr(status, "name", "") == "cudaErrorNotSupported"
            )
            if not is_not_supported:
                raise RuntimeError(
                    "cudaDeviceFlushGPUDirectRDMAWrites failed: "
                    f"{getattr(status, 'name', status)} ({int(status)})"
                )

    # NVIDIA's ordering contract also permits a CPU-initiated CUDA
    # synchronization point after the host observes the ordered CPU marker.
    cp.cuda.runtime.deviceSynchronize()
    return False


def _atomic_write_json(path, value):
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"), sort_keys=True)
        output.flush()
        os.fsync(output.fileno())
    os.chmod(temp, 0o666)
    os.replace(temp, path)


def _read_json(path):
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, ValueError, TypeError):
        return None


def _unlink(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class GdrRdmaEndpoint:
    """One role in a same-host RC GPUDirect RDMA pair."""

    def __init__(self, payload_size, tag, role):
        if payload_size <= 0 or payload_size % 4:
            raise ValueError("payload_size must be positive and uint32-aligned")
        if role not in ("prod", "cons"):
            raise ValueError("role must be 'prod' or 'cons'")
        if not SAFE_TAG.fullmatch(tag):
            raise ValueError("tag may contain only letters, digits, '.', '_' or '-'")

        self.payload_size = payload_size
        self.word_count = payload_size // 4
        self.tag = tag
        self.role = role
        self.peer_role = "cons" if role == "prod" else "prod"
        self.is_producer = role == "prod"
        self.rdma_dev = os.environ.get("RDMA_DEV", "mlx5_0")
        self.ib_port = int(os.environ.get("RDMA_IB_PORT", "1"))
        self.gid_index = int(os.environ.get("RDMA_GID_INDEX", "3"))
        self.cuda_device = int(os.environ.get("CUDA_DEVICE", "0"))
        self.timeout_s = float(os.environ.get("GDR_RDMA_TIMEOUT_S", "60"))
        if self.timeout_s <= 0:
            raise ValueError("GDR_RDMA_TIMEOUT_S must be positive")

        prefix = f"gdr_rdma_{tag}"
        self.session_path = Path(f"/dev/shm/{prefix}_session.info")
        self.own_info_path = Path(f"/dev/shm/{prefix}_{role}.info")
        self.peer_info_path = Path(f"/dev/shm/{prefix}_{self.peer_role}.info")
        self.own_ready_path = Path(f"/dev/shm/{prefix}_{role}.ready")
        self.peer_ready_path = Path(
            f"/dev/shm/{prefix}_{self.peer_role}.ready"
        )
        self.ack_path = Path(f"/dev/shm/{prefix}_ack.info")

        self.session_id = None
        self.ctx = None
        self.pd = None
        self.cq = None
        self.qp = None
        self.payload_mr = None
        self.ctrl_mr = None
        self.gpu_memory = None
        self.gpu_memptr = None
        self.gpu_array = None
        self.peer = None
        self._closed = False

        try:
            self._join_session()
            self._create_resources()
            self._publish_local_info()
            self.peer = self._wait_for_peer()
            self._connect_qp()
            self._publish_ready()
            self._wait_for_peer_ready()
        except BaseException:
            self.close()
            raise

    def _join_session(self):
        if self.role == "cons":
            # Consumer starts a new epoch before producer is launched.
            for path in (
                self.session_path,
                self.own_info_path,
                self.peer_info_path,
                self.own_ready_path,
                self.peer_ready_path,
                self.ack_path,
            ):
                _unlink(path)
            self.session_id = uuid.uuid4().hex
            _atomic_write_json(
                self.session_path,
                {
                    "session": self.session_id,
                    "created_ns": time.time_ns(),
                    "tag": self.tag,
                },
            )
            return

        _unlink(self.own_info_path)
        _unlink(self.own_ready_path)
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            session = _read_json(self.session_path)
            if session and session.get("tag") == self.tag:
                created_ns = int(session.get("created_ns", 0))
                age_s = (time.time_ns() - created_ns) / 1e9
                if 0 <= age_s <= self.timeout_s:
                    self.session_id = session.get("session")
                    if self.session_id:
                        return
            time.sleep(0.01)
        raise TimeoutError(
            f"fresh consumer session not found: {self.session_path}"
        )

    def _create_resources(self):
        device = cp.cuda.Device(self.cuda_device)
        device.use()

        # CuPy Memory directly owns a cudaMalloc allocation.  The ndarray and
        # MemoryPointer retain it through all QP traffic and MR deregistration.
        self.gpu_memory = cp.cuda.Memory(self.payload_size)
        self.gpu_memptr = cp.cuda.MemoryPointer(self.gpu_memory, 0)
        self.gpu_array = cp.ndarray(
            (self.word_count,), dtype=cp.uint32, memptr=self.gpu_memptr
        )
        self.gpu_array.fill(cp.uint32(0))
        cp.cuda.runtime.deviceSynchronize()
        gpu_ptr = int(self.gpu_memory.ptr)
        _prepare_gpu_pointer(gpu_ptr)

        self.ctx = vdev.Context(name=self.rdma_dev)
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, QUEUE_DEPTH)

        access = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        # rdma-core v57 casts address through uintptr_t into ibv_reg_mr.
        self.payload_mr = MR(
            self.pd, self.payload_size, access, address=gpu_ptr
        )
        self.ctrl_mr = MR(self.pd, CTRL_SIZE, access)
        self.ctrl_mr.write(b"\x00" * CTRL_SIZE, CTRL_SIZE, offset=0)

        cap = QPCap(
            max_send_wr=QUEUE_DEPTH,
            max_recv_wr=QUEUE_DEPTH,
            max_send_sge=1,
            max_recv_sge=1,
        )
        init_attr = QPInitAttr(
            qp_type=IBV_QPT_RC, scq=self.cq, rcq=self.cq, cap=cap
        )
        self.qp = QP(self.pd, init_attr)
        self.psn = random.SystemRandom().randrange(1 << 24)

    def _publish_local_info(self):
        port_attr = self.ctx.query_port(self.ib_port)
        gid = self.ctx.query_gid(self.ib_port, self.gid_index)
        info = {
            "session": self.session_id,
            "role": self.role,
            "payload_size": self.payload_size,
            "qpn": self.qp.qp_num,
            "psn": self.psn,
            "lid": port_attr.lid,
            "gid": str(gid),
            "payload_rkey": self.payload_mr.rkey,
            "payload_addr": self.payload_mr.buf,
            "ctrl_rkey": self.ctrl_mr.rkey,
            "ctrl_addr": self.ctrl_mr.buf,
            "created_ns": time.time_ns(),
        }
        _atomic_write_json(self.own_info_path, info)

    def _wait_for_peer(self):
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            peer = _read_json(self.peer_info_path)
            if not peer:
                time.sleep(0.01)
                continue
            if (
                peer.get("session") != self.session_id
                or peer.get("role") != self.peer_role
            ):
                time.sleep(0.01)
                continue
            if int(peer.get("payload_size", -1)) != self.payload_size:
                raise ValueError(
                    "peer payload_size mismatch: "
                    f"local={self.payload_size} peer={peer.get('payload_size')}"
                )
            return peer
        raise TimeoutError(f"peer info not published: {self.peer_info_path}")

    def _connect_qp(self):
        attr = QPAttr()
        attr.qp_state = IBV_QPS_INIT
        attr.pkey_index = 0
        attr.port_num = self.ib_port
        attr.qp_access_flags = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        self.qp.to_init(attr)

        attr = QPAttr()
        attr.qp_state = IBV_QPS_RTR
        attr.path_mtu = IBV_MTU_1024
        attr.dest_qp_num = int(self.peer["qpn"])
        attr.rq_psn = int(self.peer["psn"])
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 12
        ah_attr = AHAttr()
        ah_attr.port_num = self.ib_port
        ah_attr.is_global = 1
        ah_attr.dgid = self.peer["gid"]
        ah_attr.sgid_index = self.gid_index
        ah_attr.hop_limit = 1
        attr.ah_attr = ah_attr
        self.qp.to_rtr(attr)

        attr = QPAttr()
        attr.qp_state = IBV_QPS_RTS
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.sq_psn = self.psn
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)

    def _publish_ready(self):
        _atomic_write_json(
            self.own_ready_path,
            {"session": self.session_id, "role": self.role, "state": "RTS"},
        )

    def _wait_for_peer_ready(self):
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            ready = _read_json(self.peer_ready_path)
            if (
                ready
                and ready.get("session") == self.session_id
                and ready.get("role") == self.peer_role
                and ready.get("state") == "RTS"
            ):
                return
            time.sleep(0.001)
        raise TimeoutError(f"peer QP did not reach RTS: {self.peer_ready_path}")

    def _post_write(self, mr, length, remote_addr, remote_rkey):
        sge = SGE(addr=mr.buf, length=length, lkey=mr.lkey)
        wr = SendWR(
            opcode=IBV_WR_RDMA_WRITE,
            num_sge=1,
            sg=[sge],
            send_flags=IBV_SEND_SIGNALED,
        )
        wr.set_wr_rdma(rkey=int(remote_rkey), addr=int(remote_addr))
        self.qp.post_send(wr)

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            count, completions = self.cq.poll(num_entries=1)
            if count:
                completion = completions[0]
                if completion.status != 0:
                    raise RuntimeError(
                        "RDMA WRITE completion failed: "
                        f"status={completion.status}"
                    )
                return
        raise TimeoutError("timed out waiting for RDMA WRITE completion")

    def write_payload(self):
        """Write the entire local GPU allocation to the peer GPU MR."""
        self._post_write(
            self.payload_mr,
            self.payload_size,
            self.peer["payload_addr"],
            self.peer["payload_rkey"],
        )

    def write_sequence(self, sequence):
        """Publish an ordered sequence marker through the host control MR."""
        self.ctrl_mr.write(struct.pack("!Q", sequence), CTRL_SIZE, offset=0)
        self._post_write(
            self.ctrl_mr,
            CTRL_SIZE,
            self.peer["ctrl_addr"],
            self.peer["ctrl_rkey"],
        )

    def read_sequence(self):
        """Read only the local host control MR; never touches the GPU MR."""
        return struct.unpack(
            "!Q", self.ctrl_mr.read(CTRL_SIZE, offset=0)
        )[0]

    def wait_for_sequence(self, expected):
        deadline = time.monotonic() + self.timeout_s
        last = 0
        while time.monotonic() < deadline:
            last = self.read_sequence()
            if last == expected:
                return
            if last > expected:
                raise RuntimeError(
                    f"sequence skipped: expected={expected} observed={last}"
                )
        raise TimeoutError(
            f"timed out waiting for sequence {expected}; observed={last}"
        )

    def publish_ack(self, sequence):
        _atomic_write_json(
            self.ack_path,
            {"session": self.session_id, "sequence": sequence},
        )

    def wait_for_ack(self, expected):
        deadline = time.monotonic() + self.timeout_s
        last = 0
        while time.monotonic() < deadline:
            ack = _read_json(self.ack_path)
            if ack and ack.get("session") == self.session_id:
                last = int(ack.get("sequence", 0))
                if last == expected:
                    return
                if last > expected:
                    raise RuntimeError(
                        f"ack skipped: expected={expected} observed={last}"
                    )
            time.sleep(0.0005)
        raise TimeoutError(
            f"timed out waiting for ack {expected}; observed={last}"
        )

    def close(self):
        if self._closed:
            return []
        self._closed = True
        errors = []

        def close_one(name):
            resource_obj = getattr(self, name, None)
            if resource_obj is None:
                return
            try:
                resource_obj.close()
            except Exception as exc:
                errors.append(f"{name}.close: {exc}")
            setattr(self, name, None)

        # Deregister payload before freeing its CuPy allocation.
        for name in ("qp", "payload_mr", "ctrl_mr", "cq", "pd", "ctx"):
            close_one(name)

        self.gpu_array = None
        self.gpu_memptr = None
        self.gpu_memory = None
        gc.collect()

        _unlink(self.own_info_path)
        _unlink(self.own_ready_path)
        if self.role == "prod":
            ack = _read_json(self.ack_path)
            if ack and ack.get("session") == self.session_id:
                _unlink(self.ack_path)
        else:
            session = _read_json(self.session_path)
            if session and session.get("session") == self.session_id:
                _unlink(self.session_path)

        for error in errors:
            print(f"GDR_CLEANUP_WARNING: {error}", file=sys.stderr, flush=True)
        return errors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback_obj):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
