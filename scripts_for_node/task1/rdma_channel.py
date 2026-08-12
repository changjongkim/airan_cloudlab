"""Simple RDMA WRITE channel for cross-MIG-partition L1↔NRx transport.

Uses libibverbs (via pyverbs) directly. Both endpoints run on the same host
(loopback) via the Mellanox NIC in PHY internal loopback mode, so QP addresses
are exchanged through /dev/shm files rather than a network handshake.

RoCE resolves the selected GID through its Ethernet netdev while the QP moves
to RTR. Containers must therefore use ``--network=host`` in addition to
``--device=/dev/infiniband``; exposing only the device nodes is not sufficient.

Layout (per direction, one ring slot for simplicity):
  region 0..DATA_SZ-1    : payload buffer (rx_slot+h_hat or return LLR)
  region DATA_SZ..+8     : sequence counter (published *after* RDMA completes)

Producer.send(seq, data_bytes) → RDMA WRITE payload, then RDMA WRITE seq.
Consumer.wait(seq) → spin on local seq counter until >= seq.
"""
import os
import time
import struct
import ctypes

import numpy as np

import pyverbs.device as vdev
from pyverbs.pd import PD
from pyverbs.cq import CQ
from pyverbs.qp import QPCap, QPInitAttr, QPAttr, QP
from pyverbs.addr import AH, AHAttr, GID, GlobalRoute
from pyverbs.mr import MR
from pyverbs.enums import (
    IBV_ACCESS_LOCAL_WRITE, IBV_ACCESS_REMOTE_WRITE, IBV_ACCESS_REMOTE_READ,
    IBV_QPT_RC, IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS,
    IBV_MTU_1024, IBV_WR_RDMA_WRITE, IBV_SEND_SIGNALED,
)
from pyverbs.wr import SendWR, SGE

DEV_NAME = os.environ.get("RDMA_DEV", "mlx5_0")
GID_INDEX = int(os.environ.get("RDMA_GID_INDEX", "3"))  # RoCE v2 IPv4
IB_PORT = 1
QUEUE_DEPTH = 64


class RdmaEndpoint:
    """One side of an RDMA loopback pair; supports RDMA WRITE only."""

    def __init__(self, buf_size, tag, is_producer):
        self.buf_size = buf_size
        self.tag = tag
        self.is_producer = is_producer

        # 1. Open device / PD / CQ.
        dev_list = vdev.get_device_list()
        dev = next(d for d in dev_list if d.name.decode() == DEV_NAME)
        self.ctx = vdev.Context(name=DEV_NAME)
        self.pd = PD(self.ctx)
        self.cq = CQ(self.ctx, QUEUE_DEPTH)

        # A RoCE GID is scoped to the network namespace that owns its netdev.
        # Docker's default namespace can still open mlx5_0 and query its GID,
        # but ibv_modify_qp(... RTR ...) then fails with ENODEV. Fail early
        # with the actual remedy instead of surfacing that opaque error.
        self.netdev = self._require_gid_netdev()

        # 2. Register a fresh MR. pyverbs will allocate and return an MR whose
        # .buf attribute is the raw address; use .write()/.read() to touch it.
        self.data_size = buf_size
        self.total_size = buf_size + 8
        access = (IBV_ACCESS_LOCAL_WRITE
                  | IBV_ACCESS_REMOTE_WRITE
                  | IBV_ACCESS_REMOTE_READ)
        self.mr = MR(self.pd, self.total_size, access)
        # Zero the buffer via write() so the seq slot starts at 0.
        self.mr.write(b"\x00" * self.total_size, self.total_size)

        # 3. Create QP (RC transport).
        cap = QPCap(max_send_wr=QUEUE_DEPTH, max_recv_wr=QUEUE_DEPTH,
                    max_send_sge=1, max_recv_sge=1)
        qp_attr = QPInitAttr(qp_type=IBV_QPT_RC, scq=self.cq, rcq=self.cq,
                             cap=cap)
        self.qp = QP(self.pd, qp_attr)

        # 4. Publish local address so peer can attach.
        port_attr = self.ctx.query_port(IB_PORT)
        gid = self.ctx.query_gid(IB_PORT, GID_INDEX)
        # GID exposes only .gid (colon-separated hex string) — store that
        # string as bytes and reconstruct on the peer side via GID(gid=str).
        gid_str = str(gid)
        self.local_info = {
            "qpn": self.qp.qp_num,
            "psn": 0,
            "lid": port_attr.lid,
            "gid_str": gid_str,
            "rkey": self.mr.rkey,
            "vaddr": self.mr.buf,
        }
        self._publish_info()

        # 5. Discover peer and transition QP INIT → RTR → RTS.
        self.peer = self._wait_peer()
        self._to_init()
        self._to_rtr()
        self._to_rts()

    @staticmethod
    def _require_gid_netdev():
        ndev_path = (f"/sys/class/infiniband/{DEV_NAME}/ports/{IB_PORT}/"
                     f"gid_attrs/ndevs/{GID_INDEX}")
        try:
            with open(ndev_path, "r", encoding="ascii") as f:
                netdev = f.read().strip()
        except OSError as exc:
            raise RuntimeError(
                f"RoCE GID {DEV_NAME} port {IB_PORT} index {GID_INDEX} is not "
                "visible in this network namespace; run the container with "
                "--network=host"
            ) from exc
        if not netdev or not os.path.exists(f"/sys/class/net/{netdev}"):
            raise RuntimeError(
                f"RoCE netdev {netdev or '<unknown>'} is not visible in this "
                "network namespace; run the container with --network=host"
            )
        return netdev

    # --- QP state machine ---------------------------------------------------
    def _to_init(self):
        attr = QPAttr()
        attr.qp_state = IBV_QPS_INIT
        attr.pkey_index = 0
        attr.port_num = IB_PORT
        attr.qp_access_flags = (IBV_ACCESS_LOCAL_WRITE
                                 | IBV_ACCESS_REMOTE_WRITE
                                 | IBV_ACCESS_REMOTE_READ)
        self.qp.to_init(attr)

    def _to_rtr(self):
        attr = QPAttr()
        attr.qp_state = IBV_QPS_RTR
        attr.path_mtu = IBV_MTU_1024
        attr.dest_qp_num = self.peer["qpn"]
        attr.rq_psn = self.peer["psn"]
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 12
        ah_attr = AHAttr()
        ah_attr.port_num = IB_PORT
        ah_attr.is_global = 1
        ah_attr.dgid = self.peer["gid_str"]
        ah_attr.sgid_index = GID_INDEX
        ah_attr.hop_limit = 1
        attr.ah_attr = ah_attr
        self.qp.to_rtr(attr)

    def _to_rts(self):
        attr = QPAttr()
        attr.qp_state = IBV_QPS_RTS
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.sq_psn = 0
        attr.max_rd_atomic = 1
        self.qp.to_rts(attr)

    # --- Peer info exchange via /dev/shm files ------------------------------
    def _info_path(self, role):
        return f"/dev/shm/rdma_{self.tag}_{role}.info"

    def _publish_info(self):
        role = "prod" if self.is_producer else "cons"
        path = self._info_path(role)
        # Store the GID as its human-readable colon-string form; the peer
        # rebuilds the GID object via GID(gid=<string>).
        gid_str = self.local_info["gid_str"]
        with open(path, "wb") as f:
            f.write(struct.pack(
                "!IIH40sIQ",
                self.local_info["qpn"], self.local_info["psn"],
                self.local_info["lid"],
                gid_str.encode("ascii").ljust(40, b"\x00"),
                self.local_info["rkey"], self.local_info["vaddr"],
            ))
        os.chmod(path, 0o666)

    def _wait_peer(self):
        role = "cons" if self.is_producer else "prod"
        path = self._info_path(role)
        rec_size = struct.calcsize("!IIH40sIQ")
        for _ in range(6000):  # up to 60 s
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        data = f.read(rec_size)
                    if len(data) == rec_size:
                        qpn, psn, lid, gid_raw, rkey, vaddr = struct.unpack(
                            "!IIH40sIQ", data)
                        gid_str = gid_raw.rstrip(b"\x00").decode("ascii")
                        return {"qpn": qpn, "psn": psn, "lid": lid,
                                "gid_str": gid_str, "rkey": rkey, "vaddr": vaddr}
                except Exception:
                    pass
            time.sleep(0.01)
        raise TimeoutError(f"peer info not published: {path}")

    # --- Data plane ---------------------------------------------------------
    def rdma_write(self, local_offset, length, remote_offset):
        """Post RDMA WRITE; block on completion."""
        sge = SGE(addr=self.mr.buf + local_offset, length=length,
                  lkey=self.mr.lkey)
        wr = SendWR(opcode=IBV_WR_RDMA_WRITE, num_sge=1, sg=[sge],
                    send_flags=IBV_SEND_SIGNALED)
        wr.set_wr_rdma(
            rkey=self.peer["rkey"],
            addr=self.peer["vaddr"] + remote_offset,
        )
        self.qp.post_send(wr)
        # Poll for completion.
        while True:
            n, wcs = self.cq.poll(num_entries=1)
            if n > 0:
                if wcs[0].status != 0:
                    raise RuntimeError(f"RDMA WRITE failed: status={wcs[0].status}")
                return
