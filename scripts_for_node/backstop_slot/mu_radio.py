"""Receivers for a cell whose uplink slot carries several UEs on the same PRBs (MU-MIMO).

``MuConvPath`` is Aerial's cuPHY PUSCH pipeline decoding all UEs of the slot in one call.
``MuNrxPath`` is the NVlabs neural receiver (github.com/NVlabs/neural_rx, exported to
TensorRT) followed by pyAerial rate recovery, LDPC and CRC per UE.  Both report, per UE,
whether the CRC passed (``last_crc_mask``); ``good_mask`` adds the payload comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import cupy as cp
import numpy as np
import tensorrt as trt

import cuda.bindings.runtime as cudart

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig, _pusch_config_to_cuphy
from aerial import pycuphy
from aerial.phy5g.ldpc import CrcChecker, LdpcDecoder, LdpcDeRateMatch, get_mcs
from aerial.phy5g.ldpc.util import get_code_block_size
from aerial.pycuphy.util import get_pusch_dyn_prms_phase_2
from aerial.util.cuda import check_cuda_errors
from nrx_trt_direct import DirectNrx
from persistent_pusch_rx import PersistentPuschRx

from ul_profiles import NUM_PRBS, NUM_RX_ANT, UlProfile


def mu_pusch_configs(profile: UlProfile, tb_size: int) -> tuple[list[PuschConfig], list[PuschConfig]]:
    """One group with every UE (what a receiver of the slot sees) and one single-UE group per
    UE (the bit-level chain after the neural receiver, whose LLRs are per UE)."""
    mod_order, code_rate = get_mcs(profile.mcs_index, 1)
    ues = [PuschUeConfig(
        scid=profile.scid, layers=1, dmrs_ports=1 << port, rnti=profile.rnti,
        data_scid=profile.scid, mcs_table=0, mcs_index=profile.mcs_index,
        code_rate=int(code_rate * 10), mod_order=int(mod_order), tb_size=tb_size,
    ) for port in profile.ue_dmrs_ports]

    def group(members: list, streams: int) -> PuschConfig:
        return PuschConfig(
            ue_configs=members, num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=profile.scrm_id,
            start_prb=0, num_prbs=NUM_PRBS, prg_size=1, num_ul_streams=streams,
            dmrs_syms=profile.dmrs_syms, dmrs_max_len=1,
            dmrs_add_ln_pos=len(profile.dmrs_positions) - 1,
            start_sym=profile.start_sym, num_symbols=profile.num_symbols,
        )

    return [group(ues, len(ues))], [group([ue], 1) for ue in ues]


def good_mask(crc_mask: int, payloads: list, reference: np.ndarray, tb_bytes: int) -> int:
    """UEs whose CRC passed and whose payload equals the transmitted one."""
    mask = 0
    for ue, payload in enumerate(payloads):
        if (crc_mask >> ue) & 1 and np.array_equal(np.asarray(payload)[:tb_bytes], reference[ue][:tb_bytes]):
            mask |= 1 << ue
    return mask


class MuConvPath:
    def __init__(self, profile: UlProfile, tb_size: int, stream: cp.cuda.Stream) -> None:
        self.profile = profile
        self.configs, _ = mu_pusch_configs(profile, tb_size)
        self.tb_size = tb_size
        self.num_ue = len(profile.ue_dmrs_ports)
        self.stream = stream
        self.rx = PersistentPuschRx(
            cell_id=profile.scrm_id, num_rx_ant=NUM_RX_ANT, num_tx_ant=NUM_RX_ANT,
            enable_pusch_tdi=profile.enable_pusch_tdi, eq_coeff_algo=1,
            cuda_stream=int(stream.ptr),
        )
        self.last_cb_fail = 0
        self.last_crc_mask = 0
        self.last_cb_fail_by_ue = [0] * self.num_ue
        self.levels = {profile.name: (self.configs, tb_size)}

    def add_level(self, profile: UlProfile, tb_size: int) -> None:
        """Another MCS of the same cell (link adaptation): same pipeline, other TB parameters."""
        self.levels[profile.name] = (mu_pusch_configs(profile, tb_size)[0], tb_size)

    def use(self, name: str) -> None:
        self.configs, self.tb_size = self.levels[name]

    def run(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, list]:
        rx = self.rx
        dynamic = _pusch_config_to_cuphy(
            cuda_stream=rx.cuda_stream, rx_data=[rx_slot], slot=slot, pusch_configs=self.configs,
        )
        rx.pusch_pipeline.setup_pusch_rx(dynamic)
        required = [int(v) for v in dynamic.dataOut.harqBufferSizeInBytes]
        rx._ensure_harq(required)
        for pointer, size in zip(rx._harq_buffers, required):
            check_cuda_errors(cudart.cudaMemsetAsync(pointer, 0, size, rx.cuda_stream))
        dynamic = get_pusch_dyn_prms_phase_2(dynamic, rx._harq_buffers)
        rx.pusch_pipeline.setup_pusch_rx(dynamic)
        rx.pusch_pipeline.run_pusch_rx()
        out = dynamic.dataOut
        total = int(out.totNumCbs[0])
        cb_crcs = np.asarray(out.cbCrcs)[:total] != 0
        self.last_cb_fail = int(cb_crcs.sum())
        offsets = [int(v) for v in np.asarray(out.startOffsetsCbCrc).reshape(-1)[:self.num_ue]] + [total]
        self.last_cb_fail_by_ue = [int(cb_crcs[offsets[u]:offsets[u + 1]].sum()) for u in range(self.num_ue)]
        crc = np.asarray(out.tbCrcs).reshape(-1)
        self.last_crc_mask = sum(1 << u for u in range(self.num_ue) if int(crc[u]) == 0)
        starts = [int(v) for v in np.asarray(out.startOffsetsTbPayload).reshape(-1)[:self.num_ue]]
        payloads = [np.asarray(out.tbPayloads[start:start + self.tb_size]) for start in starts]
        return self.last_crc_mask == (1 << self.num_ue) - 1, payloads


class NvEngine(DirectNrx):
    """TensorRT runner for an exported NVlabs neural receiver; shapes come from the engine and
    the DMRS layout from the JSON written next to it at export time."""

    def __init__(self, engine_path: str, *, stream: cp.cuda.Stream) -> None:
        self.meta = json.loads(Path(str(engine_path).rsplit(".", 1)[0] + ".json").read_text())
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = stream
        self.caller_owned_stream = True
        self.graph = None
        self.inputs, self.outputs = {}, {}
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = cp.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
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


class MuNrxPath:
    def __init__(self, engine_path: str, profile: UlProfile, tb_size: int, stream: cp.cuda.Stream,
                 ldpc_iterations: int = 10, mode: str = "fast") -> None:
        # "reference": pyAerial LS estimator and the pyAerial rate-recovery / LDPC / CRC wrappers.
        # "fast": the same computation with the per-call work removed -- the LS estimate is the
        # pilot REs times a fixed factor (taken once from the pyAerial estimator; the two agree
        # to 3e-4), and the bit-level chain calls cuPHY directly on preallocated buffers.
        self.mode = mode
        self.profile = profile
        self.configs, self.per_ue = mu_pusch_configs(profile, tb_size)
        self.num_ue = len(self.per_ue)
        self.tb_size = tb_size
        self.stream = stream
        handle = int(stream.ptr)
        self.estimator = ChannelEstimator(num_rx_ant=NUM_RX_ANT, ch_est_algo=3, cuda_stream=handle)
        self.derate = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=handle)
        self.decoder = LdpcDecoder(num_iterations=ldpc_iterations, cuda_stream=handle)
        self.crc = CrcChecker(cuda_stream=handle)
        self.engine = NvEngine(engine_path, stream=stream)
        if self.engine.meta["num_tx"] != self.num_ue or self.engine.meta["prbs"] != NUM_PRBS:
            raise ValueError("engine and profile disagree on users or bandwidth")
        with stream:
            self.data_symbols = cp.asarray(
                [s for s in range(profile.num_symbols) if s not in profile.dmrs_positions], dtype=cp.int64)
            subcarriers = np.asarray(self.engine.meta["dmrs_subcarrier_pos"])[:self.num_ue]
            self.pilot_sc = cp.asarray(np.stack(
                [np.concatenate([12 * prb + subcarriers[u] for prb in range(NUM_PRBS)]) for u in range(self.num_ue)],
                axis=1))                                                   # (pilot, UE)
            self.dmrs_symbols = cp.asarray(list(profile.dmrs_positions), dtype=cp.int64)
        stream.synchronize()
        self.engine.capture_graph()
        self.last_crc_mask = 0
        self.ls_factor: dict[int, cp.ndarray] = {}
        # Constants of the bit-level chain (one entry per UE), per MCS level.
        self.levels: dict[str, dict] = {}
        self.current = profile.name
        self.add_level(profile, tb_size)
        level = self.levels[profile.name]
        self.chain, self.cb_size, self.crc_input = level["chain"], level["cb_size"], None
        length = int(level["chain"]["length"][0])
        with stream:
            self.padded = [cp.zeros((8, length // self.bits), dtype=cp.float16, order="F") for _ in range(self.num_ue)]
        stream.synchronize()

    def add_level(self, profile: UlProfile, tb_size: int) -> None:
        """Another MCS of the same cell (link adaptation).  The model and its 16QAM LLRs are the
        same; only rate recovery, LDPC and CRC depend on the TB size and code rate."""
        configs, per_ue = mu_pusch_configs(profile, tb_size)
        ue = per_ue[0].ue_configs[0]
        data_symbols = profile.num_symbols - len(profile.dmrs_positions)
        self.bits = int(ue.mod_order)
        length = data_symbols * self.bits * NUM_PRBS * 12
        users = range(len(per_ue))
        chain = {
            "tb": [np.uint32(tb_size * 8) for _ in users],
            "rate": [np.float32(ue.code_rate / 10240.) for _ in users],
            "length": [np.uint32(length) for _ in users],
            "mod": [np.uint8(self.bits) for _ in users],
            "layers": [np.uint8(1) for _ in users],
            "rv": [np.uint32(0) for _ in users],
            "ndi": [np.uint32(1) for _ in users],
            "cinit": [np.uint32((ue.rnti << 15) + ue.data_scid) for _ in users],
            "group": [np.uint32(u) for u in users],
        }
        self.levels[profile.name] = {
            "configs": configs, "per_ue": per_ue, "tb_size": tb_size, "chain": chain,
            "cb_size": int(get_code_block_size(tb_size * 8, ue.code_rate / 10240.)), "crc_input": None,
        }

    def use(self, name: str) -> None:
        if name == self.current:
            return
        self.levels[self.current]["crc_input"] = self.crc_input
        level = self.levels[name]
        self.configs, self.per_ue, self.tb_size = level["configs"], level["per_ue"], level["tb_size"]
        self.chain, self.cb_size, self.crc_input = level["chain"], level["cb_size"], level["crc_input"]
        self.current = name

    def _ls_fast(self, rx_slot: cp.ndarray, slot: int) -> cp.ndarray:
        """LS estimate in the model's layout: (DMRS symbol x pilot, UE, antenna)."""
        picked = cp.take(rx_slot[self.pilot_sc], self.dmrs_symbols, axis=2)    # (pilot, UE, symbol, antenna)
        factor = self.ls_factor.get(slot)
        if factor is None:
            channel = self.estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=self.configs)
            self.stream.synchronize()
            reference = cp.transpose(cp.asarray(channel[0]), (0, 1, 3, 2))      # (pilot, UE, symbol, antenna)
            factor = (reference * cp.conj(picked)).sum(axis=3, keepdims=True) / (
                cp.abs(picked) ** 2).sum(axis=3, keepdims=True)
            factor = cp.ascontiguousarray(cp.transpose(factor, (2, 0, 1, 3)) * cp.float32(0.70710678))
            self.ls_factor[slot] = factor
        estimate = cp.transpose(picked, (2, 0, 1, 3)) * factor                  # (symbol, pilot, UE, antenna)
        return estimate.reshape(-1, estimate.shape[2], estimate.shape[3])[None, ...]

    def _run_fast(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, list]:
        with self.stream:
            estimate = self._ls_fast(rx_slot, slot)
            window = rx_slot[None, ...]
            inputs = self.engine.inputs
            cp.copyto(inputs["rx_slot_real"], window.real)
            cp.copyto(inputs["rx_slot_imag"], window.imag)
            cp.copyto(inputs["h_hat_real"], estimate.real)
            cp.copyto(inputs["h_hat_imag"], estimate.imag)
            outputs = self.engine.launch(use_graph=True)
            llrs = cp.take(outputs["output_1"][0, ...], self.data_symbols, axis=3)   # (bit, UE, sc, symbol)
            for u in range(self.num_ue):
                # Same element order as a Fortran-order flattening of (bit, 1, sc, symbol).
                self.padded[u][:self.bits, :] = cp.transpose(llrs[:, u], (0, 2, 1)).reshape(self.bits, -1)
            c = self.chain
            coded = self.derate.pycuphy_ldpc_derate_match.derate_match(
                [pycuphy.CudaArrayHalf(a) for a in self.padded], c["tb"], c["rate"], c["length"], c["mod"],
                c["layers"], c["rv"], c["ndi"], c["cinit"], c["group"])
            decoded = self.decoder.pycuphy_ldpc_decoder.decode(coded, c["tb"], c["rate"], c["rv"], c["length"])
            blocks = [cp.asarray(d) for d in decoded]
            if self.crc_input is None:
                self.crc_input = cp.zeros((8448, sum(b.shape[1] for b in blocks)), dtype=cp.float16, order="F")
            column = 0
            for b in blocks:
                self.crc_input[:self.cb_size, column:column + b.shape[1]] = b[:self.cb_size, :]
                column += b.shape[1]
            payloads = self.crc.crc_checker.check_crc(pycuphy.CudaArrayHalf(self.crc_input), c["tb"], c["rate"])
            crcs = self.crc.crc_checker.get_tb_crcs()
        self.stream.synchronize()
        mask, out = 0, []
        for u in range(self.num_ue):
            out.append(cp.asnumpy(cp.asarray(payloads[u])))
            if int(cp.asnumpy(cp.asarray(crcs[u])).reshape(-1)[0]) == 0:
                mask |= 1 << u
        self.last_crc_mask = mask
        return mask == (1 << self.num_ue) - 1, out

    def run(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, list]:
        if self.mode == "fast":
            return self._run_fast(rx_slot, slot)
        channel = self.estimator.estimate(rx_slot=rx_slot, slot=slot, pusch_configs=self.configs)
        with self.stream:
            # LS estimate (pilot, UE, antenna, DMRS symbol) -> per DMRS symbol across pilots,
            # scaled by 1/sqrt(2): the layout the NVlabs code feeds the model (test vector
            # of the export agrees to 2e-4).
            estimate = cp.transpose(cp.asarray(channel[0]), (3, 0, 1, 2)) * cp.float32(0.70710678)
            estimate = estimate.reshape(-1, estimate.shape[2], estimate.shape[3])[None, ...]
            window = rx_slot[None, ...]
            inputs = self.engine.inputs
            cp.copyto(inputs["rx_slot_real"], window.real)
            cp.copyto(inputs["rx_slot_imag"], window.imag)
            cp.copyto(inputs["h_hat_real"], estimate.real)
            cp.copyto(inputs["h_hat_imag"], estimate.imag)
            outputs = self.engine.launch(use_graph=True)
            llrs = cp.take(outputs["output_1"][0, ...], self.data_symbols, axis=3)
            split = [cp.ascontiguousarray(llrs[:, u:u + 1]) for u in range(self.num_ue)]
            coded = self.derate.derate_match(input_llrs=split, pusch_configs=self.per_ue)
            blocks = self.decoder.decode(input_llrs=coded, pusch_configs=self.per_ue)
            tbs, crcs = self.crc.check_crc(input_bits=blocks, pusch_configs=self.per_ue)
        # Same reason as NrxPath.run: host copies must wait for the chain's stream.
        self.stream.synchronize()
        mask, payloads = 0, []
        for u in range(self.num_ue):
            crc = cp.asnumpy(crcs[u]) if isinstance(crcs[u], cp.ndarray) else np.asarray(crcs[u])
            payloads.append(cp.asnumpy(tbs[u]) if isinstance(tbs[u], cp.ndarray) else np.asarray(tbs[u]))
            if int(crc.reshape(-1)[0]) == 0:
                mask |= 1 << u
        self.last_crc_mask = mask
        return mask == (1 << self.num_ue) - 1, payloads
