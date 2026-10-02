"""Receivers for one cell of the slot-scale mode.

``ConvPath`` is Aerial's monolithic cuPHY PUSCH pipeline with persistent HARQ
buffers.  ``NrxPath`` is the pyAerial NeuralRx chain (LS channel estimate,
TensorRT, rate recovery, LDPC, CRC) on one caller-owned stream.  Both decode
the same received slot and report CRC and payload correctness.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

import cuda.bindings.runtime as cudart

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig, _pusch_config_to_cuphy
from aerial.pycuphy.util import get_pusch_dyn_prms_phase_2
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.ldpc import CrcChecker, LdpcDecoder, LdpcDeRateMatch, get_mcs
from nrx_trt_direct import DirectNrx
from persistent_pusch_rx import PersistentPuschRx

from ul_profiles import (
    DMRS_SCRM_ID, NUM_PRBS, NUM_RX_ANT, NUM_SUBCARRIERS, NUM_SYMBOLS_SLOT,
    RNTI, UlProfile,
)


def mcs_params(profile: UlProfile) -> tuple[int, float]:
    mod_order, code_rate = get_mcs(profile.mcs_index, 1)
    return int(mod_order), float(code_rate)


def pusch_configs(profile: UlProfile, tb_size: int) -> list[PuschConfig]:
    mod_order, code_rate = mcs_params(profile)
    ue = PuschUeConfig(
        scid=0,
        layers=profile.rank,
        dmrs_ports=profile.dmrs_port_mask,
        rnti=RNTI,
        data_scid=0,
        mcs_table=0,
        mcs_index=profile.mcs_index,
        code_rate=int(code_rate * 10),
        mod_order=mod_order,
        tb_size=tb_size,
    )
    return [PuschConfig(
        ue_configs=[ue],
        num_dmrs_cdm_grps_no_data=2,
        dmrs_scrm_id=DMRS_SCRM_ID,
        start_prb=0,
        num_prbs=NUM_PRBS,
        prg_size=1,
        num_ul_streams=profile.rank,
        dmrs_syms=profile.dmrs_syms,
        dmrs_max_len=1,
        dmrs_add_ln_pos=len(profile.dmrs_positions) - 1,
        start_sym=profile.start_sym,
        num_symbols=profile.num_symbols,
    )]


def fortran_slot_views(base_ptr: int, owner, count: int) -> list[cp.ndarray]:
    """Fortran-ordered (subcarrier, symbol, antenna) views into one allocation."""
    item = np.dtype(np.complex64).itemsize
    shape = (NUM_SUBCARRIERS, NUM_SYMBOLS_SLOT, NUM_RX_ANT)
    strides = (item, item * shape[0], item * shape[0] * shape[1])
    size = item * shape[0] * shape[1] * shape[2]
    memory = cp.cuda.UnownedMemory(base_ptr, size * count, owner)
    return [
        cp.ndarray(shape, dtype=cp.complex64,
                   memptr=cp.cuda.MemoryPointer(memory, index * size),
                   strides=strides)
        for index in range(count)
    ]


def conv_path(profile: UlProfile, tb_size: int, stream: cp.cuda.Stream, ldpc_iterations: int = 0):
    """The cell's cuPHY pipeline.  ``ldpc_iterations`` > 0 fixes its LDPC iteration limit;
    0 keeps cuPHY's table (10 iterations for the profiles used here)."""
    import aerial.phy5g.pusch.pusch_rx as module
    original = module.get_pusch_stat_prms

    def fixed(*args, **kwargs):
        from aerial.pycuphy import LdpcMaxItrAlgoType
        return original(*args, **kwargs)._replace(
            ldpcMaxNumItrAlgo=LdpcMaxItrAlgoType.LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED,
            fixedMaxNumLdpcItrs=np.uint8(ldpc_iterations))

    if ldpc_iterations:
        module.get_pusch_stat_prms = fixed
    try:
        if profile.num_ue > 1:
            from mu_radio import MuConvPath
            return MuConvPath(profile, tb_size, stream)
        return ConvPath(profile, tb_size, stream)
    finally:
        module.get_pusch_stat_prms = original


def nrx_path(config: dict, profile: UlProfile, tb_size: int, stream: cp.cuda.Stream):
    if profile.num_ue > 1:
        from mu_radio import MuNrxPath
        return MuNrxPath(config["engine"], profile, tb_size, stream,
                         int(config.get("nrx_ldpc_iterations", 10)), config.get("nrx_path_mode", "fast"))
    return NrxPath(config["engine"], profile, tb_size, stream, config.get("nrx_ls_input", "nvlabs"),
                   int(config.get("nrx_ldpc_iterations", 10)))


def result_masks(path, ok: bool, payload, reference, tb_bytes: int) -> tuple[int, int]:
    """Per-UE bit masks of one decode: CRC passed, and CRC passed with the right payload."""
    if path.profile.num_ue > 1:
        from mu_radio import good_mask
        return path.last_crc_mask, good_mask(path.last_crc_mask, payload, reference, tb_bytes)
    return int(ok), int(bool(ok and np.array_equal(payload[:tb_bytes], reference)))


class ConvPath:
    def __init__(self, profile: UlProfile, tb_size: int, stream: cp.cuda.Stream) -> None:
        self.profile = profile
        self.configs = pusch_configs(profile, tb_size)
        self.stream = stream
        self.rx = PersistentPuschRx(
            cell_id=DMRS_SCRM_ID,
            num_rx_ant=NUM_RX_ANT,
            num_tx_ant=NUM_RX_ANT,
            enable_pusch_tdi=profile.enable_pusch_tdi,
            eq_coeff_algo=1,
            cuda_stream=int(stream.ptr),
        )
        self.last_cb_fail = 0

    def run(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, np.ndarray]:
        """Decode one slot; ``last_cb_fail`` keeps the failed code-block count."""
        rx = self.rx
        dynamic = _pusch_config_to_cuphy(
            cuda_stream=rx.cuda_stream, rx_data=[rx_slot], slot=slot,
            pusch_configs=self.configs,
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
        cbs = int(out.totNumCbs[0])
        self.last_cb_fail = int((np.asarray(out.cbCrcs)[:cbs] != 0).sum())
        crc = np.asarray(out.tbCrcs).reshape(-1)
        size = self.configs[0].ue_configs[0].tb_size
        return int(crc[0]) == 0, np.asarray(out.tbPayloads[:size])


class NrxPath:
    def __init__(self, engine_path: str, profile: UlProfile, tb_size: int,
                 stream: cp.cuda.Stream, ls_input: str = "nvlabs", ldpc_iterations: int = 10) -> None:
        # Layout of the LS channel estimate handed to the model.  "nvlabs": per DMRS symbol
        # across pilots, scaled by 1/sqrt(2) -- what the NVlabs neural_rx code (the model's
        # origin) feeds it; verified against its test vector.  "example": per pilot across
        # DMRS symbols, unscaled, as in the pyAerial example notebook.  The example layout
        # was used in every run before 2026-10-01 and costs most of the model's gain
        # (256 weak TBs: 134 decoded vs 211 with the NVlabs layout; conventional 170).
        self.ls_input = ls_input
        if not profile.neural_eligible or profile.rank != 1:
            raise ValueError("the public NeuralRx model accepts one QPSK layer only")
        if profile.num_symbols != 12:
            raise ValueError("the NeuralRx input window is 12 symbols")
        self.profile = profile
        self.configs = pusch_configs(profile, tb_size)
        self.stream = stream
        handle = int(stream.ptr)
        self.estimator = ChannelEstimator(
            num_rx_ant=NUM_RX_ANT, ch_est_algo=3, cuda_stream=handle
        )
        self.derate = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=handle)
        self.decoder = LdpcDecoder(num_iterations=ldpc_iterations, cuda_stream=handle)
        self.crc = CrcChecker(cuda_stream=handle)
        self.engine = DirectNrx(engine_path, stream=stream)
        relative = np.asarray(profile.dmrs_positions) - profile.start_sym
        window = profile.dmrs_syms[profile.start_sym:profile.start_sym + 12]
        with stream:
            self.engine.inputs["active_dmrs_ports"].fill(1)
            self.engine.inputs["dmrs_ofdm_pos"][:] = cp.asarray(
                relative[None, :], dtype=cp.int32
            )
            self.engine.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(
                [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
            )
            self.data_symbols = cp.asarray(
                np.flatnonzero(np.asarray(window) == 0), dtype=cp.int64
            )
        stream.synchronize()
        self.engine.capture_graph()

    def run(self, rx_slot: cp.ndarray, slot: int) -> tuple[bool, np.ndarray]:
        start = self.profile.start_sym
        channel = self.estimator.estimate(
            rx_slot=rx_slot, slot=slot, pusch_configs=self.configs
        )
        with self.stream:
            window = rx_slot[None, :, start:start + 12, :]
            estimate = cp.asarray(channel[0])
            if self.ls_input == "nvlabs":
                estimate = cp.transpose(estimate, (3, 0, 1, 2)) * cp.float32(0.70710678)
            else:
                estimate = cp.transpose(estimate, (0, 3, 1, 2))
            estimate = estimate.reshape(-1, estimate.shape[2], estimate.shape[3])[None, ...]
            inputs = self.engine.inputs
            cp.copyto(inputs["rx_slot_real"], window.real)
            cp.copyto(inputs["rx_slot_imag"], window.imag)
            cp.copyto(inputs["h_hat_real"], estimate.real)
            cp.copyto(inputs["h_hat_imag"], estimate.imag)
            outputs = self.engine.launch(use_graph=True)
            llrs = cp.take(outputs["output_1"][0, ...], self.data_symbols, axis=3)
            coded = self.derate.derate_match(
                input_llrs=[llrs], pusch_configs=self.configs
            )
            blocks = self.decoder.decode(input_llrs=coded, pusch_configs=self.configs)
            tbs, crcs = self.crc.check_crc(input_bits=blocks, pusch_configs=self.configs)
        # The chain runs on a non-blocking stream; the host copies below run on
        # cupy's default stream, which does not wait for it.  Without this
        # synchronization the host can read the previous TB's CRC and payload.
        self.stream.synchronize()
        crc = cp.asnumpy(crcs[0]) if isinstance(crcs[0], cp.ndarray) else np.asarray(crcs[0])
        payload = cp.asnumpy(tbs[0]) if isinstance(tbs[0], cp.ndarray) else np.asarray(tbs[0])
        return int(crc.reshape(-1)[0]) == 0, payload
