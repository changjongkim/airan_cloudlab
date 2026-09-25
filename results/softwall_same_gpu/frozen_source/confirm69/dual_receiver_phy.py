#!/usr/bin/env python3
"""One valid PUSCH input decoded by conventional cuPHY and NeuralRx paths."""

from __future__ import annotations

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx

from aerial.phy5g.algorithms import (
    ChannelEqualizer,
    ChannelEstimator,
    NoiseIntfEstimator,
    TrtEngine,
    TrtTensorPrms,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import (
    CrcChecker,
    LdpcDecoder,
    LdpcDeRateMatch,
    get_mcs,
    get_tb_size,
)
from aerial.phy5g.ldpc.util import random_tb
from aerial.phy5g.pdsch import PdschTx
from aerial.util.cuda import get_cuda_stream


class PairedDualReceiver:
    """Generate one valid NeuralRx-compatible slot and decode either branch."""

    def __init__(
        self,
        engine_path: str,
        *,
        seed: int = 20260920,
        device: int = 0,
        enable_local_neural: bool = True,
    ) -> None:
        cp.cuda.runtime.setDevice(device)
        np.random.seed(seed)
        self.device = device
        self.slot = 2
        self.start_sym = 2
        self.num_symbols = 12
        self.dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        self.mod_order, code_rate = get_mcs(2, 1)
        self.code_rate = int(code_rate * 10)
        tb_size_bits = get_tb_size(
            mod_order=self.mod_order,
            code_rate=code_rate,
            dmrs_syms=self.dmrs_syms,
            num_prbs=273,
            start_sym=self.start_sym,
            num_symbols=self.num_symbols,
            num_layers=1,
        )
        self.reference_tb = random_tb(
            mod_order=self.mod_order,
            code_rate=code_rate,
            dmrs_syms=self.dmrs_syms,
            num_prbs=273,
            start_sym=self.start_sym,
            num_symbols=self.num_symbols,
            num_layers=1,
        )
        if len(self.reference_tb) != tb_size_bits // 8:
            raise RuntimeError("generated transport block size mismatch")

        transmitter = PdschTx(cell_id=41, num_rx_ant=4, num_tx_ant=4)
        self.rx_slot = transmitter.run(
            tb_inputs=[self.reference_tb],
            num_ues=1,
            slot=self.slot,
            num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_ids=[41],
            start_prb=0,
            num_prbs=273,
            dmrs_syms=self.dmrs_syms,
            start_sym=self.start_sym,
            num_symbols=self.num_symbols,
            scids=[0],
            layers=[1],
            dmrs_ports=[1],
            rntis=[1234],
            data_scids=[0],
            precoding_matrices=[
                np.array([[0.5 + 0.j, 0.0 + 0.5j, 0.0 + 0.5j, -0.5 + 0.j]])
            ],
            code_rates=[self.code_rate],
            mod_orders=[self.mod_order],
        )
        self.clean_rx_slot = cp.asfortranarray(cp.asarray(self.rx_slot))
        ue = PuschUeConfig(
            scid=0,
            layers=1,
            dmrs_ports=1,
            rnti=1234,
            data_scid=0,
            mcs_table=0,
            mcs_index=2,
            code_rate=self.code_rate,
            mod_order=self.mod_order,
            tb_size=len(self.reference_tb),
        )
        self.pusch_configs = [
            PuschConfig(
                ue_configs=[ue],
                num_dmrs_cdm_grps_no_data=2,
                dmrs_scrm_id=41,
                start_prb=0,
                num_prbs=273,
                prg_size=1,
                num_ul_streams=4,
                dmrs_syms=self.dmrs_syms,
                dmrs_max_len=1,
                dmrs_add_ln_pos=2,
                start_sym=self.start_sym,
                num_symbols=self.num_symbols,
            )
        ]

        self.stream_handle = get_cuda_stream()
        self.stream = cp.cuda.ExternalStream(int(self.stream_handle))
        self.conv_channel_estimator = ChannelEstimator(
            num_rx_ant=4, ch_est_algo=1, cuda_stream=self.stream_handle
        )
        self.neural_channel_estimator = ChannelEstimator(
            num_rx_ant=4, ch_est_algo=3, cuda_stream=self.stream_handle
        )
        self.noise_estimator = NoiseIntfEstimator(
            num_rx_ant=4, eq_coeff_algo=1, cuda_stream=self.stream_handle
        )
        self.equalizer = ChannelEqualizer(
            num_rx_ant=4,
            eq_coeff_algo=1,
            enable_pusch_tdi=0,
            cuda_stream=self.stream_handle,
        )
        self.conv_derate = LdpcDeRateMatch(
            enable_scrambling=True, cuda_stream=self.stream_handle
        )
        self.conv_decoder = LdpcDecoder(cuda_stream=self.stream_handle)
        self.conv_crc = CrcChecker(cuda_stream=self.stream_handle)
        self.neural_derate = LdpcDeRateMatch(
            enable_scrambling=True, cuda_stream=self.stream_handle
        )
        self.neural_decoder = LdpcDecoder(cuda_stream=self.stream_handle)
        self.neural_crc = CrcChecker(cuda_stream=self.stream_handle)
        self.neural_engine_wrapper = None
        if enable_local_neural:
            self.neural_engine_wrapper = TrtEngine(
                trt_model_file=engine_path,
                max_batch_size=1,
                cuda_stream=self.stream_handle,
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
        self.active_dmrs_ports = cp.ones((1, 1), dtype=cp.float32)
        # The TensorRT input contains symbols [start_sym, start_sym+12), so use
        # DMRS positions relative to that window.
        absolute_dmrs = np.flatnonzero(np.asarray(self.dmrs_syms))
        relative_dmrs = absolute_dmrs - self.start_sym
        self.dmrs_ofdm_pos = cp.asarray(relative_dmrs[None, :], dtype=cp.int32)
        self.dmrs_subcarrier_pos = cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        )
        window = self.dmrs_syms[
            self.start_sym : self.start_sym + self.num_symbols
        ]
        self.data_symbol_indices = cp.asarray(
            np.flatnonzero(np.asarray(window) == 0), dtype=cp.int64
        )
        self.stream.synchronize()
        self.neural_engine = None
        if enable_local_neural:
            self.neural_engine = DirectNrx(engine_path)
            self.neural_engine.capture_graph()

    def apply_rayleigh_awgn(
        self, snr_db: float, seed: int, *, noise_reference: str = "post_fading"
    ) -> None:
        """Apply block-Rayleigh fading and a declared AWGN power convention.

        ``post_fading`` preserves the historical trace: each block's noise
        power scales with its realized faded signal power. ``pre_fading`` uses
        the clean transmit-grid power, so fading changes the effective SNR.
        A campaign must record which convention it uses; the resulting traces
        are different workloads and must not be pooled as one comparison.
        """

        if noise_reference not in {"post_fading", "pre_fading"}:
            raise ValueError(f"unsupported noise reference: {noise_reference}")

        random = cp.random.RandomState(seed)
        channel = (
            random.standard_normal((1, 1, 4), dtype=cp.float32)
            + 1j * random.standard_normal((1, 1, 4), dtype=cp.float32)
        ) / cp.sqrt(cp.float32(2.0))
        faded = self.clean_rx_slot * channel
        signal_power = cp.mean(cp.abs(
            faded if noise_reference == "post_fading" else self.clean_rx_slot
        ) ** 2)
        noise_power = signal_power / cp.float32(10.0 ** (snr_db / 10.0))
        sigma = cp.sqrt(noise_power / cp.float32(2.0))
        noise = sigma * (
            random.standard_normal(faded.shape, dtype=cp.float32)
            + 1j * random.standard_normal(faded.shape, dtype=cp.float32)
        )
        self.rx_slot = cp.asfortranarray((faded + noise).astype(cp.complex64))
        self.stream.synchronize()

    def observed_channel_features(self) -> dict:
        """Measure only current-slot PHY observables before either CRC result.

        The feature extraction cost is returned and belongs inside a future
        online policy's deadline. Neither generator SNR nor transmitted TB
        truth is exposed to the decision policy.
        """

        with cp.cuda.Device(self.device), self.stream:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            channel = self.neural_channel_estimator.estimate(
                rx_slot=self.rx_slot,
                slot=self.slot,
                pusch_configs=self.pusch_configs,
            )
            estimate = cp.asarray(channel[0])
            channel_power = cp.mean(estimate.real ** 2 + estimate.imag ** 2)
            received_power = cp.mean(
                self.rx_slot.real ** 2 + self.rx_slot.imag ** 2
            )
            end.record()
            end.synchronize()
            return {
                "channel_estimate_power": float(channel_power.get()),
                "received_grid_power": float(received_power.get()),
                "gpu_ms": float(cp.cuda.get_elapsed_time(begin, end)),
            }

    def restore_clean_slot(self) -> None:
        self.rx_slot = self.clean_rx_slot

    def _verify(self, output) -> tuple[bool, int, int]:
        transport_blocks, crc_values = output
        crc_array = (
            cp.asnumpy(crc_values[0])
            if isinstance(crc_values[0], cp.ndarray)
            else np.asarray(crc_values[0])
        )
        payload = (
            cp.asnumpy(transport_blocks[0])
            if isinstance(transport_blocks[0], cp.ndarray)
            else np.asarray(transport_blocks[0])
        )
        crc_failures = int(int(crc_array.reshape(-1)[0]) != 0)
        payload_mismatches = int(not np.array_equal(payload, self.reference_tb))
        return crc_failures == 0 and payload_mismatches == 0, crc_failures, payload_mismatches

    def _measure(self, function) -> tuple[float, bool, int, int]:
        with cp.cuda.Device(self.device), self.stream:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            output = function()
            end.record()
            end.synchronize()
            gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
            correct, crc_failures, payload_mismatches = self._verify(output)
            self.stream.synchronize()
        return gpu_ms, correct, crc_failures, payload_mismatches

    def conventional_once(self):
        channel = self.conv_channel_estimator.estimate(
            rx_slot=self.rx_slot,
            slot=self.slot,
            pusch_configs=self.pusch_configs,
        )
        lw_inv, noise_var = self.noise_estimator.estimate(
            rx_slot=self.rx_slot,
            channel_est=channel,
            slot=self.slot,
            pusch_configs=self.pusch_configs,
        )
        llrs, _ = self.equalizer.equalize(
            rx_slot=self.rx_slot,
            channel_est=channel,
            lw_inv=lw_inv,
            noise_var_pre_eq=noise_var,
            pusch_configs=self.pusch_configs,
        )
        coded = self.conv_derate.derate_match(
            input_llrs=llrs, pusch_configs=self.pusch_configs
        )
        blocks = self.conv_decoder.decode(
            input_llrs=coded, pusch_configs=self.pusch_configs
        )
        return self.conv_crc.check_crc(
            input_bits=blocks, pusch_configs=self.pusch_configs
        )

    def neural_once_wrapper(self):
        channel = self.neural_channel_estimator.estimate(
            rx_slot=self.rx_slot,
            slot=self.slot,
            pusch_configs=self.pusch_configs,
        )
        rx_window = cp.asarray(self.rx_slot)[
            None,
            :,
            self.start_sym : self.start_sym + self.num_symbols,
            :,
        ]
        estimate = cp.asarray(channel[0])
        estimate_input = cp.transpose(estimate, (0, 3, 1, 2)).reshape(
            estimate.shape[0] * estimate.shape[3],
            estimate.shape[1],
            estimate.shape[2],
        )[None, ...]
        outputs = self.neural_engine_wrapper.run(
            {
                "rx_slot_real": cp.ascontiguousarray(rx_window.real).astype(cp.float32),
                "rx_slot_imag": cp.ascontiguousarray(rx_window.imag).astype(cp.float32),
                "h_hat_real": cp.ascontiguousarray(estimate_input.real).astype(cp.float32),
                "h_hat_imag": cp.ascontiguousarray(estimate_input.imag).astype(cp.float32),
                "active_dmrs_ports": self.active_dmrs_ports,
                "dmrs_ofdm_pos": self.dmrs_ofdm_pos,
                "dmrs_subcarrier_pos": self.dmrs_subcarrier_pos,
            }
        )
        llrs = cp.take(
            outputs["output_1"][0, ...], self.data_symbol_indices, axis=3
        )
        coded = self.neural_derate.derate_match(
            input_llrs=[llrs], pusch_configs=self.pusch_configs
        )
        blocks = self.neural_decoder.decode(
            input_llrs=coded, pusch_configs=self.pusch_configs
        )
        return self.neural_crc.check_crc(
            input_bits=blocks, pusch_configs=self.pusch_configs
        )

    def neural_once(self):
        """Run NeuralRx using persistent caller-owned TensorRT bindings."""

        if self.neural_engine is None:
            raise RuntimeError("local NeuralRx engine is disabled")

        channel = self.neural_channel_estimator.estimate(
            rx_slot=self.rx_slot,
            slot=self.slot,
            pusch_configs=self.pusch_configs,
        )
        # The channel estimator and TensorRT use distinct streams. Complete the
        # producer before filling the persistent TensorRT input buffers.
        self.stream.synchronize()
        with self.neural_engine.stream:
            rx_window = cp.asarray(self.rx_slot)[
                None,
                :,
                self.start_sym : self.start_sym + self.num_symbols,
                :,
            ]
            estimate = cp.asarray(channel[0])
            estimate_input = cp.transpose(estimate, (0, 3, 1, 2)).reshape(
                estimate.shape[0] * estimate.shape[3],
                estimate.shape[1],
                estimate.shape[2],
            )[None, ...]
            cp.copyto(
                self.neural_engine.inputs["rx_slot_real"],
                cp.ascontiguousarray(rx_window.real).astype(cp.float32),
            )
            cp.copyto(
                self.neural_engine.inputs["rx_slot_imag"],
                cp.ascontiguousarray(rx_window.imag).astype(cp.float32),
            )
            cp.copyto(
                self.neural_engine.inputs["h_hat_real"],
                cp.ascontiguousarray(estimate_input.real).astype(cp.float32),
            )
            cp.copyto(
                self.neural_engine.inputs["h_hat_imag"],
                cp.ascontiguousarray(estimate_input.imag).astype(cp.float32),
            )
            cp.copyto(
                self.neural_engine.inputs["active_dmrs_ports"],
                self.active_dmrs_ports,
            )
            cp.copyto(
                self.neural_engine.inputs["dmrs_ofdm_pos"],
                self.dmrs_ofdm_pos,
            )
            cp.copyto(
                self.neural_engine.inputs["dmrs_subcarrier_pos"],
                self.dmrs_subcarrier_pos,
            )
            outputs = self.neural_engine.launch(use_graph=True)
        self.neural_engine.stream.synchronize()
        with self.stream:
            llrs = cp.take(
                outputs["output_1"][0, ...], self.data_symbol_indices, axis=3
            )
            coded = self.neural_derate.derate_match(
                input_llrs=[llrs], pusch_configs=self.pusch_configs
            )
            blocks = self.neural_decoder.decode(
                input_llrs=coded, pusch_configs=self.pusch_configs
            )
            return self.neural_crc.check_crc(
                input_bits=blocks, pusch_configs=self.pusch_configs
            )

    def prepare_neural_ipc(self, forward: cp.ndarray) -> float:
        """Write this request's real NeuralRx inputs into a flat IPC buffer."""

        rx_shape = (1, 3276, 12, 4)
        ce_shape = (1, 4914, 1, 4)
        rx_elements = int(np.prod(rx_shape))
        ce_elements = int(np.prod(ce_shape))
        expected = 2 * rx_elements + 2 * ce_elements
        if forward.dtype != cp.float32 or forward.size != expected:
            raise RuntimeError(
                f"IPC forward contract mismatch {forward.shape}/{forward.dtype}"
            )

        def section(offset: int, count: int, shape: tuple[int, ...]):
            value = forward[offset:offset + count].reshape(shape, order="C")
            if not value.flags.c_contiguous:
                raise RuntimeError("IPC forward section is not contiguous")
            return value

        with cp.cuda.Device(self.device), self.stream:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            channel = self.neural_channel_estimator.estimate(
                rx_slot=self.rx_slot,
                slot=self.slot,
                pusch_configs=self.pusch_configs,
            )
            rx_window = cp.asarray(self.rx_slot)[
                None,
                :,
                self.start_sym:self.start_sym + self.num_symbols,
                :,
            ]
            estimate = cp.asarray(channel[0])
            estimate_input = cp.transpose(estimate, (0, 3, 1, 2)).reshape(
                estimate.shape[0] * estimate.shape[3],
                estimate.shape[1],
                estimate.shape[2],
            )[None, ...]
            offset = 0
            cp.copyto(section(offset, rx_elements, rx_shape), rx_window.real)
            offset += rx_elements
            cp.copyto(section(offset, rx_elements, rx_shape), rx_window.imag)
            offset += rx_elements
            cp.copyto(section(offset, ce_elements, ce_shape), estimate_input.real)
            offset += ce_elements
            cp.copyto(section(offset, ce_elements, ce_shape), estimate_input.imag)
            end.record()
            end.synchronize()
            return float(cp.cuda.get_elapsed_time(begin, end))

    def complete_neural_ipc(
        self, backward: cp.ndarray
    ) -> tuple[float, bool, int, int]:
        """Finish LDPC/CRC from an IPC NeuralRx output for this request."""

        expected_shape = (2, 1, 3276, 12)
        if backward.dtype != cp.float32 or backward.size != int(np.prod(expected_shape)):
            raise RuntimeError(
                f"IPC backward contract mismatch {backward.shape}/{backward.dtype}"
            )

        def postprocess():
            llrs_full = backward.reshape(expected_shape, order="C")
            llrs = cp.take(llrs_full, self.data_symbol_indices, axis=3)
            coded = self.neural_derate.derate_match(
                input_llrs=[llrs], pusch_configs=self.pusch_configs
            )
            blocks = self.neural_decoder.decode(
                input_llrs=coded, pusch_configs=self.pusch_configs
            )
            return self.neural_crc.check_crc(
                input_bits=blocks, pusch_configs=self.pusch_configs
            )

        return self._measure(postprocess)

    def profile_neural_once(self) -> dict:
        """Run one NeuralRx request and return GPU time for each pipeline stage."""

        names = (
            "channel_estimation",
            "tensor_preparation",
            "tensorrt",
            "dmrs_removal",
            "derate_match",
            "ldpc_decode",
            "crc",
        )
        events = [cp.cuda.Event() for _ in range(len(names) + 1)]
        with cp.cuda.Device(self.device), self.stream:
            events[0].record()
            channel = self.neural_channel_estimator.estimate(
                rx_slot=self.rx_slot,
                slot=self.slot,
                pusch_configs=self.pusch_configs,
            )
            events[1].record()
            rx_window = cp.asarray(self.rx_slot)[
                None,
                :,
                self.start_sym : self.start_sym + self.num_symbols,
                :,
            ]
            estimate = cp.asarray(channel[0])
            estimate_input = cp.transpose(estimate, (0, 3, 1, 2)).reshape(
                estimate.shape[0] * estimate.shape[3],
                estimate.shape[1],
                estimate.shape[2],
            )[None, ...]
            inputs = {
                "rx_slot_real": cp.ascontiguousarray(rx_window.real).astype(cp.float32),
                "rx_slot_imag": cp.ascontiguousarray(rx_window.imag).astype(cp.float32),
                "h_hat_real": cp.ascontiguousarray(estimate_input.real).astype(cp.float32),
                "h_hat_imag": cp.ascontiguousarray(estimate_input.imag).astype(cp.float32),
                "active_dmrs_ports": self.active_dmrs_ports,
                "dmrs_ofdm_pos": self.dmrs_ofdm_pos,
                "dmrs_subcarrier_pos": self.dmrs_subcarrier_pos,
            }
            events[2].record()
            outputs = self.neural_engine_wrapper.run(inputs)
            events[3].record()
            llrs = cp.take(
                outputs["output_1"][0, ...], self.data_symbol_indices, axis=3
            )
            events[4].record()
            coded = self.neural_derate.derate_match(
                input_llrs=[llrs], pusch_configs=self.pusch_configs
            )
            events[5].record()
            blocks = self.neural_decoder.decode(
                input_llrs=coded, pusch_configs=self.pusch_configs
            )
            events[6].record()
            output = self.neural_crc.check_crc(
                input_bits=blocks, pusch_configs=self.pusch_configs
            )
            events[7].record()
            events[-1].synchronize()
            correct, crc_failures, payload_mismatches = self._verify(output)
        stages = {
            name: float(cp.cuda.get_elapsed_time(events[index], events[index + 1]))
            for index, name in enumerate(names)
        }
        return {
            "stages_gpu_ms": stages,
            "total_gpu_ms": sum(stages.values()),
            "correct": correct,
            "crc_failures": crc_failures,
            "payload_mismatches": payload_mismatches,
        }

    def run_conventional(self) -> tuple[float, bool, int, int]:
        return self._measure(self.conventional_once)

    def run_neural(self) -> tuple[float, bool, int, int]:
        return self._measure(self.neural_once)
