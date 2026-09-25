#!/usr/bin/env python3
"""Valid paired PUSCH Tx/Rx work for SoftWall timing experiments."""

from __future__ import annotations

import cupy as cp
import numpy as np

from aerial.phy5g.algorithms import (
    ChannelEqualizer,
    ChannelEstimator,
    NoiseIntfEstimator,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import CrcChecker, LdpcDecoder, LdpcDeRateMatch
from aerial.phy5g.ldpc.util import random_tb
from aerial.phy5g.pdsch import PdschTx
from aerial.util.cuda import get_cuda_stream


class PairedConventionalRan:
    """Generate one valid 273-PRB slot, then repeatedly decode and verify it."""

    def __init__(self, *, seed: int = 20260920, device: int = 0) -> None:
        cp.cuda.runtime.setDevice(device)
        np.random.seed(seed)
        self.device = device
        self.slot = 2
        self.dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.mod_order = 6
        self.code_rate = 910
        self.reference_tb = random_tb(
            mod_order=self.mod_order,
            code_rate=self.code_rate,
            dmrs_syms=self.dmrs_syms,
            num_prbs=273,
            start_sym=0,
            num_symbols=14,
            num_layers=1,
        )
        transmitter = PdschTx(cell_id=41, num_rx_ant=4, num_tx_ant=4)
        self.rx_slot = transmitter.run(
            tb_inputs=[self.reference_tb],
            num_ues=1,
            slot=self.slot,
            start_prb=0,
            num_prbs=273,
            dmrs_syms=self.dmrs_syms,
            start_sym=0,
            num_symbols=14,
            scids=[0],
            layers=[1],
            dmrs_ports=[1],
            rntis=[1000],
            data_scids=[41],
            precoding_matrices=[
                np.array([[0.5 + 0.j, 0. + 0.5j, 0. + 0.5j, -0.5 + 0.j]])
            ],
            code_rates=[self.code_rate * 10],
            mod_orders=[self.mod_order],
        )
        ue = PuschUeConfig(
            scid=0,
            layers=1,
            dmrs_ports=1,
            rnti=1000,
            data_scid=41,
            mcs_table=0,
            mcs_index=0,
            code_rate=self.code_rate * 10,
            mod_order=self.mod_order,
            tb_size=len(self.reference_tb),
        )
        self.pusch_configs = [PuschConfig(
            ue_configs=[ue],
            num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_id=41,
            start_prb=0,
            num_prbs=273,
            prg_size=1,
            num_ul_streams=4,
            dmrs_syms=self.dmrs_syms,
            dmrs_max_len=1,
            dmrs_add_ln_pos=1,
            start_sym=0,
            num_symbols=14,
        )]
        self.stream_handle = get_cuda_stream()
        self.stream = cp.cuda.ExternalStream(int(self.stream_handle))
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=4, ch_est_algo=1, cuda_stream=self.stream_handle
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
        self.derate_matcher = LdpcDeRateMatch(
            enable_scrambling=True, cuda_stream=self.stream_handle
        )
        self.decoder = LdpcDecoder(cuda_stream=self.stream_handle)
        self.crc = CrcChecker(cuda_stream=self.stream_handle)

    def one_cell(self):
        channel = self.channel_estimator.estimate(
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
        coded = self.derate_matcher.derate_match(
            input_llrs=llrs, pusch_configs=self.pusch_configs
        )
        blocks = self.decoder.decode(
            input_llrs=coded, pusch_configs=self.pusch_configs
        )
        return self.crc.check_crc(
            input_bits=blocks, pusch_configs=self.pusch_configs
        )

    def run_cells(self, cells: int) -> tuple[float, bool, int, int]:
        outputs = []
        with cp.cuda.Device(self.device), self.stream:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            for _ in range(cells):
                outputs.append(self.one_cell())
            end.record()
            end.synchronize()
            gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
            crc_failures = 0
            payload_mismatches = 0
            for transport_blocks, crc_values in outputs:
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
                if int(crc_array.reshape(-1)[0]) != 0:
                    crc_failures += 1
                if not np.array_equal(payload, self.reference_tb):
                    payload_mismatches += 1
            self.stream.synchronize()
        return (
            gpu_ms,
            crc_failures == 0 and payload_mismatches == 0,
            crc_failures,
            payload_mismatches,
        )
