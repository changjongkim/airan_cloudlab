#!/usr/bin/env python3
"""Reusable full conventional PUSCH path for same-GPU SoftWall experiments."""

from __future__ import annotations

import time

import cupy as cp
import numpy as np
from cuda.bindings import runtime as cudart

from aerial.phy5g.algorithms import (
    ChannelEqualizer,
    ChannelEstimator,
    NoiseIntfEstimator,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import (
    CrcChecker,
    LdpcDecoder,
    LdpcDeRateMatch,
    get_mcs,
    get_tb_size,
)
from aerial.util.cuda import get_cuda_stream


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0, "mean": None, "p50": None, "p95": None,
            "p99": None, "p99_9": None, "min": None, "max": None,
        }
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "p99_9": float(np.percentile(data, 99.9)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 250_000:
            time.sleep((remaining_ns - 100_000) / 1e9)


class ConventionalRan:
    """CE + noise/interference estimation + EQ + de-rate-match + LDPC + CRC."""

    def __init__(
        self,
        *,
        device: int = 0,
        num_prbs: int = 273,
        mcs_index: int = 2,
        num_rx_ant: int = 4,
        seed: int = 20260920,
    ) -> None:
        self.device = device
        self.num_prbs = num_prbs
        self.mcs_index = mcs_index
        self.num_rx_ant = num_rx_ant
        self.seed = seed

        start_sym = 2
        num_symbols = 12
        dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        mod_order, code_rate = get_mcs(mcs_index, 1)
        tb_size = get_tb_size(
            mod_order=mod_order,
            code_rate=code_rate,
            dmrs_syms=dmrs_syms,
            num_prbs=num_prbs,
            start_sym=start_sym,
            num_symbols=num_symbols,
            num_layers=1,
        )
        ue = PuschUeConfig(
            scid=0,
            layers=1,
            dmrs_ports=1,
            rnti=1234,
            data_scid=0,
            mcs_table=0,
            mcs_index=mcs_index,
            code_rate=int(code_rate * 10),
            mod_order=mod_order,
            tb_size=tb_size // 8,
        )
        self.pusch_configs = [PuschConfig(
            ue_configs=[ue],
            num_dmrs_cdm_grps_no_data=2,
            dmrs_scrm_id=41,
            start_prb=0,
            num_prbs=num_prbs,
            dmrs_syms=dmrs_syms,
            dmrs_max_len=1,
            dmrs_add_ln_pos=0,
            start_sym=start_sym,
            num_symbols=num_symbols,
        )]

        rng = np.random.default_rng(seed)
        num_subcarriers = num_prbs * 12
        host_rx = (
            rng.standard_normal((num_subcarriers, 14, num_rx_ant))
            + 1j * rng.standard_normal((num_subcarriers, 14, num_rx_ant))
        ).astype(np.complex64)

        with cp.cuda.Device(device):
            cudart.cudaSetDevice(device)
            self.stream_handle = get_cuda_stream()
            self.stream = cp.cuda.ExternalStream(int(self.stream_handle))
            self.rx_slot = cp.asarray(host_rx, order="F")
            self.channel_estimator = ChannelEstimator(
                num_rx_ant=num_rx_ant, cuda_stream=self.stream_handle)
            self.noise_estimator = NoiseIntfEstimator(
                num_rx_ant=num_rx_ant,
                eq_coeff_algo=1,
                cuda_stream=self.stream_handle,
            )
            self.equalizer = ChannelEqualizer(
                num_rx_ant=num_rx_ant,
                enable_pusch_tdi=0,
                eq_coeff_algo=1,
                cuda_stream=self.stream_handle,
            )
            self.derate_matcher = LdpcDeRateMatch(
                enable_scrambling=True, cuda_stream=self.stream_handle)
            self.decoder = LdpcDecoder(cuda_stream=self.stream_handle)
            self.crc = CrcChecker(cuda_stream=self.stream_handle)

    def one_cell(self) -> None:
        h_est = self.channel_estimator.estimate(
            rx_slot=self.rx_slot, slot=0, pusch_configs=self.pusch_configs)
        lw_inv, noise_var = self.noise_estimator.estimate(
            rx_slot=self.rx_slot,
            channel_est=h_est,
            slot=0,
            pusch_configs=self.pusch_configs,
        )
        llrs, _ = self.equalizer.equalize(
            rx_slot=self.rx_slot,
            channel_est=h_est,
            lw_inv=lw_inv,
            noise_var_pre_eq=noise_var,
            pusch_configs=self.pusch_configs,
        )
        coded = self.derate_matcher.derate_match(
            input_llrs=llrs, pusch_configs=self.pusch_configs)
        blocks = self.decoder.decode(
            input_llrs=coded, pusch_configs=self.pusch_configs)
        self.crc.check_crc(input_bits=blocks, pusch_configs=self.pusch_configs)

    def run_cells(self, cells: int) -> float:
        with cp.cuda.Device(self.device), self.stream:
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            begin.record()
            for _ in range(cells):
                self.one_cell()
            end.record()
            end.synchronize()
            return float(cp.cuda.get_elapsed_time(begin, end))
