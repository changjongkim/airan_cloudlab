"""Persistent-HARQ adapter around Aerial's monolithic cuPHY PUSCH pipeline."""

from __future__ import annotations

from typing import List

import cupy as cp
import cuda.bindings.runtime as cudart

from aerial.phy5g.config import PuschConfig, _pusch_config_to_cuphy
from aerial.phy5g.pusch import PuschRx
from aerial.pycuphy.util import get_pusch_dyn_prms_phase_2
from aerial.util.cuda import check_cuda_errors


class PersistentPuschRx(PuschRx):
    """Keep fixed-mode HARQ buffers out of the per-request lifecycle."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._harq_buffers: list[int] = []
        self._harq_capacities: list[int] = []

    def close(self) -> None:
        buffers, self._harq_buffers = self._harq_buffers, []
        self._harq_capacities = []
        for pointer in buffers:
            check_cuda_errors(cudart.cudaFree(pointer))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_harq(self, required: list[int]) -> None:
        if not self._harq_buffers:
            self._harq_buffers = [
                check_cuda_errors(cudart.cudaMalloc(size)) for size in required
            ]
            self._harq_capacities = list(required)
        if len(required) != len(self._harq_buffers) or any(
            need > capacity for need, capacity in zip(required, self._harq_capacities)
        ):
            raise RuntimeError(
                "PUSCH mode exceeds the readiness-time persistent HARQ allocation"
            )

    def run_persistent(
        self,
        *,
        rx_slot: cp.ndarray,
        slot: int,
        pusch_configs: List[PuschConfig],
    ):
        with cp.cuda.ExternalStream(int(self.cuda_stream)):
            rx_slot = cp.array(rx_slot, dtype=cp.complex64, order="F", copy=False)
        dynamic = _pusch_config_to_cuphy(
            cuda_stream=self.cuda_stream,
            rx_data=[rx_slot],
            slot=slot,
            pusch_configs=pusch_configs,
        )
        tb_sizes = [
            ue.tb_size for group in pusch_configs for ue in group.ue_configs
        ]

        # Phase 1 computes the required buffer capacity. The allocation occurs
        # only on the first readiness call and is thereafter fail-closed.
        self.pusch_pipeline.setup_pusch_rx(dynamic)
        required = [int(value) for value in dynamic.dataOut.harqBufferSizeInBytes]
        self._ensure_harq(required)
        for pointer, size in zip(self._harq_buffers, required):
            check_cuda_errors(
                cudart.cudaMemsetAsync(pointer, 0, size, self.cuda_stream)
            )
        dynamic = get_pusch_dyn_prms_phase_2(dynamic, self._harq_buffers)
        self.pusch_pipeline.setup_pusch_rx(dynamic)
        self.pusch_pipeline.run_pusch_rx()

        start_offsets = list(dynamic.dataOut.startOffsetsTbPayload)
        start_offsets.append(int(dynamic.dataOut.totNumPayloadBytes[0]))
        transport_blocks = []
        for index, size in enumerate(tb_sizes):
            value = dynamic.dataOut.tbPayloads[
                start_offsets[index]:start_offsets[index + 1]
            ]
            transport_blocks.append(value[:size])
        return dynamic.dataOut.tbCrcs, transport_blocks
