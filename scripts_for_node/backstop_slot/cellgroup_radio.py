"""Conventional cuPHY PUSCH receiver for a group of cells in one pipeline call.

Aerial's cuPHY takes a cell group per setup/run.  ``CellGroupRx`` decodes one slot of every
cell on a GPU with a single pipeline, instead of one process and one pipeline per cell.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import cuda.bindings.runtime as cudart

from aerial import pycuphy
from aerial.phy5g.config import _pusch_config_to_cuphy
from aerial.pycuphy.types import PuschCellDynPrm, PuschEqCoefAlgoType
from aerial.pycuphy.util import get_pusch_dyn_prms_phase_2, get_pusch_stat_prms
from aerial.util.cuda import check_cuda_errors

from ul_profiles import DMRS_SCRM_ID, NUM_RX_ANT


class CellGroupRx:
    def __init__(self, cells: int, stream: cp.cuda.Stream) -> None:
        stat = get_pusch_stat_prms(
            cell_id=DMRS_SCRM_ID, num_rx_ant=NUM_RX_ANT, num_tx_ant=NUM_RX_ANT,
            enable_pusch_tdi=1, eq_coeff_algo=PuschEqCoefAlgoType(1),
        )
        stat = stat._replace(
            nMaxCells=np.uint16(cells), nMaxCellsPerSlot=np.uint16(cells),
            cellStatPrms=[stat.cellStatPrms[0]] * cells,
        )
        self.cells = cells
        self.stream = int(stream.ptr)
        self.pipeline = pycuphy.PuschPipeline(stat, self.stream)
        self.harq: dict[int, tuple[int, int]] = {}
        self.last = None

    def run(self, slots: list[cp.ndarray], slot_numbers: list[int], configs: list,
            members: list[int] | None = None) -> tuple[list[bool], list[int]]:
        """Decode one slot for each listed cell of the group.

        ``members`` are the group's cell positions being decoded this period (default: all,
        in order).  Returns (CRC pass, failed code blocks) per listed cell.
        """
        members = list(range(len(configs))) if members is None else members
        dynamic = _pusch_config_to_cuphy(
            cuda_stream=self.stream, rx_data=slots, slot=slot_numbers[0], pusch_configs=configs,
        )
        group = dynamic.cellGrpDynPrm
        dynamic = dynamic._replace(
            cellGrpDynPrm=group._replace(
                cellPrms=[PuschCellDynPrm(cellPrmStatIdx=np.uint16(members[i]), cellPrmDynIdx=np.uint16(i),
                                          slotNum=np.uint16(s)) for i, s in enumerate(slot_numbers)],
                ueGrpPrms=[g._replace(cellPrmIdx=i) for i, g in enumerate(group.ueGrpPrms)],
            ),
            dataOut=dynamic.dataOut._replace(
                cbCrcs=np.ones([4000], dtype=np.uint32),
                tbPayloads=np.zeros([1_000_000], dtype=np.uint8),
            ),
        )
        self.pipeline.setup_pusch_rx(dynamic)
        required = [int(v) for v in dynamic.dataOut.harqBufferSizeInBytes]
        buffers = []
        for member, size in zip(members, required):
            if member not in self.harq:
                self.harq[member] = (check_cuda_errors(cudart.cudaMalloc(size)), size)
            pointer, capacity = self.harq[member]
            if size > capacity:
                raise RuntimeError("PUSCH mode exceeds the persistent HARQ allocation")
            check_cuda_errors(cudart.cudaMemsetAsync(pointer, 0, size, self.stream))
            buffers.append(pointer)
        dynamic = get_pusch_dyn_prms_phase_2(dynamic, buffers)
        self.pipeline.setup_pusch_rx(dynamic)
        self.pipeline.run_pusch_rx()
        out = dynamic.dataOut
        self.last = out
        count = len(configs)
        crcs = np.asarray(out.tbCrcs).reshape(-1)
        offsets = [int(v) for v in np.asarray(out.startOffsetsCbCrc).reshape(-1)[:count]] + [int(out.totNumCbs[0])]
        cb = np.asarray(out.cbCrcs)
        ok = [int(crcs[i]) == 0 for i in range(count)]
        cb_fail = [int((cb[offsets[i]:offsets[i + 1]] != 0).sum()) for i in range(count)]
        return ok, cb_fail

    def payload(self, index: int, size: int) -> np.ndarray:
        """TB payload of UE ``index`` from the last run."""
        start = int(np.asarray(self.last.startOffsetsTbPayload).reshape(-1)[index])
        return np.asarray(self.last.tbPayloads[start:start + size])
