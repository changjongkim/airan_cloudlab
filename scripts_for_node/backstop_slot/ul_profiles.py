"""Uplink cell profiles for the slot-scale Antiphase mode.

Every cell carries one full-band PUSCH allocation per uplink slot.  A weak
(cell-edge) UE is scheduled with rank 1 and QPSK, which is the geometry the
public pyAerial NeuralRx model accepts.  A strong UE is scheduled with rank 2
(two layers, SU-MIMO) and 16QAM (MCS 12); only the conventional receiver decodes it.
Rank adaptation makes this split realistic: low-SINR UEs transmit one layer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace

NUM_PRBS = 273
NUM_SUBCARRIERS = 12 * NUM_PRBS
NUM_SYMBOLS_SLOT = 14
NUM_RX_ANT = 4
SLOT_ELEMENTS = NUM_SUBCARRIERS * NUM_SYMBOLS_SLOT * NUM_RX_ANT
DMRS_SCRM_ID = 41
RNTI = 1234


@dataclass(frozen=True)
class UlProfile:
    name: str
    rank: int
    mcs_index: int          # TS 38.214 table 5.1.3.1-1
    start_sym: int
    num_symbols: int
    dmrs_positions: tuple[int, ...]
    enable_pusch_tdi: int
    cdl_model: str
    delay_spread_ns: float
    esno_db_low: float
    esno_db_high: float
    neural_eligible: bool
    # MU-MIMO profiles: one single-layer UE per DMRS port on the same PRBs.
    ue_dmrs_ports: tuple[int, ...] = ()
    scrm_id: int = DMRS_SCRM_ID
    rnti: int = RNTI
    scid: int = 0

    @property
    def num_ue(self) -> int:
        return max(1, len(self.ue_dmrs_ports))

    @property
    def dmrs_syms(self) -> list[int]:
        mask = [0] * NUM_SYMBOLS_SLOT
        for position in self.dmrs_positions:
            mask[position] = 1
        return mask

    @property
    def dmrs_port_mask(self) -> int:
        return (1 << self.rank) - 1

    def to_dict(self) -> dict:
        return asdict(self)


# The weak profile is the P3-qualified NeuralRx contract (MCS 7, start symbol 0,
# DMRS 0/5/10, TDI on, one TX layer, four RX, CDL-D 100 ns).  Its Es/No band is
# where the two receivers disagree most often in the P3 holdout.
WEAK = UlProfile(
    name="weak_rank1",
    rank=1,
    mcs_index=7,
    start_sym=0,
    num_symbols=12,
    dmrs_positions=(0, 5, 10),
    enable_pusch_tdi=1,
    cdl_model="D",
    delay_spread_ns=100.0,
    esno_db_low=-3.8,
    esno_db_high=-3.2,
    neural_eligible=True,
)

# Rank-2 TBs above about 13 code blocks (MCS 13+, 14 KB) trigger a per-call
# device allocation in pyAerial's PUSCH setup that stalls the host for up to
# 11 ms; the production cuPHY driver preallocates this.  MCS 12 (16QAM,
# 12,549-byte TB) stays below that limit.
STRONG = UlProfile(
    name="strong_rank2",
    rank=2,
    mcs_index=12,
    start_sym=0,
    num_symbols=12,
    dmrs_positions=(0, 5, 10),
    enable_pusch_tdi=1,
    cdl_model="C",
    delay_spread_ns=100.0,
    esno_db_low=20.0,
    esno_db_high=24.0,
    neural_eligible=False,
)

# Extra weak profiles used only to test whether the failed-code-block signal
# generalizes beyond the evaluation profile.
WEAK_E = replace(WEAK, name="weak_cdl_e", cdl_model="E")
WEAK_WIDE = replace(WEAK, name="weak_wide_snr", esno_db_low=-4.6, esno_db_high=-2.4)
WEAK_MCS4 = replace(WEAK, name="weak_mcs4", mcs_index=4, esno_db_low=-6.8, esno_db_high=-4.6)

# Two UEs on the same 273 PRBs (MU-MIMO), 16QAM MCS 14, DMRS on symbols 2 and 11: the
# evaluation setting of the public NVlabs neural receiver (nrx_rt, "DoubleTDLlow" channel).
# Slots come from the NVlabs link simulation; Eb/No 2-3 dB is where the conventional
# receiver loses 12-45% of the TBs.
NV_MU2 = UlProfile(
    name="nv_mu2",
    rank=1,
    mcs_index=14,
    start_sym=0,
    num_symbols=14,
    dmrs_positions=(2, 11),
    enable_pusch_tdi=1,
    cdl_model="DoubleTDLlow",
    delay_spread_ns=0.0,
    esno_db_low=2.0,
    esno_db_high=3.0,
    neural_eligible=True,
    ue_dmrs_ports=(0, 2),
    scrm_id=1,
    rnti=1,
    scid=1,
)

# The same two-user cell at other 16QAM code rates (MCS 10-16 of table 1): the neural receiver
# outputs 16QAM LLRs whatever the code rate, so one engine serves all of them (link adaptation).
NV_MU2_BY_MCS = tuple(replace(NV_MU2, name=f"nv_mu2_m{m}", mcs_index=m) for m in range(10, 17))

PROFILES = {profile.name: profile
            for profile in (WEAK, STRONG, WEAK_E, WEAK_WIDE, WEAK_MCS4, NV_MU2) + NV_MU2_BY_MCS}
