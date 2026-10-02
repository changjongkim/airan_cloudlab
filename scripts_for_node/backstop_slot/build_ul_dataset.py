#!/usr/bin/env python3
"""Pre-generate received uplink slots for one profile.

Payloads are encoded by pyAerial's NR transmit chain and sent through a
Sionna 3GPP CDL channel with AWGN, as in the P3-qualified NeuralRx mode.  The
timed workers only read these slots, so TensorFlow never runs in the timed
path.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import json
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np
import sionna
import tensorflow as tf

from aerial.phy5g.ldpc.util import random_tb
from aerial.phy5g.pdsch import PdschTx

from slot_radio import mcs_params
from ul_profiles import DMRS_SCRM_ID, NUM_PRBS, NUM_RX_ANT, NUM_SUBCARRIERS, PROFILES, RNTI

UL_SLOTS = (4, 9, 14, 19)   # the U slot of DDDSU at 30 kHz, two per 5 ms


def build_channel(profile, batch_streams: int):
    grid = sionna.phy.ofdm.ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=4096,
        subcarrier_spacing=30e3,
        num_tx=1,
        num_streams_per_tx=batch_streams,
        cyclic_prefix_length=288,
        num_guard_carriers=(410, 410),
        dc_null=False,
        pilot_pattern=None,
        pilot_ofdm_symbol_indices=None,
    )
    mapper = sionna.phy.ofdm.ResourceGridMapper(grid)
    remove_guards = sionna.phy.ofdm.RemoveNulledSubcarriers(grid)
    ue_array = sionna.phy.channel.tr38901.Antenna(
        polarization="single" if profile.rank == 1 else "dual",
        polarization_type="V" if profile.rank == 1 else "cross",
        antenna_pattern="38.901",
        carrier_frequency=3.5e9,
    )
    gnb_array = sionna.phy.channel.tr38901.AntennaArray(
        num_rows=1, num_cols=2, polarization="dual", polarization_type="cross",
        antenna_pattern="38.901", carrier_frequency=3.5e9,
    )
    model = sionna.phy.channel.tr38901.CDL(
        profile.cdl_model, profile.delay_spread_ns * 1e-9, 3.5e9,
        ue_array, gnb_array, "uplink", min_speed=0.8333,
    )
    channel = sionna.phy.channel.OFDMChannel(
        model, grid, add_awgn=True, normalize_channel=True, return_channel=False,
    )
    return mapper, remove_guards, channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--payload-seed", type=int, required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    rank = profile.rank
    tf.random.set_seed(args.channel_seed)
    np.random.seed(args.payload_seed)
    rng = np.random.default_rng(args.payload_seed)
    mod_order, code_rate = mcs_params(profile)
    transmitter = PdschTx(cell_id=DMRS_SCRM_ID, num_rx_ant=rank, num_tx_ant=rank)
    mapper, remove_guards, channel = build_channel(profile, rank)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rx_path = args.out_dir / f"{profile.name}_rx.npy"
    received = np.lib.format.open_memmap(
        rx_path, mode="w+", dtype=np.complex64,
        shape=(args.count, NUM_SUBCARRIERS, 14, NUM_RX_ANT),
    )
    blocks, slots, esno = [], [], []
    begin = time.time()
    for first in range(0, args.count, args.batch):
        size = min(args.batch, args.count - first)
        grids = []
        for index in range(size):
            tb = random_tb(
                mod_order=mod_order, code_rate=code_rate,
                dmrs_syms=profile.dmrs_syms, num_prbs=NUM_PRBS,
                start_sym=profile.start_sym, num_symbols=profile.num_symbols,
                num_layers=rank,
            )
            slot = UL_SLOTS[(first + index) % len(UL_SLOTS)]
            grid = transmitter.run(
                tb_inputs=[tb], num_ues=1, slot=slot,
                num_dmrs_cdm_grps_no_data=2, dmrs_scrm_ids=[DMRS_SCRM_ID],
                start_prb=0, num_prbs=NUM_PRBS, dmrs_syms=profile.dmrs_syms,
                start_sym=profile.start_sym, num_symbols=profile.num_symbols,
                scids=[0], layers=[rank], dmrs_ports=[profile.dmrs_port_mask],
                rntis=[RNTI], data_scids=[0], code_rates=[int(code_rate * 10)],
                mod_orders=[mod_order],
            )
            grids.append(cp.asnumpy(cp.asarray(grid)).reshape(NUM_SUBCARRIERS, 14, rank))
            blocks.append(np.asarray(tb, dtype=np.uint8))
            slots.append(slot)
        values = rng.uniform(profile.esno_db_low, profile.esno_db_high, size)
        esno.extend(float(v) for v in values)
        tx = tf.convert_to_tensor(np.stack(grids), dtype=tf.complex64)   # B,F,T,S
        tx = tf.transpose(tx, (0, 3, 2, 1))                                # B,S,T,F
        tx = tf.reshape(tx, (size, 1, rank, -1))
        noise = tf.constant(10.0 ** (-values / 10.0), tf.float32)[:, None, None, None, None]
        rx = remove_guards(channel(mapper(tx), noise))[:, 0]               # B,A,T,F
        received[first:first + size] = tf.transpose(rx, (0, 3, 2, 1)).numpy()
        print(f"{profile.name}: {first + size}/{args.count} "
              f"({time.time() - begin:.1f} s)", flush=True)
    received.flush()
    np.save(args.out_dir / f"{profile.name}_tb.npy", np.stack(blocks))
    meta = {
        "schema": "backstop-slot-ul-dataset-v1",
        "profile": profile.to_dict(),
        "count": args.count,
        "tb_bytes": int(blocks[0].size),
        "mod_order": mod_order,
        "code_rate_x1024": code_rate,
        "slots": slots,
        "esno_db": esno,
        "payload_seed": args.payload_seed,
        "channel_seed": args.channel_seed,
        "sionna_version": sionna.__version__,
        "tensorflow_version": tf.__version__,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out_dir / f"{profile.name}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
