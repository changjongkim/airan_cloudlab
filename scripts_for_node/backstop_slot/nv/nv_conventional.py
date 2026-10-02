#!/usr/bin/env python3
"""Decode the slots written by ``nv_dataset.py`` with Aerial's conventional cuPHY receiver.

Both receivers see the same received slot.  Per Eb/No point this prints how many TBs each
receiver recovered and how many only one of them recovered.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.pusch import PuschRx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tdi", type=int, default=1)
    parser.add_argument("--eq", type=int, default=1)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    rx, payload, nrx_ok, ebno = data["rx"], data["payload"], data["nrx_ok"], data["ebno"]
    num_tx, prbs = int(meta["num_tx"]), int(meta["prbs"])
    tb_bytes = meta["tb_size_bits"] // 8
    mod_order, code_rate = get_mcs(meta["mcs_index"], 1)
    dmrs_syms = [0] * 14
    for position in meta["dmrs_ofdm_pos"][0]:
        dmrs_syms[position] = 1
    # DMRS port of each user from its first pilot subcarrier: 0 -> port 0, 1 -> port 2.
    ports = [0 if row[0] == 0 else 2 for row in meta["dmrs_subcarrier_pos"]][:num_tx]
    ues = [PuschUeConfig(
        scid=1, layers=1, dmrs_ports=1 << port, rnti=1, data_scid=1, mcs_table=0,
        mcs_index=meta["mcs_index"], code_rate=int(code_rate * 10), mod_order=int(mod_order),
        tb_size=tb_bytes) for port in ports]
    configs = [PuschConfig(
        ue_configs=ues, num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=1, start_prb=0, num_prbs=prbs,
        prg_size=1, num_ul_streams=num_tx, dmrs_syms=dmrs_syms, dmrs_max_len=1,
        dmrs_add_ln_pos=len(meta["dmrs_ofdm_pos"][0]) - 1, start_sym=0, num_symbols=14)]
    receiver = PuschRx(cell_id=meta["n_cell_id"], num_rx_ant=meta["num_rx_ant"],
                       num_tx_ant=meta["num_rx_ant"], enable_pusch_tdi=args.tdi, eq_coeff_algo=args.eq)

    conv_ok = np.zeros_like(nrx_ok)
    crc_ok = np.zeros_like(nrx_ok)
    seconds = []
    for i in range(rx.shape[0]):
        slot = cp.asarray(np.asfortranarray(rx[i]))
        start = time.perf_counter()
        crcs, tbs = receiver.run(rx_slot=slot, slot=meta["slot_number"], pusch_configs=configs)
        seconds.append(time.perf_counter() - start)
        for u in range(num_tx):
            crc_ok[i, u] = int(np.asarray(crcs).reshape(-1)[u]) == 0
            got = np.asarray(tbs[u])[:tb_bytes]
            conv_ok[i, u] = bool(crc_ok[i, u] and np.array_equal(got, payload[i, u, :tb_bytes]))

    rows = []
    print("| Eb/No | TBs | conventional | NeuralRx | both | only conventional | only NeuralRx | neither |")
    print("|---|---|---|---|---|---|---|---|")
    for e in sorted(set(ebno.tolist())):
        sel = ebno == e
        c, n = conv_ok[sel].reshape(-1), nrx_ok[sel].reshape(-1)
        row = {"ebno_db": float(e), "tbs": int(c.size), "conv": int(c.sum()), "nrx": int(n.sum()),
               "both": int((c & n).sum()), "conv_only": int((c & ~n).sum()),
               "nrx_only": int((~c & n).sum()), "neither": int((~c & ~n).sum()),
               "conv_crc_pass_wrong_payload": int((crc_ok[sel].reshape(-1) & ~c).sum())}
        rows.append(row)
        print(f"| {e:g} | {row['tbs']} | {row['conv']} | {row['nrx']} | {row['both']} | {row['conv_only']} | "
              f"{row['nrx_only']} | {row['neither']} |")
    args.output.write_text(json.dumps({"meta": meta, "rows": rows,
                                       "conv_wall_ms_p50": float(np.median(seconds) * 1e3),
                                       "nrx_tf_seconds_per_slot": float(data["nrx_seconds"])}, indent=2))


if __name__ == "__main__":
    main()
