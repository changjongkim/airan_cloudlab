#!/usr/bin/env python3
"""Generate uplink slots with the NVlabs neural_rx link simulation and decode them with its
pre-trained neural receiver (TensorFlow, Sionna 0.18).

Run with the working directory at ``third_party/neural_rx/scripts``.  For each Eb/No point it
writes the received slots in Aerial layout, the transmitted payload bits, and whether the
neural receiver recovered every payload bit per user.  ``nv_conventional.py`` then decodes the
same slots with Aerial's conventional cuPHY receiver.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", required=True, help="e.g. nrx_rt.cfg")
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--num-tx", type=int, default=2)
parser.add_argument("--channel", default="DoubleTDLlow")
parser.add_argument("--speed", type=float, default=None, help="UE speed in m/s (min = max)")
parser.add_argument("--mcs", type=int, default=None, help="MCS index (default: the one of the base config)")
parser.add_argument("--esno", default=None,
                    help="Es/No points in dB; converted to the Eb/No of the configuration (overrides --ebno)")
parser.add_argument("--ebno", default="0,1,2,3,4,5")
parser.add_argument("--slots", type=int, default=64, help="slots per Eb/No point")
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--gpu", default="")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.get_logger().setLevel("ERROR")
for device in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

from utils import Parameters, NeuralReceiverONNX, load_weights      # noqa: E402
from utils import DataGeneratorAerial, DataEvaluator                # noqa: E402

# A copy of the base configuration with our evaluation settings (the label keeps pointing at
# the original weights).
base = Path("../config") / args.base_config
text = base.read_text()
label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
text = re.sub(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}", text)
text = re.sub(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"', text)
if args.speed is not None:
    text = re.sub(r"max_ut_velocity_eval\s*=.*", f"max_ut_velocity_eval = {args.speed}", text)
    text = re.sub(r"min_ut_velocity_eval\s*=.*", f"min_ut_velocity_eval = {args.speed}", text)
if args.mcs is not None:
    text = re.sub(r"(?m)^mcs_index\s*=.*", f"mcs_index = [{args.mcs}]", text)
name = f"bs_{label}_{args.prbs}_{args.channel}_{args.num_tx}{'' if args.mcs is None else '_m' + str(args.mcs)}.cfg"
(Path("../config") / name).write_text(text)

params = Parameters(name, training=False, system="nrx", num_tx_eval=args.num_tx)
generator = DataGeneratorAerial(params)
evaluator = DataEvaluator(params)
receiver = NeuralReceiverONNX(
    num_it=params.num_nrx_iter, d_s=params.d_s, num_units_init=params.num_units_init,
    num_units_agg=params.num_units_agg, num_units_state=params.num_units_state,
    num_units_readout=params.num_units_readout,
    num_bits_per_symbol=params.transmitters[0]._num_bits_per_symbol,
    layer_type_dense=params.layer_type_dense, layer_type_conv=params.layer_type_conv,
    layer_type_readout=params.layer_type_readout, nrx_dtype=params.nrx_dtype,
    num_tx=params.max_num_tx, num_rx_ant=params.num_rx_antennas)
inputs, _, _, _ = generator(1, 10.0)
receiver(inputs)
load_weights(receiver, f"../weights/{label}_weights")
receiver._cgnn.num_it = params.num_nrx_iter_eval

tx = params.transmitters[0]
meta = {
    "base_config": args.base_config, "label": label, "prbs": args.prbs, "num_tx": args.num_tx,
    "channel": args.channel, "speed": args.speed, "num_rx_ant": int(params.num_rx_antennas),
    "mcs_index": int(params.mcs_index[0]), "mcs_table": int(params.mcs_table),
    "tb_size_bits": int(tx._tb_size), "num_bits_per_symbol": int(tx._num_bits_per_symbol),
    "target_coderate": float(tx._target_coderate),
    "dmrs_ofdm_pos": np.asarray(inputs[5]).tolist(), "dmrs_subcarrier_pos": np.asarray(inputs[6]).tolist(),
    "n_cell_id": int(params.n_cell_id), "slot_number": int(params.slot_number),
    "ebno_db": [float(e) for e in args.ebno.split(",")], "slots": args.slots,
}
if args.esno:
    # Es/No (per resource element) -> rate-adjusted Eb/No of the link simulation.
    import math
    shift = 10 * math.log10(meta["num_bits_per_symbol"] * meta["target_coderate"])
    meta["esno_db"] = [float(e) for e in args.esno.split(",")]
    meta["ebno_db"] = [round(e - shift, 3) for e in meta["esno_db"]]
print(json.dumps(meta), flush=True)

slots, bits, ok, ebnos, seconds = [], [], [], [], []
for ebno in meta["ebno_db"]:
    done = 0
    while done < args.slots:
        n = min(args.batch, args.slots - done)
        nrx_inputs, c, b, _ = generator(n, ebno)
        start = time.perf_counter()
        llr, _ = receiver(nrx_inputs)
        seconds.append((time.perf_counter() - start) / n)
        _, _, u_hat = evaluator(llr, c)
        b_np = np.asarray(b).astype(np.uint8)
        u_np = np.asarray(u_hat).astype(np.uint8)
        slots.append((nrx_inputs[0] + 1j * nrx_inputs[1]).astype(np.complex64))
        bits.append(np.packbits(b_np, axis=-1))
        ok.append((b_np == u_np).all(axis=-1))
        ebnos += [ebno] * n
        done += n
    good = np.concatenate(ok)[-args.slots:]
    print(f"ebno {ebno}: neural receiver TB pass {good.mean():.3f} ({good.sum()}/{good.size})", flush=True)

args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output, rx=np.concatenate(slots), payload=np.concatenate(bits), nrx_ok=np.concatenate(ok),
         ebno=np.asarray(ebnos, dtype=np.float32), meta=json.dumps(meta),
         nrx_seconds=float(np.median(seconds)))
print("saved", args.output, flush=True)
