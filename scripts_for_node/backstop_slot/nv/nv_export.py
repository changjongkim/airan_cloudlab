#!/usr/bin/env python3
"""Export a pre-trained NVlabs neural receiver to ONNX for a fixed bandwidth and user count,
and save one test vector (inputs, TensorFlow LLRs, payload) to validate the TensorRT engine.

Run with the working directory at ``third_party/neural_rx/scripts``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", required=True)
parser.add_argument("--prbs", type=int, default=273)
parser.add_argument("--num-tx", type=int, default=2)
parser.add_argument("--channel", default="DoubleTDLlow")
parser.add_argument("--ebno", type=float, default=6.0)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append("../")

import numpy as np                      # noqa: E402
import tensorflow as tf                 # noqa: E402
import tf2onnx                          # noqa: E402
import onnx                             # noqa: E402

from utils import Parameters, NeuralReceiverONNX, load_weights      # noqa: E402
from utils import DataGeneratorAerial, DataEvaluator                # noqa: E402

base = Path("../config") / args.base_config
text = base.read_text()
label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
text = re.sub(r"n_size_bwp_eval\s*=.*", f"n_size_bwp_eval = {args.prbs}", text)
text = re.sub(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{args.channel}"', text)
name = f"bs_{label}_{args.prbs}_{args.channel}_{args.num_tx}.cfg"
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
inputs, c, b, _ = generator(1, args.ebno)
llr, h_hat = receiver(inputs)
load_weights(receiver, f"../weights/{label}_weights")
receiver._cgnn.num_it = params.num_nrx_iter_eval
llr, h_hat = receiver(inputs)
_, _, u_hat = evaluator(llr, c)
ok = (np.asarray(b) == np.asarray(u_hat)).all(axis=-1)

symbols, ant, num_tx = params.symbol_allocation[1], params.num_rx_antennas, params.max_num_tx
signature = [
    tf.TensorSpec([1, None, symbols, ant], tf.float32, name="rx_slot_real"),
    tf.TensorSpec([1, None, symbols, ant], tf.float32, name="rx_slot_imag"),
    tf.TensorSpec([1, None, num_tx, ant], tf.float32, name="h_hat_real"),
    tf.TensorSpec([1, None, num_tx, ant], tf.float32, name="h_hat_imag"),
    tf.TensorSpec(inputs[4].shape, tf.float32, name="active_dmrs_ports"),
    tf.TensorSpec(inputs[5].shape, tf.int32, name="dmrs_ofdm_pos"),
    tf.TensorSpec(inputs[6].shape, tf.int32, name="dmrs_subcarrier_pos"),
]
model, _ = tf2onnx.convert.from_keras(receiver, signature)
args.output_dir.mkdir(parents=True, exist_ok=True)
stem = f"{label}_{args.prbs}prb_{args.num_tx}ue"
onnx.save(model, str(args.output_dir / f"{stem}.onnx"))
tx = params.transmitters[0]
meta = {
    "label": label, "prbs": args.prbs, "num_tx": args.num_tx, "num_rx_ant": int(ant), "symbols": int(symbols),
    "mcs_index": int(params.mcs_index[0]), "tb_size_bits": int(tx._tb_size),
    "num_bits_per_symbol": int(tx._num_bits_per_symbol),
    "dmrs_ofdm_pos": np.asarray(inputs[5]).tolist(), "dmrs_subcarrier_pos": np.asarray(inputs[6]).tolist(),
    "n_cell_id": int(params.n_cell_id), "slot_number": int(params.slot_number),
    "input_shapes": {n: list(np.asarray(x).shape) for n, x in zip(
        ("rx_slot_real", "rx_slot_imag", "h_hat_real", "h_hat_imag", "active_dmrs_ports",
         "dmrs_ofdm_pos", "dmrs_subcarrier_pos"), inputs)},
    "output_shapes": {"llr": list(llr.shape), "h_hat": list(h_hat.shape)},
    "outputs": [o.name for o in model.graph.output], "test_tb_pass": ok.tolist(),
}
np.savez(args.output_dir / f"{stem}_testvector.npz",
         rx_slot_real=inputs[0], rx_slot_imag=inputs[1], h_hat_real=inputs[2], h_hat_imag=inputs[3],
         active_dmrs_ports=inputs[4], dmrs_ofdm_pos=inputs[5], dmrs_subcarrier_pos=inputs[6],
         llr=np.asarray(llr), h_hat_out=np.asarray(h_hat),
         payload=np.packbits(np.asarray(b).astype(np.uint8), axis=-1), meta=json.dumps(meta))
(args.output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta))
