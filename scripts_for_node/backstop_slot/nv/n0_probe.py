import os, re, sys, json, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pathlib import Path
sys.path.append("..")
import numpy as np, tensorflow as tf
from utils import Parameters
from sionna.utils import ebnodb2no
out = {}
for channel in ("DoubleTDLhigh", "DoubleTDLlow"):
    for mcs in (10, 13, 16):
        base = Path("../config") / "nrx_large.cfg"
        text = base.read_text()
        label = re.search(r"label\s*=\s*'([^']+)'", text).group(1)
        text = re.sub(r"n_size_bwp_eval\s*=.*", "n_size_bwp_eval = 273", text)
        text = re.sub(r"(?m)^channel_type_eval\s*=.*", f'channel_type_eval = "{channel}"', text)
        text = re.sub(r"(?m)^mcs_index\s*=.*", f"mcs_index = [{mcs}]", text)
        name = f"n0probe_{channel}_m{mcs}.cfg"
        (Path("../config") / name).write_text(text)
        params = Parameters(name, training=False, system="nrx", num_tx_eval=2)
        tx = params.transmitters[0]; rg = tx._resource_grid
        qm, rate = int(tx._num_bits_per_symbol), float(tx._target_coderate)
        shift = 10 * math.log10(qm * rate)
        rows = {}
        for esno in (6.0, 14.0, 16.0, 20.0):
            no = float(ebnodb2no(esno - shift, qm, rate, rg))
            rows[esno] = no
        out[f"{channel}_m{mcs}"] = {"qm": qm, "rate": rate, "ebno_param": bool(params.ebno), "no": rows,
                                   "k": rows[20.0] * 10 ** 2.0,
                                   "rg": {"num_ofdm_symbols": int(rg.num_ofdm_symbols), "fft_size": int(rg.fft_size),
                                          "cp": int(rg.cyclic_prefix_length), "eff_sc": int(rg.num_effective_subcarriers),
                                          "num_data_symbols": int(rg.num_data_symbols), "streams": int(rg.num_streams_per_tx),
                                          "num_tx": int(rg.num_tx)}}
        (Path("../config") / name).unlink()
print(json.dumps(out, indent=1))
