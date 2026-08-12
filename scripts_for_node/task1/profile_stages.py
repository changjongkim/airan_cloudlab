"""Per-stage latency profiler for real_l1_nrx pipeline.

Runs 1 cell/slot and measures CE, TRT NRx, and LDPC path times separately
using cuda events. Helps pinpoint the 100+ ms/cell bottleneck.
"""
import os
import sys
import numpy as np
import cupy as cp
sys.path.insert(0, "/opt/nvidia/cuBB/pyaerial/src")

from aerial.phy5g.algorithms import ChannelEstimator, TrtEngine, TrtTensorPrms
from aerial.phy5g.ldpc import (
    LdpcDeRateMatch, LdpcDecoder, CrcChecker, get_mcs, get_tb_size,
)
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.util.cuda import get_cuda_stream
from cuda.bindings import runtime as cudart

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WARM = 20

mcs = 2; mod, cr = get_mcs(mcs, 1); num_prbs = 273; num_rx = 4
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
tb = get_tb_size(mod_order=mod, code_rate=cr, dmrs_syms=dmrs_syms, num_prbs=num_prbs,
                 start_sym=2, num_symbols=12, num_layers=1)
ue = PuschUeConfig(scid=0, layers=1, dmrs_ports=1, rnti=1234, data_scid=0,
                   mcs_table=0, mcs_index=mcs, code_rate=int(cr*10),
                   mod_order=mod, tb_size=tb//8)
cfg = [PuschConfig(ue_configs=[ue], num_dmrs_cdm_grps_no_data=2, dmrs_scrm_id=41,
                   start_prb=0, num_prbs=num_prbs, dmrs_syms=dmrs_syms,
                   dmrs_max_len=1, dmrs_add_ln_pos=2, start_sym=2, num_symbols=12)]

cudart.cudaSetDevice(0)
stream = get_cuda_stream()

NSC = num_prbs * 12
rx = cp.asarray((np.random.randn(NSC, 14, num_rx) +
                 1j * np.random.randn(NSC, 14, num_rx)).astype(np.complex64), order='F')

ch_est = ChannelEstimator(num_rx_ant=num_rx, ch_est_algo=3, cuda_stream=stream)

trt_file = "/tmp/neural_rx.trt"
onnx_file = "/opt/nvidia/cuBB/pyaerial/models/neural_rx.onnx"
if not os.path.exists(trt_file):
    cmd = (f"trtexec --onnx={onnx_file} --saveEngine={trt_file} --skipInference "
           f"--inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp32:chw,fp32:chw,int32:chw,int32:chw "
           f"--outputIOFormats=fp32:chw,fp32:chw "
           f"--shapes=rx_slot_real:1x3276x12x4,rx_slot_imag:1x3276x12x4,"
           f"h_hat_real:1x4914x1x4,h_hat_imag:1x4914x1x4 > /tmp/trtexec.log 2>&1")
    ret = os.system(cmd)
    if ret != 0:
        sys.exit(f"trtexec failed ({ret})")
trt_engine = TrtEngine(
    trt_model_file=trt_file, max_batch_size=1, cuda_stream=stream,
    input_tensors=[
        TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
        TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
        TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
        TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
        TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
        TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
        TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32),
    ],
    output_tensors=[
        TrtTensorPrms('output_1', (8, 1, 3276, 12), np.float32),
        TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32),
    ],
)
derm = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=stream)
dec = LdpcDecoder(cuda_stream=stream)
crc = CrcChecker(cuda_stream=stream)

active = cp.ones((1, 1), dtype=cp.float32)
dofdm = cp.array([[2, 2, 2]], dtype=cp.int32)
dsub = cp.array([[0, 2, 4, 6, 8, 10]], dtype=cp.int32)
mask = np.array([dmrs_syms[2 + k] == 0 for k in range(12)], dtype=bool)
didx = cp.asarray(np.where(mask)[0])

# Warm up.
for _ in range(WARM):
    h = ch_est.estimate(rx_slot=rx, slot=0, pusch_configs=cfg)
    rx_in = rx[None, :, 2:14, :]
    h0 = h[0]; ce_in = cp.transpose(h0,(0,3,1,2)).reshape(h0.shape[0]*h0.shape[3], h0.shape[1], h0.shape[2])[None,...]
    out = trt_engine.run({
        "rx_slot_real": cp.ascontiguousarray(rx_in.real).astype(cp.float32),
        "rx_slot_imag": cp.ascontiguousarray(rx_in.imag).astype(cp.float32),
        "h_hat_real": cp.ascontiguousarray(ce_in.real).astype(cp.float32),
        "h_hat_imag": cp.ascontiguousarray(ce_in.imag).astype(cp.float32),
        "active_dmrs_ports": active, "dmrs_ofdm_pos": dofdm, "dmrs_subcarrier_pos": dsub,
    })
    llrs = cp.take(out["output_1"][0,...], didx, axis=3)
    coded = derm.derate_match(input_llrs=[llrs], pusch_configs=cfg)
    blocks = dec.decode(input_llrs=coded, pusch_configs=cfg)
    tbs, _ = crc.check_crc(input_bits=blocks, pusch_configs=cfg)
cp.cuda.runtime.deviceSynchronize()

# Measure per-stage.
def timeit(fn):
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record(); fn(); e1.record(); e1.synchronize()
    return cp.cuda.get_elapsed_time(e0, e1)

ts_ce, ts_prep, ts_trt, ts_ldpc = [], [], [], []
buf = {}
for _ in range(ITERS):
    ts_ce.append(timeit(lambda: ch_est.estimate(rx_slot=rx, slot=0, pusch_configs=cfg)))
    h = ch_est.estimate(rx_slot=rx, slot=0, pusch_configs=cfg)

    def prep():
        rx_in = rx[None, :, 2:14, :]
        h0 = h[0]
        ce_in = cp.transpose(h0, (0,3,1,2)).reshape(
            h0.shape[0]*h0.shape[3], h0.shape[1], h0.shape[2])[None, ...]
        buf['rx_r'] = cp.ascontiguousarray(rx_in.real).astype(cp.float32)
        buf['rx_i'] = cp.ascontiguousarray(rx_in.imag).astype(cp.float32)
        buf['ce_r'] = cp.ascontiguousarray(ce_in.real).astype(cp.float32)
        buf['ce_i'] = cp.ascontiguousarray(ce_in.imag).astype(cp.float32)
    ts_prep.append(timeit(prep))

    def trt():
        buf['out'] = trt_engine.run({
            "rx_slot_real": buf['rx_r'], "rx_slot_imag": buf['rx_i'],
            "h_hat_real": buf['ce_r'], "h_hat_imag": buf['ce_i'],
            "active_dmrs_ports": active, "dmrs_ofdm_pos": dofdm,
            "dmrs_subcarrier_pos": dsub,
        })
    ts_trt.append(timeit(trt))

    def ldpc():
        llrs = cp.take(buf['out']["output_1"][0, ...], didx, axis=3)
        coded = derm.derate_match(input_llrs=[llrs], pusch_configs=cfg)
        blocks = dec.decode(input_llrs=coded, pusch_configs=cfg)
        crc.check_crc(input_bits=blocks, pusch_configs=cfg)
    ts_ldpc.append(timeit(ldpc))

for name, arr in [("CE", ts_ce), ("prep", ts_prep), ("TRT", ts_trt), ("LDPC", ts_ldpc)]:
    a = np.array(arr)
    print(f"{name:6s}  mean={a.mean():7.3f}ms  p50={np.percentile(a,50):7.3f}ms  "
          f"p95={np.percentile(a,95):7.3f}ms  p99={np.percentile(a,99):7.3f}ms")
