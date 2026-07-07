# Experiment Chain Summary — 20260622

Generated: 2026-06-28 03:53:29

## E9 — cudaDeviceSynchronize → cudaFree (H1 discriminator)

### Baseline (no shim)
Files:

### E9 (shim: sync before each cudaFree)
Files:

Shim FINAL stats (from log):

## E1 — §18 AI-side dual capture

AI-side sqlite kernel counts:
  X1_neuralrx_alone_3g_AI.sqlite: ? kernels
  X2_neuralrx_2g_L1_3g_crosspart_AI.sqlite: ? kernels
  X3_neuralrx_L1_coloc_3g_AI.sqlite: ? kernels
  X5_chanpred_2g_L1_3g_crosspart_AI.sqlite: ? kernels
  X6_chanpred_L1_coloc_3g_AI.sqlite: ? kernels

### Verification gates (manual check):
- A: every AI sqlite >0 kernels (above)
- B: X3 (neuralrx_L1_coloc) AI cudaFree > X1 (neuralrx_alone) AI cudaFree → symmetric
- C: chanpred delta(X6-X4) < NeuralRx delta(X3-X1) → PHY-AI specificity

Quick B-gate calc:
  X1_neuralrx_alone_3g AI cudaFree total: ?
  X3_neuralrx_L1_coloc_3g AI cudaFree total: ?
