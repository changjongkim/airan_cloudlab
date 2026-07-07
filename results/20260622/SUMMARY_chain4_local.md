# Chain 4 v3 — local analysis (24 conditions)

## L1 process metrics

| partition | scenario | cudaFree (n / ms) | slow% (>1ms) | memcpyAsync ms | kernel ms |
|---|---|---|---|---|---|
| 7g | alone | 2463 / 600.7 | 0.2% | 55.2 | 625.7 |
| 7g | neuralrx_coloc | 2463 / 9037.8 | 93.1% | 3580.6 | 628.1 |
| 7g | chanpred_coloc | 2463 / 3300.6 | 63.3% | 993.1 | 624.3 |
| 4g | alone | 2463 / 721.9 | 0.1% | 56.5 | 498.7 |
| 4g | neuralrx_coloc | 2463 / 9153.1 | 93.3% | 3589.1 | 499.4 |
| 4g | chanpred_coloc | 2463 / 4251.6 | 61.5% | 1232.9 | 497.6 |
| 4g | coloc_qwen | 2463 / 9105.4 | 93.1% | 3614.8 | 499.2 |
| 4g | coloc_hbm | 2463 / 9102.6 | 92.4% | 3554.3 | 503.8 |
| 4g | coloc_chanpred | 2463 / 9154.1 | 93.5% | 3622.3 | 500.1 |
| 4g | coloc_resnet | 2463 / 9213.2 | 93.8% | 3597.4 | 502.0 |
| 3g | neuralrx_coloc | 2463 / 9058.1 | 91.6% | 3616.5 | 509.2 |
| 3g | chanpred_coloc | 2463 / 4419.4 | 64.3% | 1244.5 | 509.7 |
| 3g | coloc_qwen | 2463 / 8943.9 | 90.5% | 3510.2 | 513.4 |
| 3g | coloc_hbm | 2463 / 9030.1 | 91.7% | 3579.0 | 509.9 |
| 3g | coloc_chanpred | 2463 / 9075.4 | 92.6% | 3595.8 | 514.0 |
| 3g | coloc_resnet | 2463 / 9101.3 | 92.0% | 3529.8 | 516.6 |
| 2g | alone | 2463 / 1220.6 | 32.4% | 71.7 | 539.6 |
| 2g | chanpred_coloc | 2463 / 5880.7 | 66.8% | 2983.4 | 532.3 |
| 2g | coloc_qwen | 2463 / 2717.2 | 41.9% | 659.3 | 532.3 |
| 2g | coloc_hbm | 2463 / 7242.8 | 74.1% | 2577.2 | 532.2 |
| 2g | coloc_chanpred | 2463 / 4573.6 | 52.7% | 1466.0 | 531.0 |
| 2g | coloc_resnet | 2463 / 3029.2 | 44.8% | 789.8 | 532.4 |

## AI process metrics (per-process per-condition)

| condition | workload | kernels | wall s | kernel/s |
|---|---|---|---|---|
| 2g_chanpred_coloc | chp | 3373638 | 30.0 | 112494 |
| 2g_coloc_chanpred | nrx | 15432 | 29.9 | 516 |
| 2g_coloc_hbm | nrx | 15381 | 30.0 | 512 |
| 2g_coloc_qwen | nrx | 18414 | 30.0 | 614 |
| 2g_coloc_qwen | qwen | 1251962 | 30.0 | 41748 |
| 2g_coloc_resnet | nrx | 18082 | 30.0 | 603 |
| 2g_coloc_resnet | res | 159047 | 29.8 | 5337 |
| 2g_neuralrx_coloc | nrx | 18722 | 30.0 | 624 |
| 3g_chanpred_coloc | chp | 3388668 | 30.0 | 113014 |
| 3g_coloc_chanpred | chp | 3314444 | 30.0 | 110523 |
| 3g_coloc_chanpred | nrx | 12702 | 30.0 | 423 |
| 3g_coloc_hbm | nrx | 13677 | 30.0 | 456 |
| 3g_coloc_qwen | nrx | 11972 | 28.5 | 421 |
| 3g_coloc_qwen | qwen | 1232859 | 30.0 | 41113 |
| 3g_coloc_resnet | nrx | 14394 | 29.9 | 481 |
| 3g_coloc_resnet | res | 159523 | 29.8 | 5353 |
| 3g_neuralrx_coloc | nrx | 12252 | 29.0 | 423 |
| 4g_chanpred_coloc | chp | 3399281 | 30.0 | 113364 |
| 4g_coloc_chanpred | nrx | 12363 | 29.1 | 425 |
| 4g_coloc_hbm | nrx | 12364 | 29.1 | 425 |
| 4g_coloc_qwen | nrx | 12313 | 29.0 | 424 |
| 4g_coloc_qwen | qwen | 1232274 | 30.0 | 41092 |
| 4g_coloc_resnet | nrx | 12711 | 29.9 | 425 |
| 4g_coloc_resnet | res | 159050 | 29.8 | 5338 |
| 4g_neuralrx_coloc | nrx | 12711 | 30.0 | 424 |
| 7g_chanpred_coloc | chp | 3366075 | 30.0 | 112252 |
| 7g_neuralrx_coloc | nrx | 12256 | 29.7 | 413 |
