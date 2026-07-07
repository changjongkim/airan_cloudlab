# Chain 6 — Cell size sweep on 3g partition

Generated: 2026-07-01 00:14:37

## L1 latency (cells × scenario)
| cells | scenario | mean (ms) | p99 (ms) | miss1ms |
|---|---|---|---|---|
| 4 | alone | 8.153 | 8.447 | 20/20 |
| 4 | neuralrx_coloc | 8.924 | 9.444 | 20/20 |
| 4 | chanpred_coloc | 42.309 | 48.429 | 20/20 |
| 4 | l1_qwen_only | 8.152 | 8.432 | 20/20 |
| 4 | l1_hbm_only | 8.951 | 9.266 | 20/20 |
| 4 | l1_chanpred_only | 8.145 | 8.416 | 20/20 |
| 4 | l1_resnet_only | 8.165 | 8.460 | 20/20 |
| 4 | coloc_qwen | 70.807 | 75.323 | 20/20 |
| 4 | coloc_hbm | 70.695 | 75.094 | 20/20 |
| 4 | coloc_chanpred | 70.309 | 75.176 | 20/20 |
| 4 | coloc_resnet | 70.443 | 75.202 | 20/20 |
| 10 | alone | 20.296 | 20.601 | 20/20 |
| 10 | neuralrx_coloc | 174.546 | 179.174 | 20/20 |
| 10 | chanpred_coloc | 115.232 | 121.159 | 20/20 |
| 10 | l1_qwen_only | 21.719 | 21.975 | 20/20 |
| 10 | l1_hbm_only | 22.155 | 22.665 | 20/20 |
| 10 | l1_chanpred_only | 20.236 | 20.878 | 20/20 |
| 10 | l1_resnet_only | 22.175 | 22.379 | 20/20 |
| 10 | coloc_qwen | 178.128 | 178.950 | 20/20 |
| 10 | coloc_hbm | 172.863 | 178.499 | 20/20 |
| 10 | coloc_chanpred | 175.242 | 179.189 | 20/20 |
| 10 | coloc_resnet | 174.447 | 180.463 | 20/20 |
| 40 | alone | 88.757 | 90.379 | 20/20 |
| 40 | neuralrx_coloc | 693.373 | 699.957 | 20/20 |
| 40 | chanpred_coloc | 456.525 | 479.404 | 20/20 |
| 40 | l1_qwen_only | 81.077 | 82.718 | 20/20 |
| 40 | l1_hbm_only | 86.731 | 88.387 | 20/20 |
| 40 | l1_chanpred_only | 80.427 | 81.716 | 20/20 |
| 40 | l1_resnet_only | 80.140 | 81.652 | 20/20 |
| 40 | coloc_qwen | 693.947 | 701.655 | 20/20 |
| 40 | coloc_hbm | 688.008 | 698.258 | 20/20 |
| 40 | coloc_chanpred | 695.329 | 706.171 | 20/20 |
| 40 | coloc_resnet | 690.472 | 703.303 | 20/20 |
| 60 | alone | 133.443 | 134.173 | 20/20 |
| 60 | chanpred_coloc | 516.249 | 526.231 | 20/20 |
| 60 | l1_qwen_only | 120.372 | 121.655 | 20/20 |
| 60 | l1_hbm_only | 120.797 | 122.140 | 20/20 |
| 60 | l1_chanpred_only | 133.189 | 134.054 | 20/20 |
| 60 | l1_resnet_only | 120.747 | 122.029 | 20/20 |
| 60 | coloc_hbm | 510.334 | 1038.504 | 20/20 |
| 60 | coloc_chanpred | 527.227 | 1025.901 | 20/20 |
