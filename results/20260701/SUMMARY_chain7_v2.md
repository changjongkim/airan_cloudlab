# Chain 7 v2 — All NSYS series × cell sweep

Generated: 2026-07-01 00:59:56

## L1 latency by cells × NSYS-series condition
| cells | series | scenario | mean (ms) | p99 (ms) | miss1ms |
|---|---|---|---|---|---|
| 4 | §18 | X2 NRX cross-part | 11.867 | 68.038 | 20/20 |
| 4 | §18 | X3 NRX coloc | 71.266 | 75.181 | 20/20 |
| 4 | §18 | X5 CHP cross-part | 8.172 | 8.498 | 20/20 |
| 4 | §18 | X6 CHP coloc | 34.626 | 40.979 | 20/20 |
| 4 | mech | e9 sync-first shim | crash | crash | crash |
| 4 | mech | e10 NSYS callchain | 70.479 | 75.233 | 20/20 |
| 4 | mech | e2 cudaFree-noop shim | 70.780 | 75.142 | 20/20 |
| 4 | proof | p5 extended callchain | 70.734 | 75.156 | 20/20 |
| 4 | proof | p6 defer+drain shim | 70.999 | 75.161 | 20/20 |
| 10 | §18 | X2 NRX cross-part | 27.045 | 64.008 | 20/20 |
| 10 | §18 | X3 NRX coloc | 175.052 | 179.387 | 20/20 |
| 10 | §18 | X5 CHP cross-part | 22.070 | 22.267 | 20/20 |
| 10 | §18 | X6 CHP coloc | 75.846 | 77.171 | 20/20 |
| 10 | mech | e9 sync-first shim | crash | crash | crash |
| 10 | mech | e10 NSYS callchain | 173.738 | 179.660 | 20/20 |
| 10 | mech | e2 cudaFree-noop shim | 173.377 | 179.209 | 20/20 |
| 10 | proof | p5 extended callchain | 173.439 | 178.985 | 20/20 |
| 10 | proof | p6 defer+drain shim | 173.121 | 178.484 | 20/20 |
| 40 | §18 | X2 NRX cross-part | 88.805 | 90.779 | 20/20 |
| 40 | §18 | X3 NRX coloc | 693.113 | 701.125 | 20/20 |
| 40 | §18 | X5 CHP cross-part | 80.817 | 82.362 | 20/20 |
| 40 | §18 | X6 CHP coloc | 342.962 | 358.789 | 20/20 |
| 40 | mech | e9 sync-first shim | crash | crash | crash |
| 40 | mech | e10 NSYS callchain | 685.477 | 694.821 | 20/20 |
| 40 | mech | e2 cudaFree-noop shim | 688.514 | 697.970 | 20/20 |
| 40 | proof | p5 extended callchain | 693.761 | 702.544 | 20/20 |
| 40 | proof | p6 defer+drain shim | 688.993 | 699.906 | 20/20 |
| 60 | §18 | X2 NRX cross-part | 120.501 | 122.689 | 20/20 |
| 60 | §18 | X3 NRX coloc | crash | crash | crash |
| 60 | §18 | X5 CHP cross-part | 120.192 | 121.129 | 20/20 |
| 60 | §18 | X6 CHP coloc | 488.573 | 505.697 | 20/20 |
| 60 | mech | e9 sync-first shim | crash | crash | crash |
| 60 | mech | e10 NSYS callchain | crash | crash | crash |
| 60 | mech | e2 cudaFree-noop shim | crash | crash | crash |
| 60 | proof | p5 extended callchain | crash | crash | crash |
| 60 | proof | p6 defer+drain shim | crash | crash | crash |
