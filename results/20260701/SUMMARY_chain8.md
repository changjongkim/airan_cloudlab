# Chain 8 — cudaFreeAsync + cudaMemPool mitigation

Generated: 2026-07-01 05:46:49

## L1 cudaFree by shim × cells
| cells | alone | NRx coloc | + freeasync (A) | + memPool (B) |
|---|---|---|---|---|
| 4 | 8.438 | 70.991 | 68.793 | 71.711 |
| 10 | 20.292 | 172.641 | 173.983 | 173.376 |
| 40 | 86.970 | 687.966 | 687.772 | 689.985 |
| 60 | 129.540 | ? | ? | ? |

## DONE marker
  All 16 conditions attempted.
