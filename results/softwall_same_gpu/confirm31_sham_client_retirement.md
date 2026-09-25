# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | misses after retirement | worst response (ms) | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 5000 | 0 | 17 | 17 | 94.964 | n/a | n/a | n/a | n/a | 20 | False |
| 100 | 5 | 5000 | 0 | 16 | 16 | 116.188 | n/a | n/a | n/a | n/a | 108 | False |

All frozen gates pass: False.

## Retirement diagnostics

- cap20: 17 misses after retirement; release offsets [2, 2, 3, 3, 3, 10, 45, 46, 50, 203, 310, 331, 562, 661, 730, 928, 974].
- cap100: 16 misses after retirement; release offsets [2, 3, 3, 10, 10, 10, 11, 58, 221, 230, 263, 319, 461, 462, 829, 843].

A post-retirement miss is not counted as physical worker overlap. The timing association does not by itself prove a teardown cause; compare against a worker-free control and inspect GPU-event time.
