# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | misses after retirement | worst response (ms) | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 5000 | 0 | 4 | 4 | 87.490 | n/a | n/a | n/a | n/a | 20 | False |
| 100 | 5 | 5000 | 0 | 4 | 4 | 97.232 | n/a | n/a | n/a | n/a | 108 | False |

All frozen gates pass: False.

## Retirement diagnostics

- cap20: 4 misses after retirement; release offsets [8, 8, 8, 8].
- cap100: 4 misses after retirement; release offsets [8, 8, 8, 9].

A post-retirement miss is not counted as physical worker overlap. The timing association does not by itself prove a teardown cause; compare against a worker-free control and inspect GPU-event time.
