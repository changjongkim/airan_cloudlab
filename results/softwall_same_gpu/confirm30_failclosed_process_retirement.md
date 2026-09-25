# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | misses after retirement | worst response (ms) | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 5000 | 15 | 11 | 11 | 100.002 | 11.100 | 16.832 | 143.774 | 159.408 | 20 | False |
| 100 | 5 | 5000 | 5 | 6 | 6 | 96.625 | 10.277 | 10.861 | 41.034 | 44.580 | 108 | False |

All frozen gates pass: False.

## Retirement diagnostics

- cap20: 11 misses after retirement; release offsets [5, 5, 6, 11, 11, 11, 290, 322, 420, 549, 550].
- cap100: 6 misses after retirement; release offsets [4, 12, 12, 12, 13, 255].

A post-retirement miss is not counted as physical worker overlap. The timing association does not by itself prove a teardown cause; compare against a worker-free control and inspect GPU-event time.
