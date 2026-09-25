# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 5000 | 15 | 1 | 9.990 | 10.301 | 155.933 | 159.460 | 20 | False |
| 100 | 5 | 5000 | 5 | 0 | 11.113 | 12.321 | 44.689 | 44.973 | 108 | True |

All frozen gates pass: False.
