# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 5 | 5000 | 1500 | 0 | 18.230 | 33.862 | 126.343 | 159.546 | 20 | True |
| 100 | 5 | 5000 | 500 | 1 | 20.124 | 25.091 | 44.945 | 45.057 | 108 | False |

All frozen gates pass: False.
