# SoftWall physical optional-client overrun

| cap | runs | releases | physically overlapped releases | misses | mean active p99 (ms) | worst active max (ms) | optional GPU mean (ms) | optional worst max (ms) | visible SMs | cap gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 1500 | 450 | 1 | 18.150 | 28.763 | 127.271 | 159.402 | 20 | False |
| 100 | 3 | 1500 | 150 | 2 | 19.405 | 25.970 | 44.942 | 45.060 | 108 | False |

All frozen gates pass: False.
