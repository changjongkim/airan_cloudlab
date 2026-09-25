# SoftWall S1 Qwen-b8c512-prefill overlap sweep

| condition | AI visible SMs | runs | RAN samples | misses | incorrect TB | p99 range (ms) | worst max (ms) | Qwen-b8c512-prefill/s mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alone | 0 | 5 | 5000 | 0 | 0 | 9.029–10.687 | 20.699 | 0.0 |
| cap100_p0 | 108 | 5 | 5000 | 153 | 0 | 20.871–22.363 | 26.065 | 11.4 |
| cap80_p0 | 86 | 5 | 5000 | 0 | 0 | 10.989–12.812 | 18.667 | 9.7 |
