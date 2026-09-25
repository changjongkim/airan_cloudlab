# SoftWall S2 hardware policy campaign

| policy | runs | releases | misses | incorrect | p99 range (ms) | worst max (ms) | NRx commits | conv commits | fallback | duplicate commits | admission rejects | bound violations | background/s | AI violations/crossings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conventional_only | 3 | 1500 | 0 | 0 | 9.140–15.839 | 25.441 | 0 | 1500 | 0 | 0 | 0 | 1 | 648.9 | 0/0 |
| eager_dual | 3 | 1500 | 0 | 0 | 13.979–21.214 | 33.873 | 1350 | 150 | 0 | 0 | 0 | 2 | 592.5 | 0/0 |
| nrx_only | 3 | 1500 | 0 | 150 | 3.556–4.677 | 7.537 | 1350 | 0 | 0 | 0 | 0 | 0 | 674.4 | 0/0 |
| s2_reserved | 3 | 1500 | 0 | 0 | 21.918–22.115 | 25.018 | 1350 | 150 | 150 | 0 | 0 | 0 | 646.7 | 0/0 |
