# SoftWall S2 hardware policy campaign

| policy | runs | releases | misses | incorrect | p99 range (ms) | worst max (ms) | NRx commits | conv commits | fallback | duplicate commits | admission rejects | bound violations | background/s | AI violations/crossings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conventional_only | 5 | 5000 | 0 | 3180 | 10.085–12.949 | 19.119 | 0 | 5000 | 0 | 0 | 0 | 0 | 649.3 | 0/0 |
| eager_dual | 5 | 5000 | 0 | 77 | 15.818–19.016 | 30.899 | 4893 | 107 | 0 | 0 | 0 | 3 | 596.5 | 0/0 |
| nrx_only | 5 | 5000 | 0 | 107 | 7.662–11.030 | 23.551 | 5000 | 0 | 0 | 0 | 0 | 1 | 641.8 | 0/0 |
| s2_reserved | 5 | 5000 | 0 | 81 | 26.770–27.234 | 47.460 | 4889 | 111 | 111 | 0 | 0 | 5 | 638.1 | 0/0 |
