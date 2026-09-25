# SoftWall confirmatory campaign

| condition | runs | RAN samples | misses | run p99 range (ms) | worst max (ms) | AI units/s mean | budget violations | release crossings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mps_alone | 5 | 2500 | 0 | 12.446–16.415 | 28.503 | 0.0 | 0 | 0 |
| protected_gemm | 5 | 2500 | 1 | 10.794–15.568 | 33.439 | 637.1 | 1 | 0 |
| protected_hbm | 5 | 2500 | 0 | 10.166–14.738 | 27.383 | 802.4 | 0 | 0 |
| uncontrolled_gemm | 5 | 2500 | 0 | 13.435–19.276 | 22.804 | 804.9 | 0 | 0 |
| uncontrolled_hbm | 5 | 2500 | 4 | 19.680–26.839 | 37.075 | 1046.1 | 0 | 0 |
