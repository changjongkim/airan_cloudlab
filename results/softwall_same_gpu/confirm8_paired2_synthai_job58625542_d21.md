# SoftWall confirmatory campaign

| condition | runs | RAN samples | misses | incorrect TB | run p99 range (ms) | worst max (ms) | AI units/s mean | budget violations | release crossings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mps_alone | 3 | 3000 | 0 | 0 | 9.020–9.681 | 17.752 | 0.0 | 0 | 0 |
| protected_gemm | 3 | 3000 | 0 | 0 | 8.267–12.441 | 19.156 | 316.1 | 0 | 0 |
| protected_hbm | 3 | 3000 | 0 | 0 | 8.375–11.330 | 13.024 | 406.5 | 0 | 0 |
| uncontrolled_gemm | 3 | 3000 | 17 | 0 | 19.410–20.560 | 27.965 | 791.7 | 0 | 0 |
| uncontrolled_hbm | 3 | 3000 | 2963 | 0 | 24.370–26.274 | 31.546 | 1016.9 | 0 | 0 |
