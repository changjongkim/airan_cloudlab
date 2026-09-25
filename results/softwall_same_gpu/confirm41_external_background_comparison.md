# Same-trace external S2 with bounded background AI

| condition | correct | RAN misses | background units | background units/s |
|---|---:|---:|---:|---:|
| external_s2 | 9817 | 0 | 130254 | 144.727 |
| s2_reserved | 9817 | 0 | 127231 | 141.368 |
| eager_dual | 9817 | 0 | 126390 | 140.433 |
External minus eager correct: 0.0000 pp; 95% paired CI [-0.0480, 0.0480] pp.
External minus local S2 correct: 0.0000 pp; 95% paired CI [-0.0480, 0.0480] pp.

All frozen gates pass: True.
