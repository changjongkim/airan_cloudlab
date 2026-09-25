# Same-trace external S2 with bounded background AI

| condition | correct | RAN misses | background units | background units/s |
|---|---:|---:|---:|---:|
| external_s2 | 9845 | 0 | 130177 | 144.641 |
| s2_reserved | 9841 | 0 | 127180 | 141.311 |
| eager_dual | 9843 | 0 | 126477 | 140.530 |
External minus eager correct: 0.0200 pp; 95% paired CI [-0.0354, 0.0754] pp.
External minus local S2 correct: 0.0400 pp; 95% paired CI [-0.0080, 0.0880] pp.

All frozen gates pass: True.
