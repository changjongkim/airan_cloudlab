# SoftWall S2 natural-channel noninferiority

| round | releases | eager correct | S2 correct | S2−eager (pp) | S2/eager wins | seed mismatch | NRx decision mismatch | S2 misses | S2 bound violations | AI violations S/E | S2 bg/s | eager bg/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 986 | 986 | 0.000 | 0/0 | 0 | 0 | 0 | 0 | 0/0 | 555.8 | 526.3 |
| 2 | 1000 | 986 | 985 | -0.100 | 0/1 | 0 | 0 | 0 | 0 | 0/0 | 558.0 | 531.3 |
| 3 | 1000 | 989 | 989 | 0.000 | 0/0 | 0 | 0 | 0 | 0 | 0/0 | 556.0 | 524.7 |

Paired releases: 3000; S2−eager: -0.0333 percentage points; 95% CI [-0.0987, 0.0320].
Noninferiority margin: -0.500 percentage points; pass: True.
Background: S2 556.6/s, eager 527.4/s, gain 5.53%.
Valid-pair background (3 rounds): S2 556.6/s, eager 527.4/s, gain 5.53%.
All frozen gates pass: True.
