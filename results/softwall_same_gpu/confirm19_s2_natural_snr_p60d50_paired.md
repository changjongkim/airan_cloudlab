# SoftWall S2 natural-channel paired analysis

| round | releases | conventional | NRx-only | eager | S2 | independent union | S2/eager mismatches | S2 gains over NRx | S2 losses vs eager | S2 misses | S2 bound violations | S2 bg/s | eager bg/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 352 | 973 | 980 | 978 | 981 | 2 | 6 | 2 | 0 | 1 | 636.5 | 595.5 |
| 2 | 1000 | 380 | 975 | 984 | 980 | 982 | 4 | 6 | 4 | 0 | 1 | 638.1 | 593.5 |
| 3 | 1000 | 379 | 981 | 986 | 987 | 986 | 1 | 6 | 0 | 0 | 1 | 638.6 | 600.4 |
| 4 | 1000 | 338 | 984 | 988 | 988 | 987 | 0 | 4 | 0 | 0 | 0 | 644.7 | 596.2 |
| 5 | 1000 | 371 | 980 | 985 | 986 | 987 | 1 | 6 | 0 | 0 | 2 | 632.4 | 597.0 |

Complete paired rounds: 5; releases: 5000.
S2/eager correctness mismatches: 8; S2 gains over NRx-only: 28; S2 losses versus eager: 6.
Mean background throughput: S2 638.1/s, eager 596.5/s (6.96% delta).

`independent union` combines outcomes from separate policy processes and is diagnostic at the decoder transition; eager is the actual both-branch execution result.
