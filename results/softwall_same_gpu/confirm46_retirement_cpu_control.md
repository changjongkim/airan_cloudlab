# Confirm46 CPU-sham versus MPS-client retirement control

| round | seed | order | CPU-sham misses | MPS-sham misses | CPU indices | MPS indices |
|---:|---:|---|---:|---:|---|---|
| 1 | 20320001 | cpu_sham → mps_sham | 0 | 2 | [] | [7, 8] |
| 2 | 20320002 | mps_sham → cpu_sham | 0 | 2 | [] | [7, 852] |
| 3 | 20320003 | cpu_sham → mps_sham | 0 | 2 | [] | [7, 8] |
| 4 | 20320004 | mps_sham → cpu_sham | 0 | 1 | [] | [9] |
| 5 | 20320005 | cpu_sham → mps_sham | 0 | 2 | [] | [7, 8] |
| 6 | 20320006 | mps_sham → cpu_sham | 0 | 1 | [] | [7] |
| 7 | 20320007 | cpu_sham → mps_sham | 0 | 0 | [] | [] |
| 8 | 20320008 | mps_sham → cpu_sham | 0 | 0 | [] | [] |
| 9 | 20320009 | cpu_sham → mps_sham | 0 | 1 | [] | [7] |
| 10 | 20320010 | mps_sham → cpu_sham | 0 | 1 | [] | [7] |

Paired discordant misses: MPS-only 12, CPU-only 0. Round-level sign test: MPS excess 8, CPU excess 0, one-sided exact p = 0.00390625.
All frozen gates pass: True.
