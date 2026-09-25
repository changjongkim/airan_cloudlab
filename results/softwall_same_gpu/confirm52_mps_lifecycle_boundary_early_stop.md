# Confirm52 MPS client lifecycle boundary

Same-node, same-seed, reversed-order triplets; each condition has 1,000 P60/D35 clean-PUSCH releases and zero intended optional GPU work during the timed epoch.

| triplet | order | idle misses | ack-only misses | exited misses | idle p99 ms | ack-only p99 ms | exited p99 ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | idle→ack→exited | 0 | 0 | 0 | 15.909 | 14.629 | 15.177 |
| 2 | exited→ack→idle | 0 | 0 | 0 | 18.625 | 14.654 | 15.460 |
| 3 | idle→ack→exited | 0 | 1 | 0 | 17.382 | 15.070 | 14.424 |
| 4 | exited→ack→idle | 0 | 1 | 0 | 15.488 | 16.908 | 15.977 |
| 5 | idle→ack→exited | 0 | 1 | 0 | 14.839 | 16.139 | 15.583 |
| 6 | exited→ack→idle | 0 | 1 | 0 | 14.101 | 15.149 | 16.013 |
| 7 | idle→ack→exited | 0 | 0 | 1 | 15.457 | 14.060 | 16.098 |
| 8 | exited→ack→idle | 0 | 2 | 1 | 15.302 | 17.851 | 15.413 |
| 9 | idle→ack→exited | 0 | 0 | 0 | 16.417 | 13.436 | 16.329 |

Misses idle / ack-only / confirmed-exit: 0 / 6 / 2 of 9000 each.
Primary paired signs ack-only>exited / ack-only<exited / ties: 5/1/3; exact one-sided p=0.10937500.
Completed/planned triplets: 9/10; best-case final one-sided p=0.06250000 if all remaining pairs favor ack-only excess.
Secondary diagnostic signs exited>idle / exited<idle / ties: 2/0/7; descriptive p=0.25000000.
Control errors: 0. Evidence grade: C early-stop diagnostic.
Decision: EARLY STOP: even if every remaining pair favored ack-only excess, the frozen primary p<0.05 gate could not pass. No exit-barrier or novelty claim.

mps_idle miss indices (round, zero-based release): [].
mps_ack_only miss indices (round, zero-based release): [(3, 7), (4, 7), (5, 7), (6, 7), (8, 7), (8, 8)].
mps_retired miss indices (round, zero-based release): [(7, 572), (8, 7)].

A launcher exit-confirmation timestamp after first release does not prove actual overlap. Waiting for actual exit also delays first release, so this contrast cannot separate an exit-barrier effect from a longer quiet period. It does not identify an internal CUDA/MPS operation, prove AI throughput gain, or establish a universal hard deadline bound.
