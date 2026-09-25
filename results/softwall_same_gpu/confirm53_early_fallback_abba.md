# Confirm53 known-CRC-failure early recovery ablation

| arm | pair | timing | correct/1000 | miss | AI units | NRx rejects | fallback/early | RAN p99 ms |
|---:|---:|---|---:|---:|---:|---:|---|---:|
| 1 | 1 | early_if_clear | 983 | 0 | 343 | 0 | 21/21 | 5.561 |
| 2 | 1 | latest | 968 | 0 | 197 | 19 | 19/0 | 55.746 |
| 3 | 2 | latest | 972 | 0 | 384 | 19 | 19/0 | 55.667 |
| 4 | 2 | early_if_clear | 983 | 0 | 221 | 0 | 19/19 | 5.579 |

| pair | early−latest correct TB | fewer NRx rejects | early-only correct | latest-only correct | frozen pair gate |
|---:|---:|---:|---:|---:|---|
| 1 | 15 | 19 | 16 | 1 | PASS |
| 2 | 11 | 19 | 12 | 1 | PASS |

Completed arms: 4/4. Overall frozen gate: PASS.
This only tests a one-cell known-failure timing mechanism. It is not a comparison to the strong combined baseline or a WCET proof.

AI throughput did not improve consistently: early−latest units were +146 and -163 in the two pairs. At P45, the 40 ms AI budget plus 2 ms guard leaves about 3 ms for the radio response and host decision; this makes background admission sensitive to small timing changes. Descriptive no-fallback response ≤3 ms counts by arm: 369, 219, 404, 237. These counts were inspected after the frozen gate and are not a new pass criterion.
