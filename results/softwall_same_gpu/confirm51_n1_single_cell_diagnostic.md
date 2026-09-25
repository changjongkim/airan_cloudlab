# Confirm51 N1 one-cell load diagnostic

Policies in each triple: conventional-only / eager-dual / external transaction.
This is an operating-point diagnostic; the full combined baseline and multicell N1 grid remain untested.
Evidence grade: C (one complete round; opposite-order round stopped after the diagnostic pass became impossible).
Round 2 was stopped after round 1 because no eager deadline miss at any frozen period made the two-round pass impossible.

| round | period ms | misses | correct TB | completed AI units | RAN p99 ms | eager p99 slack ms | external−eager correct pp | diagnostic |
|---:|---:|---|---|---|---|---:|---:|---|
| 1 | 90 | 0 / 0 / 0 | 364 / 981 / 979 | 13063 / 12966 / 12723 | 4.043 / 6.618 / 55.857 | 73.382 | -0.200 | FAIL |
| 1 | 45 | 0 / 0 / 0 | 370 / 980 / 964 | 722 / 0 / 149 | 3.699 / 7.142 / 55.767 | 72.858 | -1.600 | FAIL |
| 1 | 25 | 0 / 0 / 0 | 371 / 979 / 947 | 0 / 0 / 0 | 3.244 / 8.065 / 55.652 | 71.935 | -3.200 | FAIL |
| 1 | 12 | 0 / 0 / 0 | 351 / 981 / 916 | 0 / 0 / 0 | 3.005 / 7.189 / 55.628 | 72.811 | -6.500 | FAIL |

Replicated diagnostic crossover periods: none.
Eager failure point: not observed in the frozen one-cell grid; its distance below P12 remains unknown.
Input/contract errors: 0.

Decision: Do not add seeds or relax bounds. Complete the comparable multicell N1 path before deciding the system claim.

The zero-miss observations are empirical, not a worst-case guarantee.
