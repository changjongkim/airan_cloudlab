# Confirm48 Python GC tail mechanism diagnostic

| round | GC | correct | RAN misses | NRx >30 ms indices | collections | overlapping long GC events | AI units |
|---:|---|---:|---:|---|---:|---:|---:|
| 1 | gc_on | 3944 | 0 | [] | 87 | 0 | 51338 |

Completed/planned rounds: 1/4.
Prospective hypothesis gate passes: False.
The gate tests GC as a contributor to the observed tail; it does not establish a hard deadline bound.

Failed checks: r1_gc_on_hypothesis, full_frozen_order_completed, all_pass.
