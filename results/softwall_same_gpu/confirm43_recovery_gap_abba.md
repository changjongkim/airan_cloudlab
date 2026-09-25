# Confirm43 transactional recovery-gap AI ABBA ablation

| round | gap lease | correct | RAN misses | natural fallbacks | completed AI | in recovery gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | off | 9821 | 0 | 221 | 127682 | 0 |
| 2 | on | 9821 | 0 | 221 | 127981 | 659 |
| 3 | on | 9818 | 0 | 221 | 128001 | 663 |
| 4 | off | 9820 | 0 | 221 | 127231 | 0 |

Gap ON minus OFF: 1069 completed AI units (+0.419%).
Matched ON-minus-OFF radio paired 95% CI: [-0.0733, +0.0733] pp; [-0.0933, +0.0533] pp.
All frozen gates pass: False.

Failed checks: r1_off_radio_and_bounds, r2_on_radio_and_bounds, r4_off_radio_and_bounds, all_pass.
