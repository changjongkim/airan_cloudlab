# Confirm44 transactional recovery-gap AI BAAB ablation

| round | gap lease | correct | RAN misses | natural fallbacks | completed AI | in recovery gap |
|---:|---|---:|---:|---:|---:|---:|
| 1 | on | 9827 | 0 | 225 | 128887 | 675 |
| 2 | off | 9827 | 0 | 225 | 127205 | 0 |
| 3 | off | 9823 | 0 | 225 | 127237 | 0 |
| 4 | on | 9827 | 0 | 225 | 128872 | 675 |

Gap ON minus OFF: 3317 completed AI units (+1.304%).
Matched ON-minus-OFF radio paired 95% CI: [-0.0620, +0.0620] pp; [-0.0279, +0.1079] pp.
All frozen gates pass: False.

Failed checks: r1_on_radio_and_bounds, r3_off_radio_and_bounds, r4_on_radio_and_bounds, all_pass.
