# Confirm50 requalified 50 ms recovery-gap ABBA ablation

| round | pair | gap | correct | RAN misses | NRx bound violations | AI units | gap AI | admission rejects |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 1 | off | 9795 | 0 | 0 | 127026 | 0 | 1 |
| 2 | 1 | on | 9793 | 0 | 0 | 127621 | 752 | 0 |
| 3 | 2 | on | 9801 | 0 | 0 | 127581 | 746 | 0 |
| 4 | 2 | off | 9802 | 0 | 0 | 126902 | 0 | 0 |

Pair 1: ON−OFF AI +595 (+0.468%; gap 752, outside-gap change -157), admission rejection change -1, radio paired 95% CI [-0.0820, +0.0420] pp.
Pair 2: ON−OFF AI +679 (+0.535%; gap 746, outside-gap change -67), admission rejection change +0, radio paired 95% CI [-0.0538, +0.0338] pp.

All frozen gates pass: True.
