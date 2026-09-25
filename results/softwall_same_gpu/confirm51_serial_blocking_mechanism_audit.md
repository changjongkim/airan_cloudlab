# Confirm51: serial fallback blocking mechanism audit

Post-hoc, evidence grade C. The comparisons pair channel seed and release index, but the policies ran in separate time windows and receiver outcomes can vary.

| period ms | external NRx rejects | rejects immediately after fallback | eager NRx correct on rejected input | eager TB correct there | external TB correct there | external TB correct total | eager TB correct total | external AI units | consecutive rejects per fallback: length×count |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90 | 0 | 0 | 0 | 0 | 0 | 979 | 981 | 12723 | 0×27 |
| 45 | 25 | 25 | 23 | 23 | 7 | 964 | 980 | 149 | 1×25 |
| 25 | 50 | 25 | 48 | 48 | 15 | 947 | 979 | 0 | 2×25 |
| 12 | 110 | 22 | 105 | 106 | 42 | 916 | 981 | 0 | 5×22 |

- P90: external NRx-committed front+post GPU p99 2.294 ms; AI RPC n=12723, p99=3.674 ms, max=4.409 ms.
- P45: external NRx-committed front+post GPU p99 1.944 ms; AI RPC n=149, p99=3.959 ms, max=6.822 ms.
- P25: external NRx-committed front+post GPU p99 1.903 ms; AI RPC n=0.
- P12: external NRx-committed front+post GPU p99 1.829 ms; AI RPC n=0.

At P45, all 25 rejects immediately follow a fallback. The controller waits until release+53 ms for reserved recovery and then runs conventional, so the next P45 release can start more than 8 ms late even before conventional execution. However, the next release arrives at +45 ms: only 8 ms remains before the previous fallback begins, and after its 25 ms reserved interval ends at +78 ms, only 20 ms remains before the next request's +98 ms NRx cutoff. Both windows are shorter than the declared 50 ms NRx bound. Thus removing the host wait alone cannot safely recover the rejected NRx opportunities under an exclusive-GPU execution contract. Overlap requires a separately qualified interference bound or physical cancellation. The rejected inputs' eager outcomes identify potential radio value, not the correctness that a new implementation would necessarily achieve.

The NRx GPU service and AI RPC distributions are observed values, not WCETs or safe admission bounds. Reducing the 40 ms AI host budget from these maxima alone would be an unjustified contract change.

Input/summary errors: 0.
