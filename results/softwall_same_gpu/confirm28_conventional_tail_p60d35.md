# Confirm28 conventional-only tail

No AI worker or RPC socket was present. One MPS conventional full-PHY process ran 10,000 clean valid-PUSCH releases at P60/D35 after 30 warm-up iterations.

| releases | correct | deadline misses | response p99 | response max | GPU p99 | GPU max |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 10,000 | 0 | 5.323 ms | 7.642 ms | 5.237 ms | 7.571 ms |

This run did not reproduce the isolated 36.076 ms conventional GPU event seen in confirm27 while an idle/repeatedly used overrun-client process remained in the MPS campaign. Confirm27's non-overlap event is therefore classified as an unexplained MPS-campaign/residual tail rather than proven intrinsic receiver WCET. The observation is still a contract failure and is not removed from confirm27.
