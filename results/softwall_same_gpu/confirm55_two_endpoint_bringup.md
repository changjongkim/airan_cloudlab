# Confirm55 two-endpoint same-GPU MPS bring-up

Job `58693395` on `nid001153`; two pinned cap40 NeuralRx endpoints, one cap20 AI worker, one conventional lane; 1,000 simultaneous two-cell P150/D130 releases.

Correct 1958/2000 cells; RAN misses 0; NRx 50 ms violations 0; conventional GPU 25 ms violations 0; AI units 29910.
Fallbacks 48, early 48; NRx rejects 0; retime rejects 0.
Radio p99/max 7.739/13.503 ms; NRx p99/max 5.888/13.503 ms.
Natural two-cell simultaneous fallback occurred at release indices [854]; this is descriptive and was not a frozen pass gate.

Frozen gate: PASS. Failed checks: none.
This is physical integration qualification, not a strong-baseline comparison or WCET proof. Worker timestamps do not separately prove concurrent kernel overlap.
