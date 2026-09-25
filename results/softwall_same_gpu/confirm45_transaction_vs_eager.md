# Confirm45 fully transactional external S2 vs eager-dual

| round | order | eager correct | external correct | eager AI | external AI | recovery-gap AI | external−eager AI |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | eager_dual → external_transaction | 9823 | 9825 | 126498 | 130230 | 684 | +3732 |
Round 1 external−eager radio: +0.0200 pp, paired 95% CI [-0.0420, +0.0820] pp; eager NRx 20 ms branch-bound diagnostic 3.
| 2 | external_transaction → eager_dual | 9796 | 9799 | 124525 | 127528 | 744 | +3003 |
Round 2 external−eager radio: +0.0300 pp, paired 95% CI [-0.0350, +0.0950] pp; eager NRx 20 ms branch-bound diagnostic 2.

All frozen gates pass: True.
