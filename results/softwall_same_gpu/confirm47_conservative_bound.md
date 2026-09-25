# Confirm47 fully transactional external S2 vs eager-dual

| round | order | eager correct | external correct | eager AI | external AI | recovery-gap AI | admission rejections | external−eager AI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | eager_dual → external_transaction | 9837 | 9838 | 126400 | 128140 | 602 | 0 | +1740 |
Round 1 external−eager radio: +0.0100 pp, paired 95% CI [-0.0419, +0.0619] pp; eager NRx 20 ms branch-bound diagnostic 2.
| 2 | external_transaction → eager_dual | 9818 | 9823 | 126506 | 127916 | 670 | 0 | +1410 |
Round 2 external−eager radio: +0.0500 pp, paired 95% CI [-0.0088, +0.1088] pp; eager NRx 20 ms branch-bound diagnostic 3.

All frozen gates pass: True.
