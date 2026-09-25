# External same-request endpoint vs local baselines

| condition | releases | correct | deadline miss | conventional executions |
|---|---:|---:|---:|---:|
| conventional | 10000 | 3688 | 0 | 10000 |
| eager dual | 10000 | 9835 | 0 | 10000 |
| external S2 | 10000 | 9840 | 0 | 203 |

External−eager: 0.0500 pp; 95% CI [-0.0088, 0.1088]; margin −0.250 pp.
Natural recoveries: 43; external/eager wins: 7/2.
All frozen gates pass: True.
