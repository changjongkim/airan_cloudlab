# Confirm42 transactional external endpoint smoke

- Job/node: 58682735 / nid001145
- Radio: 486/500 correct; 0 deadline misses
- Natural fallbacks: 18; admission rejections: 0
- Correct conventional recoveries: 4
- Completed AI units: 6304; before reserved recovery: 52
- Synthetic post-commit NRx probes: 18; accepted duplicates: 0
- All frozen gates pass: True
- Scope: A smoke pass establishes concrete integration for this allowed workload and seed. It is not a throughput advantage, hard WCET proof or multi-cell scheduling result; same-node paired ablation follows.

## Failed checks

- None
