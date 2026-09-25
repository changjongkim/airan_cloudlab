# External recovery cold-start audit

| job | first fallback release | first conventional GPU ms | later median/max GPU ms | first response ms |
|---:|---:|---:|---:|---:|
| 58673249 | 112 | 27.289 | 2.553/2.922 | 30.669 |
| 58674868 | 103 | 30.136 | 2.520/3.072 | 33.774 |
| 58676333 | 6 | 19.163 | 2.514/2.704 | 22.390 |

The first timed conventional fallback is much slower in these runs. This is consistent with an unprimed conventional path, but causal attribution requires a frozen warmed-versus-unwarmed experiment.
