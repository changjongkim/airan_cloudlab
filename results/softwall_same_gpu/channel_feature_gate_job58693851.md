# Observable channel-gate calibration: INVALID held-out test

Job `58693851` completed on `nid001169` in 58 seconds with four planned
500-request arms. The executable finished without error, but an independent
post-run integrity check found that train channel seeds were
`20356001–20356500` and test seeds were `20356002–20356501`.
**499 of 500 channel realizations overlap.** Their observed channel-power
features match exactly for all 499 common seeds. The planned held-out AUC,
threshold, and PASS/FAIL gates therefore **cannot qualify a baseline channel
gate**. The originally emitted `diagnostic_pass=false` is preserved, but it
does not make the train/test design valid. The run must not be relabeled PASS
by replacing seeds after seeing the result.

The within-seed pre/post noise-reference comparison remains a paired
description of two different synthetic workloads. At nominal −8.5 dB, the
historical per-block post-fading normalization yielded NeuralRx 487/500 and
488/500 correct. Fixing AWGN power against the clean pre-fading grid yielded
216/500 and 213/500. Conventional correctness was 184/500 and 187/500 under
the historical convention, 184/500 and 186/500 with fixed noise. The two
seed arms heavily overlap and must not be presented as independent replications.
Within the first paired arm, NeuralRx was correct only under historical
normalization on 275 requests, and only under fixed-noise on 4 requests;
the corresponding counts in the shifted second arm were 277 and 2. The
channel/noise convention materially changes radio utility. Neither mode is
validated as a field-channel distribution by this experiment.

The observable feature call took about 0.72 ms at the median in each arm.
Its first call took 303.932 ms in the first arm and 5.48–8.02 ms in the other
arms. The frozen maximum-under-5-ms feature-cost gate fails; the first-call
cost cannot be silently excluded. Even a warmed 0.72 ms feature path alone
exceeds a 0.5 ms slot period, so this implementation cannot be used to claim
that production pacing target.

The [machine-readable integrity audit](channel_feature_gate_integrity_job58693851.json)
contains raw/source hashes, arm counts, exact overlap, and paired discordance.
The [frozen protocol](channel_feature_gate_protocol.json), raw JSON, original
emitted [gate report](channel_feature_gate_job58693851.json), and exact
[source snapshot](frozen_source/channel_feature_gate/) are retained. The active
analyzer now rejects train/test channel-seed overlap; its frozen version is in
the source snapshot. This campaign is offline PHY calibration, with no MPS
co-run, online admission, AI throughput, or deadline-superiority result.
