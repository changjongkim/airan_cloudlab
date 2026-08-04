# AI-RAN GPU Isolation Strategy: Presentation Slides (v2 · data re-verified)

*Data: 2026-07 identical-NRx grid + fault·NCU experiments · 2026-08-03 diverse-AI deployment experiment*
*Re-verification: the previous "SP + MPS pct=30 = 45 ms" claim was wrong (measured: 146 ms). Metrics also corrected to gap p99.*

---

## Slide 1 · Setup — what, where, how measured

**Hardware**
- CloudLab d8545 (Wisconsin) · NVIDIA A100-SXM4-40GB × 4 · Driver 580.173.02 · CUDA 13.0
- MIG profiles: 4g.20gb (56 SM) / 3g.20gb (42 SM) / 2g.10gb (28 SM) / Full GPU (108 SM)

**Software**
- L1: cuPHY 25.3-cubb (5G DU) + pyaerial 2026.1.dev1
- AI (7 diverse workloads): Qwen 2.5-3B vLLM · Whisper large-v3 · BERT · Qwen-VL · NRx · CsiNet · BeamPred
- Isolation tools: MIG (hardware partition) + CUDA MPS (logical multiplexing, per-client pct thread% cap)

**Metrics (corrected)**
- **L1 p99 latency (ms)** — from realL1_*.json, per-iteration 5G SLA gate · 3-trial min/max/mean
- **gap p99 (ms)** — inter-kernel tail gap · captures MPS OFF SLA risk correctly
- **launch rate (kernels/s)** — L1 starvation signal
- Duty cycle used only as a reference indicator (can mislead SLA judgement)

**Datasets**
- identical-NRx grid experiment (2026-07 · 108 conditions · 3 configs × 6 N × 2 MPS × 3 trials)
- fault injection · NCU deep-dive experiment (2026-07)
- diverse-AI deployment experiment (2026-08-03 · 273 conditions · 13 scenarios · 213 per-iter L1 measurements)

---

## Slide 2 · Problem — why this matters

![F02_EN](analysis_chain19/figures/mig_mps/F02_quadrant_ai_throughput_EN.png)

**Goal: run 5G L1 and multiple AI services on a single A100** (SoftBank AITRAS-style deployment)

- **Motivation 1 (why same GPU?)** — Multi-GPU is ideal but expensive. Operators want "1 GPU = 1 cell site + AI service pack"
- **Motivation 2 (why hard?)** — 5G TTI 500 μs SLA and AI batch throughput are competing requirements (latency vs throughput)
- **Motivation 3 (two design axes)** — Isolation tool choice (MIG hardware partition? MPS logical multiplexing? both?) × placement (co-locate L1+AI on the same partition? separate?). **Only measurement resolves it**

---

## Slide 3 · Attempt 1 — Multiple processes without MPS → L1 starves

![F_G03_EN](analysis_chain19/figures/mig_mps/F_G03_STARVE_EN.png)

**identical-NRx grid experiment · all configs · MPS off/on × N=1–8 · re-measured with launch rate**

- **Observation** — MPS OFF collapses L1 launch rate to 1000–2000 kernels/s. MPS ON holds 2000–12000. Same picture on Full GPU and MIG SP
- **Cause** — Without MPS, multiple processes on the GPU time-slice the CUDA context. L1 waits for its turn → kernel throughput plummets
- **Conclusion** — **MPS is mandatory.** Non-negotiable axis. Every subsequent experiment assumes MPS on

---

## Slide 4 · Attempt 2 — Full GPU + MPS on: **bimodal, worst-case fails**

**diverse-AI experiment Exp 1 · Full GPU + MPS on · diverse AI N=1–12** measured — two metrics side by side

| (a) Duty cycle view — "looks healthy" | (b) Actual L1 p99 latency — bimodal |
| :---: | :---: |
| ![F09b_EN](analysis_chain19/figures/mig_mps/F09b_duty_full_gpu_EN.png) | ![F09_EN](analysis_chain19/figures/mig_mps/F09_mps_alone_full_gpu_EN.png) |

- **Observation (a vs b contradiction + bimodality)** — (a) Duty is above baseline → "GPU well utilized". (b) Actual L1 p99 varies 42 → **63 ms**. Per-trial variance is high: N=6 gives **62 / 62 / 39 ms** across 3 trials (bimodal). MPS scheduler either packs well (baseline) or hits a bad pattern (63 ms)
- **Cause** — Duty cycle is a utilization metric ("how busy is the GPU"). **It is NOT an SLA metric ("did L1 kernels finish in time").** Must judge by worst-case to avoid missing SLA risk
- **Verdict** — Mean 54 ms · worst 62 ms → **worst-case fails 50 ms SLA**. Depends on unpredictable scheduler behaviour

---

## Slide 5 · Attempt 3 — MIG SP partition: counter-intuitively **worse than Full GPU**

![F_G04_EN](analysis_chain19/figures/mig_mps/F_G04_SP_PARADOX_EN.png)

**N=6 diverse AI · MPS on · L1 p99 measured across 3 trials**

- **Counter-intuitive observation** — MIG SP-4g + MPS pct=100 → **411 ms** (8× worse than Full GPU's 54 ms). "We turned MIG on and it got worse"
- **Cause** — MIG 4g partition = only **56 SMs**. Full GPU = 108 SMs. Same N AI clients on a **halved SM budget → contention intensifies**, L1 cannot secure SMs
- **Lesson** — "MIG on" by itself buys nothing. **When L1 and AI share a partition, MIG becomes a resource constraint that makes contention worse**, not better. MIG only pays off when partitions are used to **separate** workloads

---

## Slide 6 · Attempt 4 — Can MPS pct tuning save SP? **No**

![F_SP_PCT_EN](analysis_chain19/figures/mig_mps/F_SP_PCT_answer_EN.png)

**diverse-AI experiment Exp 11 · MIG SP-4g · MPS pct 100/70/50/30 × N=6 · L1 p99 measured**

- **Observation** — Lowering pct helps: 411 → 317 → 233 → **146 ms**. But **even the best (pct=30) is 3× over 50 ms SLA**
- **Cause** — The pct cap only bounds per-client SM usage. The launch queue is still shared, L1 kernels still wait. With only 56 SMs to begin with, capping AI doesn't leave enough headroom for L1
- **Conclusion** — **MIG+MPS cannot make SP meet SLA in the current dataset**. This setup fails as an answer for tight L1+AI co-work → **open problem**

---

## Slide 7 · Attempt 5 — MIG cross-partition + MPS on AI: **works**

![F_G05_EN](analysis_chain19/figures/mig_mps/F_G05_CP_WIN_EN.png)

**diverse-AI experiment Exp 5 · L1 on 4g partition · AI on 3g partition + MPS on · N=6/8/10/12/16 · 3 trials**

- **Observation** — L1 p99 mean 39–44 ms · worst 44–48 ms · **under 50 ms SLA all the way to N=16**. Tight min–max band (predictable)
- **Cause** — L1 runs alone on the 4g partition → owns its launch queue. AI is multiplexed via MPS on the 3g partition. **The two partitions' launch queues are hardware-separated** → no contention
- **Caveat** — This is **physical separation of L1 and AI**. Tightly-coupled workloads like NRx cannot live on the L1 partition → this is the **loose co-tenancy answer, not the tight co-work answer**

---

## Slide 8 · Under CP, duty and latency **agree** — evidence that isolation works

**diverse-AI experiment Exp 5 · two metrics side by side**

| (a) Duty cycle — stable (discrete per-condition bars) | (b) L1 p99 latency — baseline preserved |
| :---: | :---: |
| ![F13b_EN](analysis_chain19/figures/mig_mps/F13b_duty_cp_EN.png) | ![F_G05_EN](analysis_chain19/figures/mig_mps/F_G05_CP_WIN_EN.png) |

- **Both metrics agree** — The Slide 4 contradiction (duty ↑ but latency ↑) disappears. Duty and latency are **simultaneously** stable → evidence isolation is real
- **Mechanism** — The MIG partition boundary hardware-separates the launch queues. L1's duty is determined only by L1 kernel execution. AI activity does not leak into the metric
- **Reinterpretation** — In CP, this works **without any MPS pct tuning (pct=100 default)**. Reason: L1 and AI are on different partitions and different queues, so the pct cap is not needed for L1 protection. MPS matters only within the AI partition

---

## Slide 9 · Scale Validation — N=16 extreme, min–max band held

![F_G05_EN](analysis_chain19/figures/mig_mps/F_G05_CP_WIN_EN.png)

**diverse-AI experiment Exp 5 · N=16 diverse AI containers (Qwen · Whisper · BERT · NRx · CsiNet · BeamPred mix) · 30 min · 3 trials**

- **Numbers** — L1 p99 mean **42.7 ms** · worst 47 ms · 10 % over baseline 38.5 ms. Fault-isolated (AI crash → L1 untouched)
- **Why the number matters** — Within the 5G TTI 500 μs × 100 kernels/slot budget. Beats the SoftBank AITRAS target (5G + 6 AI services) by **2.6×**. Worst-case still passes SLA
- **Limit** — Can scale AI until the 3g partition's SMs/memory saturate. L1 remains at baseline on the 4g partition. Note: N > 16 has not been tested → **AI-partition saturation point is still open** (candidate for next experiment)

---

## Slide 10 · Verdict — Corrected decision matrix + open problem

![F_G06_EN](analysis_chain19/figures/mig_mps/F_G06_VERDICT_EN.png)

**Re-verdict based on measured data (all inaccurate figures like the earlier "45 ms" are corrected)**

- **Winners** — Multi-GPU and **MIG CP + MPS on AI** are the only topologies that pass SLA on both mean AND worst-case. Full GPU + MPS passes on the mean but fails worst-case (bimodal). SP·MIG fails at every tuning
- **Deployment recipe** — L1 on the 4g partition alone (MPS not needed). AI on the 3g partition with `nvidia-cuda-mps-control -d`. Pct tuning is for fairness across AI containers, not for L1 protection
- **Open problem** — **Tight L1+AI co-work (an NRx-like workload on the L1 partition itself) does not meet SLA in the current dataset**. Next experiments: CUDA stream priority · dedicated streams · smaller AI kernels · AI-partition saturation at N > 30

---

*Presentation · 2026-08-04 · 10 slides · re-verified figures · data sources: identical-NRx grid (2026-07 · 108 conditions) + diverse-AI deployment experiment (2026-08-03 · 273 conditions + 213 per-iter). The prior v1 numbers for SP·MPS·pct=30 were wrong — this v2 corrects them with measured values.*
