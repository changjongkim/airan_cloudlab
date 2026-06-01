# MIG Is Ineffective For AI-RAN: Defensible Claim From Current Data

Date: 2026-06-01  
Scope: 20260531 + 20260601 results

## 1. Core Claim

The defensible paper claim is:

> MIG is useful for average throughput and capacity isolation, but it is not an effective abstraction for AI-RAN real-time deployment because it cannot provide deterministic L1 latency under the resource configurations AI-RAN actually needs. The failure comes from two mechanisms: partition fragmentation reduces L1 headroom, and in-line PHY-AI co-location inside a partition creates catastrophic temporal sharing. Cross-partition raw HBM saturation, however, is not the dominant observed failure mode in the final data.

In Korean:

> MIG는 평균 처리량 격리에는 효과가 있지만, AI-RAN의 L1 real-time latency 보장에는 효과적인 추상화가 아니다. 이유는 L1을 보호하려면 partition을 나눠야 하는데 그 순간 L1 headroom이 줄고, NeuralRx 같은 in-line PHY-AI를 같은 partition에서 함께 실행하면 tail latency가 폭발하기 때문이다. 단, 최종 데이터 기준으로 “다른 partition의 raw HBM bandwidth saturation 때문에 L1이 무너진다”는 단순 주장은 지지되지 않는다.

## 2. What We Can Prove vs What We Should Not Overclaim

| Topic | Can we claim it? | Evidence | Wording |
|---|---|---|---|
| MIG fragmentation hurts L1 | Yes | 5/31 baselines: Full/7g/4g/3g/2g; 2g mean +40.2% vs Full | Strong claim |
| Small partition creates memory-subsystem pressure | Yes | 5/31 NCU: MIO throttle 7g 0.02% -> 2g 0.66%; L2 traffic changes | Strong support |
| MIG cross-partition raw HBM saturation breaks L1 | No, not with final data | 6/1 F: 39 stress conditions, 0 positive p99 inflation | Do not claim |
| Cross-partition early spikes exist | Yes, but exploratory | 5/31 Tier1/NSYS/A stage | Use as motivation, not final proof |
| Throughput metrics are misleading | Yes | AI throughput stable while L1/coloc p99 explodes | Strong claim |
| Same-partition L1 + NeuralRx is catastrophic | Yes | 6/1 G/H: p99 +373% to +537% | Strongest claim |
| MIG needs workload-aware placement | Yes | F safe cross-partition vs G catastrophic coloc; external AI type less important after coloc | Strong claim |

## 3. Storyline For A Paper

### Act 1: AI-RAN Needs Tail-Latency Isolation, Not Just Capacity Isolation

AI-RAN L1 is not a throughput workload. cuPHY L1 is a latency-critical pipeline made of many short kernels, copy/convert operations, and strict frame/slot timing. Therefore a good isolation mechanism must preserve p99/p999 latency, not just average throughput.

Evidence:

- AI throughput v2 shows AI throughput changes only around 0-2.2% with L1.
- But L1 p99 and coloc p99 can change by hundreds of percent.

Use this to say:

> Throughput isolation and real-time isolation are different. MIG mostly solves the former, but AI-RAN needs the latter.

### Act 2: MIG Partitioning Itself Reduces L1 Headroom

MIG turns one GPU into smaller static devices. This has a direct cost for L1 even before AI co-tenants are added.

5/31 clean baseline:

| L1 config | Mean | p99 | Key point |
|---|---:|---:|---|
| Full GPU v2 | 36.38 ms | 39.18 ms | baseline |
| 7g MIG | 37.21 ms | 38.60 ms | MIG mode itself is almost free |
| 4g MIG | 39.07 ms | 40.47 ms | modest overhead |
| 3g MIG | 40.20 ms | 41.32 ms | modest overhead |
| 2g MIG | 51.01 ms | 52.14 ms | mean +40.2% vs Full |

NCU support:

| Metric | 7g | 3g | 2g | Meaning |
|---|---:|---:|---:|---|
| MIO throttle | 0.02% | 0.13% | 0.66% | 2g has much less memory I/O headroom |
| L2 traffic / kernel | 12.27 | 377.86 | 289.49 | smaller partition changes L2/memory behavior |
| SM throughput | 1.65% | 3.69% | 5.08% | same work compressed into fewer SMs |

Claim:

> MIG fragmentation is not free. It preserves isolation by statically reducing each workload's available compute/cache/memory-system headroom. For L1, 2g is already too small before AI is added.

This directly supports your L2/capacity argument. The clean wording is not "L2 cache saturation is fully proven in every case," but:

> The data show memory-subsystem headroom loss under small partitions, with MIO throttle and L2 traffic changes consistent with L2/cache/memory-pipeline pressure.

### Act 3: Cross-Partition Saturation Is Not The Main Culprit

This is counterintuitive but important. We tested your initial bandwidth-sharing hypothesis hard on 6/1.

6/1 F saturation matrix:

- D2D copies: sizes/streams
- H2D copies
- GEMM compute
- chanpred intensity
- ResNet intensity
- Forecaster intensity
- stacked AI workloads
- kitchen sink stress

Result:

| Block | Conditions | Mean p99 delta | Worst p99 delta |
|---|---:|---:|---:|
| D2D | 12 | -17.2% | -15.0% |
| H2D | 6 | -13.5% | -12.2% |
| GEMM | 4 | -16.3% | -12.1% |
| chanpred intensity | 5 | -15.5% | -9.6% |
| ResNet intensity | 3 | -17.4% | -17.0% |
| Forecaster intensity | 3 | -16.7% | -16.3% |
| stacked chanpred | 3 | -15.4% | -14.4% |
| stacked ResNet | 2 | -16.8% | -16.1% |
| kitchen sink | 1 | -18.6% | -18.6% |

Positive p99 inflation: **0 / 39 conditions**.

Claim:

> The final data reject a naive raw-HBM-bandwidth contention story. MIG cross-partition steady saturation is much better isolated than expected.

Why this still helps the paper:

- It makes the paper more credible because we are not forcing a hypothesis.
- It sharpens the real failure mode: MIG is not ineffective because every partition interferes all the time; it is ineffective because AI-RAN placement requirements expose fragmentation and same-partition temporal sharing.

### Act 4: Same-Partition PHY-AI Co-Location Breaks L1 Catastrophically

This is the strongest evidence that MIG is ineffective for AI-RAN deployment.

6/1 G:

| Condition | Alone p99 | Coloc p99 | Delta |
|---|---:|---:|---:|
| 3g L1 + NeuralRx same partition | 56.14 ms | 265.32 ms | +372.6% |
| 4g L1 + NeuralRx same partition | 56.00 ms | 356.56 ms | +536.7% |
| 2g L1 + NeuralRx same partition | 61.14 ms | 369.60 ms | +504.5% |

6/1 H:

| Condition | Mean | p99 | Interpretation |
|---|---:|---:|---|
| 3g alone | 44.09 ms | 45.56 ms | clean |
| D2D max external | 40.33 ms | 43.26 ms | cross-partition safe |
| GEMM 4096 external | 42.97 ms | 44.45 ms | cross-partition safe |
| stack4 chanpred external | 44.07 ms | 45.53 ms | cross-partition safe |
| kitchen external | 42.90 ms | 44.65 ms | cross-partition safe |
| 3g coloc + chanpred external | 124.87 ms | 355.84 ms | catastrophic |
| 2g coloc + chanpred external | 139.63 ms | 368.19 ms | catastrophic |

Claim:

> MIG cannot safely host L1 and in-line PHY-AI in the same partition. The L1 distribution becomes bimodal: typical frames can look normal, but tail frames become catastrophic.

This is the core AI-RAN argument. NeuralRx is not just "another AI"; it represents in-line PHY-AI, the exact AI-RAN workload class that operators want to integrate near L1.

### Act 5: Static MIG Placement Is Not Enough

Final deployment problem:

- If L1 gets a small partition, standalone latency/headroom is bad.
- If L1 and NeuralRx are placed in the same partition, p99 explodes.
- If AI is placed in a separate partition, cross-partition isolation is good, but then capacity is fragmented and you may not have enough space for in-line processing.
- Throughput measurements miss the problem.

Claim:

> MIG provides static spatial isolation, but AI-RAN needs workload-aware temporal isolation. Static slicing cannot decide safely without knowing whether the AI is external, in-line, bursty, PHY-like, copy-heavy, or co-located with L1.

## 4. How To Use Your Original Bandwidth/L2 Intuition Correctly

Your initial intuition is still useful, but it must be narrowed.

### Original intuition

> One physical GPU has fixed bandwidth/fabric/cache resources, so slicing creates shared-resource contention and L1 latency spikes.

### What the data supports

1. **Fixed resources matter through fragmentation**
   - Smaller partition means less SM/cache/memory-pipeline headroom.
   - 2g L1 proves this.

2. **L2/cache/memory-system pressure matters within small partitions**
   - NCU MIO throttle and L2 traffic changes support this.

3. **Temporal bandwidth matters inside the same partition**
   - L1 + NeuralRx coloc causes catastrophic p99.
   - This likely reflects time-sharing of SMs, memory path, kernels, and runtime scheduling inside one MIG device.

### What the data does not support

1. **General cross-partition HBM saturation failure**
   - F saturation refutes this.

2. **External AI type is always decisive**
   - In G, once coloc is active, external AI type adds similar overhead.

3. **Mean/throughput metrics prove real-time safety**
   - They do not.

## 5. Final Claim Wording Options

### Strong but defensible

> MIG is insufficient for AI-RAN real-time consolidation. Although it provides strong cross-partition throughput isolation, it forces static partitioning that reduces L1 headroom and fails catastrophically when in-line PHY-AI shares a partition with cuPHY L1.

### More systems-paper style

> MIG solves spatial capacity isolation, but AI-RAN requires temporal latency isolation. Our data show that cross-partition saturation is not the dominant problem; instead, the combination of partition fragmentation and same-partition PHY-AI time-sharing produces unacceptable L1 tail latency.

### If you want to keep bandwidth language

> The issue is not sustained cross-partition HBM bandwidth saturation. The issue is effective temporal bandwidth available to L1: small MIG slices reduce memory-system headroom, and same-partition in-line AI consumes burst execution/memory/runtime slots that L1 needs for deterministic frame processing.

## 6. Figure / Evidence Plan

Use these as the core figures:

1. **Partition baseline figure**
   - Full / 7g / 4g / 3g / 2g L1 alone.
   - Claim: fragmentation cost.

2. **NCU small partition figure**
   - MIO throttle and L2 traffic for 7g/3g/2g.
   - Claim: memory-subsystem headroom shrinks.

3. **F saturation negative result**
   - 39 conditions, p99 delta <= 0.
   - Claim: cross-partition raw bandwidth is not the culprit.

4. **G coloc explosion**
   - 3g/4g/2g coloc p99 deltas.
   - Claim: same-partition PHY-AI is catastrophic.

5. **H dual sanity / bimodal**
   - external stress safe vs coloc catastrophic.
   - Claim: real-time tail failure, not average slowdown.

6. **Throughput vs p99 contrast**
   - AI throughput stable, L1 p99 catastrophic under coloc.
   - Claim: throughput isolation is the wrong metric.

## 7. Bottom Line

Yes, we can support the claim that MIG is not effective for AI-RAN. The cleanest version is:

> MIG is effective at cross-partition throughput isolation, but AI-RAN needs deterministic L1 tail-latency isolation. MIG fails this deployment objective because static slicing reduces L1 headroom and same-partition in-line PHY-AI co-location creates catastrophic bimodal tail latency. Therefore MIG alone is not a sufficient isolation mechanism for AI-RAN real-time L1 + PHY-AI consolidation.

