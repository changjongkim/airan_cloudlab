# MIG Is Ineffective For AI-RAN: Defensible Claim From Current Data

Date: 2026-06-01  
Scope: 20260531 + 20260601 results

## 1. Core Claim

The defensible paper claim is:

> MIG is useful for average throughput and capacity isolation, but it is not an effective abstraction for AI-RAN real-time deployment because it cannot provide deterministic L1 latency under the resource configurations AI-RAN actually needs. The failure comes from three mechanisms: partition fragmentation reduces L1 headroom, PHY-AI workloads such as NeuralRx can still create severe L1 tail inflation even when placed in a separate MIG partition, and in-line PHY-AI co-location inside a partition creates catastrophic temporal sharing. Cross-partition raw HBM saturation by generic stressors, however, is not the dominant observed failure mode in the final data.

In Korean:

> MIG는 평균 처리량 격리에는 효과가 있지만, AI-RAN의 L1 real-time latency 보장에는 효과적인 추상화가 아니다. 이유는 L1을 보호하려면 partition을 나눠야 하는데 그 순간 L1 headroom이 줄고, NeuralRx 같은 PHY-AI는 별도 MIG partition에 둬도 L1 tail을 크게 흔들 수 있으며, 같은 partition에서 함께 실행하면 tail latency가 폭발하기 때문이다. 단, 최종 데이터 기준으로 “일반적인 D2D/H2D/GEMM/raw HBM saturation 때문에 항상 L1이 무너진다”는 단순 주장은 지지되지 않는다.

## 1.1 Stronger Thesis: MIG Is The Wrong Abstraction For AI-RAN

The stronger version is not "MIG has interference." The stronger version is:

> MIG exposes a GPU as statically partitioned capacity islands, but AI-RAN needs a real-time composition abstraction between L1 and PHY-AI. These are different problems. MIG can isolate average resource ownership, but it cannot express or enforce the temporal contract that L1 needs: bounded frame-level tail latency while AI kernels, memory movement, and PHY inference phases execute nearby.

This makes the paper more robust because we do not need every workload to interfere. In fact, the 6/1 F result helps the argument:

- If generic cross-partition D2D/H2D/GEMM stress does **not** hurt L1, then raw bandwidth sharing alone is not the explanation.
- If NeuralRx separate-partition and NeuralRx same-partition cases **do** hurt L1, then the problem is more specific and more important: MIG cannot reason about AI-RAN workload semantics.
- If small L1 partitions are bad even alone, then "just isolate L1 harder" has a cost.

The result is a deployment-level failure:

> MIG does not give an operator a safe rule for consolidating L1 and PHY-AI. Separate them and you pay fragmentation/headroom cost and still see workload-specific PHY-AI risk; co-locate them and L1 p99 becomes catastrophic; use throughput metrics and you miss the failure entirely.

## 1.2 AI-RAN Deployment Trilemma

The cleanest way to frame the negative result is as a trilemma. An AI-RAN system wants all three:

1. **Protect L1 tail latency**
2. **Run PHY-AI close to L1**
3. **Use the GPU efficiently**

MIG cannot satisfy all three at once.

| Deployment choice | What it gets | What breaks | Evidence |
|---|---|---|---|
| Put L1 in a small isolated partition | spatial isolation, more room for AI | L1 headroom collapses | 2g L1 alone mean +40.2% vs Full |
| Put L1 in a larger isolated partition | better L1 headroom | less GPU left for AI; still not full guarantee | 4g/3g better than 2g but Tier1 co-tenant tails remain workload-dependent |
| Put NeuralRx in another partition | spatial separation | PHY-AI can still inflate L1 tail | 5/31 NeuralRx separate partition p99 +376% |
| Put NeuralRx with L1 in same partition | avoids cross-partition split, closer PHY-AI integration | catastrophic temporal sharing | 6/1 G p99 +373% to +537% |
| Rely on average throughput | looks isolated | misses p99/p999 failure | AI throughput changes ~0-2.2% while L1/coloc p99 explodes |

This trilemma is the paper's strongest conceptual contribution:

> MIG forces AI-RAN into a bad choice between fragmentation, unsafe co-location, and misleading throughput-only isolation.

## 1.3 What The Claim Is Not

To avoid reviewer pushback, state explicitly what we are not claiming.

We are **not** claiming:

- MIG is useless for all GPU sharing.
- MIG cross-partition throughput isolation fails generally.
- Raw HBM bandwidth saturation always hurts L1.
- Every AI workload hurts L1.
- More AI always means worse L1 latency.

We are claiming:

- MIG is insufficient for **real-time AI-RAN L1 + PHY-AI consolidation**.
- The insufficiency appears through **tail latency**, not average throughput.
- The failure is **workload-semantic and placement-dependent**, not a simple monotonic bandwidth curve.
- Therefore AI-RAN needs an isolation/scheduling mechanism beyond static MIG slicing.

## 2. What We Can Prove vs What We Should Not Overclaim

| Topic | Can we claim it? | Evidence | Wording |
|---|---|---|---|
| MIG fragmentation hurts L1 | Yes | 5/31 baselines: Full/7g/4g/3g/2g; 2g mean +40.2% vs Full | Strong claim |
| Small partition creates memory-subsystem pressure | Yes | 5/31 NCU: MIO throttle 7g 0.02% -> 2g 0.66%; L2 traffic changes | Strong support |
| MIG cross-partition raw HBM saturation breaks L1 | No, not with final data | 6/1 F: 39 stress conditions, 0 positive p99 inflation | Do not claim |
| Cross-partition early spikes exist | Yes, but exploratory | 5/31 Tier1/NSYS/A stage | Use as motivation, not final proof |
| Separate-partition NeuralRx is dangerous | Yes | 5/31 Phase4 NeuralRx: L1 p99 +376%; NSYS S7 idle/gap/memcpy inflation | Strong but workload-specific |
| Throughput metrics are misleading | Yes | AI throughput stable while L1/coloc p99 explodes | Strong claim |
| Same-partition L1 + NeuralRx is catastrophic | Yes | 6/1 G/H: p99 +373% to +537% | Strongest claim |
| Multi-AI effects are non-monotonic | Yes | 5/31 l1_multi_ai and NSYS v3: many multi-AI cases near baseline or smoothing | Important nuance |
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

### Act 4: PHY-AI Is Different From Generic AI, Even When L1 Is Isolated

This is the part that must be explicit for your argument. We did not only test same-partition co-location. We also tested L1 isolated in its own MIG partition while NeuralRx ran as a separate AI workload.

5/31 Tier1 Phase 4 used 3g L1 with real AI-RAN workloads on another partition:

| AI co-tenant | L1 mean | L1 p99 | Delta vs 3g L1 alone |
|---|---:|---:|---:|
| 3g L1 alone | 40.20 ms | 41.32 ms | baseline |
| qwen_small | 52.78 ms | 67.96 ms | p99 +64.5% |
| chanpred | 53.97 ms | 71.06 ms | p99 +72.0% |
| xapp | 52.93 ms | 68.85 ms | p99 +66.6% |
| **NeuralRx** | **60.43 ms** | **196.68 ms** | **p99 +376.0%** |

This says:

- Isolating L1 in its own MIG partition is not automatically enough.
- NeuralRx is not just a generic AI load; it is a PHY-AI workload with timing/memory/kernel behavior closer to L1.
- The p99 effect is much larger than Qwen/chanpred/xapp.

NSYS/NCU interpretation from 5/31:

- NCU per-kernel metrics changed little, so the L1 compute kernels themselves were not simply slowed down.
- NSYS showed the disturbance in inter-kernel gaps, memcpy, and copy/convert transitions.
- S7 NeuralRx had stronger total/idle inflation than Qwen in NSYS v2.
- Memory ops in NSYS detailed analysis showed memcpy total S5 140 ms -> S7 NeuralRx 585 ms, +317%.

Important nuance:

6/1 F shows generic cross-partition saturation does not break L1. That does **not** erase the 5/31 NeuralRx result. The correct interpretation is:

> Cross-partition interference is workload-specific, not universal. Generic bandwidth/compute saturation is safe in our final tests, but PHY-AI NeuralRx as a separate co-tenant produced severe L1 tail inflation in the 5/31 Tier1/NSYS data and must be treated as high risk.

This is stronger than a raw-bandwidth story because it says MIG placement cannot be decided from resource size alone. It requires workload semantics: PHY-AI vs LLM vs CNN vs synthetic GEMM/HBM.

### Act 5: Multi-AI And Partition Matrix Show Static Placement Is Unreliable

We also tested many multi-AI layouts and partition sizes. The key result is not "more AI always worse." The key result is **non-monotonicity**: static MIG placement cannot predict tail behavior from partition count or AI count alone.

5/31 Phase2/Phase3 Tier1:

| Layout | L1 partition | L1 mean | L1 p99 | Lesson |
|---|---|---:|---:|---|
| M1 3-way balanced | 3g | 53.36 ms | 71.64 ms | 3g L1 + multi-AI still high |
| M2 L1 small | 2g | 65.98 ms | 84.45 ms | small L1 partition is bad |
| M3 asymmetric | 4g | 53.96 ms | 70.90 ms | larger L1 does not remove all overhead |
| M4 4-way | 4g | 53.21 ms | 73.53 ms | more partitions, still high |
| D1 L1 starved | 2g | 66.71 ms | 82.89 ms | 2g is consistently risky |
| D2 L1 boosted | 4g | 53.67 ms | 73.10 ms | 4g helps headroom but not full guarantee |

5/31 extended `l1_multi_ai` matrix, N=5 each, shows many multi-AI combinations near baseline:

| Scenario | L1 size / AI mix | Alone p99 | Multi p99 | Delta |
|---|---|---:|---:|---:|
| M5a | 3-way balanced + sat_compute | 39.29 | 39.62 | +0.84% |
| M5b | 3-way balanced + sat_hbm | 38.43 | 38.54 | +0.31% |
| M5c | ResNet + chanpred | 41.02 | 37.77 | -7.91% |
| M7b | 4-way heterogeneous 3AI | 39.79 | 38.31 | -3.73% |
| M8a | ResNet + Forecaster | 38.69 | 38.98 | +0.74% |
| M10b | 4g L1 + NeuralRx | 39.20 | 40.69 | +3.81% |
| M11c | 2g L1 + ResNet | 51.95 | 53.40 | +2.79% |

This matters because:

- Multi-AI does not monotonically increase L1 latency.
- Some heterogeneous mixes smooth tail behavior.
- 2g L1 still has poor baseline headroom even when the multi condition itself is not much worse.
- Therefore "allocate X g to L1 and Y g to AI" is not enough; workload type and temporal behavior matter.

NSYS v3 supports the same point:

| Scenario | Finding |
|---|---|
| S27 3g + chanpred | steady p99 +8% |
| S28 3g + ResNet | steady p99 +26% |
| S29 3g + Forecaster | steady p99 +19% |
| S31 ResNet+chanpred | steady p99 +3.4%, smoothing |
| S32 ResNet+Forecaster | steady p99 +7%, smoothing |
| S35 2g + chanpred | steady p99 +66%, worst partition/workload combination |

The placement conclusion is:

> MIG static slicing is too coarse for AI-RAN. The same partition size can be safe or unsafe depending on workload phase behavior; adding more AI can sometimes smooth rather than worsen tail latency; and small L1 partitions are fragile regardless.

### Act 6: Same-Partition PHY-AI Co-Location Breaks L1 Catastrophically

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

### Act 7: Static MIG Placement Is Not Enough

Final deployment problem:

- If L1 gets a small partition, standalone latency/headroom is bad.
- If NeuralRx is placed in a separate partition, 5/31 data still shows a severe L1 p99 risk.
- If L1 and NeuralRx are placed in the same partition, p99 explodes.
- If generic AI is placed in a separate partition, cross-partition isolation is often good, but then capacity is fragmented and workload-specific PHY-AI behavior still matters.
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

4. **PHY-AI cross-partition behavior is not equivalent to synthetic bandwidth**
   - 6/1 F says D2D/H2D/GEMM saturation is safe.
   - 5/31 NeuralRx says a PHY-AI co-tenant can still be dangerous.
   - Therefore the issue is not "bandwidth amount" alone, but the temporal pattern of memory movement, kernel launches, copy/convert boundaries, and PHY-like processing phases.

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

### Full version that includes separate-partition NeuralRx

> MIG is not a sufficient isolation mechanism for AI-RAN because safety depends on workload semantics, not only resource size. Generic cross-partition saturation is well isolated, but L1 loses headroom under small slices, NeuralRx can create severe L1 p99 inflation even as a separate PHY-AI co-tenant, and same-partition L1+NeuralRx co-location produces catastrophic bimodal tail latency.

### Strongest paper-title version

> MIG is the wrong abstraction for real-time AI-RAN consolidation: it partitions capacity, but AI-RAN needs temporal latency contracts between L1 and PHY-AI.

### Reviewer-proof version

> Our results do not show that MIG fails at all forms of cross-partition isolation. On the contrary, MIG isolates generic cross-partition saturation well. This is precisely why the AI-RAN result is sharper: the failure appears when the deployment requires L1 and PHY-AI to be composed under real-time deadlines. Static capacity slicing cannot provide that composition safely.

## 5.1 Claims Ranked By Strength

Use claims in this order. The first three are the backbone; the later ones are support.

| Rank | Claim | Strength | Why |
|---:|---|---|---|
| 1 | Same-partition L1+NeuralRx is catastrophic | Very strong | 6/1 G/H, p99 +373% to +537% |
| 2 | Small L1 partition loses headroom | Very strong | 5/31 baselines, 2g +40.2% mean vs Full |
| 3 | Generic cross-partition saturation is not the culprit | Very strong | 6/1 F, 0/39 positive p99 inflation |
| 4 | Separate-partition NeuralRx is high-risk | Strong but condition-specific | 5/31 Phase4 p99 +376%, NSYS supports |
| 5 | Throughput isolation is misleading | Strong | throughput stable, p99 not stable |
| 6 | Multi-AI behavior is non-monotonic | Strong nuance | l1_multi_ai and NSYS v3 smoothing |
| 7 | L2/cache/memory headroom shrinks with partitioning | Supportive mechanism | NCU MIO/L2 evidence |

## 5.2 The One-Slide Version

If you need to explain the whole project on one slide:

```
Question:
  Can MIG safely consolidate cuPHY L1 and AI/PHY-AI for AI-RAN?

Result:
  MIG isolates generic cross-partition throughput stress.
  But AI-RAN still fails because:
    (1) L1 small slices lose headroom: 2g L1 mean +40%.
    (2) NeuralRx separate partition can inflate L1 p99 +376%.
    (3) L1+NeuralRx same partition inflates p99 +373% to +537%.
    (4) AI throughput metrics hide the failure.

Conclusion:
  MIG partitions capacity; AI-RAN needs bounded temporal latency.
  Static MIG slicing alone is not a sufficient isolation mechanism.
```

## 5.3 Anticipated Reviewer Objections

### Objection: "But your F experiment shows MIG isolation works."

Answer:

Yes. That is why the claim is not "MIG isolation is broken." The claim is that **MIG's isolation target is the wrong target for AI-RAN**. It isolates generic cross-partition saturation, but it does not provide a safe real-time composition rule for L1 and PHY-AI.

### Objection: "Why not just put NeuralRx in another MIG partition?"

Answer:

We tried the separated design. In 5/31 Phase4, 3g L1 with NeuralRx as an AI co-tenant showed L1 p99 +376%. Even if later generic saturation is safe, NeuralRx is a PHY-AI workload whose temporal pattern is high-risk. Static MIG placement does not know that.

### Objection: "Why not just allocate a larger L1 partition?"

Answer:

Larger partitions reduce fragmentation risk but consume the GPU budget that AI-RAN needs for AI workloads. More importantly, 4g does not solve same-partition coloc; 6/1 G shows 4g coloc p99 +537%, worse than 3g in that experiment.

### Objection: "If multi-AI sometimes smooths latency, is MIG actually okay?"

Answer:

No. Smoothing proves the opposite of a static guarantee. Tail behavior depends on workload timing and phase alignment. An isolation mechanism that is safe only for some accidental mixes is not a real-time scheduling abstraction.

### Objection: "Your L1 latency numbers are tens of ms, not sub-ms URLLC."

Answer:

The absolute benchmark includes the experimental cuPHY setup, 20 cells, synthetic pipeline, and measurement envelope. The paper claim should focus on **relative tail inflation and isolation failure**, not absolute production slot timing. The same system-level issue remains: p99/p999 tail is not bounded by MIG.

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

5. **5/31 NeuralRx separate-partition risk**
   - 3g L1 alone vs 3g L1 + NeuralRx separate partition.
   - Claim: L1 isolation alone does not make PHY-AI safe.

6. **Multi-AI non-monotonic matrix**
   - M5-M12 and NSYS v3 S27-S36.
   - Claim: static placement rules cannot predict tail behavior.

7. **H dual sanity / bimodal**
   - external stress safe vs coloc catastrophic.
   - Claim: real-time tail failure, not average slowdown.

8. **Throughput vs p99 contrast**
   - AI throughput stable, L1 p99 catastrophic under coloc.
   - Claim: throughput isolation is the wrong metric.

## 7. Bottom Line

Yes, we can support the claim that MIG is not effective for AI-RAN. The cleanest version is:

> MIG is effective at many forms of cross-partition throughput isolation, but AI-RAN needs deterministic L1 tail-latency isolation. MIG fails this deployment objective because static slicing reduces L1 headroom, workload-specific PHY-AI such as NeuralRx can still create severe L1 tail inflation even when separated, and same-partition in-line PHY-AI co-location creates catastrophic bimodal tail latency. Therefore MIG alone is not a sufficient isolation mechanism for AI-RAN real-time L1 + PHY-AI consolidation.
