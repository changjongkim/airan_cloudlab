# MIG는 AI-RAN에 충분한 격리 추상화가 아니다: 시각 증거

생성일: 2026-06-01  
소스 범위: `cloudlab_results/results/20260531`, `cloudlab_results/results/20260601`

이 문서는 정리된 evidence를 그림 중심으로 다시 구성한 한국어 버전이다. 핵심 결론은 단순히 "L1만 격리하면 되는가"가 아니라, **L1 tail latency와 AI workload service quality를 동시에 만족해야 하는 AI-RAN에서 MIG의 정적 파티셔닝이 잘못된 제어 추상화**라는 점이다.

## 1. 작은 MIG slice는 L1 headroom을 줄인다

![Partition baseline](figures/fig01_partition_baseline.png)

5/31 baseline에서 2g L1은 Full GPU v2 대비 mean latency가 약 +40.2% 증가한다. 7g/4g/3g는 비교적 안정적이지만, slice가 작아질수록 L1이 사용할 수 있는 SM/cache/memory-system headroom이 줄어든다. 이 결과는 "MIG가 capacity는 나눠주지만 real-time headroom까지 보장하지는 않는다"는 첫 번째 증거다.

## 2. NeuralRx는 generic AI가 아니라 PHY-AI co-tenant다

![Phase4 NeuralRx](figures/fig02_phase4_neuralrx_risk.png)

5/31 Phase4에서 3g L1 기준 NeuralRx co-tenant는 L1 p99를 약 +376%까지 키웠다. Qwen-small, ChanPred, XApp도 p99를 올리지만 NeuralRx는 훨씬 크다. 따라서 주장은 "모든 외부 AI가 항상 위험하다"가 아니라, **AI-RAN에서 실제로 붙이고 싶은 PHY-AI의 temporal behavior가 L1에 매우 위험할 수 있다**는 쪽이 더 정확하다.

## 3. 하지만 generic cross-partition saturation이 주범은 아니다

![F saturation](figures/fig03_f_saturation_negative_result.png)

6/1 F saturation은 D2D/H2D/GEMM/ResNet/ChanPred/Forecaster/stack/kitchen stress를 걸었지만, non-baseline 39개 조건에서 positive L1 p99 inflation이 0개였다. 이 negative result가 중요하다. 우리의 주장은 "MIG의 cross-partition isolation이 전부 깨졌다"가 아니다. 오히려 generic saturation은 잘 막히기 때문에, AI-RAN failure가 더 선명해진다.

## 4. 같은 partition에 L1과 NeuralRx를 넣으면 p99가 폭발한다

![G coloc](figures/fig04_g_coloc_explosion.png)

6/1 G에서 same-partition L1+NeuralRx coloc은 3g p99 +372.6%, 4g p99 +536.7%, 2g p99 +504.5% 수준의 catastrophic tail을 만든다. 특히 4g에서도 해결되지 않는다. 즉 "큰 partition을 주면 coloc이 안전해진다"는 단순한 해법이 아니다.

## 5. H dual sanity check: 외부 stress는 안전, coloc은 위험

![H dual](figures/fig05_h_dual_sanity.png)

6/1 H에서는 같은 3g L1 기준으로 외부 D2D/GEMM/stack/kitchen stress는 p99가 baseline 근처에 머문다. 반면 coloc 조건은 p99가 350ms 이상으로 튄다. 이것은 원인을 더 좁혀준다. 문제는 sustained cross-partition HBM bandwidth 하나가 아니라, **same-partition temporal sharing, runtime scheduling, copy/kernel phase alignment**가 L1 deadline과 충돌하는 구조다.

## 6. AI workload도 partition에 자유롭지 않다

![AI partition scaling](figures/fig06_ai_partition_scaling.png)

AI workload는 leftover slice에 그냥 고정 배치할 수 없다. Qwen-small은 1g에서 실패했고, Qwen-7B 계열은 4g에서만 의미 있게 실행된다. ResNet, sat_compute, sat_hbm, Forecaster는 partition size에 따라 throughput이 크게 달라진다. 따라서 L1을 보호하려고 큰 slice를 주면 AI capacity가 줄고, AI throughput을 보존하려면 L1 headroom이 줄어든다.

## 7. AI 평균 throughput이 안정적이어도 AI p99는 흔들린다

![AI per-op p99](figures/fig07_ai_per_op_p99_delta.png)

5/31 AI per-op latency에서 ChanPred는 2g 기준 p99 +27.0%, 3g +23.7%, 4g +19.6%까지 증가한다. NeuralRx 3g도 +12.9%, Qwen도 partition별로 +3.2~+6.7% 수준의 p99 증가가 보인다. 즉 throughput만 보면 양쪽이 안전해 보일 수 있지만, real-time 관점에서는 AI side도 tail-latency tradeoff를 가진다.

## 8. 최종 tradeoff

![Tradeoff](figures/fig08_tradeoff_summary.png)

MIG에서 선택지는 모두 비용을 가진다.

- L1을 작은 slice에 두면 L1 headroom이 줄어든다.
- L1을 보호하려고 큰 slice를 주면 AI가 fit/throughput/p99 비용을 낸다.
- NeuralRx를 separate partition에 둬도 5/31에서는 L1 p99 risk가 컸다.
- L1과 NeuralRx를 같은 partition에 두면 6/1 G/H에서 p99가 catastrophic하게 폭발한다.
- 여러 AI workload를 나눠 배치하면 multi-AI phase behavior가 non-monotonic해진다.

이 마지막 그림은 단일 실험의 raw plot이 아니라 evidence synthesis plot이다. L1 축은 관측된 latency inflation을 사용했고, AI 축은 fit failure, AI p99 inflation, coloc NeuralRx throughput 감소 정황을 하나의 risk axis에 모은 것이다. 따라서 논문 본문에서는 앞선 개별 figure들을 primary evidence로 쓰고, 이 그림은 전체 tradeoff를 설명하는 summary figure로 쓰는 것이 안전하다.

## 9. Multi-AI placement는 단조롭지 않다

![Multi AI matrix](figures/fig09_l1_multi_ai_matrix.png)

5/31 `l1_multi_ai` sweep은 같은 L1이라도 AI 조합과 partition 배치에 따라 p99가 증가하기도 하고 감소하기도 함을 보여준다. 이 결과는 "AI가 많으면 항상 나쁘다"도 아니고 "AI를 분리하면 항상 안전하다"도 아니라는 점을 말한다. MIG static slicing은 workload phase alignment와 runtime behavior를 모르기 때문에, 미리 정한 고정 배치 규칙만으로 real-time safety를 보장하기 어렵다.

## 10. AI throughput과 AI p99는 같은 이야기를 하지 않는다

![AI throughput vs p99](figures/fig10_ai_throughput_vs_p99.png)

5/31 `ai_throughput_v2`에서는 L1 background가 있어도 AI mean throughput은 거의 변하지 않는다. 하지만 `ai_per_op_latency`에서는 ChanPred, NeuralRx, Qwen의 per-op p99가 증가한다. 이 그래프는 throughput-only isolation metric이 AI-RAN에는 부족하다는 점을 보여준다. AI-RAN에서는 L1뿐 아니라 AI inference 자체도 tail-sensitive service가 될 수 있다.

## 11. P5 sustained run에서도 workload-dependent behavior가 남는다

![P5 sustained](figures/fig11_p5_sustained_l1.png)

P5 sustained run은 짧은 micro test가 아니라 긴 시간 동안 L1과 co-tenant AI를 같이 돌린 결과다. p99는 workload마다 다르게 움직이며, NeuralRx와 다른 AI workload의 pattern이 동일하지 않다. 이것은 단발성 outlier가 아니라, co-tenant temporal behavior가 L1 tail에 계속 관여한다는 보조 증거다.

## 12. NSYS gap 분석: tail은 inter-kernel gap에서 보인다

![NSYS gap](figures/fig12_nsys_gap_summary.png)

NSYS deep analysis는 L1 kernel 자체의 평균 실행시간만으로는 tail을 설명하기 어렵고, kernel 사이 gap과 burst idle 구간이 중요하다는 점을 보여준다. 특히 2g L1 + ChanPred는 p99 gap이 크게 나타난다. 이 결과는 대역폭을 단순 sustained GB/s로만 보면 놓치는 **temporal bandwidth / scheduling gap** 문제를 뒷받침한다.

## 13. Memory/copy pressure는 workload별로 다르다

![Memory ops](figures/fig13_memory_ops_pressure.png)

NSYS memory-op summary를 보면 NeuralRx, ResNet, Forecaster 등은 memcpy total이나 p99 pattern이 다르다. 이 차이는 왜 synthetic D2D/H2D saturation만으로는 PHY-AI interference를 재현하기 어려운지 설명한다. 문제는 총량 bandwidth 하나가 아니라 copy/convert/kernel phase가 L1과 어떻게 겹치는가다.

## 14. Dv2 replication은 negative result를 강화한다

![Dv2 sanity](figures/fig14_dv2_sanity.png)

Dv2 반복 실험은 H2D, D2D, compute, launch, ChanPred stress가 baseline 근처에 머무는 것을 보여준다. 이 그림은 주장을 더 조심스럽고 강하게 만든다. 즉, "MIG cross-partition이 항상 깨진다"가 아니라, **generic cross-partition stress는 대체로 안전하지만 AI-RAN PHY-AI composition에서는 정적 MIG가 충분하지 않다**가 맞다.

## 결론

데이터 기반으로 가장 강한 주장은 다음이다.

> MIG는 generic cross-partition throughput isolation에는 효과적일 수 있다. 그러나 AI-RAN은 L1 tail latency와 AI workload service quality를 동시에 보장해야 한다. MIG는 static capacity slicing만 제공하므로, L1 headroom, AI fit/throughput/p99, PHY-AI co-location tail latency 사이의 tradeoff를 안전하게 제어하지 못한다. 따라서 MIG 단독으로는 real-time L1 + PHY-AI consolidation을 위한 충분한 isolation mechanism이 아니다.

## 생성된 source tables

- `data/partition_baseline.csv`
- `data/phase4_phy_ai_p99.csv`
- `data/f_saturation_block_summary.csv`
- `data/g_coloc_l1_p99.csv`
- `data/h_dual_p99.csv`
- `data/ai_throughput_parsed.csv`
- `data/ai_partition_scaling.csv`
- `data/ai_per_op_latency_parsed.csv`
- `data/ai_per_op_p99_delta.csv`
- `data/tradeoff_summary.csv`
- `data/l1_multi_ai_matrix.csv`
- `data/ai_throughput_v2_parsed.csv`
- `data/ai_throughput_vs_p99.csv`
- `data/p5_sustained_l1.csv`
- `data/nsys_gap_summary.csv`
- `data/memory_ops_pressure.csv`
- `data/dv2_sanity.csv`
