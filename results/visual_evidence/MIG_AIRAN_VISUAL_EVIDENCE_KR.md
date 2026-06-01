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

## 9. 정적 partition plan은 L1과 AI 사이의 tradeoff를 드러낸다

![Static partition sweep](figures/fig09_static_partition_sweep.png)

5/31 Phase2/Phase3는 여러 MIG partition plan을 직접 바꿔가며 L1 p99를 본 sweep이다. 그림의 라벨은 코드명 대신 실제 배치 의미로 풀어썼다. 예를 들어 `L1 3g / Qwen-small on 2g+2g`는 L1이 3g slice를 쓰고, Qwen-small 계열 AI가 두 개의 2g slice에 배치된 조건이다. `L1 2g / Qwen-small on 3g`는 AI 쪽에 더 큰 slice를 주는 대신 L1이 2g로 줄어든 조건이다.

여기서 가장 중요한 점은 L1이 2g로 작아지는 순간 p99가 82~84ms대로 올라간다는 것이다. 반대로 L1을 4g 또는 약 3g로 키워도 p99가 완전히 baseline으로 돌아오지는 않는다. 즉, L1에 더 큰 slice를 주면 headroom은 좋아지지만 AI가 쓸 수 있는 GPU budget이 줄고, AI에 더 큰 slice를 주면 L1이 작아져 real-time tail이 악화된다. 이것이 MIG의 정적 slicing이 만드는 가장 기본적인 운영 tradeoff다.

이 그림은 내부 실험 코드명 대신 실제 partition과 workload 의미로 바꿔 표시했다. 빨간 막대는 L1이 2g로 starved된 조건이고, 노란 막대는 L1이 3g/4g를 받았지만 여전히 AI co-tenant와 함께 실행된 조건이다. 핵심은 특정 AI 하나가 아니라, **partition plan 자체가 L1 safety와 AI capacity를 동시에 결정한다**는 점이다.

## 10. AI throughput과 AI p99는 같은 이야기를 하지 않는다

![AI throughput vs p99](figures/fig10_ai_throughput_vs_p99.png)

5/31 `ai_throughput_v2`에서는 L1 background가 있어도 AI mean throughput은 거의 변하지 않는다. ChanPred, NeuralRx, Qwen-small, XApp 모두 평균 처리량 변화만 보면 0~2% 수준으로 안전해 보인다. 하지만 같은 계열의 `ai_per_op_latency`를 보면 ChanPred는 최대 +27%, NeuralRx는 +12.9%, XApp은 +12.3%, Qwen은 +6.7%까지 per-operation p99가 증가한다.

이 그림의 목적은 AI side도 단순 throughput으로 평가하면 안 된다는 점을 보여주는 것이다. AI-RAN에서 AI inference가 scheduling loop 안에 들어오거나 near-real-time decision에 쓰이면, 평균 throughput이 아니라 per-op tail latency가 중요해진다. 그러면 MIG의 문제는 L1만의 문제가 아니다. L1을 보호하기 위한 partitioning이 AI의 fit/throughput/tail latency와 충돌하고, AI를 보호하기 위한 partitioning이 L1 headroom과 충돌한다.

## 11. Coloc이 시작되면 외부 AI 종류는 2차 문제가 된다

![Coloc external dominance](figures/fig11_coloc_external_dominance.png)

6/1 G 실험은 L1과 NeuralRx를 같은 MIG partition에 coloc한 뒤, 바깥 partition에 다른 AI workload를 추가로 올린 조건들을 비교한다. 결과는 매우 선명하다. 외부에 ChanPred, Forecaster, Qwen-small, ResNet, HBM saturation, XApp, 또는 복수 AI를 올려도 L1 p99는 대체로 356~371ms 근처에 머문다. 즉, 이 구간에서는 외부 AI 종류가 핵심 원인이 아니다. 이미 같은 partition 안에서 L1과 NeuralRx가 temporal resources를 공유하는 순간, tail failure가 지배적이 된다.

이 그림은 "어떤 AI가 외부에 있으면 위험한가?"라는 질문보다 "L1과 in-line PHY-AI를 같은 partition에 넣어도 되는가?"라는 질문이 더 중요하다는 점을 보여준다. 답은 데이터상 명확히 아니다. MIG는 partition 사이 capacity isolation은 줄 수 있지만, 같은 MIG device 안에서 L1과 PHY-AI가 runtime, kernel launch, copy, SM/memory path를 시간적으로 나눠 쓰는 문제는 해결하지 못한다.

## 12. NSYS SQLite 재분석: kernel-only gap은 idle이 아니다

![NSYS gap](figures/fig12_nsys_kernel_vs_activity_gap.png)

처음 NSYS 결과를 kernel-to-kernel gap만으로 보면 "L1 kernel 사이에 긴 idle이 있다"처럼 보인다. 그런데 SQLite에서 `CUPTI_ACTIVITY_KIND_KERNEL`, `MEMCPY`, `MEMSET` interval을 다시 합쳐보면 해석이 달라진다. 많은 long gap은 진짜로 GPU가 논 것이 아니라, 다음 L1 kernel로 넘어가기 전 boundary가 memcpy/memset 활동으로 채워진 구간이다.

따라서 여기서의 문제 지점은 단순 idle gap이 아니다. 더 정확히는 **L1 pipeline의 convert/copy/memset boundary가 partition size와 co-tenant에 따라 길어지는 현상**이다. 이게 중요한 이유는 AI-RAN L1이 sustained 평균 throughput만 필요한 workload가 아니라 frame cadence를 맞춰야 하는 workload이기 때문이다. 평균 GPU busy가 낮아도, 특정 boundary가 수백 us~ms 단위로 흔들리면 p99/p999 deadline이 바로 깨진다.

그림에서 빨간 막대는 kernel만 보고 잰 p99 gap이고, 파란 막대는 kernel/memcpy/memset을 모두 activity로 merge한 뒤 잰 p99 gap이다. 두 값의 차이가 클수록 kernel 사이가 "빈 시간"이 아니라 memory op로 채워졌다는 뜻이다. 노란 선은 1ms 이상 kernel gap 중 memcpy/memset이 포함된 비율이다.

가장 선명한 조건은 `L1 2g alone`과 `L1 2g + ChanPred`다. kernel-only p99 gap은 각각 1444us, 1404us까지 커지지만 all-activity p99 gap은 243us, 175us로 훨씬 작다. 동시에 1ms 이상 kernel gap의 96.7%, 99.1%가 memcpy/memset을 포함한다. 즉 2g에서는 L1이 alone이어도 memory/setup boundary가 매우 조밀하게 끼어들고, ChanPred를 같이 두면 그 구조가 더 강해진다.

`L1 3g + NeuralRx`도 같은 방향의 증거다. 3g baseline은 1ms 이상 kernel gap이 run당 19.3개이고 그중 memory 포함 비율이 45.3%인데, NeuralRx가 붙으면 run당 83.0개, memory 포함 비율 72.4%로 늘어난다. 이 결과는 NeuralRx가 단순히 "외부 AI 하나"가 아니라 L1의 copy/convert boundary와 시간적으로 충돌하는 PHY-AI workload라는 해석을 뒷받침한다.

| Condition | kernel-only p99 gap us | all-activity p99 gap us | >=1ms kernel gaps / run | with memcpy/memset | memory fraction inside big gaps |
| --- | --- | --- | --- | --- | --- |
| L1 7g MIG alone | 567 | 242 | 43.0 | 45.6% | 2.2% |
| L1 3g alone | 808 | 189 | 19.3 | 45.3% | 3.6% |
| L1 3g + NeuralRx | 986 | 229 | 83.0 | 72.4% | 37.8% |
| L1 2g alone | 1444 | 243 | 682.0 | 96.7% | 87.5% |
| L1 2g + ChanPred | 1404 | 175 | 650.0 | 99.1% | 92.2% |
| L1 3g + ResNet | 925 | 226 | 69.3 | 77.8% | 43.6% |
| L1 3g + ResNet+ChanPred | 874 | 175 | 37.0 | 62.9% | 24.3% |
| L1 3g + ResNet+Forecaster | 919 | 172 | 10.3 | 59.7% | 16.5% |
| L1 4g + ResNet | 819 | 171 | 10.7 | 48.3% | 11.4% |

## 13. NSYS가 가리키는 실제 문제 지점: convert/copy/memset boundary

![Memory ops](figures/fig13_nsys_memory_activity_breakdown.png)

memory activity를 보면 원인이 더 구체화된다. 같은 L1 pipeline에서 memcpy call 수와 memset call 수는 거의 같지만, 총 duration은 partition/workload에 따라 크게 바뀐다. `L1 3g alone`의 memcpy 총 시간은 46.8ms인데 `L1 3g + NeuralRx`에서는 195.1ms로 약 4.2배 증가한다. `L1 3g + ResNet`도 155.2ms, `L1 3g + ResNet+Forecaster`도 140.4ms까지 커진다. 반대로 2g 조건은 memcpy보다 memset 쪽이 더 직접적이다. `L1 3g alone`의 memset은 408.2ms인데 `L1 2g alone`과 `L1 2g + ChanPred`는 각각 814.5ms, 813.9ms로 거의 2배다.

이 차이가 중요하다. 만약 문제가 단순 sustained HBM bandwidth 포화라면 H2D/D2D/GEMM synthetic stress가 일관되게 L1 p99를 망가뜨려야 한다. 하지만 6/1 F와 Dv2에서는 generic cross-partition stress가 대부분 baseline 근처에 머물렀다. 반면 NeuralRx, ResNet 계열, 작은 2g L1에서는 copy/memset boundary가 길어진다. 즉 우리가 주장해야 할 bandwidth 문제는 "평균 GB/s를 많이 썼다"가 아니라, **고정된 device bandwidth와 memory path를 시간적으로 나눠 쓰는 상황에서 L1 boundary가 deterministic하게 보호되지 않는다**는 것이다.

| Condition | memcpy total ms | memcpy vs 3g alone | memset total ms | memset vs 3g alone |
| --- | --- | --- | --- | --- |
| L1 7g MIG alone | 82.2 | 1.8x | 207.4 | 0.5x |
| L1 3g alone | 46.8 | 1.0x | 408.2 | 1.0x |
| L1 3g + NeuralRx | 195.1 | 4.2x | 407.8 | 1.0x |
| L1 2g alone | 81.6 | 1.7x | 814.5 | 2.0x |
| L1 2g + ChanPred | 48.9 | 1.0x | 813.9 | 2.0x |
| L1 3g + ResNet | 155.2 | 3.3x | 407.6 | 1.0x |
| L1 3g + ResNet+ChanPred | 103.5 | 2.2x | 407.9 | 1.0x |
| L1 3g + ResNet+Forecaster | 140.4 | 3.0x | 407.8 | 1.0x |
| L1 4g + ResNet | 79.5 | 1.7x | 407.7 | 1.0x |

transition level에서도 같은 구조가 보인다. NeuralRx 조건에서는 `copy_complex64_kernel -> convert_kernel`, `convert_kernel -> noise_intf_est`, `convert_kernel -> eq_coef` 같은 L1 stage boundary의 p99 gap이 수십 ms까지 벌어진다. ResNet+ChanPred/Forecaster 조건에서는 `convert_kernel -> ch_est_pre`가 1920번 반복되고, 그 boundary의 평균 memory fraction이 88~90% 수준이다. 즉 tail은 하나의 거대한 kernel이 느려져서 생기는 것이 아니라, 반복적인 L1 stage 사이에서 copy/memset이 끼어드는 방식으로 만들어진다.

| Condition | Boundary transition | count | p50 gap us | p99 gap us | max gap ms | mean mem fraction |
| --- | --- | --- | --- | --- | --- | --- |
| L1 2g alone | convert_kernel -> noise_intf_est | 32 | 1522 | 91139 | 91.1 | 7.9% |
| L1 2g alone | copy_float32_kernel -> convert_kernel | 26 | 2005 | 39580 | 39.6 | 0.0% |
| L1 2g alone | convert_kernel -> eq_coef | 26 | 3426 | 37847 | 37.8 | 1.4% |
| L1 3g + NeuralRx | copy_complex64_kernel -> convert_kernel | 58 | 1377 | 76267 | 76.3 | 0.0% |
| L1 3g + NeuralRx | convert_kernel -> noise_intf_est | 36 | 1388 | 69277 | 69.3 | 13.2% |
| L1 3g + NeuralRx | convert_kernel -> eq_coef | 36 | 1380 | 62632 | 62.6 | 10.0% |
| L1 3g + ResNet+ChanPred | convert_kernel -> ch_est_pre | 1920 | 746 | 1019 | 1.0 | 88.8% |
| L1 3g + ResNet+Forecaster | convert_kernel -> ch_est_pre | 1920 | 846 | 993 | 2.0 | 89.7% |
| L1 4g + ResNet | convert_kernel -> ch_est_pre | 1920 | 688 | 950 | 1.8 | 89.1% |

그래서 NSYS 근거로 써야 할 문장은 더 좁고 강해야 한다. "MIG가 bandwidth isolation을 전혀 못 한다"가 아니라, **MIG의 정적 capacity isolation은 L1 kernel boundary의 temporal memory activity를 제어하지 못한다**가 맞다. 이것이 AI-RAN에서 치명적인 이유는 L1은 throughput workload가 아니라 deadline workload이고, NeuralRx/ChanPred/ResNet 같은 PHY-AI workload도 평균 처리량뿐 아니라 per-op tail latency와 fit constraint를 동시에 갖기 때문이다.

## 14. Dv2 replication은 negative result를 강화한다

![Dv2 sanity](figures/fig14_dv2_sanity.png)

Dv2 반복 실험은 H2D, D2D, compute, launch, ChanPred stress가 baseline 근처에 머무는 것을 보여준다. 이 그림은 주장을 더 조심스럽고 강하게 만든다. 즉, "MIG cross-partition이 항상 깨진다"가 아니라, **generic cross-partition stress는 대체로 안전하지만 AI-RAN PHY-AI composition에서는 정적 MIG가 충분하지 않다**가 맞다.

아래 표는 Dv2 반복 실험의 숫자다. H2D, D2D, compute, launch, ChanPred 모두 p99 mean이 baseline 주변에 있고 CI도 크게 분리되지 않는다. 이 표는 논문에서 매우 중요하다. 왜냐하면 reviewer가 "그냥 MIG가 isolation을 못 하는 것 아닌가?"라고 물을 때, 우리는 "아니다. generic cross-partition stress는 꽤 잘 막힌다. 문제는 AI-RAN의 L1+PHY-AI composition과 static placement다"라고 답할 수 있기 때문이다.

| Dv2 condition | n | p99 mean us | p99 CI us | p999 mean us | max mean us |
| --- | --- | --- | --- | --- | --- |
| 0 alone | 10 | 950 | 920-980 | 1006 | 2691 |
| 1 H2D 8MB | 10 | 952 | 908-997 | 1131 | 14395 |
| 2 D2D 32MB | 10 | 892 | 853-930 | 942 | 2518 |
| 3 compute | 10 | 918 | 875-961 | 967 | 2399 |
| 4 launch | 10 | 952 | 921-982 | 1043 | 23500 |
| 5 chanpred | 10 | 913 | 871-955 | 985 | 2016 |

## 결론

데이터 기반으로 가장 강한 주장은 다음이다.

> MIG는 generic cross-partition throughput isolation에는 효과적일 수 있다. 그러나 AI-RAN은 L1 tail latency와 AI workload service quality를 동시에 보장해야 한다. MIG는 static capacity slicing만 제공하므로, L1 headroom, AI fit/throughput/p99, PHY-AI co-location tail latency 사이의 tradeoff를 안전하게 제어하지 못한다. 따라서 MIG 단독으로는 real-time L1 + PHY-AI consolidation을 위한 충분한 isolation mechanism이 아니다.

## NSYS까지 포함한 최종 해석

지금까지의 데이터는 다음 순서로 읽는 것이 가장 강하다.

1. **MIG는 capacity isolation에는 의미가 있다.** 6/1 F와 5/31 Dv2에서 generic D2D/H2D/GEMM/launch/ChanPred stress는 baseline 주변에 머물렀다. 따라서 "MIG가 모든 cross-partition isolation에 실패한다"는 주장은 데이터와 맞지 않는다.

2. **하지만 AI-RAN이 원하는 것은 capacity isolation만이 아니다.** L1은 frame deadline을 맞춰야 하고, AI workload도 throughput뿐 아니라 per-op p99와 fit constraint를 가진다. 2g L1은 standalone부터 headroom이 작고, AI workload는 작은 slice에서 fit 실패나 throughput scaling 문제를 보인다.

3. **NSYS는 failure mechanism이 단순 idle gap이 아니라 memory-filled kernel boundary라는 점을 보여준다.** kernel-only gap만 보면 긴 idle처럼 보이지만, kernel/memcpy/memset activity를 merge하면 2g 조건의 1ms 이상 kernel gap 대부분이 memory op로 채워져 있다. 즉 문제는 GPU가 놀아서가 아니라, L1의 convert/copy/memset boundary가 temporal하게 보호되지 않는다는 것이다.

4. **copy/convert/runtime boundary가 workload별로 다르게 흔들린다.** NeuralRx는 3g L1의 memcpy total을 4.2배로 키우고, 2g L1은 memset duration을 3g 대비 거의 2배로 키운다. 이것은 synthetic HBM stress와 PHY-AI workload가 같지 않다는 뜻이다.

5. **가장 치명적인 지점은 same-partition L1+PHY-AI coloc이다.** 6/1 G/H에서 L1과 NeuralRx가 같은 partition에 들어가면 p99가 수백 ms로 폭발한다. 외부 AI 종류를 바꿔도 coloc 이후에는 p99가 이미 높은 영역에 머문다. 이 결과는 MIG가 partition 사이 격리는 줄 수 있어도, 같은 MIG device 내부의 temporal sharing 문제는 해결하지 못한다는 점을 보여준다.

따라서 논문에서 최종 메시지는 이렇게 가져가야 한다.

> MIG는 GPU를 공간적으로 나누는 좋은 capacity isolation 도구지만, AI-RAN의 real-time L1 + PHY-AI consolidation에는 부족하다. 이유는 L1과 AI가 모두 tail-sensitive하고, static partition은 workload phase, copy/memset/runtime boundary, kernel launch gap, PHY-AI coloc behavior를 제어하지 못하기 때문이다. AI-RAN에는 MIG 위에 workload-aware temporal scheduling 또는 admission/control layer가 추가로 필요하다.

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
- `data/static_partition_sweep.csv`
- `data/ai_throughput_v2_parsed.csv`
- `data/ai_throughput_vs_p99.csv`
- `data/g_coloc_external_dominance.csv`
- `data/nsys_gap_summary.csv`
- `data/memory_ops_pressure.csv`
- `data/nsys_kernel_vs_all_activity_summary.csv`
- `data/nsys_memory_activity_breakdown.csv`
- `data/nsys_selected_rootcause_transitions.csv`
- `data/dv2_sanity.csv`
- `data/nsys_profile_matrix.csv`
- `data/nsys_gap_detail_selected.csv`
- `data/nsys_kernel_gap_selected.csv`
- `data/nsys_runtime_selected.csv`
- `data/dv2_sanity_table.csv`
