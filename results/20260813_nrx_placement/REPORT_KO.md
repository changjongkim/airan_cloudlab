# AI-RAN NRx placement·transport 실험 최종 보고서

실험일: 2026-08-12~13  
노드: CloudLab Wisconsin d8545, A100-SXM4-40GB ×4, ConnectX-6 Dx 200 Gb/s  
대상: real cuPHY channel estimation/LDPC + NVIDIA pretrained NeuralRx TensorRT + Qwen 7B

## 1. 한 줄 결론

크로스-MIG P2P는 **L1 격리에 실제로 효과가 있고**, NIC GPUDirect RDMA는 zero-CPU-copy dependency 전달이 가능함을 확인했다. 그러나 A100 40GB에서 `3g Qwen + 2g L1 + 2g NRx`로 분할하면 NRx의 서비스 시간이 길어져, 통신 절감분보다 NRx compute-capacity 손실이 더 크다. 따라서 현재 결과의 지배 병목은 transport가 아니라 **NRx에 배정한 GPU slice의 처리 용량과 queue stability**다.

이 결과는 “cross-partition이 무의미하다”는 뜻이 아니다. L1 active time은 같은 4g에 L1+NRx를 배치했을 때 baseline 대비 1.62×로 늘었지만, 2g|2g P2P로 분리하면 1.04×로 거의 복구됐다. 다만 dependency-carrying slot end-to-end는 6.19 ms에서 6.38 ms로 소폭 나빠졌다.

## 2. 기존 105 ms 결과를 다시 해석해야 하는 이유

Nsight Systems와 caller-owned TensorRT binding으로 기존 Aerial wrapper를 분해했다.

| NRx 경로, 4g MIG | mean GPU time | 해석 |
|---|---:|---|
| `pycuphy` raw wrapper | 104.456 ms | generic layout conversion kernel이 지배 |
| direct TensorRT binding | 1.413 ms | caller-owned input/output buffer |
| direct TensorRT + CUDA Graph | 1.340 ms | enqueue 약 2.5 μs |

Nsight에서 `convert_kernel<float,float>`가 wrapper GPU kernel time의 98.4%를 차지했다. TensorRT 계산 자체는 약 1.4 ms였다. Direct TensorRT의 두 output은 wrapper output과 bit-for-bit 일치했고 `max_abs_difference=0`이었다.

또한 MCS 2/QPSK의 실제 LLR output은 `(1, 2, 1, 3276, 12)`이므로 backward payload는 314,496 B다. 기존 코드의 8-bit 가정 1,257,984 B는 이 모델 contract와 맞지 않았다. 따라서 과거 약 106~112 ms 결과는 wrapper 포함 legacy measurement로 보존하되, placement와 transport의 실시간 성능 결론에는 새 direct-TensorRT 결과를 사용해야 한다.

## 3. 공정한 placement 비교

P2P/MIG/MPS 비교는 동일 모델, direct TensorRT, CUDA Graph, synthetic input, warm-up 후 1,000 iterations, 3 trials를 사용했다. NIC GDR은 같은 optimized model contract로 2 repeats를 수행했다. MIG same과 cross P2P 모두 별도 3g MIG에서 Qwen을 실행했다.

| 구성 | 자원 배치 | L1 active mean | L1 slowdown | NRx mean | slot e2e mean | 처리량 | Qwen |
|---|---|---:|---:|---:|---:|---:|---:|
| MIG same | 4g: L1+NRx · 3g: Qwen | 2.975 ms | 1.621× | 1.782 ms | **6.191 ms** | 322.8 slot/s | 10.22 it/s |
| MIG+MPS same | 4g: L1+NRx one MPS client · 3g: Qwen | 3.100 ms | 1.702× | 1.832 ms | 6.383 ms | 313.1 slot/s | 10.22 it/s |
| Cross P2P | 2g: L1 · 2g: NRx · 3g: Qwen | **2.533 ms** | **1.043×** | 3.114 ms | 6.383 ms | 312.9 slot/s | 10.23 it/s |
| Cross NIC GDR | 2g: L1 · 2g: NRx · 3g: Qwen | — | — | — | 6.326 ms | 158.1 slot/s* | 10.24 it/s |

\* 현재 GDR pipeline은 request/response depth 1이고 P2P pipeline은 ring depth 2다. 따라서 GDR의 e2e latency는 유효하지만 이 표의 GDR 처리량을 P2P 처리량과 직접 비교하면 안 된다. 별도 depth-1 P2P 대조 결과를 함께 보존한다.

동일 depth 1 transport 대조는 다음과 같다.

| depth 1 구성 | slot e2e mean | e2e p99 | 처리량 | Qwen |
|---|---:|---:|---:|---:|
| MIG same, 4g | 3.338 ms | 3.501 ms | 300.3 slot/s | 10.22 it/s |
| Cross P2P, 2g\|2g | **5.888 ms** | 6.224 ms | **169.8 slot/s** | 10.22 it/s |
| Cross NIC GDR, 2g\|2g | 6.326 ms | 6.846 ms | 158.1 slot/s | 10.24 it/s |

같은 2g|2g 조건에서 NIC loopback은 native P2P보다 평균 0.438 ms, 약 7.4% 느렸다. 즉 예상했던 5~10 μs NIC 경로는 관측되지 않았다. 다만 그 차이도 전체 6 ms대 pipeline의 일부이므로 compute가 여전히 지배적이다. P2P depth를 1에서 2로 늘리면 처리량은 169.8에서 312.9 slot/s로 거의 두 배가 되지만 e2e는 5.888에서 6.383 ms로 늘어난다.

핵심 관찰:

1. **L1 격리 가설은 맞았다.** Cross P2P에서 L1 slowdown이 62.1%에서 4.3%로 줄었다.
2. **end-to-end 개선 가설은 현재 분할에서는 맞지 않았다.** 2g NRx가 4g NRx보다 약 1.33 ms 느려져 P2P 비용 76.8 μs보다 훨씬 큰 손실을 만들었다.
3. **NIC GDR도 실패한 것은 아니다.** 실제 GPU MR, zero-CPU-copy input/output binding으로 6.33 ms e2e를 달성했다. 하지만 현재 구현은 depth 1이라 concurrent L1-isolation의 증거로 쓰지 않고 transport feasibility와 latency 결과로만 사용한다. NIC를 통과한다고 NRx compute가 빨라지지는 않는다.
4. **MIG+MPS는 자동 추가 격리가 아니다.** L1과 NRx가 하나의 combined MPS client로 같은 4g에서 실행되면 MPS가 두 stream 사이를 격리하지 않으므로 MIG-only와 거의 같은 결과다.
5. 4g+3g와 2g+2g+3g 모두 GPU0의 slice를 전부 사용한다. Cross placement는 새 유휴자원을 만들지 않고, 4g RAN slice를 L1 2g와 NRx 2g로 재분배한다.

## 4. Full-GPU MPS의 RAN–AI utility trade-off

Full A100에서 L1+NRx combined client는 100%로 두고 Qwen client cap만 sweep했다.

| Qwen MPS cap | slot e2e mean | L1 active p99 | 처리량 | Qwen |
|---:|---:|---:|---:|---:|
| 30% | **5.865 ms** | 3.121 ms | 340.9 slot/s | 7.92 it/s |
| 50% | 6.226 ms | 3.425 ms | 321.0 slot/s | 11.14 it/s |
| 70% | 6.656 ms | 4.312 ms | 300.4 slot/s | 17.24 it/s |
| 100% | 8.569 ms | 6.487 ms | 233.3 slot/s | **21.11 it/s** |

따라서 same-partition이 항상 잘못된 배치는 아니다. 충분한 전체 GPU와 admission control이 있으면 MPS가 가장 낮은 e2e를 만들 수 있다. 대신 AI utility를 높일수록 L1 tail과 slot e2e가 악화되는 명확한 Pareto trade-off가 생긴다. Cross P2P의 장점은 최저 평균 latency가 아니라 **예측 가능한 L1 isolation**이다.

## 5. NRx replica와 1 ms arrival queue

동일 MIG slice 안에서 TensorRT execution context 수를 1, 2, 4, 8, 16으로 늘렸다. replica는 동일한 compute resource를 공유하므로 총 처리량을 늘리지 못했고 대체로 감소시켰다.

| NRx 자원 | N=1 service mean | N=1 closed-loop capacity | 확인된 안정 open-loop rate | 1,000 slot/s |
|---|---:|---:|---:|---|
| 2g | 2.984 ms | 334.9 slot/s | 250 slot/s | 불안정 |
| 4g | 1.340 ms | 745.1 slot/s | 700 slot/s | 불안정 |
| full A100 | 0.892 ms | 1,130.5 slot/s | 1,000~1,100 slot/s | 안정 |

2g에서 400 slot/s 이상, 4g에서 750~1,000 slot/s 구간부터 queue가 누적됐다. full A100은 1,000 slot/s에서 p99 약 0.92 ms로 안정적이었다. 즉 1 ms마다 NRx request가 하나 도착하는 시스템이라면 현재 모델/engine은 **한 2g 또는 4g slice로는 부족**하다.

같은 slice에 replica를 더 올리는 것은 해결책이 아니다. 독립 GPU/slice에 sharding하거나, model 축소·precision 변경·batching/temporal pipeline 최적화로 단일-request service demand를 낮춰야 한다.

## 6. 현실적인 mixed-AI 조건

Qwen만 사용한 새 dependency-carrying 비교 외에 기존 Chain 19가 Qwen, Whisper, BERT, NRx, CsiNet, BeamPred를 섞어 다음 범위를 이미 3 trials로 측정했다.

- Full-GPU MPS: N=1, 3, 6, 8, 10, 12
- Cross-partition AI side: N=6, 8, 10, 12, 16
- MPS cap × N sweep과 fault/recovery 실험

이 결과는 cross-MIG가 인접 AI workload 증가와 crash로부터 L1을 격리한다는 사실을 뒷받침한다. 단, 해당 실험은 L1 결과를 실제 NRx가 소비하는 dependency-carrying chain이 아니므로, 새 P2P/GDR 결과와 역할이 다르다. 논문에서는 Chain 19를 **background co-tenant scalability**, 이번 결과를 **RAN dependency placement**로 구분해야 한다.

## 7. 연구 주장으로 가져갈 것

강한 주장은 다음과 같다.

> AI-RAN에서 통신 가능한 accelerator partition을 만드는 것만으로는 안정적인 RAN pipeline이 되지 않는다. Placement는 L1 isolation, dependency transport, NRx service capacity, open-loop queue stability, 그리고 monetizable AI utility를 함께 최적화해야 한다.

시스템 문제는 다음처럼 표현할 수 있다.

```text
maximize   background AI utility
choose     MIG layout, L1/NRx placement, MPS cap, transport, replica placement
subject to L1 p99 <= deadline
           NRx utilization rho = arrival_rate / service_capacity < 1
           dependency e2e <= budget
```

이번 측정은 이 최적화가 단순하지 않다는 반례를 제공한다.

- MIG same: end-to-end는 좋지만 L1 slowdown이 큼.
- Cross P2P: L1은 격리되지만 작은 NRx slice 때문에 e2e 이득이 사라짐.
- Full MPS: 평균은 가장 빠를 수 있지만 AI load에 따라 tail이 변함.
- NIC GDR: CPU staging을 없앨 수 있지만 compute-capacity 부족을 해결하지 않음.

## 8. 한계

- 입력은 synthetic slot이고 RF/BLER 정확도 평가는 이번 범위가 아니다.
- direct TensorRT output은 기존 wrapper와 동일함을 검증했지만, 전체 decoder 결과의 통신 전후 bit-level golden vector 검증은 추가 가능하다.
- P2P는 ring depth 2, 현재 GDR pipeline은 depth 1이다. latency와 throughput 주장을 분리했다.
- 현재 GDR 결과는 한 노드/한 NIC의 physical loopback이며 multi-node fabric 결과가 아니다.
- 3 trials는 기능·시스템 경향 검증에는 충분하지만 논문 최종본에는 더 긴 run과 confidence interval이 바람직하다.
- A100 40GB의 가능한 MIG geometry 때문에 Qwen 3g를 유지한 상태에서 L1과 NRx를 각각 3g로 줄 수 없다. 이 topology constraint 자체가 결과를 좌우한다.
- 공정성을 위해 모든 배치에서 같은 4g-built TensorRT engine을 사용했다. 별도 sensitivity gate에서 2g/full native-build engine은 각각 2.961/0.857ms였고, shared engine의 2.984/0.882ms보다 0.8%/2.9% 빨랐다. 절대값은 조금 변하지만 2g·4g의 1,000 slot/s 불안정과 placement 결론은 바뀌지 않는다.

## 9. 산출물

- `PLACEMENT_SUMMARY.csv`: 8개 placement/utility 조건
- `DEPTH1_TRANSPORT_COMPARISON.csv`: 동일 in-flight depth의 P2P/GDR 대조
- `NRX_CAPACITY.csv`: 2g/4g/full × replica N=1~16
- `NRX_OPEN_LOOP.csv`: 70개 open-loop queue 조건
- `NRX_TACTIC_SENSITIVITY.csv`: 2g/full native-build engine 민감도
- `figures/nrx_wrapper_decomposition.png`
- `figures/nrx_replica_capacity.png`
- `figures/nrx_open_loop_queue.png`
- `figures/placement_latency_utility.png`
- `analyze_results.py`: raw JSON/log에서 CSV와 figure 재생성 및 correctness assertion
- `raw/`: profile, Nsight, P2P, MPS, MIG+MPS, NIC GDR 원본

재생성:

```bash
python3 analyze_results.py
```
