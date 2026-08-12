# Fair direct-P2P overlap experiment

## Outcome

현재 CloudLab A100/R580 환경에서는 sibling MIG device 사이 CUDA P2P가 실제로
동작한다. `cudaDeviceCanAccessPeer`가 양방향 1을 반환했고, 실제 forward/backward
크기의 `cudaMemcpyPeerAsync`를 GPU-side first/last/checksum으로 10/10 검증했다.

실제 slot overlap을 만들면 MIG 분리 효과도 명확하다. 같은 4g에서 L1과 NRx를
동시에 실행할 때 L1 CUDA-stream elapsed time은 자체 baseline의 평균 61.12배가
됐지만, 2g L1과 2g NRx를 분리하고 direct P2P로 연결하면 1.49배에 그쳤다.

## Fairness contract

- 동일 총 GPU 자원: `4g shared` 대 `2g L1 + 2g NRx`
- 두 topology 모두 별도 3g에서 Qwen 7B 실행: 10.22–10.23 it/s
- 같은 FP16 NRx engine, tensor shape/dtype, synthetic IQ, warm-up 20, measured 30
- ring depth 2로 slot N의 NRx와 다음/이전 slot의 L1을 실제 동시 실행
- L1 자원이 4g와 2g로 다르므로 각각의 L1-only baseline으로 정규화
- 독립 3회 반복; 매 반복 후 `4g + 3g` Topology A 복구

`l1_active_ms`라는 raw field는 L1 front와 back을 각각 감싼 CUDA event elapsed의
합이다. 같은 MIG에서 NRx가 stream scheduling을 지연시키는 시간도 포함하며,
순수 kernel instruction 실행시간이라는 뜻은 아니다. 이 지연 자체가 여기서
측정하려는 L1 completion interference다.

## Three-trial aggregate

| Condition | Own L1-only baseline | L1 mean | L1 p99 | Slowdown | NRx mean | E2E mean/p99 | Slot/s | Qwen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Same 4g · L1+NRx overlap | 1.586 ms | 96.935 ms | 197.823 ms | **61.12×** | 104.409 ms | 206.850/303.635 ms | 9.508 | 10.22 |
| Cross 2g+2g · direct P2P | 2.234 ms | **3.316 ms** | **3.695 ms** | **1.49×** | 105.916 ms | 208.714/213.893 ms | 9.421 | 10.23 |

Cross-MIG P2P는 same-MIG 대비 L1 mean을 96.58% 줄였고 29.23배 낮았다.
L1 p99는 98.13% 감소해 53.54배 낮았다. 세 trial의 normalized slowdown 범위는
same-MIG 59.88–61.82배, cross-MIG 1.39–1.54배다.

반면 pipeline mean/throughput은 거의 그대로다. NRx가 약 104–106ms로 지배하므로
cross-MIG의 E2E mean은 0.90% 늘고 throughput은 0.92% 낮았다. 이 실험의 승리는
NRx service time 단축이 아니라 **L1 latency isolation과 tail 안정화**다. E2E p99는
303.635ms에서 213.893ms로 29.56% 개선됐다.

## Direct-P2P transport gate

복사와 stream completion만 측정하고, 검증은 측정 후 수행했다.

| Direction | Payload | Mean | p99 | Integrity |
|---|---:|---:|---:|---:|
| 2g L1 → 2g NRx | 1,415,232 B | 64.93 µs | 80.53 µs | 10/10 |
| 2g NRx → 2g L1 | 1,257,984 B | 59.15 µs | 61.22 µs | 10/10 |

Overlap pipeline에서 양방향 복사 합계는 평균 111.23µs였다. NIC GDR staging의
동일 payload별 약 0.7ms보다 이 topology의 direct P2P가 훨씬 짧다.

## Why the earlier result looked wrong

기존 106–112ms 실험은 `CE → NRx → LDPC`의 closed-loop request/response라 같은
slot의 L1과 NRx를 직렬 실행했다. 겹치는 kernel이 없으므로 MIG가 제거할
same-partition contention 자체가 없었고, 전체 수치는 거의 104–107ms NRx 시간이다.
따라서 4g co-location 106ms와 cross-MIG 110ms가 비슷했던 것은 격리 실패가 아니다.

또한 기존 `config5 cuPHY-only ≈39ms`는 `real_l1.py`의 **20-cell loop**인 반면,
`config1/config6` integrated pipeline은 **1 cell + NRx**다. 39ms와 106ms를 동일
workload의 baseline/slowdown으로 직접 비교하면 안 된다. 이번 실험은 동일한
single-cell L1 front/back을 각 topology의 standalone과 overlap 양쪽에서 사용했다.

## Architectural boundary

Direct P2P 결과는 한 process가 두 MIG CUDA context와 양쪽 buffer를 소유하는
구조다. 별도 L1/NRx process가 상대 process의 GPU pointer를 쓰려면 CUDA IPC 같은
주소 공유가 필요한데, cross-MIG CUDA IPC는 이 환경에서 사용할 수 없다. 따라서:

- 단일 orchestrator process/threads로 통합 가능하면 direct P2P가 최단 경로다.
- L1과 NRx의 강한 process/container 분리가 필수면 현재 구현한 NIC GPUDirect
  RDMA가 필요하다.

즉 P2P는 불가능하지 않고 L1 격리 가설도 확인했다. 다만 P2P와 NIC GDR은 같은
문제의 완전한 대체재가 아니라 process-isolation 요구가 다른 두 설계점이다.

Ring depth 2의 E2E 약 208ms는 NRx 두 slot이 queue에 걸린 pipeline latency이므로,
기존 직렬 108ms와 직접 비교하지 않는다. NRx wrapper 자체가 약 105ms인 현재
구현으로는 L1 isolation과 별개로 실시간 slot throughput을 달성할 수 없다.

## Artifacts

- Per-trial raw data: root (`trial1`), [`trial2/`](trial2/), [`trial3/`](trial3/)
- Strict summary: [`P2P_FAIR_TRIALS.csv`](P2P_FAIR_TRIALS.csv), [`P2P_FAIR_AGGREGATE.csv`](P2P_FAIR_AGGREGATE.csv)
- Analysis: [`analysis/summarize_p2p_fair.py`](analysis/summarize_p2p_fair.py)
- Figure: [`figures/p2p_fair_l1_isolation.png`](figures/p2p_fair_l1_isolation.png)
- Topology-B integrity gate: [`p2p_gate_2g2g/`](p2p_gate_2g2g/)
- Supplementary Topology-A gate: [`p2p_gate/`](p2p_gate/)
- Node scripts: [`../scripts_for_node/task1/`](../scripts_for_node/task1/)
