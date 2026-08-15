# DART-Rx novelty/realism pivot

작성일: 2026-08-13  
상태: 실험 진행 중 · claim 미확정

> **Canonical novelty 문서:** `DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`  
> 이 문서는 pivot 당시의 실험 gate 기록이다. 최종 novelty와 architecture claim은
> canonical 문서의 validity-scoped transaction 및 dual-reservation 정의를 따른다.

## 1. 지금 무엇을 연구하는가

`MIG/MPS/MIG+MPS/P2P/GDR 중 무엇이 빠른가`가 논문의 문제는 아니다. 이 비교는
아래 모순이 실제 hardware에서 발생하는지 보이는 characterization이다.

> **고정 격리로 보호된 real-time PHY가, 재구성 없이 격리 밖의 optional NRx
> capacity를 빌리면서도 deadline과 output validity를 어떻게 보존할 것인가?**

여기서 isolation-elasticity는 관찰할 systems 현상이고, completion-oriented command
interface는 안전한 borrowing을 막는 root cause이며, expiry/fallback/commit은 그
원인에서 도출되는 transaction requirement다. 세 개의 독립 problem이 아니다.

제안은 **DART-Rx** 하나다. DART-Rx는 optional neural receiver invocation을
`radio utility + deadline + result validity`를 가진 transaction으로 바꾼다.

```text
                         fixed, protected L1 partition
grant / channel state -> CE -> DART-Rx transaction ----------------------+
                              |                                          |
                              | utility <= 0 or infeasible                |
                              +--> conventional receiver                  |
                              |                                          |
                              | utility > 0 and feasible                  |
                              v                                          |
                    resident isolated NRx pool                           |
                 local / P2P / GPUDirect RDMA                            |
                              | result(slot, epoch)                       |
                              v                                          v
                   deadline-safe single-winner commit -> LDPC/CRC/output
                              ^
                              +--- just-in-time conventional fallback
```

P2P와 GDR은 novelty가 아니라 이 transaction을 isolated endpoint까지 연결하는
transport adapter다. Background AI는 NRx pool의 미예약 slack만 사용하는 secondary
consumer이며, Qwen 하나가 아니라 LLM, video, speech, training으로 검증한다.

## 2. 논문 architecture 구성

### 2.1 Utility-aware admission and placement

같은 transmitted TB/channel realization을 conventional과 NRx 양쪽으로 복호해
`U = P(success_NRx) - P(success_conventional)` utility map을 만든다. 요청의 utility가
양수이고 endpoint의 보수적 predicted finish가 deadline 안일 때만 NRx를 실행한다.
그 외에는 처음부터 conventional path를 선택한다.

단순 shortest queue와 다른 점은 endpoint queue tail, registered tensor credit,
transport, NRx graph service bound, commit guard를 한 번에 reserve한다는 것이다.

### 2.2 Isolated receiver service fabric

NRx engine과 input/output ring은 고정 MIG 또는 full-GPU endpoint에 상주시킨다.
L1 partition은 재구성하거나 재시작하지 않는다. 동일 request/engine/payload에서
local, host copy, P2P, CPU-buffer RDMA, GPUDirect RDMA를 교체해 transport와 placement
효과를 분리한다. 여러 cell의 요청은 이 resident pool로 수평 확장한다.

### 2.3 Deadline-safe fallback and commit

각 request에 `(slot_id, epoch, deadline, latest_fallback_start)`를 붙인다. NRx가 마지막
안전 시점까지 valid result를 만들지 못하면 conventional graph를 시작한다. 두 경로는
private output에 쓰고, `(slot, epoch)`가 맞는 첫 번째 timely result만 atomic commit한다.
늦은 result는 drop하므로 다음 slot buffer를 오염시킬 수 없다.

ISCA용 hardware question은 이 transaction을 host polling으로 구현할 때의 jitter와
상시 polling resource를, GPU command/copy completion 근처의 작은 deadline queue와
commit table이 줄일 수 있는가이다. 이 효과가 측정되지 않으면 DART-Q hardware
contribution은 주장하지 않는다.

## 3. 무엇이 realistic한가

### Traffic

- 0.5/1.0 ms periodic slot arrival
- synchronized/staggered 2/4/8-cell aggregate stream
- 10/25/50/75/100% selective NRx invocation
- IID control과 burst-correlated hard-channel sensitivity
- open-loop arrival, queue drain, generator lateness, per-endpoint idle time 기록

### Radio correctness

- Aerial 25.3 PUSCH chain과 실제 deployed TensorRT NRx engine
- 같은 TB와 같은 Sionna Rayleigh/CDL channel realization을 두 receiver에 입력
- per-slot decoded-TB equality, CRC, paired contingency, BLER/goodput delta
- MCS 2/7, Rayleigh/CDL-A/CDL-C, Es/N0 -8..0 dB, 세 seed,
  condition당 최대 500 paired slots와 error target 50
- actual OTA bundled sample은 두 cell의 correctness smoke로만 사용하며 일반화하지 않음

### Background co-tenants

- real prompt LLM prefill/decode
- decoded video + pretrained detector
- streaming speech chunks
- real-data training microbatches

각 workload는 자체 SLO와 throughput을 같이 측정한다. random tensor/Qwen-only 결과를
일반 AI-RAN claim으로 사용하지 않는다.

### 현재 realism의 한계

- CloudLab에 RU link가 없어 main radio result는 reproducible link simulation이다.
- 현재 pretrained NRx engine은 실험상 MCS 2/7 범위이므로 전체 scheduler MCS를 대표하지 않는다.
- traffic trace와 paired radio outcome을 결합한 단계는 measured-outcome replay이고, 실제
  P2P/GDR payload + concurrent conventional fallback까지 묶은 end-to-end run과 구분한다.
- 5 ms NRx deadline은 아직 sensitivity 값이다. 실제 Aerial slot schedule에서 stage
  budget을 도출하기 전에는 production deadline이라고 부르지 않는다.

## 4. novelty 경계

이미 존재하는 것:

- fixed MIG의 distributed execution/fragmentation 완화: Flex-MIG
- AI-RAN의 deadline-aware placement/resource allocation: HAF
- PHY의 channel-aware AI/conventional expert selection: ARCHES
- generic predictable DNN serving과 GPU sharing/preemption: Clockwork, REEF, Tally

따라서 아래 각각만으로는 novelty가 아니다.

- MIG가 MPS보다 L1을 잘 격리한다.
- GPUDirect가 host copy보다 빠르다.
- idle GPU로 NRx를 route한다.
- deadline-aware shortest queue를 쓴다.
- 채널이 어려울 때 NRx를 선택한다.

검증할 새로운 조합은 다음이다.

> **A slot-validity transaction that jointly admits a radio-useful optional PHY
> expert, executes it over a fixed isolated accelerator fabric without MIG
> reconfiguration, and guarantees bounded conventional fallback and stale-safe
> commit under multi-cell burst demand.**

이 문장은 결과가 아래 kill gates를 통과할 때만 논문 claim이 된다.

## 5. 실험 ladder와 현재 상태

| Gate | 실험 | 판정하려는 것 | 현재 상태 |
|---|---|---|---|
| G0 | MIG/MPS/MIG+MPS/P2P/GDR fair comparison | isolation/transport 원인 분해 | 기존 결과 있음, v2 반복 진행 |
| G1 | raw TRT/wrapper/CUDA Graph/profile | 실제 service bottleneck | 완료 |
| G2 | 1/2/4/8 replica + open-loop rate | service capacity/queue cliff | 장시간 campaign 진행 |
| G3 | realistic background qualification | kernel residual과 co-tenant SLO | 진행 중 |
| G4 | multi-cell periodic/selective/burst hardware replay | idle+miss 동시 발생 | runner 대기 중 |
| G5 | paired conventional/NRx radio sweep | NRx가 유용한 radio bin 존재 | 1-slot/coverage smoke 통과, full sweep 대기 중 |
| G6 | radio-labelled hardware-timed policy replay | utility-aware routing의 system value | G4/G5 뒤 구현·실행 |
| G7 | integrated L1→transport→NRx→fallback→commit | end-to-end mechanism | 미완료 |
| G8 | host vs device transaction control | architecture primitive의 필요성 | 일부 micro-gate, 통합 미완료 |

현재 smoke에서 MCS2/Rayleigh/-4 dB는 두 receiver 모두 성공했지만,
MCS2/CDL-A/-4 dB에서는 conventional만 성공했다. 표본 10개라 결론은 아니지만
NRx가 항상 유용하다는 가정이 틀릴 수 있음을 보였다. 따라서 utility-aware admission은
장식이 아니라 필수 gate다.

## 6. 사전 등록한 kill gates

1. 같은 realistic selective-burst trace에서 static placement의 timely-NRx 부족이
   1% 이상이고 eligible endpoint idle time이 10% 이상이어야 한다.
2. paired radio sweep에서 통계적으로 양의 NRx utility bin이 있어야 한다. 없다면
   neural-receiver-specific claim을 중단한다.
3. DART-Rx는 `shortest queue + conservative admission`보다 deadline miss, delivered
   radio utility, idle GPU-seconds, background SLO의 Pareto front를 개선해야 한다.
4. late completion/failure injection에서 wrong/stale commit은 0이어야 하고 fallback
   completion bound를 만족해야 한다.
5. device-side/DART-Q가 host implementation보다 p99 control delay 또는 reserved-SM
   cost를 유의미하게 줄이지 않으면 hardware contribution을 중단한다.
6. 적어도 세 종류의 실제 background workload에서 protected-L1 결과가 재현되어야 한다.

## 7. 다음 자동 실행 순서

```text
background qualification
 -> capacity/five-way comparison
 -> DART mechanism gates
 -> multi-cell traffic hardware gate
 -> paired radio-utility full sweep
 -> radio-labelled policy replay
 -> integrated P2P/GDR fallback/commit
```

현재 자동 캠페인은 paired radio full sweep까지 연결되어 있다. 그 뒤 결과에 양의 utility
bin이 존재하는지 확인한 다음 G6를 실행한다. 결과가 없는데도 임의 확률로 NRx request를
만들어 성공한 것처럼 진행하지 않는다.

## 8. 가장 중요한 figure

동일 radio-labelled multi-cell trace의 한 시간축에 다음을 겹친다.

1. channel/MCS별 predicted NRx utility와 arrivals
2. 각 endpoint queue/utilization
3. static placement에서의 idle capacity와 deadline loss
4. DART-Rx의 local/remote/conventional 선택
5. NRx/conventional winner와 late-drop
6. protected L1 tail, radio goodput, background SLO

이 그림에서 `한 endpoint는 deadline을 놓치는데 다른 endpoint는 idle`인 problem과,
DART-Rx가 그 idle capacity를 deadline-safe하고 radio-useful한 work에만 쓰는 solution이
한 번에 보여야 한다.

## 9. 관련 자료

- Flex-MIG: <https://arxiv.org/abs/2511.09143>
- HAF: <https://arxiv.org/abs/2605.07547>
- ARCHES: <https://arxiv.org/abs/2604.23397>
- NVIDIA Aerial multi-cell PUSCH example:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/content/notebooks/datalake_pusch_multicell.html>
- NVIDIA Aerial multi-cell capacity:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/release_notes/multicell_capacity.html>
