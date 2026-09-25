> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

**SoftWall 연구 계획 검토 — 2026-09-20**

대상: [RESEARCH_PLAN_SOFTWALL_KO.md](RESEARCH_PLAN_SOFTWALL_KO.md). 계획서, 인용 문서, 관련 CSV/JSON, 정책·측정 코드를 대조한 검토다. GPU 실험을 새로 실행하지 않았다. 파일 접근과 결과 작성은 `/pscratch/sd/s/sgkim/kcj/airan_cloudlab` 안에서만 수행했다.

**판정: 연구 문제는 타당하지만, 현재 주장을 그대로 두고 6개월 구현에 들어가기에는 핵심 전제가 덜 검증되어 있다.**

유지할 강점은 optional NRx의 radio utility, conventional fallback의 예약 비용, 늦은 결과의 처리, 간섭에 따른 background 제어를 하나의 문제로 묶었다는 점이다. P0와 실패 분기를 명시한 것도 적절하다. 다만 가장 중요한 질문은 “MPS 실패가 최적화 후에도 남는가”에 앞서 “한 요청의 필수 경로가 실제 deadline 안에 들어오는가”다. 또한 L1 전용 GPU를 쓰는 기본 구성에서는 보호의 상당 부분이 물리 GPU 분리에서 온다. 이때 새 기여는 같은 자원 예산에서 NRx의 유효 결과와 background 처리량을 얼마나 더 확보하는지로 입증해야 한다.

**1. 먼저 고쳐야 할 다섯 가지**

| 우선순위 | 계획서 위치 | 문제 | 필요한 수정 |
|---|---|---|---|
| 최우선 | §0, §3.2, P0, P3 | slot 주기·완료 deadline·20셀 직렬 frame 측정값이 혼용됨 | release/expiry/측정 단위를 정의하고 L1 및 단일 NRx의 시간 예산부터 검증 |
| 최우선 | §0, §4.1, §4.5, C5 | 소프트웨어 격리라는 주장에 비해 기본 보호 수단은 GPU0 전용화 | 기본 기여를 전용 L1 + 공유 NRx pool 제어로 정리하거나, 동일 GPU 공유 성공을 필수 게이트로 승격 |
| 높음 | §1.1, P3 | capacity 표의 출처와 수치가 일치하지 않음 | full/4g와 서로 다른 실험의 숫자를 분리하고 원본 경로·조건 명시 |
| 높음 | §4.3, P2, P4 | 새 모델과 실제 actuator의 정의가 불충분 | 기존 background 프로파일 대비 무엇을 추가하는지와 실행 중 바꿀 수 있는 동작 명시 |
| 높음 | §8.1, §8.2 | 같은 자원 수의 단순 대안 및 통계 기준 부족 | 같은 4-GPU 예산의 단순 pool baseline, utility 하한, miss 신뢰구간 추가 |

**2. deadline 타당성을 L1과 NRx 양쪽에서 먼저 확인해야 한다**

계획서가 P0에서 현재 harness의 한계를 인식한 것은 맞다. 그러나 `38.5ms / 20 cells = 1.92ms`는 직렬 측정의 평균 비용일 뿐, 실제 여러 셀을 병렬 처리할 때의 slot 완료 지연은 아니다. 반대로 `0.5ms 간격으로 요청을 넣었다`는 사실만으로 실제 PHY deadline을 재현했다고 할 수도 없다.

최소한 다음 네 시각을 먼저 정의해야 한다: IQ/input 준비 시각, NRx 분기 가능 시각, conventional fallback의 latest start, CRC/상위 계층 공개 expiry. slot 도착 주기와 release-to-expiry 예산은 별도 항목으로 둔다. NVIDIA cuPHY 문서도 파이프라인의 생성·설정·실행을 구분하고 여러 셀을 묶어 실행하는 구조를 설명한다. 기존 wrapper의 전체 시간과 이 실행 경로를 대응시켜야 한다. [공식 cuPHY 구성 설명](https://docs.nvidia.com/aerial/cuda-accelerated-ran/24-3/aerial_cubb/cubb_developer_guide/cubb_components.html)

인용된 [NRX_CAPACITY.csv](../../results/20260813_nrx_placement/NRX_CAPACITY.csv)의 단일 replica full GPU는 평균 service **0.882ms**, p99 **1.107ms**다. 따라서 NRx 분기 이후의 예산까지 0.5ms라면 현재 측정은 deadline 충족을 지지하지 않는다. 전송·대기·LDPC를 더하기 전부터 평균 service가 예산보다 길다. 평균만으로 모든 요청의 실패를 단정할 수는 없지만, 운영 가능한 성공률을 입증한 상태는 아니다.

GPU 수 증가는 주로 동시 처리 용량을 늘린다. 같은 모델의 단일 요청 service가 자동으로 짧아지지는 않는다. 배치는 처리율을 높여도 batch 대기와 실행 지연을 늘릴 수 있다. 따라서 P3의 통과 조건은 “batch 처리율 증가”보다 **실제 expiry 안의 유효 완료량 증가**여야 한다.

P0를 다음 순서로 바꾸는 것을 권한다.

1. 실제 release/expiry와 단위 정의: cell, slot, PUSCH/TB, frame을 구분.
2. 상주 버퍼를 쓰는 L1 단독 및 NRx 단독의 경로별 지연 측정.
3. 전송 + NRx + LDPC/CRC + commit까지 포함한 무간섭 E2E feasibility 확인.
4. 원래 wrapper / 상주 버퍼 / CUDA Graph를 단계적으로 비교해 간섭 원인을 재검증.
5. 그 뒤 slot-paced open-loop 부하와 공유 GPU 실험으로 이동.

단독 E2E부터 실패하면 모델·텐서 크기·실행 경로를 개선하거나, 실배포에 근거한 다른 deadline을 명시해야 한다. 임의로 deadline만 늘려 production 주장을 유지해서는 안 된다.

**3. 제목과 기본 설계의 관계를 정리해야 한다**

§4.1의 GPU0 전용화는 합리적인 운영 선택이다. 다만 “MIG 미사용”과 “물리적 격리 없이 소프트웨어로 보호”는 같은 주장이 아니다. GPU 하나를 전용으로 주면 보호는 물리 자원 분리에도 의존한다. §4.5의 대체물은 MIG와 동일한 자원 비용·보장 범위도 아니다.

추천하는 기본 연구 질문은 다음과 같다.

> **MIG를 사용하지 않는 고정 GPU 예산에서, L1의 시간 예산과 conventional recovery 여력을 보존하면서 selective NRx의 유효 결과와 background 처리량을 얼마나 함께 높일 수 있는가?**

이 경우 GPU0 회수와 1-GPU cell-site는 별도 확장 실험으로 둔다. 반대로 “같은 GPU 위 소프트웨어 격리”를 핵심으로 유지하려면 E4-b와 S-C를 선택적 확장으로 둘 수 없다. 이 실험의 실패는 제목과 주장의 변경으로 연결되어야 한다. 현재 계획은 이 두 방향을 동시에 주장하는 부분이 있다.

평가에서는 “MIG CP와 p99가 비슷하다”보다 **동일 GPU 수·CPU 예산·무선 부하에서의 L1 miss, delivered radio utility, background goodput**를 함께 비교하는 것이 중요하다. 4-GPU 제안을 1-GPU MIG 구성과 latency만으로 비교하면 추가 자원 효과를 분리할 수 없다.

**4. 수치와 증거 수준을 정정해야 한다**

실제 파일 대조 결과는 다음과 같다.

| 항목 | 확인한 내용 | 해석 |
|---|---|---|
| Exp11, pct=30, N=4/6/8 | 원본 JSON 각 3개 run의 p99 평균: **72.895 / 145.897 / 287.340ms** | 계획서의 해당 수치는 재현됨 |
| Exp11, pct=100, N=6 | 같은 방식으로 **411.320ms** | 해당 수치는 재현됨 |
| 45ms 그림 상수 | `analyze_mig_mps_combined.py`의 Pareto 데이터와 여러 캡션·표에 존재 | §1.3의 정정 필요성은 맞음. 본문만이 아니라 파생 그림·표도 정정 대상 |
| full A100 capacity | 인용 CSV는 **1130.516 req/s, 평균 0.882ms** | 계획서의 `1164.1 req/s, 1.34ms` 묶음과 불일치 |
| 1.34ms service | 같은 CSV의 **4g MIG, 745.103 req/s** 행에 해당 | full GPU service로 인용하면 안 됨 |
| 1164.1 req/s | 선행 synthesis 문서에 있음 | 다른 campaign 결과일 수 있음. 값 자체가 틀렸다고 단정하지 말고 대응 원본·설정을 연결 |
| P2P 76.84μs | `PLACEMENT_SUMMARY.csv`의 **same-GPU MIG P2P** 구성 | 새 설계의 물리 GPU 간 NVLink 측정값으로 사용할 수 없음 |
| 3 replica 97.2% timely | 선행 synthesis는 1 request/ms, 실험용 5ms expiry, single trial 중간 결과로 명시 | 0.5ms production timing의 증거가 아님 |
| utility 75/100, correct-TB 0.80 | 선행 synthesis는 실험용 **12ms expiry**로 명시 | selective admission 가능성의 예시이며 production SLA 검증은 아님 |

관련 파일: [분석 코드](../../results/20260803/analysis_chain19/analyze_mig_mps_combined.py), [capacity CSV](../../results/20260813_nrx_placement/NRX_CAPACITY.csv), [placement CSV](../../results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv), [선행 synthesis](MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md).

확인한 Exp11 JSON은 run당 `iterations=100`, `num_cells=20`, `miss_1ms=100`이다. 기존 p99는 각 run p99의 평균이며 전체 표본을 합친 p99와 다르다. p99.9와 낮은 miss 확률의 근거로 재사용하기에는 표본이 부족하다.

계획서에 명시된 `task1_final/gdr_pool_20260814T014651Z/`와 `results/isca_v2/mig_causal_20260813T1138Z/`는 허용된 프로젝트 루트에서 존재하지 않았다. 외부 경로는 탐색하지 않았다.

**5. 가장 유망한 기여는 admission·recovery·lease의 결합이다**

“co-tenant를 넣은 예측 모델”만으로 새로움을 설명하기에는 기존 코드와 선행 연구가 가깝다. [dart_runtime.py](../../scripts_for_node/task1/isca_v2/dart_runtime.py)의 현재 구현에는 이미 다음이 있다.

- `ProfileTable`: endpoint/tensor/graph/**background_mode**별 프로파일.
- `ServiceProfile`: forward/service/backward/control 및 positive error 비용.
- `predict_finish`: `max(now, tail_ns, blocking_until_ns) + profile.bound_ns`.
- `FallbackCalendar`: conventional recovery 예약.
- lease 예약과 endpoint health epoch.

따라서 §4.3의 “현재는 큐 길이 기반”이라는 설명은 불완전하다. 정확한 차이는 **고정 background 프로파일**에서 **endpoint별 동적 상태·현재 남은 lease·상태 전환·예측 불확실성**을 반영하는 모델로 발전하는 것이다. 기존 엔진을 강한 baseline으로 삼아야 한다.

구체적인 C2/C3 제안은 다음과 같다.

> 요청을 보낼 때 NRx의 deadline 내 성공 가능성과 conventional recovery 자원을 함께 예약하고, 그 두 예약 이후 남는 시간 안에서만 background work를 발급한다. 상태 변화로 예측 오차가 커지면 admission과 lease를 보수적으로 줄인다.

이 구조에서는 predictor가 스케줄러의 제어 결과에도 영향을 받는다. 기존 정책의 고정 trace에 대한 오차만으로 평가를 끝내지 말고, 실제 제어 루프를 닫았을 때 큐와 간섭 분포가 바뀌는 효과를 측정해야 한다. 모델은 all-reject 정책보다 의미 있는 utility를 제공하면서 miss를 낮춰야 한다.

추가할 비교는 queue-only, 기존 background profile, profile + lease residual, online adaptation의 순서가 적절하다. 평가 데이터는 같은 run의 인접 샘플을 무작위로 나누는 방식보다 workload/seed/부하 단계/실행 날짜를 분리한다. 평균 예측 오차보다 상한 초과율, 수락률, radio utility, 전환 직후의 실패율을 본다.

현재 `ProfileTable.get()`은 해당 background 프로파일이 없으면 `isolated`로 돌아간다. 공유 상태에서 isolated 지연을 보수적 상한으로 간주할 수는 없다. production 제어에서는 미지 상태의 동작을 별도로 정해야 한다: NRx 거부·fallback, 검증된 보수 프로파일, background drain 중 하나를 명시한다.

**6. MPS actuator와 bounded lease를 더 구체화해야 한다**

§2.2의 “유일한 knob은 active thread %”는 수정해야 한다. 우선순위와 다른 제어도 존재하며, 계획서 §6.2도 이를 열거하고 있다. 주장 가능한 범위는 **테스트한 정적 active-thread cap만으로는 테스트한 부하의 목표를 충족하지 못했다**이다. cap이 launch/HBM 간섭을 직접 예약·격리하지 않는다는 것과 간섭을 전혀 줄일 수 없다는 것도 구분해야 한다.

Legacy MPS v2에서 `set_active_thread_percentage`와 기본 client priority 변경은 기존 client에 즉시 적용되지 않는다. 따라서 이를 기존 상주 client를 μs/ms 단위로 조절하는 actuator로 가정하면 안 된다. 실제 driver/CUDA/MPS 버전을 고정하고 지원 범위를 확인해야 한다. [NVIDIA Legacy MPS v2 interface](https://docs.nvidia.com/deploy/mps/mpsv2-interface.html)

fast path는 work 발급 중단, outstanding graph 제한, batch 크기·발급 시점 조절 등 실제 가능한 동작으로 정의하는 편이 타당하다. stream/client priority는 보조 최적화로 평가하되 deadline 보장의 근거로 쓰지 않는다. NVIDIA는 priority를 실행 순서 보장이 아닌 hint로 명시한다. [NVIDIA MPS priority 설명](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html#client-priority-level-control)

[background_gated.py](../../scripts_for_node/task1/isca_v2/background_gated.py)는 한 모델 호출 뒤 `torch.cuda.synchronize()`하고 다음 unit 전에 파일 gate를 확인한다. 중간에 이미 발급된 GPU 작업을 즉시 회수하는 구조가 아니다. 따라서 “bounded”의 근거가 되려면 다음을 측정해야 한다.

- 최악 잔여 work-unit 실행 시간, 이미 쌓인 GPU queue, gate 전달·확인 지연.
- 다른 NRx/AI와 경합할 때 늘어난 drain 시간.
- 제어 지연과 mandatory/recovery guard를 뺀 실제 가용 slack.
- LLM prefill/decode, 길이·batch 변화에서의 quantum과 background 자체 latency.

lease 종료까지의 보수적 drain 시간은 실제 사용 가능한 slack 이하여야 한다. 관측된 평균 unit 시간만으로 hard upper bound를 선언할 수는 없다. 측정으로 얻은 통계적 보호와 가정 아래의 결정적 보장을 분리한다.

또한 duty 31.6%의 여집합인 68.4%는 곧바로 회수 가능한 용량이 아니다. allocator/동기화 대기나 흩어진 짧은 gap일 수 있으며 P0 최적화 후 사라질 수도 있다. 회수할 수 있는 연속 시간 구간과 거기 들어가는 work-unit 분포를 다시 측정해야 한다.

**7. correctness 계약에는 물리 buffer 수명과 radio 실패의 의미가 필요하다**

expiry·epoch·single commit은 적절한 기본 장치다. 그러나 epoch 검사는 오래된 결과의 논리적 채택을 막을 뿐, 이미 발급된 DMA가 재사용된 메모리에 쓰는 것을 자동으로 막지는 않는다.

추가로 문서화할 불변식은 다음과 같다.

| 항목 | 요구되는 계약 |
|---|---|
| 요청 identity | cell/slot/PUSCH 또는 TB를 유일하게 구분. `slot_id`에 전역 identity를 인코딩한다면 명시 |
| payload 공개 | 실제 copy/연산 완료 및 visibility 확인 후 completion 공개 |
| buffer 재사용 | 이전 writer/reader의 종료 또는 안전한 격리 확인 전 재할당 금지 |
| restart | 새 epoch 발급뿐 아니라 이전 IPC/DMA의 종료·buffer 교체 절차 |
| downstream 계산 | NRx 반환 LLR 이후 LDPC/CRC와 commit 비용도 deadline 예산에 포함 |
| fallback credit | 논리적 expiry와 실제 실행 완료를 구분해 자원 반환 |
| radio 실패 | deadline 내 CRC-fail/NACK와 계산이 늦은 deadline miss를 구분 |

특히 “CRC pass한 결과만 commit”으로 정의하면 채널이 나빠 두 receiver 모두 실패한 요청을 어떻게 정상 종료할지 빠져 있다. 무선 복호 실패를 시간 예산 위반과 합치면 L1 SLA와 BLER가 혼동된다. 유효 TB의 승자 결정과 deadline 내 PHY 응답/NACK의 완료를 별도로 정의해야 한다.

현재 순수 정책 엔진의 `try_commit()`에는 CRC 판정 인자가 없고, `restart()`는 ring slot을 다시 사용 가능하게 만든다. 이는 상위 integration이 맡을 수 있는 영역이므로 그 자체로 전체 구현 버그라고 단정하지 않는다. 다만 이 엔진의 unit test만으로 GPU payload 안전성과 radio 계약까지 검증됐다고 볼 수 없다.

fault injection은 예정된 graceful restart와 비정상 종료를 구분한다. NVIDIA는 outstanding GPU work가 남은 MPS client의 단순 signal 종료가 다른 client/server에 영향을 줄 수 있다고 설명한다. epoch만으로 이 공유 fault domain을 없앨 수 없다. [NVIDIA MPS 종료·fault 설명](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html#client-early-termination)

**8. 근거보다 강한 표현을 줄이고 반증 실험을 추가해야 한다**

| 현재 표현 | 더 정확한 표현 또는 필요한 검증 |
|---|---|
| “간섭은 두 개다”, “메커니즘은 닫혔다” | 관측한 주요 실패 패턴 두 가지. cache/CPU/allocator/driver/전력 등의 대안 설명을 통제하기 전 완전한 분류로 선언하지 않음 |
| cudaFree와 gap 85% 겹침 → 원인 확정 | 시간적 상관. free 제거, submit rate 고정, host affinity 통제 등 개입으로 인과 확인 |
| “정적 knob 공간 탐색 종료” | 기존 harness와 workload 범위의 cap sweep 종료. P0에서 경로를 바꾸면 대표 조건은 재측정 |
| “독립 client 증가가 launch queue를 포화시킨다” | client 수와 제출률·총 GPU 부하를 분리한 실험 필요 |
| “NRx는 GPU 단위로만 확장된다” | 측정한 동일-device replica 방식의 scaling 실패. stream/batch/graph까지 보편적으로 배제하지 않음 |
| “pull이면 GPU0 copy engine/queue를 비운다” | 검증할 가설. 요청을 받는 쪽에서 copy를 발급해도 GPU0 메모리/NVLink 접근은 남음 |
| “MIG가 P2P를 금지한다” | driver와 동일/다른 physical GPU, GI/CI 조합을 구분 |
| “C1만으로도 논문이 된다” | 가능한 별도 방향. P0 후 효과 크기, 새 원인, 재현성과 선행 연구 차이를 확인해야 판단 가능 |

MIG 공식 문서는 R570에서 같은 physical GPU의 MIG instance 간 P2P를 지원하되 다른 physical GPU의 MIG instance 간 P2P와 GI 간 CUDA IPC에는 제약이 있다고 명시한다. 계획서 §1.2의 구체적 구분은 유지하되 §2.3처럼 포괄적으로 쓰는 부분을 좁혀야 한다. MIG instance 관리 권한도 항상 관리자만 가능한 것은 아니므로 실제 배포 정책을 명시한다. [NVIDIA MIG deployment considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)

§2.2의 “고전 실시간 스케줄링은 preemption을 가정”, “단일 utility로 환원되지 않음”도 과도하다. 비선점 스케줄링과 제약을 둔 최적화로 문제를 표현할 수 있다. 여기서 어려운 점은 숨은 간섭 때문에 실행·blocking 비용의 유효 상한을 만들기 어렵고, optional 실행에 recovery 예약까지 결합된다는 점이다.

**9. 평가에서 추가해야 할 baseline과 통계**

가장 필요한 baseline은 **같은 4-GPU 구성에서 GPU0는 L1 전용, GPU1–3은 단순 NRx pool, static background 배치**다. 여기에 기존 DART의 background profile/admission/lease를 적용한 버전도 포함한다. 이 비교 없이 MIG 대비 latency만 보여주면 전용 배치와 기존 정책의 효과를 새 시스템 효과와 구별하기 어렵다.

| 비교 축 | 최소 비교 |
|---|---|
| 배치의 효과 | 같은 GPU 예산의 단순 전용 배치 + pool vs 제안 |
| admission의 효과 | RR/queue-only/기존 DART/새 정책, 동일 radio utility 목표 |
| lease의 효과 | static cap/priority/cooperative gate/제안된 bound 기반 제어 |
| prediction의 효과 | 고정 background profile vs online state, 동등한 수락률 또는 utility에서 비교 |
| recovery의 효과 | recovery 예약 유무, 여러 endpoint가 동시에 늦는 fallback burst |
| 구현 비용 | 같은 CPU·메모리·모델 상주량, instrumentation 포함/제외 |

선행 시스템 비교는 “한 종 재구현”이라고만 쓰지 말고 대상과 지원 조건을 초기에 정해야 한다. REEF는 inference preemption/concurrency, XSched는 XPU preemptive scheduling을 다루므로 기여 경계를 점검할 직접적인 후보다. 실제 적용 가능 API·CUDA Graph·TensorRT 호환성을 먼저 확인하고, 원본을 실행하지 못해 정책만 재현한다면 그 범위를 밝힌다. [REEF, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/han), [XSched, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)

통계와 목표 함수는 다음처럼 보완한다.

- `miss=0` 대신 **관측 miss 수 / 전체 요청 수 / 실행 시간 / 신뢰구간**을 기록한다. 독립 Bernoulli 가정에서 0회 실패의 단측 95% 상한은 약 `3/N`이다. 예를 들어 `10^-6` 수준을 논하려면 약 300만 표본이 필요하고, burst 상관이 있으면 이 단순 계산만으로 충분하지 않다.
- p99.9·max와 함께 반복 run의 분포, 연속 miss 길이, burst 단위 실패율을 보고한다. 5회 중 2회 붕괴는 현상의 단서이며 정밀한 발생 확률 추정은 아니다.
- deadline miss 감소를 admission 수락률·delivered utility와 함께 비교한다. 모두 거부해 miss를 줄이는 정책을 이득으로 세면 안 된다.
- `cells per node`에는 PRB/MCS/안테나/UL 비율/slot당 요청 수/NRx 선택률과 utility 요구를 명시한다.
- request rate는 대략 `cells × UL-slot fraction × requests per UL slot × NRx-selection fraction / slot period`다. 2000 req/s/cell은 매 0.5ms slot당 요청 한 개를 모두 NRx로 보내는 특정 조건이다.
- utility 정책은 이미 알고 있는 정답·CRC 결과를 admission 입력으로 쓰지 않도록 정보 시점을 명시하고, SNR 추정 오차·MCS·채널 모델·이동성 변화에 검증한다.
- 6종 AI 전수 조합을 우선하기보다 compute-heavy, bandwidth-heavy, phase-changing LLM, correlated burst를 핵심 축으로 두고 현실 workload를 대표 사례로 붙인다.

hard deadline을 결정적으로 보장하려면 스케줄러 가정과 실행·blocking 상한이 필요하다. tail 측정과 0 observed miss는 그 증거를 보강할 수 있지만 단독으로 보장을 증명하지 않는다. 실험적 SLO를 목표로 삼는 경우에는 허용 miss 수준과 적용 workload 범위를 먼저 정한다.

**10. 연구 순서와 성공 기준을 좁히는 제안**

| 순서 | 먼저 답할 질문 | 통과 기준/실패 시 조치 |
|---|---|---|
| 첫 1–2주, G0 | 실제 deadline과 측정 단위가 무엇인가? L1/NRx 단독 및 무간섭 E2E가 가능한가? | 재현 가능한 시간 예산. 실패하면 모델·경로·주장 범위를 먼저 수정 |
| 다음, G1 | 상주 버퍼/Graph 적용 후에도 어떤 간섭이 남는가? | 남은 간섭의 크기·조건·원인 개입 결과. 사라지면 측정 주장을 수정 |
| 다음, G2 | 실제 background work를 충분히 빨리 drain할 수 있는가? | 가용 slack 안의 drain, 공유 상태의 검증. 불가하면 GPU0 회수 제거 |
| 다음, G3 | 같은 GPU 예산의 단순 pool/기존 DART보다 이득이 있는가? | 같은 L1 miss/utility 요구에서 background 처리량 증가 또는 같은 background에서 utility 증가 |
| 이후 | online 모델·batch·full fault path가 그 이득을 더하는가? | ablation으로 각 추가 복잡도의 이득 확인 |

P0에 NRx feasibility와 최소 lease 실험을 포함하고, P2를 본격 구현하기 전에 단순 pool과 기존 DART의 경쟁력을 확인하는 것이 좋다. node 간 확장, 단일 GPU, 여러 GPU 세대는 주 기여의 초기 검증이 끝난 뒤 확장한다. 현재 계획의 시간 합산은 약 27–31주이며, 새 harness·transport·batch engine·baseline 이식까지 고려하면 “약 6개월”은 빠듯한 추정이다.

초기 성공 기준은 임의의 개선율을 사후 선택하지 말고 실험 전에 정한다. 최소한 **L1 miss 허용치, radio utility 하한, 측정할 background work, 같은 총 자원 수**를 고정해야 한다. “기존 대비 p99 1.05× 이내”는 보조 기준이고, 실제 expiry 충족 여부를 대체하지 못한다.

현재 자산을 가장 잘 살리는 방향은 **전용 L1을 기반으로 NRx 실행·fallback 예약·background lease를 함께 제어하는 시스템**이다. 먼저 이 구성에서 단순한 대안보다 이득이 있는지 보이고, 같은 GPU의 L1 유휴 시간 회수는 그 결과에 따라 주 기여로 승격하는 편이 계획의 위험과 주장의 강도를 맞춘다.
