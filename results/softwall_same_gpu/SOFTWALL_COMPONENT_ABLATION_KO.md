# SoftWall 구성 요소별 근거와 남은 ablation

**기준일:** 2026-09-20 UTC  
**범위:** 한 A100, MIG OFF, NVIDIA MPS, valid 273-PRB PUSCH full receiver

## 핵심 판정

SoftWall의 주장 단위는 MPS 격리가 아니다. MPS는 같은 GPU에서 별도 process와 cap을 제공하는 실행 기반이고, 보호 계약은 다음 결합에서 나온다.

```text
MPS process/cap
  + bounded work-unit qualification
  + 실제 GPU 완료 기반 admission/credit
  + conventional recovery calendar
  + single commit / epoch fence
  + logical quarantine와 maintenance teardown을 분리한 lifecycle
```

임의로 contract를 넘긴 AI kernel까지 현재 RAN deadline을 살리는 것은 보장하지 않는다. 보장 대상은 사전에 허용한 bounded unit과 명시한 workload state다.

## 현재 구성 요소별 실험 근거

| 단계 | 추가한 요소 | 직접 확인한 결과 | 판정 |
|---|---|---|---|
| RAN baseline | AI 없음 | Confirm7 paired 2-cell P25/D21에서 0/5,000 miss | contract가 RAN-alone에서 실행 가능 |
| MPS continuous | cap100 NeuralRx | 같은 Confirm7 protocol의 독립 run에서 2/5,000 miss, 931.1 unit/s | 처리량은 높지만 tail 보호 실패 |
| MPS cap | cap80 NeuralRx | 0/5,000 miss, 860.3 unit/s | workload별 통계 qualification; hard isolation 아님 |
| Bounded admission | S0 quiet-window, 완료 ACK, 6 ms budget | 0/5,000 miss, budget/crossing 0, 380.9 unit/s | compute/HBM 간섭을 포괄하는 안전 baseline |
| Recovery transaction | optional NRx + conventional calendar + single commit | Confirm24 two-cell 3,000 paired releases에서 모든 gate 통과, eager 대비 background +5.53%, utility 차이 −0.0333 pp | 단순 eager 중복 실행보다 recovery-aware slack 회수 이득 |
| Fault semantics | epoch/ID/credit/fallback invariant | Confirm25 5,000 faults에서 wrong/stale/duplicate accept, over-admission, leak 모두 0 | 논리적 transaction 정확성 확인 |
| Physical overrun | 별도 MPS client가 timeout 뒤 계속 실행 | Confirm26 P60/D25에서 cap100 2/1,500, cap20 1/1,500 miss | cap만으로 완전 격리 불가 |
| Lifecycle | quarantine, live teardown, maintenance teardown | Confirm30 live teardown은 17/10,000 miss; Confirm31 maintenance 자료는 다른 campaign과 겹쳐 무효 | logical quarantine 필요성은 확인; maintenance 복귀 gate는 단독 재실행 필요 |
| Physical same-request endpoint | cap80 process + CUDA IPC request/LLR | Confirm32 clean 10,000/10,000 correct, miss/timeout 0, p99 4.125 ms | 별도 process vertical slice 성립 |
| External natural recovery | 위 endpoint + noisy request + conventional fallback | Confirm33 10,000 releases, fallback 221 중 47 recovery, miss/timeout 0, p99 6.004 ms | 별도 process에서 자연 failure/recovery vertical slice 성립 |
| External paired strong baseline | 같은 node/job/MPS epoch/10,000 trace에서 conventional-only/eager-dual과 비교 | Confirm36 external 9,840, eager 9,835, conventional 3,688 correct; 모두 miss 0. External−eager +0.050 pp, 95% CI [−0.0088,+0.1088], 마진 −0.25 pp. Conventional 203회 vs eager 10,000회 | external 경로의 조건부 fallback·무선 utility 비열등성과 9,797회 conventional 실행 제거 확인; 명시적 recovery calendar는 이 경로에 아직 미연결 |
| External concurrent background smoke | 세 번째 cap20 MPS NeuralRx worker의 slack lease | Confirm37의 6 ms RPC budget은 76 units 중 1건 6.208 ms로 실패·admission 중단; 새 seed Confirm37b의 40 ms budget은 500 RAN releases miss/timeout 0, 6,519 AI units, AI violation/crossing 0 | 허용한 workload의 양의 유효 처리량과 fail-closed 동작 확인; 강한 paired throughput은 다음 gate |
| External background strong baseline | 동일 GPU/node/job/channel trace, cap20 worker와 40 ms RPC budget, external conditional-fallback vs local S2 vs eager | Confirm38 각 10,000 releases correct 9,845/9,841/9,843, RAN miss·AI budget/crossing/fault 0; AI units 130,177/127,180/126,477. External−eager utility +0.020 pp, CI [−0.0354,+0.0754] pp; AI +2.93% | 물리적으로 분리된 MPS endpoint에서도 conditional radio/AI 목표가 동시 성립; 외부 경로의 원자 예약·commit 상태기계는 별도 구현·검증 필요 |

Confirm7의 S0 campaign과 cap80 sweep은 동일한 workload·P25/D21·5×1,000 규모지만 독립 실행이다. 따라서 위 표는 mechanism-level evidence이며 모든 단계의 완전한 paired ablation으로 과장하지 않는다. Confirm24는 동일 trace paired comparison으로 S2와 eager의 차이를 직접 검정했다.

**구현 경계:** Confirm32–38의 외부 CUDA-IPC controller는 자연 실패를 검출해 같은 요청을 conventional로 재실행하고, RAN 뒤 여유에 bounded AI를 배치하지만 로컬 S2의 `DartRuntime.submit()` 원자 예약·fallback calendar·single-commit state를 호출하지 않는다. 따라서 외부 경로의 수치로 이 세 메커니즘의 결합 효과를 입증했다고 주장하지 않는다. 이를 실제 외부 경로에 연결한 별도 Confirm42부터 통합 검증한다.

## MPS와 결합해야 하는 최소 요소

1. **Workload certificate:** tensor shape, phase, cap, 미완료 unit 수, copy/HBM 범위를 key로 하여 실제 완료 비용과 RAN 영향 범위를 고정한다.
2. **Admission과 credit:** 다음 보호 구간 전에 완료 가능한 unit만 하나씩 발급하고, CUDA event와 응답을 확인하기 전 credit·buffer를 반환하지 않는다.
3. **Recovery reservation:** optional NRx를 받을 때 같은 요청의 conventional 처리 구간도 원자적으로 확보한다. 여러 셀에 같은 fallback 시간을 중복 판매하지 않는다.
4. **Commit fence:** request ID, endpoint health epoch, buffer generation을 확인한 한 결과만 공개한다. Late/stale completion은 물리적으로 drain하되 commit하지 않는다.
5. **Lifecycle:** bound 위반 시 새 admission을 막고 `QUARANTINED_IDLE`로 둔다. Live RAN 중 process/context를 파괴하지 않고, RAN-free maintenance window에서 teardown한 뒤 새 epoch로 requalification한다.

## 현재 가장 강한 주장과 제한

현재 데이터가 지지하는 문장은 다음과 같다.

> SoftWall은 MPS를 격리 장치로 가정하지 않고, 허용된 bounded AI unit의 admission·physical completion과 같은 요청의 conventional recovery·commit·endpoint lifecycle을 하나의 deadline transaction으로 묶는다. 실제 full-PHY에서 이 결합은 RAN 요구를 만족하는 구성과 양의 AI 처리량을 만들었고, eager dual보다 3.17–5.53% 더 많은 background 작업을 회수했다.

아직 다음 문장은 사용할 수 없다.

- 임의 AI kernel에 대한 완전 격리 또는 강제 선점
- 측정 max를 수학적 WCET로 간주한 hard-real-time 증명
- live timeout 뒤 현재 요청을 항상 conventional로 구조한다는 주장
- 실제 0.5/1 ms NR production slot 달성

## 최종 ablation을 위해 남은 실험

1. Confirm36의 외부 utility 비열등성, Confirm37b의 integration smoke, Confirm38의 동일조건 AI 처리량 비교가 통과했다. 다음에는 역순·독립 seed의 counterbalanced 반복으로 +2.93%가 순서 효과인지 점검한다.
2. 같은 external endpoint에서 `cap only`, `bounded admission`, `recovery reservation`, `lifecycle`을 동일 trace와 seed로 순차 제거한다.
3. 각 조건에서 radio utility, deadline miss, 유효 background throughput, buffer/credit invariant를 동시에 비교한다.
4. 독립 cell/channel trace와 다른 MCS, Qwen decode/training phase로 certificate key의 범위를 넓힌다.
5. Maintenance gate를 전역 experiment lock 아래 단독 재실행하고 quiet-time을 sweep해 cleanup 완료 조건을 정의한다.

Authoritative 원자료는 `results/softwall_same_gpu/raw/`, frozen protocol과 gate 판정은 같은 결과 디렉터리에 보존한다.
