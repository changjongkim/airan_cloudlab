# SoftWall 스케줄링 논문 재구성

**기준 원고:** `paper/softwall_sigmetrics27/main.tex`  
**현재 중심 주장:** certified multi-GPU scheduling for conditional recovery in AI-RAN  
**주장하지 않는 것:** 이미 실행된 dynamic placement optimizer의 처리량 우위, WCET, production HARQ 보장

## 1. 한 문장 프레이밍

> SoftWall은 optional NeuralRx, 그 결과에 조건부인 conventional recovery, deadline을 가진
> external AI를 함께 배치하면서, 모든 미해결 NeuralRx가 실패해도 실행 가능한 radio
> schedule을 물리 완료까지 유지하는 certified multi-GPU scheduler다.

핵심은 `substrate` 자체가 아니라 **기존 스케줄러에 없던 job state와 scheduling contract**다.
SoftWall은 다음 결정을 내린다.

1. optional NeuralRx를 어느 endpoint에 admit할지
2. 아직 발생하지 않았지만 미래 failure branch에서 mandatory인 recovery를 언제 어느 lane에 둘지
3. recovery를 retime한 뒤 어떤 external-AI unit을 언제 시작할지
4. CUDA IPC/P2P ownership, latest-start, fence와 single commit을 만족할 때만 그 schedule을 실행할지

현재 구현은 qualified placement 안에서 위 결정을 수행한다. 1--4 GPU에 대한 flexible recovery
placement의 이득은 모델 상한으로만 확인했으며, arbitrary dynamic placement를 물리 구현했다고
주장하지 않는다.

## 2. Cascade식 서술 규칙

원고의 각 논리 단위는 다음 순서를 따른다.

1. **Claim:** 이 문단에서 답할 문제나 관찰을 첫 문장에 둔다.
2. **Mechanism:** 그 문제가 왜 발생하는지 job, queue, lane 또는 deadline 관계로 설명한다.
3. **Evidence:** 그림, 식, 비교군 또는 측정값을 연결한다.
4. **Implication:** scheduler 설계나 claim boundary에 미치는 의미로 문단을 닫는다.

Background는 시스템과 용어만 설명한다. Motivation은 문제 하나마다 그림과 Takeaway를 둔다.
Design은 문제에서 도출된 scheduler state와 decision transaction을 설명한다. Evaluation은 실험
번호나 구현 순서 대신 scheduler 질문 순서로 구성한다.

## 3. 변경된 논문 구조

### Background

- 같은 TB에서 optional NeuralRx와 conventional recovery가 갖는 조건부 의존성
- 여러 RAN home의 local endpoint와 shared cuPHY/AI lane
- MPS가 제공하는 concurrency와 제공하지 않는 timing authority
- mode provenance와 qualification 범위

새 `softwall_airan_background` 그림은 위 두 dependency를 solution 없이 정의한다.

### Motivation: Missing Scheduling State

새 `softwall_scheduling_problem` 그림은 세 반례를 분리한다.

- **Current-idle false-safe:** 현재 queue가 비어도 unresolved recovery가 미래에 release될 수 있다.
- **Static under-utilization:** 모든 recovery를 고정 예약하면 NeuralRx success가 만든 capacity를 잃는다.
- **Local/global mismatch:** home별로 안전한 schedule도 shared recovery lane에서 충돌할 수 있다.

각 반례 뒤에는 current idleness, static reservation, local admission에 대한 Takeaway를 둔다.

### Section 3: SoftWall Design

Model과 Design을 분리하지 않고, 스케줄러의 입력 상태가 실제 GPU 실행으로 정제되는 순서로
한 section에 배치한다.

| subsection | 독자가 답을 얻어야 하는 질문 | 핵심 시각 자료 |
|---|---|---|
| **3.1 Overall architecture** | 입력 event가 어떻게 recovery debt, candidate schedule, 검증, GPU dispatch, terminal event로 순환하는가 | 세 layer와 1--6번 동작으로 구성한 CASCADE식 전체 구조도 |
| **3.2 Scheduling conditional recovery** | 아직 queue에 없는 future mandatory work를 어떻게 예약하고, NRx 결과 뒤 recovery와 AI를 어떻게 함께 다시 배치하는가 | mandatory-first outcome transition + atomic recovery-retiming/AI-lease 그림 |
| **3.3 Refining schedules to physical completion** | host schedule이 IPC/P2P, CUDA launch, fence, credit retirement까지 어떻게 효력을 유지하는가 | controller→ownership→persistent workers→terminal evidence 정제 그림 |
| **3.4 Certified event-scheduling algorithm** | event마다 어떤 순서로 UQ/MI/QSN/QSU를 판정하고 첫 action을 실행하는가 | `CertifyEvent` 6단계 알고리즘과 decision-window/capacity 식 |

이 구조에서 recovery debt와 all-fail certificate는 독립적인 모델 논문 장이 아니라
**스케줄러가 다루는 job state**다. Atomic transaction과 physical ownership은 별도 substrate
소개가 아니라 **accepted schedule을 실제 실행까지 보존하는 두 번째 스케줄링 단계**다.

새 TikZ 그림은 `paper/softwall_sigmetrics27/figures`의 다음 파일에 있다.

- `softwall_overall_architecture.tex`
- `softwall_atomic_replan.tex`
- `softwall_physical_refinement.tex`
- `softwall_tikz_style.tex`

## 4. 변경된 Evaluation 순서

| RQ | 스케줄링 질문 | 핵심 증거 |
|---|---|---|
| RQ1 | debt-aware scheduling이 어떤 이점을 만드는가 | static보다 safe capacity가 큰 3,600/13,716 point, debt-blind false-safe 4,338 point, physical 40/40 대 0/120 |
| RQ2 | online decision은 oracle에 얼마나 가까운가 | headroom 362 point, greedy 361/362 within 5%, multi-GPU placement upper-bound gain 203/362 |
| RQ3 | 실행 전에 boundary를 예측하고 scale할 수 있는가 | exact 16,023 state 일치, physical 180/180, debt64 p99 1.494 ms |
| RQ4 | accepted schedule이 실제 GPU에서 유지되는가 | NeuralRx 4,800, recovery 1,312, Qwen 1,077, commit 4,800, 관측 위반 0 |
| RQ5 | fault 뒤에도 schedule ownership이 유지되는가 | state audit와 A0--A6 physical campaign |
| RQ6 | MPS/lifecycle이 왜 mode provenance인가 | cap-only miss와 lifecycle matrix |
| RQ7 | deployment claim은 어디서 끝나는가 | testMAC 995/1,000으로 gate 실패, Aerial TDL-A UQ, Sionna CDL-D/E conditional value |

이 순서에서 MPS failure는 논문의 주인공이 아니다. 첫 결과는 **안전한 capacity reclamation과
decision quality**이며, MPS와 lifecycle은 accepted schedule을 적용할 수 있는 mode boundary를
설명하는 뒤쪽 결과다.

## 5. 현재 스케줄링 이득의 정확한 해석

- **입증됨:** static reservation보다 넓은 safe-capacity 영역
- **입증됨:** current-idle admission이 허용 bound 안에서 false-safe가 될 수 있음
- **입증됨:** fixed placement에서는 단순 safe greedy가 exact oracle을 거의 닫음
- **음성 결과:** 원래 `P180/D155`, one-Qwen-per-epoch trace에서는 recovery-first와 SoftWall이
  349,387 timely token으로 동일
- **남은 기회:** flexible recovery placement는 2-GPU 145/175, 4-GPU 58/85 point에서 양의
  upper-bound gain을 보임
- **미입증:** movement cost와 새 service tail을 포함한 physical placement-aware throughput gain

따라서 현재 논문은 “복잡한 AI ordering optimizer” 논문이 아니다. **conditional future work를
처음부터 scheduling state로 만들고, 그것을 멀티 GPU의 admission·placement·ordering·physical
ownership까지 연결한 스케줄러** 논문이다.

## 6. 다음 물리 실험: placement-aware scheduling gate

추가 성능 실험은 C175가 찾은 유일한 구조적 headroom만 겨냥한다.

1. C175의 2-GPU 후보 중 predicted gain이 5% 이상인 한 영역을 사전 고정한다.
2. `fixed recovery placement`와 `certificate-preserving placement-aware SoftWall`에 동일한
   max-radio 선택, BurstGPT window, Qwen class, failure stream과 service profiles를 준다.
3. placement-aware arm은 debt와 AI를 함께 lane에 배치한 뒤 기존 independent verifier를 통과한
   schedule만 실행한다.
4. P2P movement와 destination cuPHY service를 포함한 새 mode vector를 먼저 재자격한다.
5. 두 node에서 ABBA 순서를 사용하고, 모든 safety gate와 timely-token paired CI를 함께 판정한다.
6. 2-GPU gate가 통과할 때만 4-GPU로 확장한다.

성공 기준은 각 node/순서에서 safety 위반 0, fixed placement 대비 timely token +5% 이상,
paired bootstrap 95% CI 하한이 0보다 큰 것이다. 실패하면 “현재 physical movement cost가
model headroom을 상쇄한다”는 음성 결과로 남기고, scheduler의 safety/capacity claim은 유지한다.
