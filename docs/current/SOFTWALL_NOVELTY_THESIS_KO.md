**SoftWall: 방어할 novelty와 이를 입증하는 기준 — 2026-09-21**

**최신 판정:** [노벨리티 판정과 종료 기준](SOFTWALL_NOVELTY_DECISION_KO.md)이 이 문서의
초기 기여 후보보다 우선한다. Confirm55까지의 결과로는 독립 공동 정책과 탑컨퍼런스급
노벨리티가 입증되지 않았으며, 현재 실험 단계는 종료했다. 아래의 C1–C3는 검증할 가설이다.

관련 문서: [선행 연구 검토](SOFTWALL_RELATED_WORK_AUDIT_KO.md), [구체 스킴](SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md), [측정 타당성 검토](RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md).

**범위 확정:** 현재 필수 목표는 **같은 GPU / MIG OFF / MPS에서 허용된 AI의 실행량을 제어하여 RAN deadline을 지키고 여유 자원을 회수하는 것**이다. [최신 타당성·실험안](SOFTWALL_MPS_ISOLATION_FEASIBILITY_KO.md)의 단일 GPU S0/S1/S2 vertical slice를 실행했다. 단순 MPS 공유·작은 work unit·시간 분리·최신 static SM 기능 자체만으로 novelty를 주장하지 않는다. 아래의 GPU0/pool 표기는 후속 다중 GPU 확장의 표현이며, 전용 GPU 보호를 동일 GPU 격리의 증거로 삼지 않는다.

**2026-09-21 현재 판정:** 조건부 시스템 구성의 효과를 뒷받침하는 핵심 실험은 완료됐다. 별도 MPS endpoint에 실제 원자 fallback 예약·single commit·복구 전 bounded AI lease를 연결한 Confirm45는 같은 채널 입력의 eager-dual 기준을 두 독립 seed·반대 실행 순서에서 비교했고, 완료 AI +2.95%/+2.41%, 무선 효용 비열등성, RAN deadline miss 0 및 모든 사전 gate 통과를 얻었다. 새 50 ms NRx admission 계약과 35 ms AI socket timeout을 사전 고정한 Confirm47도 독립 seed·역순 두 쌍에서 frozen PASS, AI +1.38%/+1.11%, RAN miss 0이었다. 같은 새 계약에서 gap lease만 ON/OFF한 Confirm50은 독립 seed·역순 모두 frozen PASS, AI 순증 +0.468%/+0.535%, 네 조건 40,000건 RAN miss·NRx bound 위반 0이었다. 이전 Confirm43/44의 gap ON/OFF는 +0.419%/+1.304%였으나 NRx 30 ms 상한 위반 각 3건으로 전체 gate 실패다. 이는 MPS 자체의 독립성이 아니라 **요청별 복구 자원 예약과 slack 회수를 결합한 방식의 조건부 성능 근거**다. Confirm47의 외부 방식 RAN p99 55.803/55.828 ms는 eager 16.036/16.332 ms보다 크지만 D80 안이다. 현재 한 셀·한 endpoint 경로는 고정 profile·예약·시간 기반 AI 허가이며 새 공동 최적화 알고리즘이 아니다. 이 자료로 무조건적 hard deadline 보장, 모든 AI workload의 완전 격리, 강한 선행 기법 결합 대비 우월성을 주장하지 않는다. [실험 gate ledger](../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md)가 최종 판정 출처다.

**Confirm51 N1 진단 판정:** [1셀 부하 결과](../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md)에서 P90/45/25/12·D80의 eager miss는 모두 0이고 external의 무선 정답은 P45부터 더 낮아졌다. P25/12의 external AI 완료량은 0이다. 복구 시각까지 직렬로 기다리는 현 구조는 다음 release를 늦게 시작시켜 NRx admission을 거절하므로, 저부하의 조건부 이득을 고부하 시스템 기여로 확장할 수 없다. 강한 결합 baseline 대비 노벨리티는 여전히 미확보다. 이 결과는 첫 라운드의 등급 C 진단이며, 전체 다중 셀 N1의 실패로 해석하지 않는다.

**Confirm46 해석 정정 및 Confirm52 결과:** Confirm46은 worker의 `stop` 응답 뒤 첫 RAN release를 예약했지만, worker 프로세스가 실제로 종료했는지 기다리지 않았다. 따라서 기존의 “종료 후 GPU/MPS 잔류 상태가 miss를 유발했다”는 설명은 인과적으로 입증되지 않았다. [Confirm52 결과](../../results/softwall_same_gpu/confirm52_mps_lifecycle_boundary_early_stop.md)는 유휴 상태 유지, stop 응답만 확인, 실제 worker 실행 래퍼 종료 확인을 9개 역순 triplet·27,000회에서 비교했다. Deadline miss는 각각 0/6/2였지만 짝 단위 정확검정 p=0.109375이고 남은 한 짝의 최선도 p=0.0625여서 사전 primary gate 실패로 조기 중단했다. 종료 확인 조건은 추가 약 2초 대기를 포함하므로 장벽 고유 효과와 quiet time 효과도 분리되지 않는다. 이를 독립 노벨리티 근거로 계산하지 않는다.

**저부하 이득 확인 라운드 종료:** Confirm47의 자격 검증 뒤 별도 사전 고정한 Confirm50의 50 ms gap lease 비교까지 판정하고 추가 seed 없이 종료했다. Confirm48 GC 원인 진단과 Confirm49 AI timeout 고장 주입은 계획과 코드만 준비했으며 이번 결론의 필수 성공 조건으로 소급하지 않는다. 이 둘 없이도 위의 **조건부 시스템 구성과 독립 seed의 반복된 처리량 이득**은 보고할 수 있지만, hard-real-time 보장·GC 원인 특정·고장 상황의 fallback 시작 보장은 주장하지 않는다. 다음 핵심은 [전체 선행 기법 결합 baseline](SOFTWALL_STRONG_BASELINE_SPEC_KO.md) 및 원 계획 P0의 실제 slot 계약이며, 완료하지 않은 항목은 [목표 완료 여부 점검](SOFTWALL_COMPLETION_AUDIT_KO.md)에 명시했다.

| 논문 기여 후보 | 직접 근거 | 지금 사용할 수 있는 표현 |
|---|---|---|
| 같은 요청의 NRx와 conventional을 별도 MPS client·원자 fallback 예약·single commit으로 연결 | Confirm42 통합 smoke, Confirm45 두 독립 seed/역순 강한 eager 비교 모두 frozen PASS | 완성된 transaction 경로가 이 시험 계약에서 AI 완료량 +2.95%/+2.41%, radio 비열등성과 RAN deadline miss 0을 달성 |
| 실패한 NRx의 예약 복구 전 slack을 bounded AI에 임대 | Confirm50의 새 50 ms 계약 ON/OFF +0.468%/+0.535%, 두 seed·역순 전체 frozen PASS. Confirm43/44 +0.419%/+1.304%이나 두 실험의 30 ms bound gate는 FAIL | gap lease의 작은 추가 처리량을 자격 계약에서 재현; 이전 30 ms 실패는 유지 |
| GPU client 수명주기를 RAN epoch와 분리 | Confirm39 live sham 실패·maintenance subgroup 통과, Confirm46 CPU 0/10,000 대 GPU/MPS 12/10,000, p=0.00390625. 단, Confirm46은 stop 응답만 확인 | 실행 중 겹치는 optional AI가 없어도 GPU client 수명주기 근처의 deadline 위험 신호가 있다. 실제 종료와 잔류 상태의 원인은 미분리 |

이 표의 novelty는 개별 부품의 최초성 선언이 아니라, **RAN 복구 가능성을 보존하는 공동 예약, 그 slack의 AI 회수, 그리고 MPS client 수명주기 경계**를 같은 PHY 요청과 같은 GPU에서 함께 다루는 문제 정의·구현·반례·측정이다.

**권고:** 연구의 중심을 “여러 GPU 기법의 통합”에서 **optional NRx의 무선 이득을 얻으면서, 같은 요청의 복구 가능성을 유지하는 공동 자원 제어**로 정한다. 요소의 결합도 시스템 논문의 기여가 될 수 있다. 다만 결합 과정에서 생기는 새로운 제약, 이를 해결하는 원리, 기존 기법을 충분히 결합해도 남는 비용을 보여야 한다. 아래는 확정된 성과가 아니라, 논문 기여로 방어하기 위한 가설과 검증 기준이다.

**2026-09-20 실험 업데이트:** 이 가설의 동일 GPU vertical slice는 이제 실측 근거가 있다. 같은 noisy PUSCH에서 conventional-only와 NeuralRx-only 성공 trace가 모두 관측됐고, optional NRx+reserved conventional+single commit+bounded background lease를 한 runtime에 연결했다. 새 seed 10,000 one-cell releases에서 eager 대비 radio utility 비열등성과 background +3.17%, 3,000 two-cell correlated fallback releases에서 양 정책 AI violation 0 조건의 background +5.53%를 얻었다. 100건의 late admission은 모두 conventional로 fail-closed했다. 상세 수치와 실패한 초기 gate는 [S2 자연 채널 보고서](../../results/softwall_same_gpu/S2_NATURAL_CHANNEL_REPORT_KO.md)에 있다.

별도 MPS client가 timeout 뒤 45–126 ms 물리 실행되는 실험에서는 D25가 cap100과 cap20 모두 실패했다. 이는 “MPS cap이 완전 격리를 제공한다”는 더 단순한 주장과, fallback 함수만 추가하면 된다는 설계를 직접 반증한다. SoftWall의 기여 후보는 이 실패를 숨기는 것이 아니라 **physical outstanding work와 campaign residual state를 admission/drain/quarantine 계약에 포함해 unsafe optional work를 reject하는 것**까지 확장해야 한다.

Confirm30은 late work를 drain한 직후 process/MPS client를 종료하는 직관적인 fail-closed 방식도 반증했다. 종료 뒤 cap100 6/5,000, cap20 11/5,000 D35 misses가 발생했고 active-overlap miss는 0이었다. 따라서 새 시스템 기여 후보는 논리 quarantine과 물리 teardown을 분리하고, RAN-free maintenance window에서만 context를 파괴하는 lifecycle-aware contract까지 포함한다. 이 현상은 원인 분해와 재현 범위 확장이 필요하지만, 단순 cap·timeout·kill 결합과 구별되는 구체적 설계 제약을 제공한다.

기존 Confirm31 maintenance 결과는 별도 sham-retirement campaign과 release 구간이 겹친 사실이 확인돼 PASS 근거에서 제외했다. Confirm39의 전역 GPU0 lock 단독 재실행에서는 RAN-free maintenance 후 6,000회 conventional deadline miss 0이었지만, live-epoch sham client retirement에서 cap100/cap20 각각 4/5,000 miss가 나와 전체 gate는 실패했다. Confirm46은 GPU context를 만들지 않는 CPU socket retirement와 MPS/GPU client retirement를 같은 node·seed·교대 순서의 10짝으로 비교해 CPU 0/10,000 대 GPU/MPS 12/10,000 deadline misses, 짝 단위 단측 정확검정 p=0.00390625로 모든 frozen gate를 통과했다. 다만 GPU warmup·context 생성·MPS 등록·종료가 묶인 대비라 단일 내부 단계 원인은 특정되지 않았다. 이 lifecycle만으로 novelty가 완성되는 것은 아니며, admission·recovery와 결합해야 한다.

Confirm32는 request-specific PUSCH/channel estimate와 LLR을 same-device CUDA IPC로 교환하는 별도 persistent cap80 endpoint에서 clean 10,000/10,000 correct와 miss/timeout 0을 보였다. Confirm33은 같은 external path의 −8.5 dB 10,000 releases에서 자연 fallback 221건 중 47건을 conventional로 복구하고 miss/timeout 0을 유지했다. Confirm36의 세 방식 동일 node·trace 비교에서 외부 S2의 eager 대비 무선 효용 비열등성을 확인했다. Confirm37의 6 ms background host-RPC 계약은 실패한 채 보존했고, 독립 seed·40 ms 계약의 Confirm37b는 통과했다. Confirm38은 별도 cap20 background client와 외부 S2 endpoint를 결합하여 local S2/eager와 각각 동일 node·trace 10,000회 비교했다. 세 정책 모두 RAN miss·AI 계약 위반 0이고 외부 S2가 eager보다 완료 AI +2.93%였다. Confirm41의 역순 독립 seed에서 +3.06%로 반복됐지만, 두 외부 경로 모두 실제 원자 예약 전 구현이므로 완성된 SoftWall 효과는 Confirm45로 따로 판단한다.

Confirm32–38/41의 외부 경로는 자연 실패 후 conditional fallback을 수행했지만, 로컬 S2의 원자 fallback calendar와 `DartRuntime` single-commit 상태를 호출하지 않았다. 따라서 그 처리량 이득은 **별도 process 통합의 근거**로만 남긴다. Confirm42가 실제 transaction과 recovery-gap AI lease를 연결한 500회 smoke를 통과했고, Confirm45가 완성된 경로의 강한 eager 비교를 두 새 seed에서 통과했다.

현재 novelty 판정은 **실측으로 뒷받침되는 조건부 시스템 결합**이다. 완성된 transaction 경로의 eager 대비 AI 처리량 우위와 gap lease 구성요소의 이득은 확인됐다. 별도 process에서 물리적으로 계속 실행되는 timed-out 작업, NRx worst-case 상한, GPU client lifecycle의 원인 단계, 다중 GPU/전송 credit은 완료되지 않았다. 따라서 top-conference 주장은 성능·기능 기여로 좁히고, hard guarantee는 후속 검증 전까지 제외한다.

**1. “우리와 같은 연구가 없다”의 현재 판정**

- **확인됨:** GPU RAN/AI 공유, deadline-aware pool 배치, AI/conventional 선택, fallback, 작은 background unit을 각각 또는 일부 함께 다루는 연구가 있다.
- **이번에 검토한 공개 본문에서는 확인하지 못함:** 같은 PUSCH/TB의 optional neural 경로와 conventional recovery를 보존하면서, 여러 셀의 로컬 후처리·공유 GPU queue·background 허가량·만료 후 미완료 작업을 함께 제어하는 시스템 전체.
- **미확정:** 이 전체 모델이 외부의 실시간 scheduling 기법으로 이미 표현·해결되는지, 우리가 제안할 정책이 그 adaptation 이상인지, 실제 PHY 시간 예산에서 이득이 남는지.

따라서 현재 사용할 문장은 “검토한 AI-RAN 선행 연구는 아래 공동 제어 문제를 같은 범위로 해결하지 않는다”이다. “유사 연구가 전혀 없다” 또는 “최초임을 확정했다”는 문장은 근거 범위를 넘는다. 공개 자료의 범위를 넓히면 부재에 대한 신뢰도는 높일 수 있지만 미공개 연구까지 배제할 수는 없다.

기존 DART의 `FallbackCalendar`, admission, lease, commit은 **우리의 내부 자산**이다. 코드에 이미 있다는 사실만으로 논문 novelty가 사라지지 않는다. 외부 선행 연구와 기발표한 우리 논문에 대해 무엇이 새로운지가 기준이며, 내부 v0/v1 비교는 각 메커니즘의 기여를 설명하는 ablation이다. 이번 검토로 DART의 기발표 여부를 확정하지는 않았다.

**2. 논문의 중심 질문과 기여 문장**

중심 질문:

> 실제 PHY 완료 기한이 있는 요청에 대해, 선택적인 neural receiver의 실행시간이 공유 GPU의 상태에 따라 흔들릴 때, conventional 처리를 제때 끝낼 여력을 보존하면서 neural 개선분과 background AI 처리량을 함께 얻을 수 있는가?

검증 후 사용할 기여 문장의 초안:

> SoftWall은 요청의 radio utility, 공유 GPU의 실행 상태, 로컬 recovery 및 후처리 여력을 함께 고려하여 NRx admission·배치·background 실행을 결정한다. 논리적으로 만료된 작업의 물리적 자원 점유까지 추적하여 복구 가능 상태를 유지하고, 동일 GPU 예산과 RAN 품질 요구에서 독립 제어보다 더 많은 유효 작업을 처리한다.

여기서 “유효”는 PHY expiry 전에 올바른 요청에 귀속되어 실제 사용할 수 있는 결과를 뜻한다. NRx가 모든 요청에서 성공한다는 보장도, conventional이 모든 무선 오류를 복구한다는 보장도 아니다. conventional의 제시간 CRC-fail 응답과 계산 deadline miss는 구분한다.

**3. 기존 접근의 범위 밖에서 우리가 해결해야 할 문제**

아래의 “남는 문제”는 논문 전체에 대한 결함 판정이 아니라, 우리가 제안하는 작업 조건에 적용할 때 검증해야 할 공백이다.

| 기존 접근과 근거 | 그 접근이 제공하는 것 | 우리 조건에서 남는 문제 / 필요한 해결 |
|---|---|---|
| [YinYangRAN, INFOCOM 2024, §III–V](https://neclab.eu/fileadmin/user_upload/YinYangRAN_Resource_Multiplexing_in_GPU-Accelerated_Virtualized_RANs_pre-print.pdf) | RAN/ML GPU 할당, 신뢰도 모델, 재설정 비용 고려 | 할당 비율과 신뢰도 제어를 넘어, 이미 받은 요청의 NRx 실패·지연 때 사용할 로컬 처리 구간을 확보해야 함 |
| [CloudRIC, MobiCom 2024, §4](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1795/mobicom24-final428_authors_v.pdf?sequence=1) | queue·처리시간을 예측하는 이종 pool 배치와 compute-aware radio grant | 하나의 처리 경로를 선택하는 문제에서, 품질 이득이 있는 선택 경로와 별도 recovery 의무가 공존하는 문제로 확장해야 함 |
| [ARCHES, 2026 preprint, §3](https://arxiv.org/html/2604.23397v1) | 채널 기반 AI/conventional expert 선택, concurrent/selected-only 모드 | 무선 조건이 같아도 queue·잔여 background·후처리 여력에 따라 NRx를 다르게 선택하고, 현재 요청의 복구 가능성을 유지해야 함 |
| [OCUDU, 2026 preprint, Table I·§IV-A](https://arxiv.org/html/2609.07843v1) | resident inline contract, conventional 경로와 validation | Class A의 enqueue 후 late completion은 사용·기록·반복 시 차단하는 동작. 별도 pool에서 지연된 NRx를 포기하고 현재 요청을 시한 내 마치는 계약은 추가 검증 대상 |
| [AI-and-RAN Efficient and Safe, MobiUK 2026 초록](https://www.mobiuk.org/2026/abstracts/S2_On_Making_AI_and_RAN_Efficient_and_Safe.pdf) | RAN이 공개하는 여유 계산량과 그에 맞춘 작은 training unit | 여유 계산량 자체가 optional NRx 선택과 recovery/후처리 예약에 따라 바뀌는 경우의 공동 제어. 공개 초록만으로 세부 알고리즘의 부재를 단정하지 않음 |

YinYangRAN에는 GPU 재설정 중 CPU fallback이 있고, CloudRIC에는 빠른 요청별 의사결정이 있다. ARCHES에는 선택한 expert만 실행하는 모드도 있다. 이들을 빼고 비교하면 차이를 과장하게 된다. Concordia의 예측·간섭 보정과 Nuberu의 필수 RAN 경로 보존도 함께 인용한다. 상세 출처와 발표 구분은 [검토 문서](SOFTWALL_RELATED_WORK_AUDIT_KO.md)를 따른다.

**추가 확인: 복구 예약 자체도 기존 이론과 비교해야 한다.** Primary–backup scheduling에는 서로 다른 처리기의 primary/backup을 예약하고 자원 사용을 개선하는 연구가 있다. Optional/mandatory 계산의 품질–시간 절충도 오래된 문제이며, optional 뒤에 필수 wind-up 처리를 두는 모델까지 존재한다. Neural Simplex는 neural controller와 안전한 baseline 사이의 전환을 다룬다. 따라서 “복구 예약”, “optional + mandatory”, “AI + 안전한 baseline” 각각을 최초라고 주장할 수 없다. [Primary–backup, 2004, 출판사 초록](https://www.sciencedirect.com/science/article/pii/S0743731504000541), [동적 PB scheduling, 2008, 공식 초록](https://www.jstage.jst.go.jp/article/transinf/E91.D/3/E91.D_3_796/_article), [wind-up 모델, 2003, 공식 초록](https://globals.ieice.org/en_transactions/information/10.1587/e86-d_10_2040/_p), [Neural Simplex, 원문](https://www3.cs.stonybrook.edu/~stoller/papers/nfm2020.pdf)

**4. 방어할 기여는 세 개로 묶는다**

| 기여 후보 | 실제로 새로워야 하는 부분 | 가장 강한 반론 | 필요한 증거 |
|---|---|---|---|
| **C1. 무선 이득과 복구 비용의 결합을 규명** | NRx 선택이 pool 부하뿐 아니라 로컬 후처리·fallback 부하를 바꾸고, background 간섭이 다시 그 선택의 가치를 바꾸는 현상 | “일반적인 noisy neighbor나 backup scheduling 아닌가?” | 실제 채널·MCS·셀 부하에서 독립 제어가 손해를 보는 원인과 빈도, 회복 가능한 이득을 분해 |
| **C2. 복구 가능 상태를 유지하는 공동 제어** | NRx 수락·endpoint·후처리·복구 구간·background unit을 같은 제약으로 결정하고, 예측 오류와 미완료 작업 중에도 계약 유지 | “기존 queue scheduler에 fallback 하나 붙이면 되지 않나?” | 강한 결합 baseline 대비 정책 차이, 조건부 불변식 또는 schedulability 분석, 작은 문제의 최적해 대비 비용과 실행 오버헤드 |
| **C3. 실제 PHY 경로에서 유효성·효율 검증** | 결과가 단지 빨리 끝나는 것을 넘어 유효 TB와 background 유효 처리량을 함께 늘림 | “toy timeout·합성 kernel·GPU 추가로 얻은 이득 아닌가?” | 실제 paired radio 입력과 LDPC/CRC·전송·commit 포함, 동일 노드 자원 및 요구 조건, 반복·tail·실패 주입 평가 |

C3는 검증의 기여이며 모든 구현 요소가 새 알고리즘이라는 뜻은 아니다. Epoch, buffer generation, timer, queue, P2P 자체는 기반 수단이다. C1에서 중요한 현상이 발견되지 않거나 C2가 기존 결합보다 개선되지 않으면, 구성 요소를 더 추가해 연구 주장을 유지하지 않는다.

**5. 공동 제어의 구체적 설계 후보**

상태는 요청의 관측 가능한 radio feature, GPU0의 필수/후처리/recovery calendar, endpoint queue, 미완료 background unit, 전송 credit, 실제 미완료 DMA·kernel의 buffer 점유로 구성한다. 요청마다 후보 `(endpoint, NRx 후처리 구간, conventional 시작 시각, background 허가)`를 만들고 다음을 확인한다.

```text
예상 NRx LLR 반환 <= 예약된 NRx LDPC/CRC 시작
NRx LDPC/CRC 완료 + guard <= conventional 시작 f
f + conventional 비용 상한 + 최종 guard <= PHY expiry d

추가 의무:
  이미 받은 모든 요청의 필수·복구 calendar와 충돌하지 않음
  기존 NRx 예약을 새 background 허가가 뒤로 밀지 않음
  전송/버퍼 credit은 물리 완료 전까지 계속 점유
  NRx 거부 요청도 conventional 부하에 포함
```

위 식은 [v0 스킴](SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md)의 보수적 실행 조건이며 그 자체로 새 알고리즘은 아니다. 새 정책의 후보는 **이 조건을 만족하는 여러 계획 중 radio 요구를 채우면서 background 손실과 로컬 예약 비용을 줄이는 선택**이다. 항상 가장 빠른 endpoint가 이 선택과 같지는 않다.

목적은 고정 노드 예산에서 `background 유효 처리량 최대화`로 두고, `L1 miss <= ε`, `delivered radio utility >= Rmin`, `결과/버퍼 불변식`을 제약으로 둔다. `ε`와 `Rmin`은 결과를 본 뒤 유리하게 바꾸지 않는다. 여러 값을 sweep해 유효한 trade-off도 제시한다.

Radio utility는 요청별로 `E[선택 정책의 deadline-valid 유효 비트 − conventional 정책의 유효 비트 | 현재 관측]` 형태로 평가한다. 미래 CRC 정답을 admission에 사용하지 않는다. 무선 성공 가능성과 시간 내 완료 가능성을 독립이라고 놓고 곱하지 않으며, 실제로 joint/conditional 추정이 필요한지 검증한다. 채널이 나쁘다고 NRx가 항상 유리하거나 같은 모양의 NRx kernel이 항상 더 느려진다고 가정하지 않는다.

추천하는 첫 구현은 유한한 endpoint와 work-unit 크기에 대해 가능한 계획을 열거하고, 고정 utility table과 측정 비용으로 선택하는 것이다. 작은 trace에서는 완전탐색/최적화로 얻은 상한과 비교한다. 그 뒤 병목이 확인된 경우에만 predictor·온라인 보정·선택적 동시 실행을 추가한다. 특정 신경망 predictor의 사용 자체는 기여로 삼지 않는다.

**보호와 예측의 역할을 분리한다.** 공유 endpoint의 NRx가 예상보다 늦더라도 GPU0에 남겨 둔 conventional 경로를 실행할 수 있는 것이 목표다. 단, 이는 로컬 필수·후처리 실행 비용과 제어 지연이 정한 범위 안에 있고, NRx의 전송 간섭이 그 경로를 막지 않도록 제한됐다는 조건 아래에서만 성립한다. GPU0에서 이미 실행한 optional LDPC가 예상을 넘으면 fallback을 막을 수 있으므로 이 비용도 별도로 검증한다. 단순한 predictor p99로 hard real-time을 증명하지 않는다.

만료 후에도 kernel/DMA가 실행 중이면 endpoint와 buffer credit을 계속 차지한다. 다음 admission은 그 점유를 반영해야 한다. 논리적인 cancel만으로 용량을 되돌려 주면 안 된다. 반복적인 미완료 작업이 자원을 소진하면 해당 endpoint의 새 NRx 수락을 멈춘다. Late work를 drain한 뒤에도 실시간 구간에서는 process/context를 유지한 `QUARANTINED_IDLE` 상태로 두고, RAN-free maintenance window에서만 teardown한다. 각 상태에서 conventional 기본 용량이 유지되는지 별도 검증한다.

모든 요청이 동시에 fallback할 수 있는 상태를 기본으로 다룬다. 평균 성공률이 높다는 이유로 같은 recovery 시간을 여러 요청에 중복 예약하지 않는다. 그런 통계적 공유를 연구하려면 별도의 상관관계 모델·위험 예산·비교가 필요하며 초기 기여에 포함하지 않는다.

GPU0에 background를 두지 않는 현재 배치에서 NRx 성공으로 해제한 conventional 시간은 **GPU0의 시간**이다. 이를 바로 GPU1–3의 AI 처리량 증가로 계산해서는 안 된다. 늘어난 후속 admission 여력·로컬 부하 감소와 pool 측 처리량을 각각 측정한다.

**6. “요소를 합쳐야 한다”를 설명하는 반례**

아래는 실측 결과가 아니라 논리적 설명이다. 개별 논문이 이 실수를 한다는 주장도 아니다.

- **요청별 판단의 충돌:** 두 요청의 expiry가 10이고 conventional 시간이 각각 3이라고 하자. 둘 다 7에 fallback하면 된다는 개별 판단은 단일 직렬 처리기에서 불가능하다. 실제로는 4–7과 7–10 같은 서로 다른 구간이 필요하며, 첫 요청의 NRx 판단 시한도 앞당겨진다. 이는 공통 calendar의 필요성을 설명하지만 기존 primary–backup scheduling도 해결할 수 있으므로 단독 novelty는 아니다.
- **안전하지만 비효율적인 결합:** utility gate와 pool scheduler, background controller가 각자 고정 여유를 남기면 deadline은 지킬 수 있다. 그러나 NRx 이득이 작은 요청 때문에 가치 있는 background unit을 중단하거나, 서로 다른 자원에 중복 여유를 두어 일을 거부할 수 있다. 공동 제어가 그 손실을 줄이는지는 실제 자료와 최적화 상한으로 검증해야 한다.
- **만료와 점유의 불일치:** NRx가 기한을 놓쳐 conventional로 끝난 뒤에도 이전 DMA가 계속될 수 있다. 이때 buffer를 재사용하거나 endpoint를 비었다고 판단하면 후속 요청의 데이터 또는 시간 예측이 틀어진다. 해결책 자체의 일반성은 인정하되, 이 점유를 admission/lease에 연결하지 않았을 때 발생하는 연쇄 효과를 측정한다.

**7. 비교 구성과 통과 기준**

| 비교 구성 | 확인하는 질문 |
|---|---|
| conventional-only + 같은 전체 GPU 예산의 background | NRx를 끄는 단순 정책보다 radio 측 이득이 필요한가? |
| 전용 NRx 배치 + 남은 자원의 background | 공유로 얻는 이득이 실제로 있는가? |
| conventional 결과를 먼저 확보하거나 두 경로를 함께 실행하는 정책 | 복구 예약을 정교하게 관리할 필요가 있는가? 단순 중복 실행의 비용과 비교 |
| 채널 gate + queue-aware 배치 + 충분히 튜닝한 recovery 예약 + bounded background | 기존 기법을 결합하면 이미 충분한가? 같은 상태 정보·안전장치를 제공 |
| 기존 DART / 보수적 v0 | 우리 내부 자산의 어떤 메커니즘이 효과를 만드는가? |
| 공동 제어: 고정 모델 / 적응 모델 | 공동 의사결정과 예측 정확도의 기여를 분리할 수 있는가? |

Primary–backup 계열은 실제 fault/cancel 가정과 우리 실행 모델을 대조한 뒤 적용 가능한 정책을 결합 baseline에 포함한다. 원본을 실행하지 못한 경우 재현 범위와 adaptation을 표시한다. 모든 baseline이 반드시 같은 정책 구조를 가져야 하는 것은 아니며, 같은 자원·관측 정보·실제 요청과 품질 요구를 제공하는 것이 우선이다.

가장 중요한 실험은 `radio 조건 × 셀의 동시 도착/부하 × background phase × 예측 오차`의 교차 실험이다. NRX tail과 fallback burst가 실제로 함께 발생하는지 먼저 관찰하고, controlled delay/failure injection은 관측 결과와 구분해 stress evidence로 제시한다. 평균적인 독립 지연만 실험해서 동시 fallback 문제를 입증했다고 하지 않는다.

성공 기준은 (a) full-path가 실제 PHY expiry에 들어오고, (b) 추가 제어 오버헤드를 포함해 강한 결합 baseline보다 의미 있는 효율 이득이 반복되며, (c) 그 이득이 어느 결합 제약을 해결해서 나오는지 설명되는 것이다. 모든 상황에서 우월할 필요는 없다. 유리한 조건의 넓이, 불리한 조건, 예약·복제·buffer 점유의 비용을 같이 밝힌다.

**8. novelty의 확실성을 높이는 순서**

1. C1–C3를 논문의 고정 주장으로 두고, 각 주장에 가장 가까운 AI-RAN 원문과 일반 scheduling 원문을 대응시킨다. 읽은 절·문제 모델·실패 가정·공개 상태를 기록한다.
2. 초록만 있는 자료는 “미확인”으로 남긴다. 특히 MobiUK의 후속 본논문, AoRA의 세부 구현, primary–backup 정책의 cancel/fault 가정은 현재 확신의 한계다.
3. 실제 PHY timing을 확인한 뒤, 두 셀·두 GPU 정도에서 결합 baseline의 안전성/효율 한계를 재현한다. 차이가 없으면 모델을 복잡하게 만들기 전에 가설을 수정한다.
4. 차이를 만드는 최소 정책과 조건부 불변식을 작성한다. 작은 문제의 최적해, 정책 overhead, 각 메커니즘 제거 실험으로 설명을 검증한다.
5. 논문 초안에서 “안 다뤘다”는 주장마다 원문의 근거와 우리 실험을 연결하고, 투고 전에 최신 공개 자료를 다시 확인한다.

지금 확정할 것은 “새 기능 다섯 개”가 아니라 **C1의 문제를 실제로 확인하고, C2로 해결하며, C3으로 입증한다는 연구 구조**다. 이 과정이 성공하면 기존 요소를 포함한 시스템 구성 자체도 방어 가능한 기여가 된다.
