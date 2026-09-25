**SoftWall의 AI-RAN 선행 연구와 차별화 검토 — 2026-09-24**

대상: [원 계획서](RESEARCH_PLAN_SOFTWALL_KO.md), [구체 스킴](SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md), [타당성 검토](RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md).

후속 검토: [novelty thesis와 입증 기준](SOFTWALL_NOVELTY_THESIS_KO.md). AI-RAN 밖의 primary–backup scheduling, optional/mandatory 및 wind-up 모델, Neural Simplex도 추가 비교 대상으로 확인했다. 복구 예약 자체를 새 기여로 확정하지 않으며, 내부 DART 구현의 존재와 외부 선행 연구 중복을 구분한다.

**판정: 유사 연구가 있다. AI-RAN, GPU 공유, deadline-aware admission, conventional 경로,
bounded background 실행, P2P endpoint는 각각 새 기여가 아니다.** 이번 공개 본문 검토에서
확인하지 못한 좁은 결합은 동일 TB의 optional NeuralRx가 만든 conventional recovery debt를
여러 셀의 executable all-fail schedule로 유지하고, 그 debt의 retiming과 local/remote
endpoint·transport credit 및 외부 AI compute lease를 물리 완료까지 원자 lifecycle로
관리하는 방식이다. C102와 C113은 optimizer·추가 처리량 우위를 기각했다. C114는 이 계약이
same-device CUDA IPC에 묶이지 않고 remote NVLink-P2P NeuralRx에서도 동작한다는 유한 표본
증거를 추가했고, C119/C120은 네 physical GPU·네 endpoint까지 lifecycle을 확장했다.
멀티 GPU와 P2P 자체는 여전히 새 기여가 아니다. C123은 global AI request ownership을
두 local recovery certificate와 직렬화했고, C125/C126은 계산된 certificate가 실제
executor 순서로 refinement되어야 한다는 조건과 반례를 추가했다. 이 역시 일반적인
distributed lease나 EDF 자체가 아니라 conditional-recovery state와의 결합으로 범위를
좁힌다. 따라서 novelty 주장은 **새 정책의 우월성**이 아니라 conditional-recovery
substrate, certificate-conformant execution과 safe/useful/infeasible envelope에 한정한다.
C135/C136은 V12가 예측한 AI40 class를 물리 실행했지만, C138은 faulting RPC의
controller-return이 5 ms socket timeout을 넘는 반례를 만들었다. 따라서 V12의
`AI40+control15+guard2=57 ms` timing contract는 UQ다. V13 synchronous reference는 timeout 5 ms와
admission wall bound 7 ms를 분리하고 `AI35+control21+guard2=58 ms` class만 유지한다.
C139의 6개 fault arm과 C140의 8개 conditional exchange가 이 수정된 사슬을 지지한다.
C141은 독립 A100 node에서 control 6 arm과 conditional exchange7을 추가해 synchronous
corrected mode를 same-family 두 node로 확장했다. V14는 `prepare/abort/complete`를
RAN executor 밖으로 pipeline하고, staged ownership token을 launch 직전에 현재 all-fail
certificate로 재검증한 뒤 bounded `commit` ACK가 있을 때만 Qwen을 제출한다.
`AI45+commit7+guard2=54 ms` class는 C142/C144 두 A100 node에서 exchange11회로
물리화됐고, C143/C144의 9개 fault arm은 deferred RPC가 7 ms를 50회 넘은 상황에서도
local RAN safety와 at-most-once execution을 유지했다. Async RPC, prefetch, at-most-once
protocol 각각은 일반 분산·GPU 시스템의 새 요소가 아니다. 차별점 후보는 이들을
same-TB recovery debt와 launch-time certificate에 결합한 정확한 실행 계약이다. 후속
two-request 감사는 V14의 retained-token ownership 공백을 찾아 mode를 UQ로 내렸다.
V15 runtime은 home별 launch 전 token을 하나로 제한하며, C145/C146 두 새 A100 node에서
AI45 exchange12, fault6 arm, prepare 억제537회와 maximum unlaunched token1을
safety·duplicate 위반 없이 통과했다. C147/C148은 물리 matrix에서 빠졌던 abort
post-apply reply-loss를 두 추가 node에서 닫았다. V16은 prepare/abort/commit/complete마다
두 physical arm을 요구하며, C145--C148의 radio15,360·fault 뒤10,125에서 선언 위반0이다.

검토 범위는 2026-09-24까지 공개된 자료다. SIGCOMM·NSDI·MobiCom·INFOCOM·SIGMETRICS의 RAN 연구와 OSDI·SOSP 계열 GPU scheduling을 키워드 및 참고문헌으로 검색했다. 관련성이 높은 논문은 원문 설계 부분을 확인했고, 초록만 확인한 자료는 아래에 표시했다. 모든 학회·연도의 전체 논문을 전수 조사한 결과나 미공개 투고작의 부재를 뜻하지 않는다. 출처는 학회·출판사·저자·소속기관·arXiv의 1차 자료다. 9월 24일에는 `NeuralRx/conventional fallback`, `all-fail recovery`, `recovery credit`, `MPS`의 교차 검색도 다시 수행했으며 아래 OAI testbed 외에 정확한 계약 조합은 추가로 확인되지 않았다.

**1. 정식 학회 논문 중 직접 비교할 연구**

| 논문 / 발표 구분 | 이미 다루는 문제 | SoftWall에 대한 의미 |
|---|---|---|
| **YinYangRAN — INFOCOM 2024 본논문** | GPU에서 RAN PHY와 ML을 함께 실행하고 신뢰도 제약 아래 자원 분배 | “RAN + AI GPU 공유와 deadline 보호”는 선행 연구가 있음 |
| **CloudRIC — MobiCom 2024 본논문** | 여러 DU가 CPU/GPU 등 이종 pool을 공유하며 요청별 배치와 무선 부하를 제어 | “대기시간을 예측하는 RAN accelerator pool”도 선행 연구가 있음 |
| **Concordia — SIGCOMM 2021 본논문** | CPU vRAN deadline 보호와 일반 작업용 유휴 자원 회수 | 예측·예약·잔여 용량 회수라는 기본 논리의 직접 선행 연구 |
| **Nuberu — MobiCom 2021 본논문** | 계산 자원 부족 시 필수 신호를 보존하고 나머지 처리를 조절 | 필수 경로와 지연 가능한 처리를 구분하는 RAN 설계가 이미 존재 |
| **AoRA — SIGCOMM 2025 short paper, 3쪽** | RAN의 여유 계산 자원에서 edge AI를 기회적으로 서비스 | “AI-on-RAN으로 여유 GPU를 활용”하는 넓은 주장과 겹침 |
| **SMEC — NSDI 2026 본논문** | RAN MAC의 무선 자원과 별도 MEC 서버의 CPU/GPU 작업을 각각 application SLO에 따라 관리; MEC GPU에서는 MPS stream priority 사용 | “5G에서 MPS를 쓰고 deadline-aware GPU 작업을 조절”하는 넓은 주장도 선행 연구와 겹침. 동일 GPU PHY의 optional NRx/recovery 예약과 문제 수준은 다름 |

**YinYangRAN.** A100에서 PHY FEC와 ML 서비스를 공유하고 MPS의 SM 할당을 조절한다. §IV의 제어 주기는 1초 이상이며, 재설정 비용을 최적화에 포함한다. §III-D에는 GPU 재설정 중 CPU 소프트웨어로 전환하는 fallback도 있다. 따라서 “기존 연구에는 fallback이 없다”는 비교는 틀리다. 검토한 방식의 중심은 할당 비율 제어이며, 같은 PUSCH의 optional NRx를 위해 별도 conventional·후처리 시간을 확보하는 현재 스킴과는 예약 대상이 다르다. [논문 원문, §III-D–V](https://neclab.eu/fileadmin/user_upload/YinYangRAN_Resource_Multiplexing_in_GPU-Accelerated_Virtualized_RANs_pre-print.pdf), [IEEE 발표 정보](https://ieeexplore.ieee.org/document/10621380/)

**CloudRIC.** AAL broker가 처리기별 queue와 실행시간을 예측하고, deadline과 guard를 충족하는 처리기를 고른다. RT-RIC는 compute 상황에 맞춰 radio grant도 조정한다. 따라서 “YinYangRAN은 느리지만 우리는 요청 단위라 새롭다”만으로 부족하다. 여러 셀의 부하를 함께 보거나 보수적 지연 모델로 endpoint를 선택하는 것도 단독 차별점이 아니다. 검토한 §4의 중심은 FEC 요청의 이종 처리기 배치이며, optional neural 개선분과 같은 요청의 conventional recovery, 비-RAN AI 허가량을 묶는 정책은 별도로 입증해야 한다. [논문 원문, §4 및 Algorithm 1](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1795/mobicom24-final428_authors_v.pdf?sequence=1), [본논문 DOI](https://doi.org/10.1145/3636534.3649381)

**Concordia.** 입력 조건에 따른 실행시간 예측과 온라인 간섭 보정을 사용해 vRAN용 CPU를 예약하고 나머지를 일반 작업에 돌려준다. CPU에서의 선점 가능성과 SoftWall의 GPU work-unit blocking은 다르지만, 단순히 CPU를 GPU로 바꾼 것만으로 충분한 기여라고 보기는 어렵다. [SIGCOMM 2021 원문, §1·설계](https://conferences.sigcomm.org/sigcomm/2021/files/papers/3452296.3472894.pdf)

**Nuberu.** 계산 부족 중에도 사용자 동기화를 유지하는 minimum viable subframe을 확보한다. deadline 분리, predictive HARQ, 부하 제어가 핵심이다. 이는 같은 TB를 conventional receiver로 다시 처리하는 정책과 같지는 않지만, “필수 RAN 동작을 먼저 보호한다”는 원칙의 선행 연구다. [MobiCom 2021 원문, §1.2](https://agsaaved.github.io/files/papers/2021_ggarcia_mobicom_nuberu.pdf)

**AoRA.** container 기반 AI를 RAN에 넣어 통신 동작을 보존하면서 유휴 자원을 활용한다. 저자 목록에서 short paper임을 확인했다. 이번 비교는 공식 초록과 서지정보 수준이므로 세부 admission/fallback이 없다고 단정하지 않는다. [SIGCOMM 공식 프로그램](https://conferences.sigcomm.org/sigcomm/2025/program/papers-info/), [저자의 short 표기](https://sites.google.com/view/kyunghanlee/publications), [DOI](https://doi.org/10.1145/3718958.3750517)

**SMEC.** NSDI 2026의 실제 5G MEC testbed에서 RAN MAC scheduler와 별도 edge server가 application 요청의 남은 SLO 시간을 독립적으로 계산하고, edge GPU 작업에는 MPS stream priority를 적용한다. RAN은 srsRAN 서버, AI/video workload는 L4 edge 서버에서 실행되므로 동일 물리 GPU의 PHY optional NRx·conventional recovery 계약은 평가하지 않는다. 하지만 “MPS와 deadline-aware RAN/AI 자원 관리”라는 포괄적 기여 문장은 SMEC와 명백히 겹친다. SoftWall은 PHY 처리 자체의 radio utility·동일 요청 복구 예약·물리 미완료 작업 처리로 범위를 좁혀야 한다. [NSDI 2026 원문, §1·§6–7](https://www.usenix.org/system/files/nsdi26-zhang-xiao.pdf)

**2. 본논문과 구분해야 하지만, novelty 검토에서 중요한 최신 자료**

| 연구 | 확인된 공개 상태 | 겹치는 부분 |
|---|---|---|
| **ARCHES** | arXiv, 2026-04-25 | 채널 상황에 따른 AI/conventional PHY expert 선택과 slot 경계 전환 |
| **OCUDU dApp Platform** | arXiv, 2026-09-07; 플랫폼 preview | GPU inline AI의 timing contract, conventional 경로, 결과 검증과 수명 관리 |
| **Real-Time dApps for AI-RAN** | arXiv, 2026-09-07; OCUDU 동반 preview | 39개 AI-RAN use case의 DU 인터페이스·100 µs 제어 tail·live-cell 경로 계측 |
| **On Making AI-and-RAN Efficient and Safe** | MobiUK 2026 발표용 1쪽 초록 | RAN deadline 보호, 안전한 여유 계산량 공개, 작은 학습 연산 단위 |
| **HAF: Deadline-Driven Hierarchical Agentic Resource Sharing…** | arXiv, 2026-05-08 | AI-RAN의 deadline 기반 GPU/CPU 할당과 서로 다른 시간 규모의 제어 |
| **Interplay of AI-and-RAN** | INFOCOM 2025 workshop | SAC로 latency-sensitive RAN과 AI 사이의 MIG를 동적 배분; trace-driven simulation |
| **CAORA** | arXiv, 2025-07-12 | O-RAN KPI forecasting·anomaly detection·SAC를 결합한 proactive MIG orchestration |
| **AI-RAN: Transforming RAN…** | arXiv, 2025-01-15; architecture/PoC article | GH200에서 RAN·AI 동시 실행과 AI-for/on/and-RAN reference architecture |

**ARCHES.** GPU PHY에 AI와 conventional expert를 상주시켜 채널에 따라 선택하고, slot 경계에서 출력 경로를 바꾼다. 따라서 utility-aware neural/conventional 선택 자체는 새 기여가 아니다. 평가의 concurrent 모드와 별도로 selected-only 모드도 설명하므로, 모든 expert를 항상 실행해야만 하는 방식으로 묘사해서도 안 된다. 원고에는 MobiHoc ’26 표기가 있지만 DOI·ISBN이 placeholder이고 이번 검색으로 공식 채택을 확인하지 못했다. 여기서는 preprint로 분류한다. 기존 [DART novelty 문서](../architecture/DART_RX_CANONICAL_NOVELTY_THESIS_KO.md)에도 이미 포함된 연구다. [원문, §3](https://arxiv.org/html/2604.23397v1)

**OCUDU.** Class A의 shape 범위 이탈·bypass·enqueue 실패에는 같은 호출의 conventional 처리를 사용한다. 반면 이미 enqueue된 GPU 작업이 늦게 끝나면 그 결과를 사용하고 incident를 기록하며, 반복 시 breaker를 연다. Class B의 late-result 폐기/rollback과 구분해야 한다. 이 차이는 SoftWall의 “같은 요청의 expiry 전에 별도 recovery 완료”를 검증할 비교 지점이다. 문서의 lease는 Class C 데이터 view 등의 수명이므로 background compute lease와 같은 뜻은 아니다. [원문, Table I–II·§IV-A·§XI](https://arxiv.org/html/2609.07843v1), [공개일·상태](https://arxiv.org/abs/2609.07843)

**OCUDU 동반 인터페이스 논문.** DU가 실제로 기다리는 제어와 관찰 전용 작업을 구분하고, live-cell 부하에서 100 µs 제어 tail 및 in-process 경로를 측정한다. 따라서 AI-RAN 앱의 실시간 DU 인터페이스나 유휴 시간의 AI 실행을 일반적 새로움으로 주장하기 어렵다. SoftWall에는 같은 PHY TB의 선택적 NeuralRx·모든 실패 분기의 conventional 복구 예약·AI 실행권을 함께 결정한다는 더 좁은 검증이 필요하다. 이번 비교는 공개 초록 수준이며 구체적인 GPU 예약 알고리즘의 부재를 단정하지 않는다. [공개 초록](https://arxiv.org/abs/2609.07805)

**MobiUK의 AI-and-RAN 연구.** RAN MAC scheduler에서 여유 계산량을 제어하고, training의 행렬 연산을 다음 RAN slot을 막지 않는 단위로 나누며, site 사이에 micro-batch를 분배한다. 따라서 “작은 work unit으로 background를 제어한다”도 독립 novelty로 보기 어렵다. MobiCom 본논문과는 다른 발표이며 공개 자료가 1쪽이라 세부 알고리즘의 동일성까지 판단할 수는 없다. 다만 직접 경쟁하는 연구 방향이 공개됐다는 증거다. [공식 발표 초록](https://www.mobiuk.org/2026/abstracts/S2_On_Making_AI_and_RAN_Efficient_and_Safe.pdf)

**HAF.** 느린 placement/migration과 빠른 GPU/CPU allocation을 분리하고 RAN deadline을 위한 최소 할당량을 둔다. 공개 원고의 평가는 discrete-event simulation이다. 따라서 두 시간 규모 제어나 RAN 우선 자원 할당 자체를 새로움으로 삼기 어렵지만, 실제 CUDA 실행·전송·복구의 계약을 입증하는 과제와는 검증 수준이 다르다. 기존 DART 문서에도 등장한다. [원문, §III–IV](https://arxiv.org/html/2605.07547v1)

**Interplay/CAORA.** Interplay는 O-RAN orchestrator의 SAC agent가 latency-sensitive RAN과
AI 사이에서 MIG 자원을 동적으로 배분하며 INFOCOM 2025 workshop proceedings에 실렸다.
CAORA는 같은 계열의 구조에 KPI forecasting과 anomaly detection을 더하고 Barcelona 5G
traffic trace simulation을 사용한다. 따라서 dynamic AI-and-RAN resource allocation,
MIG repartitioning, RAN-priority fulfillment과 proactive orchestration은 SoftWall의 독립
novelty가 아니다. 두 공개 원고의 추상화는 workload-level MIG allocation이며, 같은 TB의
optional NeuralRx가 만든 future conventional recovery obligation, physical GPU fence가
붙은 external-AI lease, multi-home shared recovery calendar는 모델링하지 않는다.
[Interplay 원문·게재 상태](https://arxiv.org/abs/2503.07420),
[CAORA 원문](https://arxiv.org/abs/2507.09124)

**AI-RAN reference architecture.** 이 글은 AI-for-RAN, AI-on-RAN, AI-and-RAN을 정리하고
GH200에서 RAN과 AI workload의 concurrent proof-of-concept를 제시한다. 따라서 AI-RAN의
정의, 동일 infrastructure 실행과 utilization 동기 자체는 배경이다. Request-level recovery
certificate나 deadline-safe lease의 선행 구현으로 해석하지 않는다.
[원문](https://arxiv.org/abs/2501.09007)

**3. 인접 연구와 조사 범위**

| 연구 | 확인한 범위 / 관련성 |
|---|---|
| [Distributed AI Platform for the 6G RAN, OpenRan 2025](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/10/distributed_ai_ran.pdf) | 원문 §5–6: far-edge AI 앱이 CPU 시간·GPU 자원을 요청하고 runtime이 admission과 배치를 수행하며, srsRAN 공유 메모리 이벤트 경로를 평가한다. 따라서 RAN 근처 AI의 실시간 자원 admission 자체는 선행 연구다. 검토한 설계는 같은 PUSCH의 optional NeuralRx와 conventional 복구 interval을 함께 예약하는 문제와는 다름 |
| [EdgeRIC, NSDI 2024](https://www.usenix.org/conference/nsdi24/presentation/ko) | 공식 초록: RAN 시간 규모의 실시간 지능형 제어. GPU recovery 예약과 같은 문제로 취급하지 않음 |
| [Towards Energy Efficient 5G vRAN Servers, NSDI 2025](https://www.usenix.org/conference/nsdi25/presentation/kalia) | 공식 초록: RAN 부하·deadline slack을 이용하는 에너지 제어. slack 활용이라는 주장에 관련 |
| [Fair Resource Allocation in Virtualized O-RAN Platforms](https://arxiv.org/abs/2402.11285), [SIGMETRICS 2024 프로그램](https://sigmetrics.org/opentoc/sigmetrics24toc.html) | 초록·서지: 공유 O-RAN 계산 자원의 장기 공정성·자원 배분. optional NRx recovery의 직접 비교는 아님 |
| [SAGE 및 RANPilot, SIGCOMM 2026 공식 프로그램](https://conferences.sigcomm.org/sigcomm/2026/program/papers/) | 각각 UL 수요 예측·선제 무선 자원 배분과 O-RAN 재구성에 대한 AI 모델 적응. AI-RAN 논문이지만 서로 다른 제어 문제 |
| [REEF, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/han), [XSched, OSDI 2025](https://www.usenix.org/conference/osdi25/presentation/shen-weihang) | 일반 GPU/XPU 선점·스케줄링 비교 후보. RAN 특화 연구와 함께 검토해야 함 |
| [DARIS, DAC 2025 본논문](https://doi.org/10.1109/DAC63849.2025.11132423), [저자 공개 원고](https://arxiv.org/abs/2504.08795) | MPS·CUDA stream·stage 경계로 두 우선순위의 실시간 DNN을 배치하고 낮은 우선순위 작업을 admission함. 실험에서 높은 우선순위 miss 0을 관측했지만 저자도 보편적 deadline 보장은 주장하지 않는다. RAN PHY·동일 요청 NRx/conventional recovery는 다루지 않지만 MPS+bounded AI admission 자체의 정식 학회 선행 연구다 |
| [nvtaskset, ECRTS 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2025.21) | MPS와 MIG보다 강한 공간 계산 분할을 제시한 실시간 GPU 시스템. MPS active-thread cap을 물리적 hard isolation으로 설명할 수 없다는 경계와, 대체 partition baseline을 제공 |
| [GCAPS, ECRTS 2024 저자 원문](https://yidiwang.net/files/2024/ecrts24_gcaps_paper.pdf) | 실시간 GPU 작업의 context scheduling·preemption 비용을 모델링하고 driver runlist 변경 및 context switch overhead를 측정한다. 따라서 GPU **상태 전이 비용을 처음 발견**했다는 N3 주장은 불가. SoftWall의 CPU-sham 대 MPS client retirement 비교가 다른 단계의 인과인지 추가 분해해야 함 |
| [NVIDIA MPS architecture 공식 문서](https://docs.nvidia.com/deploy/mps/architecture.html) | client 연결/종료와 server 상태 전이·자원 정리를 명시한다. Confirm46의 deadline miss가 문서화되지 않은 새로운 현상인지, 알려진 client detach의 PHY deadline 파급을 새로 정량화한 것인지 구분해야 함 |
| [Ettus/OAI Neural Receiver testbed](https://kb.ettus.com/5G_OAI_Neural_Receiver_Testbed_with_USRP_X410) | Neural receiver와 traditional MMSE receiver를 runtime flag로 전환하고 conventional fallback mode를 제공한다. 따라서 NRx와 conventional 경로의 공존·전환도 단독 novelty가 아니다. 공개 설명에는 per-TB failure 뒤 multi-cell recovery interval을 executable schedule로 선예약하거나 이를 외부 AI lease와 원자 교환하는 계약은 확인되지 않음 |

“특정 학회에서 정확히 같은 제목의 논문을 찾지 못했다”는 것은 novelty 증거가 아니다. 특히 AI-RAN이라는 이름을 쓰지 않은 vRAN·O-RAN 논문도 포함해야 한다. preprint나 workshop/short 자료도 발표 등급과 별개로 이미 공개된 아이디어의 증거가 된다.

현재 공개 범위를 설계 축으로 비교하면 다음과 같다. `없음`은 논문이 약하다는 평가가 아니라
그 연구가 해결한 문제가 다르다는 뜻이다.

| 연구 | RAN+외부 AI 공유 | 요청별 accelerator/AI 경로 선택 | 같은 TB의 조건부 conventional debt | Multi-cell all-fail executable schedule | Recovery retime+외부 AI lease 원자 교환 | Local/remote transport credit을 물리 완료까지 추적 | Multi-home ownership·bounded control-fault 격리 | 여러 home의 shared mandatory-recovery calendar/path | Provenance 포함 predictive QSU/QSN/MI/UQ |
|---|---|---|---|---|---|---|---|---|---|
| Concordia | CPU 일반 작업 | 해당 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | 공개 범위에서 없음 | 없음 |
| YinYangRAN | 예 | 자원 비율/CPU fallback | 별도 형태 | 없음 | 없음 | 없음 | 공개 범위에서 없음 | 공개 범위에서 없음 | 공개 범위에서 없음 |
| CloudRIC | 이종 accelerator pool | 예 | 없음 | 없음 | 없음 | 일부 queue/placement | 공개 범위에서 없음 | accelerator pool은 있으나 same-TB recovery debt calendar는 없음 | 공개 범위에서 없음 |
| Interplay/CAORA | MIG 기반 AI-and-RAN | workload/MIG 배분 | 없음 | 없음 | 없음 | 없음 | orchestration은 있으나 동일 물리 transaction은 없음 | 공개 범위에서 없음 | workload 수요 예측은 있으나 recovery-debt envelope는 없음 |
| ARCHES | AI/conventional PHY | 예 | conventional expert 존재 | 없음 | 없음 | 없음 | 없음 | 공개 범위에서 없음 | 없음 |
| OCUDU | inline dApp/PHY | shape·class admission | conventional path와 late handling | 공개 평가에서 없음 | 공개 범위에서 없음 | CUDA event/lifecycle은 있음 | 공개 범위에서 없음 | 공개 범위에서 없음 | 공개 범위에서 없음 |
| DARIS | 우선순위 DNN 공유 | stage admission | 없음 | 없음 | 없음 | 없음 | 없음 | 공개 범위에서 없음 | 없음 |
| **SoftWall** | MIG-off MPS Qwen+PHY | max-radio+finite-ring endpoint admission | **요청별 recovery credit** | **예** | **예; AI45+commit7+guard2를 C145/C146 두 node에서 검증** | **C114--120 local IPC+최대 3 remote P2P** | **C123 ownership+C145--C148 single-token revalidation·pipelined 4-point fail-stop** | **C159 actual-NRx+C161 A0--A6 qualified-node fault** | **C162 exact16,023+physical180+64-debt certified scheduling** |

표의 마지막 세 열을 한 계약으로 연결한 점이 현재의 차별점 후보다. P2P, CUDA IPC,
fallback, MPS 또는 EDF 중 하나를 떼어 novelty라고 쓰면 이 표가 방어되지 않는다.

**4. 현재 스킴에서 줄여야 할 주장과 남는 가설**

| 약한 주장 | 이번 조사 후의 처리 |
|---|---|
| RAN과 AI를 같은 GPU에서 안전하게 공유 | YinYangRAN 등과 중복. 문제 배경으로 사용 |
| deadline·queue를 보고 GPU pool에 admission | CloudRIC 등과 중복. 기본 구성 요소로 사용 |
| 좋은 채널 조건에서만 NRx를 사용 | ARCHES와 비교해야 함. utility gate만으로 novelty 주장 불가 |
| conventional 경로·late-result 검사·epoch가 새롭다 | OCUDU 등과 비교 필요. correctness의 필요조건과 연구 기여를 구분 |
| background를 작은 단위로 나누고 남는 시간에 실행 | 최신 AI-and-RAN 자료까지 고려하면 단독 기여로 약함 |
| 한 셀의 예약된 recovery 시작 전 gap에 AI를 더 배치 | 실시간 slack 회수와 DARIS의 stage 기반 시간 분리의 자연스러운 적용이다. Confirm42의 52개 gap unit만으로는 부족했으나 Confirm43/44의 ON/OFF 두 seed·반대 순서에서 AI 순증 +0.419%/+1.304%를 관측했다. 두 실험 모두 NRx 30 ms 상한 gate 실패이므로 hard deadline 주장과 분리 |
| MPS와 우선순위·stage 경계를 결합해 높은 우선순위 deadline을 지킨다 | DARIS의 실시간 DNN 평가와 겹침. SoftWall은 radio utility/recovery 및 실제 PHY expiry를 공동 제어하는 차이를 입증해야 함 |
| 간섭을 반영하는 새로운 predictor | Concordia와 기존 DART의 online/profile 보정 대비 차이를 명시해야 함 |

남겨 둘 연구 가설은 다음과 같다. **예측이 흔들리는 공유 GPU pool에서 optional NRx의
radio 이득을 사용하되, 같은 요청의 conventional recovery와 LDPC/CRC가 PHY expiry 안에
끝날 수 있는 실행 가능한 자원을 먼저 확보한다. Local/remote endpoint와 외부 AI의 물리
완료까지 이 debt를 보존하면 transport와 failure outcome에 독립적인 안전 계약을 제공할 수
있고, 그 계약의 보수성·memory·work-unit 크기가 safe/useful/infeasible 경계를 결정한다.**

이 가설을 위 연구 중 하나가 같은 범위로 해결한 설계는 이번에 검토한 공개 본문에서 확인하지
못했다. 그러나 **요소의 조합 자체가 충분한 novelty라는 뜻은 아니다.** C113에서 원자
exchange의 strong work-conserving 대비 추가 처리량 우위가 없었으므로 성능 알고리즘 주장은
제거한다. 남는 독립 기여는 failure-contingent demand의 안전 불변식, 실행 가능한 certificate,
physical-credit lifecycle과 조건별 envelope를 예측·검증하는 데서 나와야 한다. 기존 DART의
recovery calendar와 lease는 내부 baseline으로 명확히 분리한다.

C127은 여기에 좁은 fault semantics를 추가한다. Global AI commit이 broker에는 적용됐지만
home이 reply를 못 받은 경우, request를 재시도하지 않고 affected home의 새 AI admission만
닫아 local RAN과 다른 home을 계속 실행한다. 검토한 AI-RAN 선행 연구의 공개 본문에서는
same-TB recovery certificate와 이 global ownership ambiguity를 함께 다룬 설계를 확인하지
못했다. 다만 이는 일반 distributed exactly-once protocol의 최초성 주장이 아니며, broker
restart와 durable reconciliation은 아직 구현하지 않았다.

C129/C132는 이 차이를 더 좁고 검증 가능하게 만들었다. C129에서 Qwen physical return 뒤
무제한 global `complete` RPC가 recovery를 D155 밖으로 밀었다. C132는 synchronous
`prepare`, `commit`, `complete` 각각을 5 ms로 제한하고 총 15 ms를 AI admission에
포함해 broker process fail-stop 뒤에도 양 home의 local RAN을 계속했다. 따라서 새로움의
대상은 일반적인 RPC timeout이 아니라, **same-TB conditional recovery certificate와
multi-home external-AI ownership protocol의 전체 control path를 하나의 horizon inequality로
합성하고 실패 반례에서 그 필요성을 검증한 것**이다. 일반 분산시스템의 fail-stop 또는
at-most-once 기법 자체에 대한 최초성은 주장하지 않는다. C133은 같은 full-budget mode에서
post-prepare와 post-complete를 두 arm씩 추가해, C132 post-commit과 함께 세 synchronous
control point를 각각 두 번 검증했다.
C134는 같은 과거 contract와 runtime을 다른 allocation·A100 node에서 세 지점 각각 두 arm으로
재실행해 16,320 TB의 safety 위반 0을 얻었다. C138이 이후 `timeout=wall bound` 가정을
반증했으므로 이 결과는 fail-closed와 radio continuity의 물리 기록으로 보존하되 15 ms timing
자격으로 사용하지 않는다. C139는 7 ms admission bound로 세 control point를 다시 검증했다.

Envelope V9은 cross-home slack 오류를, V10은 용량 기하와 결정시각의 혼동을 교정했다.
이어 V10이 실제 `ring_depth=1` endpoint를 반복 completion slot처럼 계산하고 executor
phase를 합친 오류를 발견했다. V11은 finite ring과 rejected-recovery ordering을 mode
vector에 넣었다. 독립 exact oracle은 39개 endpoint pool과 10,920개 작은 상태에서
deadline-slot matching의 수락 수와 checker가 일치함을 확인했다. 이 자기반증과 교정은
선행 연구 부재의 증거가 아니라, SoftWall envelope가 실제 physical credit을 추상화한다는
근거다. C138의 반례 뒤 V13은 과거 AI40 후보를 UQ로 내리고 AI35를 새 경계 class로
예측했다. C139는 `T_sock=5`, `B_rpc=7`의 fault path를, C140은 완전한 58 ms transaction을
물리 검증했다. 새 optimizer를 추가한 것이 아니라, 모델이 자신의 false-safe 가정을
반증하고 수정된 service class를 다시 예측·실행한 결과다.

V14는 여기서 control 경로의 구조 자체를 다시 모델링했다. 모든 RPC를 동기 합산하는 대신
ownership maintenance는 RAN 경로 밖에서 진행하고, physical launch를 허가하는 commit만
현재 certificate horizon에 청구한다. Prepared token은 실행권이 아니고 current horizon이
짧을 때도 request deadline까지 staged로 유지되며, launch-time revalidation을 통과해야
한다. 첫 finite model은 네 operation의 before/after-apply failure를 전수했고 C142--C144는
두 node에서 AI45 class와 세 fault point를 검증했다. 이후 two-request 반례가 untracked
held token을 찾아 V14를 UQ로 내렸고, V15는 single-token invariant와 4-state/8-edge
ownership model을 추가했다. C145/C146은 이 구현 분기를 두 새 node에서 직접 계측했다.
C147/C148은 abort 적용 뒤 ACK만 사라지는 남은 분기를 두 추가 node에서 계측했다.
검토한 공개 AI-RAN 본문에서는
optional same-TB recovery debt, executable multi-cell all-fail schedule, external-AI ownership,
launch-time revalidation과 physical fence/quarantine을 이 형태로 함께 다룬 설계를 확인하지
못했다. 이 문장은 공개 범위의 조사 결과이며 유사 연구의 절대 부재를 뜻하지 않는다.

V17은 disjoint home이라는 기존 합성 가정을 더 일반화한다. Local calendar가 각각
feasible이어도 한 shared recovery lane에서는 infeasible인 상태가 25개 조합 중 10개였고,
global exact certificate는 capacity1/2의 3,400개 variable state에서 독립 oracle와 차이0이었다.
C151/C152는 두 home의 실제 PUSCH를 한 GPU의 persistent cuPHY conventional worker로 보내는
P2P path를 두 node·1,000 request에서 오류0으로 실행했다. 이 결과는 표의 마지막 열을
model과 data-path 수준에서 지지한다. C153은 global fifth reject, two-success Qwen35
lease/fence와 certificate-ordered recovery 두 건을 D155 아래 한 runtime transaction으로
처음 결합했다. C154/C155는 같은 source를 두 새 node에서 독립 seed와 반대 branch 순서로
실행해 여덟 arm, global reject4, Qwen lease4, physical recovery20, correct home commit20,
deadline miss0을 얻었다. 이로써 controlled-outcome integrated path는 두 노드 finite-sample
gate를 통과했다. 이후 C159-Q1/Q2가 actual TensorRT NeuralRx를 P180에서 두 node·actual
NRx4,800까지 반복했고, C161은 A0--A6 physical fault에서 actual NRx2,800·recovery942·
commit2,800·miss0을 얻었다. 같은 source에서 NRx45를 깬 한 node/lifecycle은 UQ로 보존한다.

C156/C156b는 host 기록에 머물지 않고 두 node의 shared-mode GPU timeline을 검사했다. 이 과정에서 fixed
lease가 launch control 시간을 누락한 반례를 먼저 만들었고, 실제 decision 시각에
`control5+AI35`를 선납하는 V17.1로 수정했다. 교정 arm은 Qwen/recovery kernel 합계1,330,
금지 overlap0 ns와 certificate/transport phase 순서를 통과했다. 이 반례→수정→물리 trace
연결이 표 마지막 열의 실행 계약을 더 강하게 지지한다.

C162는 이 계약의 실행 가능 영역을 결과 뒤 설명하는 데 그치지 않고 사전에 분류한다.
Debt 수, 실제 decision time, Qwen class와 bound/lifecycle/node provenance를 입력으로
QSU/QSN/MI/UQ를 만들었다. Qualified state16,023개의 exact 판정 차이0, 기존 physical
decision1,200/1,200 일치, 두 새 node의 prespecified boundary180/180 일치를 얻었다.
Candidate scheduler와 independent verifier는 64 debt까지 반환 certificate 실패0과
decision p991.494ms를 보였다. 검토한 공개 연구에서 same-TB conditional recovery debt,
all-fail executable calendar, external-AI lease와 provenance-qualified predictive envelope를
이 조합으로 평가한 사례는 확인하지 못했다. 이는 공개 범위의 비교이며 절대 부재 증명이
아니다.

핵심적인 차이의 후보는 “NRx가 timeout되면 fallback 함수를 호출한다”가 아니다. 여러 셀이 동시에 fallback을 요구할 수 있으므로, 그때 쓸 **실행 시간과 후처리·전송 여력**까지 이미 확보되어 있어야 한다. 이 예약을 과도하게 잡으면 공유 이득이 사라진다는 비용까지 연구 대상이다. 단, 모델로 추정한 비용 상한을 검증된 WCET로 간주해서는 안 되며, timing bound·실패 가정 밖의 GPU/driver 정지까지 보호한다고 주장하지 않는다.

**5. 기존 기법을 결합한 강한 baseline부터 만든다**

비교 대상을 priority-only나 무제어 MPS에 한정하지 않는다. 다음 두 축을 모두 포함한다.

1. **실행 가능한 선행 구현/정책:** YinYangRAN의 할당 정책, CloudRIC의 queue-aware 선택, ARCHES의 채널 기반 선택 중 실험 환경에 이식 가능한 부분을 평가한다. 원본 실행과 정책 adaptation을 명확히 구분한다. 서로 다른 원래 문제·하드웨어의 수치를 직접 대조하지 않는다.
2. **결합 baseline:** 같은 utility 정보·queue 정보·GPU 수·buffer 안전장치를 주고, 채널 gate + queue-aware 배치 + 고정 recovery 여유 + 고정 work-unit background 제어를 결합한다. 각 제어기를 충분히 튜닝한다. SoftWall은 이것과 같은 deadline/utility 요구를 만족하면서 joint admission/reservation/lease 결정의 추가 이득을 보여야 한다.

결합 baseline은 특정 논문의 원본 구현이 아니라 우리가 정의하는 강한 비교 구성이다. 안전장치를 일부러 빼서 실패하게 만든 구성은 메커니즘 ablation으로만 사용한다. 원본 논문의 잘못으로 해석하지 않는다. 보수적 reservation이 이득의 전부인지, adaptive predictor나 joint policy가 더 필요한지도 분리한다.

**6. 진행 전에 확인할 핵심 실험**

| 순서 | 질문 / 실험 | 얻어야 할 증거 |
|---|---|---|
| 0 | 실제 release→전송→NRx→LDPC/CRC→commit과 conventional 경로가 deadline에 들어오는가? | [타당성 검토](RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md)의 P0를 먼저 통과. 실제 PHY expiry와 실험용 timeout을 구분 |
| 1 | 여러 셀의 NRx tail이 동시에 늘어나면 recovery가 밀리는가? | 독립/고정 예약 대비 joint 예약의 효과; recovery burst와 예약 비용을 함께 측정 |
| 2 | 동일 무선 입력에서 GPU 간섭·queue만 바뀌면 어떤 선택이 이득인가? | radio-only gate 및 queue-only placement로 충분하지 않은 구간과 원인을 제시 |
| 3 | 늦은 DMA·kernel이 다음 요청 버퍼에 영향을 주는가? | 물리 완료 전 재사용 금지, 만료 결과 격리, 중복 commit 방지. correctness 증거로 분류 |
| 4 | 같은 GPU 예산·같은 miss 허용치·같은 radio utility 하한에서 AI가 얼마나 더 처리되는가? | 전용 배치·결합 baseline·기존 DART·v0·v1의 비교. 거부된 요청을 포함한 전체 offered load 기준 |
| 5 | joint 제어와 predictor의 각각의 이득은 무엇인가? | fixed/adaptive predictor × independent/joint control을 분리한 ablation; 예약했지만 쓰지 않은 GPU 시간도 보고 |

측정은 timely CRC-fail과 계산 deadline miss를 분리하고, correct-TB/유효 bit, NRx 이득, background 유효 처리량, fallback burst, buffer/credit 점유, sample 수·반복·불확실성을 함께 보고한다. 하나의 미세한 latency 개선만으로 연구 가설이 입증됐다고 보지 않는다.

**진행 판단:** 같은 GPU strong baseline, 최대 4-GPU endpoint, 4-home/12-cell composition,
global queue, certificate executor, V16-qualified single-token pipelined broker control과
AI45 conditional class의 four-point 물리 검증까지 완료됐다. V17 non-separable certificate와
shared cuPHY/P2P path, controlled two-node integration과 V17.1 two-node GPU timeline도 완료됐다.
다음 핵심 gate는 actual-NRx outcome과 integrated fault holdout,
실제 DU `d_MAC`, V16 mode의 cross-family/production-mode 재자격이다. 성능 우위는 추가하지 않고 substrate의 적용 조건과 비용을
결과로 남긴다. “AI-RAN 최초”, “유사 연구 없음”은 사용하지 않고, 공개 본문 범위에서
확인되지 않은 정확한 계약만 주장한다.
