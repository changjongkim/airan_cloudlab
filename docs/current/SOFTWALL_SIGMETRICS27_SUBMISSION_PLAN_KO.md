# SoftWall SIGMETRICS 2027 제출 전환 계획

**기준일:** 2026-09-25  
**권고 track:** Systems primary, Measurement & Applied Modeling secondary  
**공식 일정:** [SIGMETRICS 2027 CFP](https://www.sigmetrics.org/sigmetrics2027/pages/cfp.html) Winter 기준
abstract 2027-01-04 23:59 AoE, paper 2027-01-11 23:59 AoE  
**재확인:** 2026-09-25 UTC, notification 2027-03-10과 conference 2027-06-07–11 Atlanta까지
[기계 판독 snapshot](../../results/softwall_multigpu/sigmetrics27_cfp_recheck_v1.json)에 기록  
**형식:** CFP의 `\documentclass[acmsmall, screen, review]{acmart}`에 double-anonymous용
`anonymous` option을 추가한 review 원고, technical content 최대 20쪽, references 무제한,
correctness용 appendix 길이 제한 없음

## 1. Venue fit 판정

SIGMETRICS Systems track은 구현·실증 전에 system modeling과 performance analysis를 요구한다.
SoftWall은 이 요구와 직접 맞는다.

| SIGMETRICS 요구 | SoftWall 대응 |
|---|---|
| Well-motivated systems problem | MPS 공유 GPU에서 optional NRx가 만드는 future mandatory recovery debt |
| Principled model | all-fail dominance, global shared-recovery certificate, decision-window condition |
| Implementation | MPS, actual TensorRT NRx, Aerial/cuPHY, Qwen, CUDA IPC/NVLink P2P |
| Model validation | exact16,023, retrospective1,200, prespecified physical boundary180 |
| Performance analysis | QSU/QSN/MI/UQ envelope, debt64 scheduler latency, lifecycle provenance |
| Negative/diagnostic insight | strong oracle headroom0, false-safe counterexamples, UQ mode 분리 |

NSDI는 systems aspects of networking hardware/PHY를 일부 허용하지만 GPU resource scheduling과
PHY 자체를 명시적으로 제외한다. 현재 원고는 네트워크 stack contribution보다 model/measurement
비중이 크므로 SIGMETRICS가 더 자연스럽다.

## 2. 제출용 한 문장

> SoftWall turns unresolved optional NeuralRx executions into executable conventional-recovery debt,
> preserves that debt through an atomic physical GPU lifecycle, and predicts the safe/useful/infeasible
> envelope of MIG-off AI-RAN sharing before execution.

제출 원고에서는 증거 강도 순서로 기여를 제시한다.

1. 새 systems contract: 측정 반례에서 도출한 certificate-to-physical-execution refinement
2. 새 modeling result: provenance-qualified predictive envelope
3. 위 둘을 표현하는 AI-RAN 특화 future same-TB mandatory debt

Conditional debt 자체를 새로운 일반 primary/backup scheduling 이론으로 주장하지 않는다.

## 3. 20쪽 body 배분

| Section | 목표 쪽수 | 핵심 내용 |
|---|---:|---|
| Abstract | 0.4 | 문제, 세 기여, Q2/C161/C162 핵심 수치, 음성 성능 결과 |
| Introduction | 1.6 | motivating example, local-safe/global-unsafe 반례, contributions |
| Background/Motivation | 1.5 | MPS 경계, optional NRx semantics, threat model |
| Failure-derived requirements | 1.5 | 일곱 측정 반례와 refinement chain |
| Model | 3.0 | debt, certificate, Lemma 1/1c, atomic state, decision-window theorem |
| Design | 3.0 | admission, outcome batch, lease/fence, ownership/fault, provenance |
| Implementation | 1.5 | topology, workers, IPC/P2P, instrumentation, verifier |
| Evaluation | 6.0 | MPS-only, Q2, C161, C162, necessity witness, scalability, oracle, lifecycle |
| Related Work | 1.5 | 세 cluster와 claim matrix |
| Limitations/Conclusion | 1.0 | production timing, WCET, cross-family, UQ, no throughput claim |
| **현재 컴파일** | **15.0** | references 포함; technical body 20쪽 제한 내 여유 |

Appendix에는 상세 proof, 전체 fault/lifecycle matrix, protocol/hash, 실패 history를 넣는다.
핵심 novelty와 가장 강한 evidence는 body에서 빠지면 안 된다.

## 4. Body에 필요한 도표

| ID | 도표 | 목적 | 상태 |
|---|---|---|---|
| Fig. 1 | System contract + all-fail/conditional timeline | 문제와 SoftWall transaction을 첫 1분 안에 설명 | SVG/PDF/PNG 완료 |
| Fig. 2 | Certificate-preserving state transition | common-cutoff batch, replan, launch, fence, retire | manuscript diagram 필요 |
| Fig. 3 | QSU/QSN/MI/UQ envelope + scalability | 88/89 boundary와 debt64 control cost | C162 figure 존재 |
| Fig. 4 | End-to-end evaluated topology | 두 home, GPU3 NRx, GPU2 cuPHY+Qwen, IPC/P2P | Fig.1과 병합 가능 |
| Fig. 5 | Model vs physical boundary outcomes | 180/180 match와 각 case 결과 | 기존 JSON에서 작성 필요 |
| Table 1 | Closest-work claim matrix | broad sharing과 exact conditional contract 구분 | 완료 |
| Table 2 | Counterexample→refinement chain | 단순 결합이 실패하는 이유 | 완료 |
| Table 3 | Q2/C161/C162 headline results | 평가 전체를 한눈에 연결 | 원고에 일부 완료 |

색상만으로 상태를 구분하지 않고 hatch, label, marker를 함께 사용한다.

## 5. 제출 전 필수 문서 작업

1. Markdown 초고를 anonymous `acmart` LaTeX로 옮긴다.
2. Lemma 1, shared-recovery necessity, decision-window sufficiency를 정식 번호와 가정으로 쓴다.
3. 모든 평가 문장에 artifact citation을 연결하고 claim audit를 CI처럼 실행한다.
4. Related work bibliography를 DOI/공식 proceedings 기준 BibTeX로 변환한다.
5. Methods에 generative-AI 사용 범위와 저자의 검증 책임을 공개한다.
6. Introduction contribution paragraph 뒤에 artifact release 계획을 한 문단 넣는다.
7. Double-anonymous 관점에서 node/job/user path와 소속을 드러내는 표현을 제거한다.

위 일곱 항목은 현재 모두 구현·감사됐다. 남은 편집은 prose와 figure 배치의 reviewer 검토다.

## 6. Artifact/reproducibility 문구 초안

> Upon acceptance, we plan to release the SoftWall controller and verifier, finite-state and exact
> model checkers, experiment protocols and analyzers, sanitized raw timing traces, and the manifests
> that bind source and result artifacts. GPU binaries or models whose redistribution is restricted will
> be accompanied by build and acquisition instructions. Production DU traces are not part of the
> artifact because they were not available for this study.

실제 공개 가능 라이선스와 NVIDIA component 재배포 조건은 camera-ready 전에 확인한다.

## 7. 제출 전 중단/추가 실험 규칙

새 실험은 다음 세 경우에만 연다.

1. 원고의 핵심 theorem 가정과 구현 assertion 사이에 대응하지 않는 항목이 발견됨
2. Headline 수치를 뒷받침하는 권위 artifact가 없거나 서로 모순됨
3. Reviewer-style audit에서 현재 claim 범위 안의 false-safe 가능성이 구체적으로 발견됨

`recovery-first`는 초기 all-fail admission 뒤 unresolved recovery를 먼저 처리하는
certificate-preserving 기준선이다. 이를 failure correlation로 깨는 실험은 안전 비교군의
정의를 바꾸므로 열지 않는다. C162 E4/E6b의 두 prespecified QSN 상태는 debt-blind
current-idle admission이 radio guard를 12/1ms 넘길 수 있음을 보이고, SoftWall은 두 node에서
합계 60/60회 GPU launch 전에 거절했다. 상세 판정은
[필요성 반론 감사](../archive/SOFTWALL_NECESSITY_GAP_DECISION_KO.md)에 있다.

Production `d_MAC`, cross-family, lifecycle 나머지 5개를 단순히 채우기 위한 campaign은 열지
않는다. 이들은 현재 원고의 limitation/future work다.

### Reviewer risk 우선순위

| 반론 | 위험 | 현재 답 | 마감 전 조치 |
|---|---|---|---|
| Synthetic `P180/D155` | 높음 | `P`는 slot이 아닌 harness release period; C163 bridge/validator 24 tests PASS, 실제 trace UQ | Abstract와 Limitations에서 production 보장 제외 및 재자격 절차 명시 |
| 외부 TDL-A에서 NeuralRx 0/10 | 높음 | 두 canary 모두 conventional 10/10, NeuralRx 0/10; synthetic path만 qualified | 외부-channel gain 제외, compatibility 해결 뒤 같은 envelope gate 재실행 |
| Strong baseline 대비 처리량 0 | 중간 | 385,262-token 동률로 optimizer claim 기각 | 안전성, 사전 mode 분류와 launch rejection을 실무 가치로 제시 |
| Necessity가 `B_conv=25 ms`에 의존 | 중간 | E6b는 24ms, E4는 19ms가 경계; Q2 표본 최대13.465ms를 가상 bound로 쓰면 두 고정 witness 소멸 | 보수성을 공개하고, 양의 recovery charge에서는 false-safe decision window가 이동함을 Proposition과 연결 |

첫 두 항목은 실제 DU trace와 NeuralRx input-contract 호환성이라는 외부 의존이다. 현재
데이터로 해결한 것처럼 쓰지 않는다. 마감 전에는 새 workload/deadline을 골라 성능 이득을
만드는 실험 대신 claim boundary와 재현 가능한 전환 절차를 선명하게 유지한다.

## 8. 현재 제출 readiness

| 항목 | 상태 |
|---|---|
| Claim/evidence freeze | 완료 |
| 영문 full-structure draft | 완료 |
| Machine-readable claim audit | 13/13 PASS, necessity witness 포함 |
| Novelty/closest-work matrix | 완료, Interplay/CAORA 추가 |
| Fig.3 envelope/scalability | 완료 |
| Fig.1 system contract | 완료 |
| LaTeX/acmart conversion | 완료, 익명 review 원고 15쪽 |
| BibTeX 정규화 | 완료, 인용 16개 모두 resolve |
| Anonymous/reproducibility/AI-disclosure/CFP audit | 26/26 PASS |

현재 새로운 optimizer·lifecycle 증거를 더 늘리는 단계는 끝났다. P2 frozen 개발 gate는
persistent-input/stream-ordered path에서 995/1,000으로 실패해 holdout을 열지 않았다. Winter
제출 전 production 확장은 양 radio path의 native cuPHY fast path와 P1 외부 trace 수용으로
제한한다. 그 결과와 무관하게 원고의 claim boundary, 도표와 bibliography를 정확히 고정한다.
