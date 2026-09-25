# SoftWall 현재 목표 완료 감사

**상태:** 2026-09-24, C145--C148 V16 four-point single-token fault 자격까지 반영  
**현재 논문축:** certified conditional-recovery substrate + feasibility envelope  
**대체된 과거 감사:** [C91까지의 원문](../archive/SOFTWALL_COMPLETION_AUDIT_THROUGH_C91_KO.md)

## 판정

현재의 좁은 시스템 주장은 **구현과 유한 표본 검증까지 완료**됐다.

> SoftWall은 MIG를 사용하지 않는 MPS 기반 AI-RAN에서 optional per-TB NeuralRx가
> 남기는 multi-cell all-fail conventional recovery 의무를 executable certificate로
> 유지한다. Recovery·endpoint/transport·external-AI credit을 원자적으로 갱신하고,
> global ownership token을 launch 직전에 현재 certificate로 재검증하고 bounded commit
> 비용을 admission에 포함해, 자격화된 단일·다중 GPU
> mode에서 RAN 복구를 보존하면서 bounded Qwen 작업을 남는 구간에 실행한다.

이 판정은 production hard-real-time 보증이나 새 optimizer의 처리량 우위를 뜻하지 않는다.
실제 DU d_MAC, 방어 가능한 WCET와 cross-family 일반화가 없으므로 **탑티어 제출용 전체
실증은 아직 완료가 아니다.**

## 사용자 목표와의 대조

| 목표 | 현재 증거 | 판정 |
|---|---|---|
| MIG OFF, MPS에서 실제 PHY와 AI 동시 실행 | C68/C70 kernel overlap, C100/C113 Qwen, C114 remote NRx | 완료 |
| 허용된 AI 작업 안에서 RAN deadline 보호 | mode별 bound, all-fail certificate, C80 4셀, C113 strong baseline | synthetic qualified mode에서 완료 |
| NeuralRx 실패 시 conventional 복구 보존 | C73/C76/C80 atomic multi-credit, C101 conditional cost | 완료 |
| 남는 자원에 AI 작업 허가 | bounded AI lease와 physical fence, BurstGPT→Qwen replay | 완료 |
| MPS만으로 안전하다는 주장 배제 | C26 deadline miss, C46 lifecycle 대조 | 완료 |
| Multi-GPU 확장 | C114 P2P NRx, C116 3 endpoint, C119/120 4 endpoint, C121/122 8/12셀 sharding | 제한된 한-node mode 완료 |
| Global AI request ownership | C123 unique lease, C126 certificate-order execution | 완료 |
| Broker fail-stop가 RAN을 막지 않음 | V14의 timing/fault 증거 뒤 ownership 반례를 교정; C145/C146 V15 fault6 arm·9,600 TB·post-fault7,463 TB·max token1·위반0 | V15 bounded volatile mode 완료 |
| 복잡한 optimizer의 추가 성능 | C78/79, C96, C102, C113, C115, C124/126 | 우위 없음으로 판정 종료 |
| Production DU deadline 보장 | 실제 IQ-ready→FAPI/MAC consumption 계약 없음 | 미완료 |
| 다른 node/GPU 일반화 | V15가 새 `nid001244`와 `nid003417`에서 재현 | same-family 두 node 통과; cross-family는 미완료 |

## 완료된 논리 사슬

1. **반례:** MPS cap과 priority는 동시 실행을 제공하지만 deadline 상한은 만들지 않는다.
2. **도메인 의무:** Optional NeuralRx는 결과가 확정되기 전까지 같은 TB의 conventional
   recovery debt를 남긴다.
3. **안전 불변식:** 모든 미해결 NeuralRx가 실패해도 각 expiry 전에 실행 가능한
   conventional schedule이 항상 남아 있어야 한다.
4. **원자 상태 전이:** Recovery retiming과 AI lease는 함께 commit하거나 둘 다
   이전 상태에 남는다.
5. **실행 refinement:** Ready recovery는 계산된 certificate 순서대로 실행한다. C125가
   임의 실행 순서의 반례를 제공했고 C126이 이를 수정했다.
6. **물리 lifecycle:** CUDA/IPC/P2P 작업은 completion fence 전까지 credit을 점유한다.
7. **분산 합성:** Recovery home이 분리된 경우 local certificate의 곱과 global AI
   request ownership을 합성한다.
8. **제어 경로 예산:** AI 전에 호출되는 prepare/commit/complete의 전체 상한도 recovery
   horizon에서 차감한다. C129는 이를 빼면 false-safe가 됨을 보였다.
9. **Fault containment:** C132/C133은 broker가 세 state transition 중 어느 지점에서
   reply 전에 죽어도 global AI를 닫고 local RAN을 계속하는 bounded mode를 검증했다.
10. **독립 재자격:** C134는 같은 contract를 다른 A100 node에서 세 fault point 각 두 arm으로 통과했다.
11. **Envelope:** 시간, memory, executor, lifecycle, transport, control fault와 budget을
    mode 축으로 분리해 QSU/QSN/MI/UQ로 판정한다.
12. **모델 자기감사:** V4의 cross-home conditional-slack 오류를 비대칭 반례로 찾아 V5에서
    home별 NRx 수락 수에 연결했다. V10이 실제 `ring_depth=1`을 무제한 completion slot로
    취급하고 executor phase를 합친 오류도 V11에서 교정했다. V11이 별도 AI completion
    guard 2 ms를 `B_eff`에서 빠뜨린 문제는 V12에서 교정했다.
13. **결정시각 감사:** `W+Delta`는 용량 필요조건일 뿐이다. V8 checker는 observe-all NRx
    bound, finite ring, exchange 전 실제 실행한 recovery, control budget과 AI completion
    guard를 모두 차감한다. Exact oracle은 39개 pool과 10,920개 bounded 상태에서 endpoint
    수락 cardinality와 일치했다.
14. **예측 후 물리 검증:** V12는 full-control mode에서 AI40+control15+completion-guard2의
    완전한 57 ms transaction이 58 ms decision window에 들어가는 두 후보를 유지했다.
    C135는 protocol을 사전 동결한 뒤
    독립 두 arm에서 정확한 후보 분기 288회와 AI40 exchange 5회씩, 2,560 TB 안전 위반 0을
    확인했다. 사건별 감사는 실제 10개 교환 모두 static margin −4 ms, conditional margin
    +1 ms와 physical guarded-horizon 통과를 확인했다.
15. **Mechanism cross-node 재자격:** C136은 기존 node를 제외한 `nid002817`에서 새
    seed/process lifecycle 6개 arm을 통과했다. C135와 합친 두 node·8 arm은 10,240 TB,
    candidate branch 1,196회, AI40 exchange 38회와 선언 위반 0이다. Node-level 통계 상한은
    여전히 약하므로 same-family 일반화나 WCET로 해석하지 않는다.
16. **Control-budget 물리 계측:** C137은 기존 source를 보존한 instrumentation mode에서
    세 번째 A100 node의 6 arm을 실행했다. Prepare/commit/complete 5,993회가 모두 5 ms
    안에 반환됐고, 같은 7,680 TB·AI40 exchange28의 safety 위반도 0이었다. 계측 mode와
    비계측 mode는 exact reliability 표본으로 합치지 않는다.
17. **반례 기반 control correction:** C138은 faulting prepare/complete가 5.205/5.831 ms에
    반환됨을 보여 5 ms timeout=wall bound 가정을 기각했다. C139는 5 ms timeout과 7 ms
    admission bound를 분리해 6 arm·16,320 TB·RPC2,020에서 7 ms 초과와 safety 위반 0을 얻었다.
18. **Corrected class 예측 검증:** V13은 AI40을 제거하고
    `AI35+control21+completion2=58 ms`를 exchange-only로 예측했다. C140 두 arm은
    target branch309와 AI35 exchange8에서 static margin−5/conditional margin0을 확인했다.
19. **Corrected exact-mode 독립 재자격:** C141은 이전 corrected node를 제외한
    `nid001372`에서 control 6 arm과 AI35 두 arm을 통과했다. C139/C140과 합친 두 node의
    control RPC4,030회는 최대5.775660 ms로 7 ms 초과0이었고, AI35 exchange15와 선언
    safety 위반0을 얻었다.
20. **Pipelined control 모델:** V14는 prepare/abort/complete를 RAN 경로 밖으로 옮기고,
    launch-time certificate revalidation 뒤 bounded commit만 동기화한다. Finite model은
    16 state·20 transition의 before/after-apply ambiguity에서 invariant 위반0을 얻었다.
21. **AI45 conditional class 검증:** `AI45+commit7+guard2=54 ms`는 static slack53 밖,
    conditional window58 안이다. C142 두 arm은 branch305·exchange9·2,560 TB에서 위반0이었다.
22. **Pipelined fault와 독립-node 재자격:** C143은 fault6 arm에서 deferred RPC 7 ms 초과50을
    실제 만들고도 16,320 TB·post-fault14,184 TB의 safety/duplicate 위반0을 얻었다. C144는
    독립 node에서 AI45 한 arm과 세 fault point를 재자격했다. 두 node 합계는 AI45 exchange11,
    fault9 arm·21,120 TB·post-fault17,914 TB다.
23. **Ownership 반례:** V14 two-request 회귀는 broker-held token 2개 중 client-tracked token
    1개인 상태를 재현했다. 관측 campaign은 drain됐지만 branch gate가 없었으므로 V14 mode를
    UQ로 강등했다.
24. **V15 model→physical closure:** V15는 staged/offered token 뒤 prepare를 금지한다.
    Model v2의 fault16/20+ownership4/8 상태에서 violation0을 확인했고, C145/C146 두 새
    node의 8 arm·12,160 TB·AI45 exchange12·fault 뒤7,463 TB에서 prepare 억제537회,
    maximum unlaunched token1, safety·duplicate 위반0을 얻었다.
25. **V16 four-point closure:** V15 물리 matrix에서 빠진 abort post-apply reply-loss를
    C147/C148의 서로 다른 두 새 A100 node에서 주입했다. 3,200 TB·post-fault2,662,
    target 실행0·prepare 억제129·max token1·위반0이다. C145--C148 결합은 네 operation별
    arm2, 노드4개, radio15,360·post-fault10,125·억제666이다.

## 핵심 증거

| 범주 | 가장 강한 현재 결과 |
|---|---|
| Runtime correctness | 새 A100 allocation에서 Aerial155+Qwen3+DART30, 총188개와 fault regression 10,000회 PASS |
| 4셀 atomic recovery | C80, 동시 all-fail→4 conventional 37회, 위반 0 |
| 실제 GPU concurrency | C68 AI–NRx 8.932 ms, C70 conventional–NRx 12.309 ms overlap |
| Trace strong baseline | C113 10 arm safety/radio parity PASS; 추가 처리량 gate FAIL |
| Multi-GPU transport | C114 P2P payload 10,000회 오류 0, actual remote NRx 1,000/1,000 |
| Multi-home scale | C122 4 GPU/12셀 8,160 TB, 위반 0 |
| Certificate executor | C126 21,760 TB, exchange 2,496, safety 위반 0 |
| Full control fault matrix | C132/C133 첫 node 6 arm과 C134 독립 node 6 arm, 총32,640 TB, 위반 0 |
| Envelope v14/C142 | 당시 QSU6/QSN0/MI3/UQ11, timing/fault arm 통과; 후속 ownership 반례로 현재 V14 UQ |
| Corrected control telemetry | C138 5 ms FAIL; C139 새 node·6 arm, RPC2,020회 max5.653ms<7ms, 16,320 TB·위반0 |
| Corrected V13 독립 재자격 | C141 새 node·8 arm·18,880 TB; control RPC2,010 max5.776ms<7ms, AI35 exchange7, 위반0 |
| V15 single-token pipelined control | C145/C146 두 node; 12,160 TB, AI45 exchange12, fault6 arm·post-fault7,463, 억제537·max token1, safety/duplicate 위반0 |
| V16 four-point physical fault | C145--C148 네 node; prepare/abort/commit/complete 각2 arm, 15,360 TB·post-fault10,125·억제666·max token1·위반0 |

## 확정적으로 종료한 연구축

- Exact/joint optimizer가 max-radio를 이긴다는 주장은 C102의 39/39 동일 결과로 종료했다.
- SoftWall이 safe work-conserving baseline보다 처리량을 높인다는 주장은 C113/C115의
  사전 +2% 및 CI gate 실패로 종료했다.
- NRx 개수를 AI slowdown scalar로 쓰는 모델은 C103–105 부호 불일치로 기각했다.
- Global routing 처리량 우위는 C124/C126의 작은 효과와 comparison gate 실패로 주장하지
  않는다.

이 음성 결과는 실패한 연구축을 숨기지 않고 현재 기여를 optimizer가 아닌 substrate와
feasibility envelope로 한정하는 근거다.

## 아직 남은 제출 gate

### 1. Production timing — 최우선

실제 DU에서 t_IQ_ready, PHY submit, CRC/FAPI publish, MAC consume를 같은 clock으로
계측하고 request별 d_MAC을 정의해야 한다. 현재 P180/D155는 synthetic contract다.
이 자료가 없으면 production deadline 또는 hard-real-time이라고 쓰지 않는다.
G7 입력 schema와 validator, 9개 단위시험은 완료했으며 상태는 UQ_NO_PRODUCTION_TRACE다.

### 2. Service-bound 방법론

현재 bound는 exact warm mode의 유한 표본 qualification이다. C138은 fault path의 5 ms
가정을 기각했고 V16-qualified V15 runtime은 동기 commit 7 ms만 critical budget으로 유지한다. Cold start, GC,
MPS client restart, driver 상태를 별도 mode로 분리하고, 충분한 독립 반복과 분석 가능한
강제 상한 또는 명시적 확률 계약을 제시해야 한다. 표본 최대를 WCET라고 부르지 않는다.

### 3. 독립 재자격

C145--C148의 서로 다른 네 새 A100 node에서 V15 AI45, retained-token ownership branch와
네 fault point를 operation별 두 arm으로 통과해 same-family 재자격을 완료했다. 다음 일반화 gate는 다른 GPU family 또는 실제 production
DU mode다. 목표는 모두 통과시키는 것이 아니라 envelope의 false-safe 여부를 검사하는 것이다.

### 4. 선행 연구 제출 직전 갱신

현재 공개 문헌 감사에서는 동일 TB의 optional NeuralRx가 남기는 multi-cell recovery
certificate, external AI lease, transport credit와 launch-time revalidated ownership을
함께 다룬 연구를 찾지 못했다. “선행 연구가 전혀 없다”가 아니라 이 정확한 계약 조합과
실증 범위가 공개 본문에서 확인되지 않았다고 표현한다.

## 지금 추가하지 않을 범위

Shared mandatory recovery 또는 shared NRx pool은 local certificate의 단순 합성을 깨고
non-separable multi-resource proof를 요구한다. 이를 넣으려면 broker restart, durable
ambiguous-token reconciliation과 새 theorem이 먼저 필요하다. 현재 논문의 핵심 claim에는
필수가 아니며, production timing과 cross-family 재자격보다 우선하지 않는다.

Remote Qwen을 추가하는 G6도 현재 novelty를 바꾸지 않는다. C114–126은 transport와
ownership 합성에 필요한 근거를 이미 제공한다. 새 기능을 넓히기 전에 G7/G8을 닫는 편이
논문 타당성에 더 직접적이다.

## 현재 완료도

| 축 | 완료도 | 해석 |
|---|---:|---|
| 문제 정의·차별점 | 90% | precise claim은 고정, 제출 직전 검색 갱신 필요 |
| 설계·구현 | 98% | V16-qualified bounded volatile scope와 single-token telemetry 완료 |
| 형식 모델 | 97% | pipelined critical-path lemma, reply-loss와 ownership model v2 완료; production bound 연결 남음 |
| 실험 substrate 증거 | 99% | V14 반례→V15 교정→C145--C148 four-point 물리 branch coverage 고리 확보 |
| 외적 타당성 | 80% | V16 mode는 A100 네 node; synthetic deadline와 단일 hardware family가 남음 |
| 논문 제출 준비 | 90% | G7 production timing, deterministic/probabilistic commit bound, cross-family와 figure 정리가 필요 |

현재 더 많은 optimizer seed나 작은 처리량 차이를 추적하는 것은 중단한다. 다음 실행은
실제 DU timing trace 확보가 최우선이며, 그 다음은 cross-family 또는 production mode 재자격이다.
