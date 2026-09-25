# SoftWall service-bound qualification 방법론

**상태:** 2026-09-24, V16 four-point physical fault gate와 C147/C148 재자격 완료  
**현재 판정:** V14 `UQ_OWNERSHIP_BRANCH`; V15 three-point row `UQ_FAULT_COVERAGE`; V16 `FINITE_SAMPLE_TELEMETRY_PASS/QSU`, hard-real-time/WCET는 `UNPROVEN`

## 1. 왜 별도 방법론이 필요한가

SoftWall certificate는 `B_NRx`, `B_conv`, `B_AI`, broker control budget을 전제로 안전성을
판정한다. 이 값이 실제 mode에서 깨지면 정확한 스케줄러도 false-safe가 된다. 반대로
관측 최대에 큰 임의 배수를 붙이면 안전해 보이지만 조건부 slack이 사라져 정책이 퇴화한다.
따라서 bound는 평균이나 p99 한 개가 아니라 **mode 정의, 사전 고정, 독립 holdout,
초과 시 강등**을 함께 가진 qualification object여야 한다.

현재 mode는 다음 항목이 모두 같을 때만 하나로 취급한다.

```text
hardware family/node role, cells, P/D, MPS caps, endpoint/ring topology,
GC/cold/restart state, NRx/conv/AI work class, executor order,
broker fault model, socket timeout, admission wall bound, trace hash
```

한 항목이라도 달라지면 기존 bound를 재사용하지 않고 새 mode를 UQ에서 시작한다.

## 2. 데이터 분리

- **C135:** V12가 찾은 conditional AI40 class의 최초 prospective realization이다.
- **C136:** bound와 source를 바꾸지 않고 기존 node를 제외한 새 A100 node에서 수행한
  6-arm holdout이다.
- C136에서 초과가 한 건이라도 나오면 C135를 포함한 전체 mode를 UQ로 내린다. 관측 뒤
  bound를 넓혀 같은 campaign을 통과로 바꾸지 않는다.

## 3. Clustered evidence

TB는 같은 release, persistent process와 node 상태를 공유한다. 따라서 event 수만 독립
표본으로 세지 않고 `event < arm/process lifecycle < physical node < GPU family`를 함께
보고한다. 0회 초과를 관측한 `N_u`개 IID 단위에서 one-sided 95% 상한은
`1-0.05^(1/N_u)`다. 어떤 단위를 IID로 가정했는지 명시하지 않은 failure probability는
보고하지 않는다.

### 3.1 C136 holdout

| Component | 선언 bound | 표본 | holdout 최대 | headroom | arm | node | 초과 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeuralRx response | 45 ms | 3,312 | 28.097520 ms | 16.902480 ms | 6 | 1 | 0 |
| Conventional host path | 25 ms | 5,217 | 6.722176 ms | 18.277824 ms | 6 | 1 | 0 |
| Qwen context-128 execution | 40 ms | 123 | 28.452898 ms | 11.547102 ms | 6 | 1 | 0 |

### 3.2 C135+C136 결합

| Component | 표본 | 결합 최대 | arm | node | event-IID 95% 상한 | arm-IID 상한 | node-IID 상한 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeuralRx response | 4,389 | 28.097520 ms | 8 | 2 | 0.0682% | 31.23% | 77.64% |
| Conventional host path | 6,984 | 6.722176 ms | 8 | 2 | 0.0429% | 31.23% | 77.64% |
| Qwen context-128 execution | 165 | 28.452898 ms | 8 | 2 | 1.7992% | 31.23% | 77.64% |

Event-IID 수치는 민감도다. Persistent process와 node 안의 상관 때문에 논문의 보수적
결론은 두 A100 node·8 arm의 finite-sample qualification이다. 관측 headroom으로 bound를
45→28.1, 25→6.8, 40→28.5 ms로 낮추지 않는다.

## 4. 제어 경로의 별도 판정

과거 broker는 `prepare/commit/complete` 각각에 5 ms socket timeout을 설정하고 총 15 ms를
admission에서 차감했다. C129는 timeout이 없을 때 complete RPC가 97.259 ms 더 막혀
deadline을 깬 반례이고, C132--134는 15 ms를 모두 청구한 fail-stop path의 safety를 통과했다.

하지만 C135/C136 raw artifact에는 각 RPC의 시작·종료 latency가 없다. 이 공백을 C137의
별도 instrumentation mode에서 닫았다. 새 A100 node의 6개 arm에서 prepare 5,745회,
commit 124회, complete 124회를 기록했고 각 최대는 2.593595/0.388313/1.046257 ms였다.
5 ms 초과와 RPC fault는 0이었다. 하지만 이는 정상 경로일 뿐 faulting call을 포함하지 않았다.

C138은 같은 계측을 post-apply fail-stop 경로에 붙였다. 첫 사전 고정 prepare arm에서
faulting prepare 5.205092 ms와 concurrent complete 5.830741 ms가 나와 5 ms wall-bound
gate만 실패했다. 따라서 timeout 설정과 controller 복귀 상한을 같은 값으로 둘 수 없다.

V13은 `T_sock=5 ms`, `B_rpc=7 ms`로 분리하고 세 RPC 21 ms를 admission에 청구한다.
C139는 새 A100 node의 6 arm에서 2,020개 attempted call을 기록했다. Faulting call 12개가
5 ms보다 늦게 반환됐지만 전체 최대는 5.653235 ms이고 7 ms 초과는 0이었다. C140의
정상 corrected mode도 1,995개 호출 최대 2.021658 ms, 7 ms 초과 0이었다.

C137은 measurement code가 추가된 mode다. 따라서 C135/C136의 uninstrumented 결과를
소급해 RPC 계측 완료로 바꾸지 않는다. Mechanism 수준 합계는 세 A100 node·14 arm·17,920
TB·AI40 exchange 66회지만, exact-mode bound 증거는 uninstrumented 2 node와 instrumented
1 node로 분리한다. C138–C140은 별도 V13 mode이며 과거 표본과 합쳐 7 ms의 독립-node
reliability 수치를 만들지 않는다.

C141은 corrected V13의 exact contract를 `nid001372`에서 독립 재자격했다. Control fault
6 arm은 16,320 TB와 attempted RPC 2,010회에서 최대 5.775660 ms, 7 ms 초과 0이었다.
AI35 두 arm은 2,560 TB·target branch293·exchange7에서 static/conditional margin
−5/0 ms와 safety 위반 0을 유지했다. C139/C140과 합치면 corrected mode는 두 A100 node,
control 12 arm·RPC4,030·AI35 4 arm·exchange15의 유한 표본 증거를 가진다.

### 4.1 V14 critical/deferred control 분리

V14는 세 RPC에 같은 시간 계약을 요구하지 않는다. `prepare`, `abort`, `complete`는 별도
connection/worker에서 처리하고 RAN executor가 반환을 기다리지 않는다. Prepared token을
launch 직전에 현재 certificate로 재검증한 뒤 `commit`만 동기적으로 실행한다.

```text
B_critical = B_commit = 7 ms
B_effective = B_AI + B_commit + g_AI
```

C142의 AI45 두 arm은 commit 최대0.612807 ms, deferred 최대9.780886 ms에서 2,560 TB와
exchange9를 위반 없이 실행했다. C143의 six-fault matrix에서 commit 340회는 최대
5.148988 ms로 7 ms 안이었고, deferred RPC 1,584회는 최대9.614963 ms이며 50회가 7 ms를
넘었다. 그럼에도 16,320 TB와 fault 뒤14,184 TB에서 safety·duplicate 위반은 0이었다.
따라서 “deferred도 7 ms 안이어야 안전하다”는 계약은 데이터가 지지하지도, V14가 요구하지도
않는다. 대신 적용 여부가 불명확한 token은 quarantine되고 신규 global AI가 닫혀야 한다.

C144는 이전 node를 제외한 `nid002049`에서 AI45 한 arm과 prepare/commit/complete fault
한 arm씩을 재자격했다. C142--C144 두 node 합계는 AI45 3 arm·exchange11과 fault9 arm·
21,120 TB·fault 뒤17,914 TB의 유한 표본 근거다. 이 결과는 commit 7 ms의 WCET를 만들지
않으며, deferred worker가 CPU·memory 자원을 통해 RAN을 간접 방해하지 않는다는 보편적
상한도 아니다. Exact mode fingerprint에는 worker affinity, connection 분리와 queue depth를
포함해야 한다.

### 4.2 V15 single-token ownership 재자격

V14의 timing/fault 표본은 통과했지만, 한-request finite model은 retained staged token 뒤
두 번째 prepare를 표현하지 않았다. 결정적 two-request 회귀는 broker-held token 2개 중
client가 1개만 추적하는 상태를 재현했다. 따라서 V14의 관측 arm이 최종 drain됐다는
사실만으로 ownership 자격을 소급하지 않고 mode를 UQ로 내린다.

V15는 `staged` 또는 `offered` token이 있으면 prepare를 억제한다. C145와 C146은 기존
node를 제외한 `nid001244`, `nid003417`에서 정상 AI45와 prepare/commit/complete fault를
각각 실행했다. 두 node 합계 12,160 TB, AI45 exchange12, fault 뒤 radio7,463건에서
safety·duplicate 위반은 0이었다. 문제 분기는 537회 관측됐고 maximum unlaunched token은
모든 home-arm에서 1이었다. 이는 ownership과 control timing을 포함한 exact warm mode의
유한 표본 자격이며, durable restart나 commit7 WCET를 뜻하지 않는다.

### 4.3 V16 four-point fault coverage

V15 finite model은 abort reply-loss를 포함했지만 C145/C146은 prepare/commit/complete만
물리 주입했다. V16은 `four_point_control_fault_required`를 mode fingerprint에 추가한다.
C147/C148은 서로 다른 `nid003197`, `nid001005`에서 abort 적용 뒤 reply-loss를 주입했고,
합계 3,200 TB, fault 뒤 radio2,662, target AI 실행0, prepare 억제129회, maximum
unlaunched token1과 위반0을 기록했다. C145--C148 결합 matrix는 네 operation마다 두 arm,
서로 다른 A100 node4개, radio15,360, fault 뒤10,125다. 따라서 V15 runtime의 시간·ownership
근거를 유지하면서 물리 fault 자격만 V16으로 승격한다.

## 5. Qualification 상태전이

| 상태 | 조건 |
|---|---|
| `UQ` | mode fingerprint 또는 component telemetry 누락, holdout 없음, 초과 발생 |
| `FINITE_SAMPLE_PASS` | 사전 고정 bound, 독립 holdout, 모든 component와 safety gate 초과 0 |
| `PROBABILISTIC_QUALIFIED` | 독립 단위와 목표 exceedance probability를 사전 정의하고 필요한 표본 수 충족 |
| `DETERMINISTIC_BOUND` | 분석 가능한 실행 상한 또는 검증된 강제 중단/격리 메커니즘이 있음 |
| `PRODUCTION_QUALIFIED` | 위 bound와 실제 DU `d_MAC`을 같은 clock에서 연결 |

현재는 component holdout과 pipelined commit-path/fault telemetry를 통과한
`FINITE_SAMPLE_TELEMETRY_PASS`다.
MPS에는 강제 kernel preemption 상한이 없으므로 표본을 늘리는 것만으로
`DETERMINISTIC_BOUND`로 승격하지 않는다.

## 6. 다음 gate

1. Silent stall과 OS scheduling을 포함한 commit 7 ms 강제 상한 또는 확률 계약을 설계한다.
2. Cold start, GC ON, MPS client restart를 별도 mode로 유지한다.
3. Production DU `d_MAC`을 확보하면 component bound와 같은 clock으로 연결한다.
4. 다른 GPU family에서 mode 전체를 다시 자격화한다.
5. 어떤 경우에도 유한 표본 최대를 WCET로 바꾸지 않는다.

## Artifact

- [Clustered qualification result](../../results/softwall_multigpu/softwall_v12_clustered_bound_qualification_v1.json)
- [Analyzer](../../scripts_for_node/softwall_same_gpu/analyze_softwall_clustered_bound_qualification.py)
- [Unit tests](../../scripts_for_node/softwall_same_gpu/test_analyze_softwall_clustered_bound_qualification.py)
- [C136 result](../../results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json)
- [C135/C136 combined audit](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [C137 RPC telemetry 결과](../archive/SOFTWALL_CONFIRM137_SERVICE_BOUND_TELEMETRY_RESULT_KO.md)
- [C137 campaign result](../../results/softwall_multigpu/confirm137_service_bound_telemetry_result.json)
- [Service-bound qualification v2](../../results/softwall_multigpu/softwall_service_bound_qualification_v2.json)
- [C137 artifact manifest](../../results/softwall_multigpu/confirm137_artifact_manifest.json)
- [Service-bound v2 manifest](../../results/softwall_multigpu/softwall_service_bound_v2_manifest.json)
- [C138–C140 correction](../archive/SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md)
- [C139 corrected control result](../../results/softwall_multigpu/confirm139_qualified_control_bound_result.json)
- [C140 corrected conditional class](../../results/softwall_multigpu/confirm140_v13_ai35_result.json)
- [V13 validation summary](../../results/softwall_multigpu/softwall_envelope_v13_validation_summary.json)
- [Service-bound qualification v3](../../results/softwall_multigpu/softwall_service_bound_qualification_v3.json)
- [Qualification v3 analyzer](../../scripts_for_node/softwall_same_gpu/analyze_softwall_service_bound_qualification_v3.py)
- [Qualification v3 tests](../../scripts_for_node/softwall_same_gpu/test_analyze_softwall_service_bound_qualification_v3.py)
- [V13 corrected-control manifest](../../results/softwall_multigpu/softwall_v13_corrected_control_manifest.json)
- [C141 독립-node 결과](../archive/SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md)
- [C141 combined result](../../results/softwall_multigpu/confirm141_v13_independent_requalification_result.json)
- [Service-bound qualification v4](../../results/softwall_multigpu/softwall_service_bound_qualification_v4.json)
- [V13 cross-node manifest](../../results/softwall_multigpu/softwall_v13_cross_node_manifest.json)
- [V14 pipelined-control 설계](../archive/SOFTWALL_PIPELINED_CONTROL_DESIGN_KO.md)
- [C142--C144 V14 결과](../archive/SOFTWALL_CONFIRM142_144_V14_PIPELINED_RESULT_KO.md)
- [C142 AI45 result](../../results/softwall_multigpu/confirm142_v14_pipelined_result.json)
- [C143 fault result](../../results/softwall_multigpu/confirm143_v14_control_faults_result.json)
- [C144 independent-node result](../../results/softwall_multigpu/confirm144_v14_independent_node_result.json)
- [V14 validation](../../results/softwall_multigpu/softwall_envelope_v14_validation_summary.json)
- [V14 manifest](../../results/softwall_multigpu/softwall_v14_pipelined_control_manifest.json)
- [V14 ownership 반례와 V15 교정](../archive/SOFTWALL_V14_OWNERSHIP_CORRECTION_KO.md)
- [C145--C146 V15 결과](../archive/SOFTWALL_CONFIRM145_146_V15_SINGLE_TOKEN_RESULT_KO.md)
- [V15 finite model v2](../../results/softwall_multigpu/softwall_pipelined_control_model_v2.json)
- [V15 validation](../../results/softwall_multigpu/softwall_envelope_v15_validation_summary.json)
- [V15 manifest](../../results/softwall_multigpu/softwall_v15_single_token_manifest.json)
- [C147--C148 V16 four-point 결과](../archive/SOFTWALL_CONFIRM147_148_V16_FOUR_POINT_RESULT_KO.md)
- [V16 validation](../../results/softwall_multigpu/softwall_envelope_v16_validation_summary.json)
- [V16 manifest](../../results/softwall_multigpu/softwall_v16_four_point_manifest.json)
