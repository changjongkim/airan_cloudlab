# SoftWall C164: lifecycle token, idle, MPS restart와 Qwen reload 검증

**판정:** `C164_IDLE30_TWO_NODE_PASS`, `C164_MPS_RESTART_TWO_NODE_PASS`,
`C164_QWEN_RELOAD_TWO_NODE_PASS`, `C164_RECONNECT_TWO_NODE_PASS`  
**범위:** (1) warm persistent client가 모두 ready인 뒤 schedule 공개 전에 30초 idle한 첫
요청, (2) quiescent MPS daemon restart와 fresh-client requalification 뒤 첫 RAN 요청 및
C162 boundary subset, (3) 같은 MPS GPU에서 60회 Qwen reload 중 4-cell mandatory continuity,
(4) same-worker epoch에서 120회 channel reconnect와 durable token reconciliation  
**제외:** restart 중 무중단 service, reload 중 optional inference availability, full six-class
lifecycle vector, NeuralRx process/model cold path, worker process replacement, 5분·30분 idle,
production `d_MAC`, cross-family와 WCET

## 1. 왜 이 실험이 필요한가

C162는 lifecycle이 `warm`이고 node preflight가 통과한 mode만 QSU/QSN/MI로 승격했다.
그러나 runtime이 warm qualification을 restart·long-idle 뒤에도 재사용하면 모델은 안전해도
실행 mode provenance가 틀릴 수 있다. C164는 qualification을 다음 fingerprint에 묶는다.

```text
(node/GPU, software, placement, lifecycle, component bounds, sample targets)
```

Complete evidence 뒤에만 epoch/generation-tagged token을 발급한다. Restart, token/fingerprint
mismatch, idle expiry 또는 runtime bound violation은 optional NRx/AI admission을 fail-closed로
막는다. [Lifecycle state model](../../results/softwall_multigpu/c164_lifecycle_model_v1.json)은
9개 lifecycle과 8개 restart 전이에서 unsafe optional admission 0으로 통과했다. 단위시험
10개는 incomplete evidence, stale epoch/generation, wrong fingerprint, idle expiry와 bound
violation을 각각 검사한다.

## 2. Development 실패와 교정

두 실패를 결과에서 제거하지 않고 별도 artifact로 보존했다.

| Attempt | 관측 | 판정과 교정 |
|---|---|---|
| A1 | `salloc`은 `nid001169`를 배정했지만 shell이 `login03`에 남아 Qwen이 compute GPU를 보지 못함 | timed radio 전 orchestration 실패. `srun --pty`로 allocation node 진입 |
| A2 | schedule 공개 뒤 32.997초 idle, NRx/owner의 5초 request timeout이 release 전에 발동 | lifecycle 설계 실패. 모든 readiness 뒤 schedule 공개 **전** idle하고 기존 3초 release lead·5초 timeout 유지 |

근거는 [A1 실패](../../results/softwall_multigpu/c164a_attempt1_allocation_entry_failure.json)와
[A2 실패](../../results/softwall_multigpu/c164a_attempt2_schedule_timeout_failure.json)에 있다.
A2는 “release lead를 늘리는 것”과 “준비된 persistent mode를 idle시키는 것”이 다르다는
실제 반례다.

## 3. 최종 실험 설계

모든 owner, persistent TensorRT NeuralRx, shared cuPHY recovery와 Qwen worker가 ready가 된
시각을 coordinator의 `time.perf_counter_ns`로 기록한다. Coordinator가 30초 sleep을 마친
뒤 lifecycle marker를 확정하고, 그 다음에 기존 C162 schedule을 3초 lead로 공개한다.
따라서 per-request timeout과 release semantics는 C162와 같다.

두 node는 같은 frozen source를 사용했다.

| 항목 | Development | Holdout |
|---|---:|---:|
| Job / node | 58861564 / `nid001169` | 58861662 / `nid001137` |
| Seed | 45,000,000 | 46,000,000 |
| Case order | forward | reverse |
| Rounds | 60 | 120 |
| Case당 표본 | 10 | 20 |
| 실제 ready→schedule idle | 30.029131 s | 30.029712 s |

Holdout protocol은 development node를 exclusion에 포함했다. C162의 여섯 case를 그대로
사용해 QSU 내부, QSN, context class와 88/89 ms decision boundary를 검사했다.

## 4. 결과

| Metric | 두 node 합계/최대 |
|---|---:|
| Physical rounds | 180 |
| Actual TensorRT NeuralRx | 720 |
| Injected failure의 shared cuPHY recovery | 330 |
| Physical Qwen units | 90 |
| Radio single commits | 720 |
| Deadline misses | **0** |
| NRx release-to-complete max | 25.071750 ms |
| Recovery path max | 6.598797 ms |
| Qwen execution max | 54.214612 ms |
| Radio commit max | 126.037620 ms |

Development와 holdout 모두 C162의 12개 base gate와 C164의 lifecycle gate를 통과했다.
E6a는 conservative 88 ms에서 허가됐고 E6b는 89 ms 이상에서 거절됐다. Model/physical
lease mismatch, physical-fence violation과 source-hash mismatch는 0이었다.

[두-node 결합 결과](../../results/softwall_multigpu/c164_idle30_two_node.json)는 independent
job/node, frozen source, 반대 order, 독립 seed, development-node exclusion과 combined safety를
모두 검사한다. 전체 source/raw/result hash는
[artifact manifest](../../results/softwall_multigpu/c164_idle30_manifest.json)에 고정한다.

## 5. Quiescent MPS restart 뒤 재자격

Restart 실험은 단순히 runner를 다시 실행한 것이 아니다. 같은 allocation 안에서 먼저 네
GPU 모두에 실제 CUDA client를 연결해 이전 MPS epoch를 만들고, daemon을 종료했다. Control
socket과 server가 모두 사라진 것을 확인한 뒤 새 daemon을 시작하고 네 GPU probe를 다시
실행했다. 이전/새 control PID, server PID와 control-socket inode가 모두 달라야만 RAN
process를 시작했다. Optional NRx와 AI는 이 구간에 닫혀 있고, 모든 fresh client의 readiness와
full-path preflight가 끝난 뒤 첫 schedule을 공개했다.

| 항목 | Development | Holdout |
|---|---:|---:|
| Job / node | 58861897 / `nid001192` | 58861911 / `nid001200` |
| Seed / case order | 47,000,000 / forward | 48,000,000 / reverse |
| MPS control PID | 1677299→1677602 | 1120647→1120947 |
| MPS server PID | 1677538→1677837 | 1120882→1121181 |
| Control socket inode | …590→…600 | …714→…724 |
| Boundary rounds | 60 | 120 |

두 node를 합치면 actual NeuralRx720, injected shared recovery330, Qwen90, radio commit720이며
deadline miss는 0이다. 최대 NRx/recovery/Qwen/commit은 각각
22.713/10.720/67.907/126.226 ms였다. 88 ms context64 허가와 89 ms 이상 거절도 유지됐다.
[두-node 결합 판정](../../results/softwall_multigpu/c164_mps_restart_two_node.json)은 독립
job/node, holdout node exclusion, 독립 seed, 반대 case order, frozen C162/C164 source와 두
restart proof를 함께 검사한다. 전체 artifact는
[restart manifest](../../results/softwall_multigpu/c164_mps_restart_manifest.json)에 고정한다.

이 결과는 **restart 중 service continuity**를 보이지 않는다. 증명한 것은 restart 때 이전
lifecycle epoch를 사용하지 않고 optional work를 닫은 뒤, 새 epoch의 물리 evidence를 다시
만들어 첫 RAN 요청부터 동일 boundary contract를 만족한다는 것이다.

## 6. Qwen unload/reload 중 mandatory continuity

Qwen process가 자격화되기 전이라고 해서 GPU를 사용하지 않는 것은 아니다. Model weight load,
CUDA allocation과 여섯 context class의 warmup이 recovery GPU2에서 실제 kernel과 memory traffic을
만든다. 따라서 optional inference admission만 닫는 것으로 충분한지 물리적으로 검사했다.

각 node에서 하나의 warm mandatory cuPHY process가 `P180/D155`로 4개 conventional cell을
계속 처리했다. 그 위에서 Qwen2.5-1.5B process를 30번 새로 시작해 context
16/32/64/128/256/512를 각각 3회 warmup하고 종료했다. Qwen inference request는 한 건도
허가하지 않았다. 각 reload의 launch→ready interval과 RAN release interval이 실제로 겹쳐야
표본으로 인정했다.

첫 attempt는 두 node 모두 timed release 전에 실패했다. MPS daemon을 physical GPU2 하나로
제한하면 그것이 logical GPU0으로 재번호화되는데 client가 다시 device2를 요청해
`cudaErrorNoDevice`가 발생했다. 두 실패는 qualification 표본에서 제외하고
[development 실패](../../results/softwall_multigpu/c164e_qwenreload_dev_j58862152_prelaunch_failure.json)와
[holdout 실패](../../results/softwall_multigpu/c164f_qwenreload_holdout_j58862161_prelaunch_failure.json)에
보존했다. 새 source는 daemon visibility를 기존 qualified mode와 같은 GPU0--3으로 유지하고
client placement만 physical GPU2로 고정했다.

| Metric | 두 node 합계/최대 |
|---|---:|
| Job / node | 58862152 / `nid001288`, 58862161 / `nid001381` |
| Qwen fresh reload episode | **60** |
| Mandatory release / cell decode | **3,334 / 13,336** |
| Reload-overlap / quiet release | 2,658 / 676 |
| Optional inference unit | **0** |
| Deadline miss / cell25 위반 | **0 / 0** |
| Reload load+warmup max | 19,615.393 ms |
| Mandatory response max | 52.674 ms |
| Reload-overlap response max | 34.835 ms |
| Reload-overlap cell GPU max | 12.469 ms |

첫 reload는 두 node에서 19.615/19.576초였고 이후 episode의 중앙값은 약 7.44초였다. 이는
reload 시간을 steady-state AI unit bound처럼 다룰 수 없음을 보여준다. 그 긴 interval에도
mandatory path는 D155와 cell당 25 ms를 지켰지만, finite-sample 결과이며 cold-load WCET는
아니다. [두-node 결합 판정](../../results/softwall_multigpu/c164_qwen_reload_two_node.json)과
[artifact manifest](../../results/softwall_multigpu/c164_qwen_reload_manifest.json)에 source,
raw interval, worker output과 실패 chain을 고정한다.

## 7. 해석

이 결과는 warm A100 mode가 짧은 pause에만 의존한 우연한 결과가 아님을 한 단계 강화한다.
Ready persistent clients가 30초 idle한 뒤의 첫 actual NRx/recovery/Qwen 경계에서도 예측과
실행이 일치했다. 동시에 A2 실패는 lifecycle의 정확한 정의가 필요하다는 근거다. Timeout을
늘려서 통과시킨 것이 아니라 idle을 request release 전 상태로 옮겨 기존 timeout을 유지했다.

현재 승격되는 mode는 `idle_30s_first boundary subset`과
`mps_restart_first-after-requalification boundary subset`, 그리고
`qwen_reload_mandatory_continuity subset`이다. 마지막 것은 reload 중 AI 처리량을 보장하지
않고 mandatory RAN만 보장한다. Context16/32의 admitted physical class 표본,
NeuralRx process/model cold path, reconnect와 긴 idle은 아직 별도 UQ다. 따라서 C164 전체
완료라고 표현하지 않는다.

## 8. Same-worker channel reconnect와 durable reconciliation

C161의 pre-fence channel loss는 AI를 quarantine하고 ambiguous lease를 유지했지만 다시 여는
절차는 없었다. C164는 worker가 `prepared`를 fsync한 뒤에만 ACK하고, GPU launch 전에
`launched`, CUDA completion 뒤에 `fenced`를 같은 journal에 순서대로 기록한다. Reconnect는
token, lifecycle epoch, payload hash, worker epoch와 journal sequence를 모두 검사한다.

- `prepared`: worker token lock 아래 `nonlaunch_fenced`로 만든 뒤 retire
- `launched`: completion fence 전에는 계속 hold
- `fenced`: matching terminal record를 확인한 뒤 한 번만 retire
- absent, identity mismatch, rollback, illegal transition: fail-closed

순수 상태모델은 [43-state audit](../../results/softwall_multigpu/c164_reconnect_model_v1.json)에서
retire2, fail-closed41, invariant violation0을 얻었다. 물리 실험은 두 독립 A100 node에서
context128/512와 두 단절 위치를 균등하게 반복했다.

| Metric | 두 node 합계/최대 |
|---|---:|
| Token / channel reconnect | **120 / 120** |
| prepare-loss / post-fence-loss | 60 / 60 |
| Physical Qwen launch / terminal fence | 60 / 120 |
| Mandatory release / cell decode | 145 / 580 |
| Launch-call→fence와 mandatory wall-overlap release | 60 |
| Deadline miss / cell25 위반 | **0 / 0** |
| Qwen GPU max | 66.794 ms |
| Mandatory response / cell GPU max | 19.353 / 7.588 ms |

[두-node 결합 판정](../../results/softwall_multigpu/c164_reconnect_two_node.json)은 frozen source,
서로 다른 node/job/seed와 development-node exclusion을 확인한다. 첫 attempt는 실행 위상이
엇갈려 overlap0이었고, 두 번째 attempt는 enqueue-return timestamp를 launch 시작으로 잘못
사용해 overlap0이었다. 두 FAIL은 결과에서 제거하지 않고 manifest에 보존한다. 최종 실험은
사전 정렬된 release와 별도 `launch_called_ns`를 사용했다. Overlap은 wall interval 증거이며
새 Nsight kernel-overlap 주장이 아니다. Worker process replacement와 임의 crash window는 UQ다.
[290-file artifact manifest](../../results/softwall_multigpu/c164_reconnect_manifest.json)는 두
passing run, 두 development failure, 240개 fsync journal record와 source hash를 함께 고정한다.

## 9. Claim-scoped lifecycle qualification matrix

개별 결과를 [machine-readable lifecycle matrix](../../results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json)로
합쳤다. 입력 artifact의 hash와 각 결과의 PASS 상태를 다시 검사하며, 하나의 subset을 다른
lifecycle 보장으로 승격하지 않는다.

| Lifecycle mode | 현재 판정 | 보장되는 범위 |
|---|---|---|
| warm persistent | Qualified subset | mandatory continuity와 optional boundary |
| 30초 idle 뒤 첫 요청 | Qualified subset | mandatory continuity와 optional boundary |
| MPS restart 뒤 첫 요청 | Qualified after requalification | restart 구간을 제외한 optional boundary |
| Qwen reload | Qualified mandatory-only | reload 중 mandatory RAN continuity |
| process/model cold | UQ | whole-mode evidence 없음 |
| same-worker channel reconnect | Qualified subset | durable prepared/fenced reconciliation과 mandatory continuity |
| worker process replacement | UQ | single-node development canary만 PASS; 독립 holdout과 arbitrary crash-window proof 없음 |
| 5분·30분 idle | UQ | 물리 표본 없음 |
| GC ON | UQ, observed failure | C89에서 선언 bound 위반 |

현재 합계는 qualified 또는 부분-qualified 5개, UQ 5개이며 전체 matrix 상태는
`C164_LIFECYCLE_MATRIX_PARTIAL`, 논문 gate는
`C164_CLAIM_SCOPED_LIFECYCLE_COMPLETE`다. 이 5/10은 미완성 checklist가 아니라 논문 claim에
맞춘 의도적 경계다. 남은 mode를 채우기 위해 새 recovery mechanism을 추가하지 않는다.

이미 시작했던 process-replacement 단일-node development canary는 8 episode에서 MPS
`terminate_client` 8회, mandatory release486·cell decode1,944, miss·component-bound 위반0으로
끝났다. 그러나 독립 holdout을 열지 않았고 lifecycle matrix에도 승격하지 않는다. Durable
journal의 crash-window semantics와 임의 GPU/driver failure는 계속 UQ이며, 해당 결과는
future-work exploratory evidence다. Claim/evidence freeze와 영문 초고를 완료했으며, 다음
내부 gate는 venue 원고 압축, 도표·bibliography와 reviewer-objection audit이다.
Production `d_MAC`, WCET와 다른 GPU family는 이 matrix 밖이다.
