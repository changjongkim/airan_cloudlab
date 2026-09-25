> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# C138–C140: socket timeout과 admission wall bound 분리

**상태:** 2026-09-24, V13 판정 완료  
**결론:** 기존 5 ms wall-bound 가정은 기각했고, 5 ms socket timeout과 7 ms admission
wall bound를 분리한 뒤 AI35 조건부 교환 class를 다시 물리 검증했다.

## 왜 이 실험이 필요했는가

C132–C134는 broker의 `prepare`, `commit`, `complete`에 각각 5 ms socket timeout을 두고
세 호출 15 ms를 AI admission에 선납했다. Fault 뒤 RAN 연속성, 중복 방지, credit drain은
통과했지만 실제 장애 호출이 Python controller로 돌아오는 wall time은 기록하지 않았다.
C137은 정상 경로 5,993회를 계측해 최대 2.594 ms를 얻었지만 broker fail-stop 호출은
포함하지 않았다.

Socket timeout `T_sock`은 커널이 socket wait를 깨우는 기준이다. Runtime이 certificate에
청구해야 하는 값은 예외 생성, Python 복귀와 quarantine까지 포함한 `B_rpc`다. 두 값을
같다고 두면 다음 implication이 성립하지 않는다.

```text
socket.settimeout(5 ms)  =>  controller가 5 ms 안에 반드시 복귀
```

## C138: 사전 고정 5 ms gate의 실패

C138은 기존 C134 fail-stop 경로를 보존하고, 실제 socket을 시도한 호출만 계측했다.
Quarantine 이후의 no-op은 기록하지 않고 장애를 처음 감지한 호출은 기록했다. 새 A100
node `nid001144`, job `58831304`에서 첫 prepare-crash arm을 실행했고, 사전 규칙에 따라
첫 실패에서 campaign을 중단했다.

| 항목 | 결과 |
|---|---:|
| Radio TB | 2,720 |
| Crash 탐지 뒤 radio TB | 2,482 |
| 실제 broker RPC | 351 |
| 정상 RPC 최대 | 약 1.540 ms |
| Home0 faulting prepare | 5.205092 ms |
| Home1 in-flight faulting complete | 5.830741 ms |
| 5 ms 초과 | 2 |
| Deadline/PHY/AI/credit/duplicate 위반 | 0 |

`rpc_wall_bound`만 실패했고 나머지 17개 gate는 통과했다. 따라서 fail-closed 메커니즘은
유지되지만 `5 ms × 3 = 15 ms`를 wall-time certificate로 쓰는 mode는 UQ다.

## 설계 수정

두 시간을 다음처럼 분리했다.

```text
T_sock = 5 ms             # 장애 감지용 socket wait 설정
B_rpc  = 7 ms             # controller 복귀까지 admission이 선납하는 후보 wall bound
B_control = 3 × B_rpc = 21 ms
G_AI = 2 ms
B_eff = B_AI + 21 + 2
```

7 ms는 C138 결과를 본 뒤 선택한 새 후보이므로 C138으로 검증됐다고 주장하지 않았다.
별도 source, 새 node, 새 seed와 “첫 실패 중단·사후 확대 금지” protocol을 C139 전에
동결했다.

## C139: corrected control bound

C139는 이전 모든 검증 node와 C138 node를 제외한 `nid001252`, job `58831575`에서
prepare/commit/complete 각 두 arm을 실행했다.

| 항목 | 결과 |
|---|---:|
| Arm | 6/6 PASS |
| Radio TB | 16,320 |
| Crash 탐지 뒤 radio TB | 14,919 |
| Containment 전 atomic AI | 100 |
| 실제 broker RPC | 2,020 |
| Faulting RPC | 12 |
| Socket-timeout 5 ms보다 늦은 반환 | 12 |
| 최대 반환시간 | 5.653235 ms |
| Admission wall bound 7 ms 초과 | 0 |
| Safety/credit/duplicate 위반 | 0 |

12개 faulting call이 5 ms 뒤에 돌아왔다는 결과는 C138의 반례를 재현한다. 동시에 모두
7 ms 안에 복귀해 `B_control=21 ms` 후보를 이 node와 mode의 유한 표본에서 통과시켰다.

## C140: 보정 뒤에도 conditional recovery가 필요한가

기존 AI40 class는 보정 뒤 다음처럼 58 ms window를 넘는다.

```text
AI40 + control21 + completion2 = 63 ms > conditional window 58 ms
```

따라서 AI40 주장을 유지하지 않았다. 기존에 자격화된 context64 Qwen class의 raw bound
35 ms를 사용하면 다음 경계가 생긴다.

```text
static all-fail slack                 = 53 ms
AI35 + control21 + completion2        = 58 ms
conditional decision-time window      = 58 ms
static margin                         = -5 ms
conditional margin                    =  0 ms
```

C140은 이 계산을 본 뒤 context64-only trace와 새 두 seed를 고정했다. 같은 C139 node에서
실행한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| Arm | 2/2 PASS |
| Radio TB | 2,560 |
| Target conditional branch | 309 |
| Atomic exchange | 18 |
| AI35 target exchange | 8 |
| Target exchange의 실제 guarded horizon 최소 여유 | 43.647277 ms |
| 정상 broker RPC | 1,995 |
| 정상 RPC 최대 | 2.021658 ms |
| Safety/bound/credit 위반 | 0 |

사건별 반사실 감사에서 8개 target exchange 모두 static contract는 거절하고 conditional
contract만 수락했다. 물리 여유가 모델의 0 ms보다 큰 이유는 모델이 NRx 45 ms와 AI 35 ms
상한 전체를 청구하지만 실제 실행은 더 빨랐기 때문이다. 이는 상한을 줄이는 근거가 아니다.

## V13 envelope 판정

V13은 C132–C134의 5 ms wall-bound mode를 `QSU→UQ`로 내리고 C139/C140 mode를 새 QSU로
추가한다. 전체 count는 `QSU5/QSN0/MI3/UQ11`이다.

- AI35: `35+21+2=58 ms`, static 불가, conditional exchange-only 가능
- AI40: `40+21+2=63 ms`, decision-time window 밖
- C132–C134: safety/fault-semantics의 역사적 증거로 보존하되 15 ms admission 계약은 폐기

이 correction chain의 의미는 더 큰 AI throughput이 아니다. **제어 장애 시간까지 포함해
certificate를 닫고, 그 보수적 계약에서도 conditional credit 반환만 열 수 있는 실제 AI
class가 남는지 모델 예측과 GPU 실행을 함께 확인한 것**이다.

## 주장 한계

- 7 ms는 한 A100 node와 지정 software/lifecycle mode의 유한 표본 qualification이다.
- C140은 C139와 같은 node에서 실행했으므로 독립-node AI35 재자격은 남아 있다.
- Socket stall, OS scheduling의 결정론적 상한과 WCET는 증명하지 않았다.
- 실제 DU `d_MAC`, 다른 GPU family, broker restart와 durable reconciliation은 미검증이다.

## 권위 artifact

- `results/softwall_multigpu/confirm138_service_bound_fault_telemetry_result.json`
- `results/softwall_multigpu/confirm139_qualified_control_bound_result.json`
- `results/softwall_multigpu/confirm140_v13_ai35_result.json`
- `results/softwall_multigpu/softwall_sharded_home_envelope_grid_v13.json`
- `results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v13.json`
- `results/softwall_multigpu/softwall_envelope_v12_v13_control_bound_regression_v1.json`
- `results/softwall_multigpu/softwall_envelope_v13_validation_summary.json`
- `results/softwall_multigpu/softwall_service_bound_qualification_v3.json`
- `results/softwall_multigpu/confirm138_artifact_manifest.json`
- `results/softwall_multigpu/confirm139_artifact_manifest.json`
- `results/softwall_multigpu/confirm140_artifact_manifest.json`
- `results/softwall_multigpu/softwall_v13_corrected_control_manifest.json`
