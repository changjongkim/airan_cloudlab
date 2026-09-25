> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall C128–C132 broker fail-stop와 제어 경로 예산 판정

**상태:** 2026-09-24, C132 prospective 2-arm PASS  
**대상 mode:** 2 GPU, GPU당 4 RAN cell·2 local NeuralRx endpoint·1 Qwen worker,
P180/D155, conventional 25 ms, NeuralRx 45 ms, MIG OFF/MPS ON

## 1. 결론

Broker fault를 `AI만 fail-closed`하는 것만으로는 RAN deadline이 보존되지 않는다. Recovery
전에 동기적으로 호출하는 global `prepare`, `commit`, `complete` 각각에 유한 상한을 두고,
세 상한의 합을 물리 AI bound 및 guard와 함께 **admission 전에** recovery horizon에
포함해야 한다.

```text
B_prepare + B_commit + B_AI(class) + B_complete + G_physical
    <= min(AI deadline, earliest certified recovery start)
```

C132는 각 broker RPC를 5 ms로 제한하고 `3×5=15 ms`를 2 ms physical guard와 합쳐
17 ms를 admission guard로 사용했다. 두 독립 arm의 5,440 TB에서 broker가 home 0의
30번째 commit을 적용한 직후 `os._exit(86)`으로 종료됐지만, crash 탐지 후에도 두 home이
4,976 TB를 처리했고 모든 deadline·service-bound·AI-horizon·credit gate가 0이었다.

## 2. 실패에서 계약을 도출한 과정

| 단계 | 변경 | 물리 결과 | 판정 |
|---|---|---|---|
| C128 | broker crash 전에 상태와 marker를 `/pscratch`에 동기 기록 | 양 home release 28 cell 3이 156.904/157.142 ms에 commit | fault-path logging도 timed mode의 일부이므로 FAIL |
| C129 | marker I/O 제거, commit 적용 직후 직접 `os._exit` | home 1 complete RPC가 Qwen 반환 후 97.259 ms 더 막혀 fallback을 +160.844 ms에 시도 | fail-closed 상태만으로 부족; 무제한 제어 RPC는 UQ |
| C130 | 모든 broker RPC에 5 ms socket timeout | 5,440 TB, crash 뒤 4,992 TB, safety 위반 0 | 물리 PASS이나 admission 식에 제어 비용이 없음 |
| C131 | commit+complete 10 ms와 guard 2 ms를 admission에 반영 | 5,440 TB, crash 뒤 4,971 TB, safety 위반 0 | 물리 PASS이나 동기 prepare 5 ms가 증명에서 누락 |
| C132 | prepare+commit+complete 15 ms와 guard 2 ms를 모두 반영 | 5,440 TB, crash 뒤 4,976 TB, safety 위반 0 | 현재 선언 fault class의 QSU |

C128과 C129는 삭제하거나 warmup noise로 취급하지 않는다. C128은 관측 장치 자체도
critical path에 들어오면 자격화해야 한다는 반례이고, C129는 bounded GPU execution만으로
안전 증명이 닫히지 않는다는 직접 반례다. C130/C131은 물리 통과와 논리적 자격을 구분한
proof-audit 단계다.

## 3. C132 frozen gate

| 항목 | 결과 |
|---|---:|
| 독립 formal arm | 2 |
| 총 RAN TB | 5,440 |
| crash 탐지 뒤 RAN TB | 4,976 |
| containment 전 완료 AI unit | 114 |
| deadline miss | 0 |
| NRx bound 위반 | 0 |
| conventional bound 위반 | 0 |
| AI bound / recovery horizon 위반 | 0 / 0 |
| duplicate AI execution | 0 |
| ambiguous request 실행 | 0 |
| 잔여 local recovery/endpoint/AI credit | 0 |
| frozen source hash mismatch | 0 |
| 단위시험 / runtime fault regression | 99개 PASS / 10,000회 PASS |

두 arm 모두 home 0 commit reply가 모호해졌고, home 1도 broker 상실을 prepare 또는 complete
timeout으로 감지했다. 두 client는 새로운 global AI admission을 닫고 ambiguous operation을
재시도하지 않았다. Qwen은 commit 성공 응답 뒤에만 launch되므로 home 0의 모호한 token은
실행되지 않았으며 다른 home에 재배정되지 않았다.

## 4. 보장되는 것과 남은 것

현재 보장은 이 exact mode와 다음 fault model에 한정된다.

- 한 broker process가 commit 적용 뒤 응답 전에 fail-stop한다.
- 각 synchronous broker RPC는 5 ms 안에 성공 또는 timeout한다.
- timeout 후 global AI admission은 닫고 모호한 token을 재시도·재할당하지 않는다.
- local recovery lane, endpoint/IPC credit과 RAN certificate는 home별로 분리돼 있다.

Broker restart, durable token recovery, network partition, exactly-once completion, GPU/driver
hang과 production WCET는 입증하지 않았다. 현재 결과의 정확한 주장은 **bounded control
transaction을 포함한 volatile fail-stop containment**다.

## 5. 논문에서의 의미

이 결과는 단순한 fault-handler 추가가 아니다. SoftWall certificate가 GPU kernel 시간만
예약하는 표가 아니라, optional NeuralRx의 조건부 recovery와 외부 AI ownership protocol의
동기 제어 비용까지 포함하는 end-to-end executable contract임을 보여준다. C129의 실제
deadline 실패와 C132의 사전 고정 복구를 함께 제시해야 이 설계 선택의 필요성이 드러난다.

권위 artifact는
[C132 protocol](../../results/softwall_multigpu/confirm132_fully_budgeted_broker_crash_protocol.json),
[C132 aggregate](../../results/softwall_multigpu/confirm132_fully_budgeted_broker_crash_result.json),
[검증 요약](../../results/softwall_multigpu/confirm132_validation_summary.json),
[C128–132 manifest](../../results/softwall_multigpu/confirm128_132_artifact_manifest.json)다.
