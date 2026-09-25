# C147–C148 V16 four-point control-fault 자격 결과

**상태:** 2026-09-24 완료  
**판정:** V15 runtime 동작은 유지하고, 네 state-changing broker operation의 물리 fault
coverage를 요구하는 V16 envelope에서 `Qualified-safe-useful`  
**범위:** A100 동일 family 유한 표본; WCET, production `d_MAC`, durable broker 복구와
cross-family 일반화는 미입증

## 1. 왜 추가 실험이 필요했나

V15 finite model은 `prepare`, `abort`, `commit`, `complete`의 before/after-apply reply-loss를
모두 포함한다. 그러나 C145/C146 물리 fault arm은 `prepare`, `commit`, `complete`만 직접
주입했다. `abort`는 정상 경로에서 양쪽 home에 관측됐지만, **abort가 broker에 적용된 뒤
ACK만 사라지는 분기**는 물리적으로 검증하지 않았다.

이 차이를 숨기지 않고 V16 checker에 새 gate를 추가했다.

```text
four_point_control_fault_required = true
four_point_control_fault_qualified = false  -> V15 row UQ
four_point_control_fault_qualified = true   -> V16 row 판정 가능
```

V15 runtime source와 C145/C146 protocol hash는 변경하지 않았다. 별도 four-point broker
fault harness가 abort 전이를 적용하고 세 번째 home-0 abort의 응답 직전에 `os._exit(86)`로
종료한다.

## 2. 판정 규칙

각 새 노드 arm은 다음을 모두 만족해야 한다.

1. abort 적용 뒤 reply-loss가 정확히 한 번 주입된다.
2. ACK를 받지 못한 target request는 어느 home에서도 실행되지 않는다.
3. 두 home 모두 global AI를 fail-closed하고 local RAN certificate 실행을 계속한다.
4. 4셀×2 home의 1,600 radio record가 모두 생성된다.
5. deadline, NRx/conventional/AI bound, credit, physical worker gate를 모두 통과한다.
6. `prepare/abort/commit/complete`가 양쪽 home의 RPC telemetry에 모두 존재한다.
7. `maximum_unlaunched_tokens <= 1`이고 retained-token prepare 억제 분기가 양쪽에서
   실제 실행된다.
8. frozen source hash가 현재 파일과 일치한다.

Abort는 비동기 control worker에서 일어난다. 따라서 이 실험은 abort 시간을 RAN critical
path에 추가하지 않는다. Launch를 허가하는 동기 `commit`만 7 ms admission budget을 계속
사용한다.

## 3. 독립 두 노드 결과

| Campaign | Node / job | Radio | Fault 뒤 radio | Prepare 억제 | Max unlaunched | Target 실행 | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| C147 | `nid003197` / `58835729` | 1,600 | 1,332 | 66 | 1 | 0 | PASS |
| C148 | `nid001005` / `58835753` | 1,600 | 1,330 | 63 | 1 | 0 | PASS |
| 합계 | 새 A100 두 노드 | **3,200** | **2,662** | **129** | **1** | **0** | **PASS** |

각 arm의 21개 gate가 모두 통과했다. C147은 correct 373/800과 385/800, C148은
365/800과 390/800을 기록했다. Correct count는 채널/decoder 결과이고 safety gate는 모든
TB의 single commit과 deadline/bound/credit 위반 0으로 별도 판정한다.

물리 실행 전 두 차례 allocation에서 GPU 요청 옵션이 빠져 `nvidia-smi`가 device 0개를
보고했고 runner가 protocol freeze 직후 종료 코드 100으로 중단됐다. PHY/Qwen arm은 시작되지
않았으며 해당 빈 로그와 protocol은 덮어쓰지 않았다. 이후 과거 C145/C146과 같은
`--gpus 4` A100 allocation으로 C147/C148을 새로 실행했다.

## 4. C145–C148 결합 fault matrix

| Operation | 물리 arm | 독립 노드 | 의미 |
|---|---:|---|---|
| prepare | 2 | `nid001244`, `nid003417` | anonymous held-token 가능성을 home quarantine로 봉쇄 |
| abort | 2 | `nid003197`, `nid001005` | release 적용 여부가 모호해도 target launch 없이 fail-closed |
| commit | 2 | `nid001244`, `nid003417` | ACK 없는 request를 launch하지 않음 |
| complete | 2 | `nid001244`, `nid003417` | 완료 적용 여부가 모호해도 token 재사용/중복 실행 없음 |

결합 결과는 다음과 같다.

- 서로 다른 A100 노드 4개
- post-apply fault arm 8개, operation별 2개
- radio record 15,360개
- fault 탐지 뒤 radio record 10,125개
- AI45 conditional exchange 12개
- retained-token prepare 억제 666회
- maximum unlaunched token 1
- 선언 safety 또는 duplicate 위반 0

Operation별 두 arm이라는 뜻이며, 네 operation의 Cartesian product를 각 노드에서 모두
주입했다는 뜻은 아니다. Finite model도 fault protocol과 ownership submodel을 분리해 감사한
bounded audit이다.

## 5. Envelope 판정

V16은 C145/C146의 timing·AI45·single-token 근거를 sealed base로 사용하고 C147/C148의
abort evidence를 추가한다.

```text
V14 retained-token ownership 미검증: UQ
V15 prepare/commit/complete physical coverage: UQ under four-point requirement
V16 all-four physical coverage: QSU
counts: QSU 6 / QSN 0 / MI 3 / UQ 13
```

V16 validation 14개 gate가 모두 통과했다. Manifest는 V15의 139개 파일을 다시 검증하고
V16 extension을 합쳐 총 190개 파일을 transitively 고정했으며, 재생성 결과는
byte-identical이다.

## 6. 현재 주장

이 결과로 말할 수 있는 것은 다음 범위다.

> 두 GPU home의 bounded volatile broker mode에서 single-token ownership, launch-time
> certificate 재검증, 동기 commit admission, 비동기 prepare/abort/complete와 fail-closed
> quarantine를 결합하면, 네 state-changing control operation의 post-apply reply-loss에도
> local RAN execution을 계속할 수 있다.

아직 말할 수 없는 것은 hard-real-time 보장, broker 재시작 뒤 durable reconciliation,
GPU/driver hang, 실제 DU `d_MAC`, 다른 GPU family, shared mandatory recovery resource다.

## 7. 권위 산출물

- [C147 result](../../results/softwall_multigpu/confirm147_v16_abort_result.json)
- [C148 result](../../results/softwall_multigpu/confirm148_v16_abort_result.json)
- [V16 envelope prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v16.json)
- [V16 validation](../../results/softwall_multigpu/softwall_envelope_v16_validation_summary.json)
- [V16 immutable manifest](../../results/softwall_multigpu/softwall_v16_four_point_manifest.json)
- [V15 sealed base manifest](../../results/softwall_multigpu/softwall_v15_single_token_manifest.json)
