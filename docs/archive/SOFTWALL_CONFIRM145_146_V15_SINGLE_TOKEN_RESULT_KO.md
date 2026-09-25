> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# C145–C146 V15 single-token pipelined-control 재자격 결과

**상태:** 2026-09-24, 두 독립 A100 node의 모든 사전 고정 gate 통과  
**사후 판정:** C145/C146 결과 자체는 PASS. V16의 강화된 four-point physical-fault gate에서는
abort post-apply arm이 없으므로 V15 envelope row는 `UQ`; C147/C148을 합친 V16 row가
`Qualified-safe-useful`이다. Hard-real-time/WCET와 production `d_MAC`은 미입증

## 왜 다시 실행했는가

V14의 물리 campaign C142–C144는 최종 broker outstanding token 0으로 끝났지만,
후속 two-request 감사에서 다음 미검증 전이가 발견됐다.

```text
staged token을 짧은 recovery horizon 때문에 보유
  -> 같은 home이 두 번째 prepare 발행
  -> broker는 두 번째 token도 held
  -> client의 단일 staged slot이 응답을 버림
```

결정적 fake-broker 회귀에서 V14는 broker-held token 2개 중 1개만 추적했다. 이는 기존
물리 결과가 실제로 token을 누출했다는 뜻은 아니지만, V14 finite model과 gate가 허용된
구현 전이를 완전히 덮지 않았다는 뜻이다. 따라서 V14 envelope mode를 UQ로 내리고,
V15에서 `staged ∨ offered`인 동안 새 prepare를 금지했다.

## V15 계약

Home별 ownership 불변식은 다음과 같다.

```text
unlaunched_tokens_h = staged_h + |offered_h| <= 1
staged_h or offered_h  =>  prepare_h is suppressed
physical launch        =>  bounded synchronous commit ACK
physical completion    =>  local fence retire, then asynchronous complete
ambiguous control      =>  global AI admission fail-closed; local RAN continues
```

`unlaunched_tokens` 계수는 client가 알고 있는 staged/offered token을 뜻한다. Prepare가
broker에 적용된 뒤 reply가 유실되면 client가 token ID를 알 수 없으므로 그 token을 tracked
count로 가장하지 않는다. 대신 home 전체를 즉시 quarantine해 이후 prepare를 금지한다.
Control worker가 prepare를 직렬화하므로 이 순간 생길 수 있는 anonymous ambiguous hold도
최대 하나다.

정상·fault arm은 모두 `suppressed_prepare_due_owned_token > 0`과
`maximum_unlaunched_tokens <= 1`을 사전 gate로 요구했다. 종료 시 우연히 drain됐다는
관측만으로 통과시키지 않고, 문제가 된 retained-token branch를 실제 실행 중에 밟게 했다.

## 사전 고정 구성

| 항목 | 값 |
|---|---|
| Node | C145 `nid001244`, C146 `nid003417`; 기존 사용 node와 C146의 C145 node 제외 |
| Topology | A100 2 GPU, GPU별 4-cell recovery home, home별 NRx endpoint 2개와 Qwen 1개 |
| Timing | P180/D155, NRx45, conv25, commit7, AI completion guard2 |
| AI45 transaction | `45+7+2=54 ms`; static slack53 밖, conditional window58 안 |
| 정상 arm | node별 160 release, 총 1,280 TB |
| Fault arm | prepare/commit/complete post-apply fail-stop 각각 200 release, node별 총 4,800 TB |
| Fault semantics | ambiguous token 격리, 신규 global AI 중단, local all-fail RAN 계속 |
| 재현성 | protocol과 source SHA-256을 각 arm 전에 고정; 첫 실패 중단·bound 사후 확장 금지 |

## 결과

| Campaign | Radio records | AI45 exchange | Fault arms | Fault 뒤 radio | 억제된 prepare | 최대 unlaunched token | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| C145 / `nid001244` | 6,080 | 7 | 3 | 3,731 | 273 | 1 | PASS |
| C146 / `nid003417` | 6,080 | 5 | 3 | 3,732 | 264 | 1 | PASS |
| 합계 | **12,160** | **12** | **6** | **7,463** | **537** | **1** | **PASS** |

정상 AI45 arm 두 개는 각각 8셀 1,280 TB에서 conditional exchange를 7회와 5회
실행했다. Broker는 각 arm에서 timely request 9개를 중복 없이 완료했고 최종 outstanding
token은 0이었다. Commit 최대는 C145 0.343877 ms, C146 0.826932 ms로 선언한 7 ms 안이었다.

여섯 fault arm 모두 다음 gate를 통과했다.

- 두 home의 1,600 TB 전체 처리와 fault 감지 뒤 radio continuation
- deadline, NRx/conventional/AI bound와 physical-credit 위반 0
- commit ACK 없는 target의 launch 금지와 complete target의 exactly-once 실행
- global AI fail-closed 뒤 local certificate의 계속 실행
- duplicate request execution 0
- 각 home에서 retained-token 억제 분기 관측, 최대 unlaunched token 1
- source hash mismatch 0

## 모델과 envelope의 연결

결정적 회귀는 V14의 `untracked_held_tokens=1`을 재현하고 V15에서 0으로 만든다. Finite
model v2는 기존 reply-loss protocol의 16 state/20 edge와 single-token ownership의
4 state/8 edge를 함께 검사했고 violation은 0이다. 물리 branch counter와 두 node 결과를
추가한 envelope v15는 다음처럼 분류한다.

두 하위 모델은 각각의 bounded state space를 검사하며 모든 thread interleaving의 완전한
Cartesian product 증명은 아니다. Fault arm의 fail-closed 결과와 retained-token counter가
두 모델이 실제 구현에서 만나는 지점을 보완한다.

```text
V14 pipelined mode: UQ  (ownership branch 미자격)
V15 single-token mode 당시 판정: QSU
V15 grid 당시: QSU 6 / QSN 0 / MI 3 / UQ 12
```

Model verifier를 16개 Python hash seed로 다시 실행하자 raw state-list 순서는 16가지였지만
canonical state/edge multiset, count와 invariant 판정은 모두 동일했다. 검증 summary의
13개 gate가 모두 통과했고, 139개 파일을 묶은 manifest는 재생성해도 byte-identical했다.
후속 coverage 감사는 finite model의 네 operation 중 abort reply-loss가 물리 fault arm에서
빠졌음을 찾았다. C145/C146을 실패로 바꾸지는 않지만 더 강한 V16 gate 아래 V15 row를
UQ로 내린다. [C147/C148](SOFTWALL_CONFIRM147_148_V16_FOUR_POINT_RESULT_KO.md)이 이 공백을
두 새 node에서 닫는다.

## 논문에서 말할 수 있는 것

V15는 두 추가 A100 node의 유한 표본에서 다음 인과 사슬을 닫는다.

1. Conditional recovery가 static slack 밖의 AI45 class를 연다.
2. Global prepare/complete tail은 RAN critical path 밖으로 분리된다.
3. Launch 권한은 bounded commit ACK 뒤에만 생긴다.
4. Retained token이 있어도 home당 미실행 ownership은 하나를 넘지 않는다.
5. 각 control point의 ambiguity는 신규 AI만 닫고 local RAN certificate를 훼손하지 않는다.

이는 production deadline이나 WCET 증명이 아니다. Broker restart 뒤 durable
reconciliation, shared NRx/recovery 자원을 가진 non-separable certificate, 다른 GPU
family도 아직 자격화하지 않았다.

## 권위 산출물

- [C145 result](../../results/softwall_multigpu/confirm145_v15_single_token_result.json)
- [C146 result](../../results/softwall_multigpu/confirm146_v15_single_token_result.json)
- [V14→V15 deterministic regression](../../results/softwall_multigpu/softwall_v14_v15_staged_ownership_regression_v1.json)
- [Finite model v2](../../results/softwall_multigpu/softwall_pipelined_control_model_v2.json)
- [Finite model semantic reproducibility](../../results/softwall_multigpu/softwall_pipelined_control_model_v2_reproducibility_v1.json)
- [Envelope v15 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v15.json)
- [Envelope v15 validation](../../results/softwall_multigpu/softwall_envelope_v15_validation_summary.json)
- [V15 immutable manifest](../../results/softwall_multigpu/softwall_v15_single_token_manifest.json)
- [C147--C148 V16 four-point 결과](SOFTWALL_CONFIRM147_148_V16_FOUR_POINT_RESULT_KO.md)
