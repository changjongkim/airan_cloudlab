> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall V14 staged-token ownership 반례와 V15 교정

**상태:** 2026-09-24, V14 반례 재현·V15 CPU/model/두-node 물리 gate PASS; V16 four-point 사후 감사 반영  
**판정:** C142--C144의 관측 실행은 모두 drain됐지만, V14 구현에는 실험이 직접 gate하지
않은 ownership branch가 있어 해당 mode의 현재 envelope 자격은 UQ다. 이 branch를 막고
직접 계측한 V15는 C145/C146 당시 QSU로 자격화됐다. 후속 V16은 abort post-apply 물리
fault coverage를 추가로 요구하므로 V15 three-point row는 현재 UQ이고, C147/C148을 합친
V16 row가 QSU다.

## 반례

V14는 현재 recovery horizon이 짧아도 AI request deadline이 남아 있으면 prepared token을
`staged`로 유지한다. 이 규칙 자체는 필요하다. 그러나 V14 `peek_edf_fitting`은 staged token이
맞지 않는 같은 poll에서 `_prepare_pending == false`이면 두 번째 `prepare`도 제출할 수 있었다.
Broker가 두 번째 request를 `held`로 바꾸어 반환하면 control worker는 첫 staged slot이 이미
차 있어 그 reply를 저장하지 않았다.

```text
broker: held(token0), held(token1)
client: staged(token0)
                    ^ token1의 local owner state가 없음
```

[결정적 두-request 감사](../../results/softwall_multigpu/softwall_v14_v15_staged_ownership_regression_v1.json)는
V14에서 prepare 21회, broker-held 2개, client-tracked 1개, untracked 1개를 재현했다.
C142와 C144의 AI45 physical arm은 최종 `outstanding_tokens=0`이므로 이 누수가 실제 campaign에
발생했다는 증거는 없다. 그러나 “발생하지 않았음”을 요구하는 frozen gate가 없었으므로
상태공간 안전 주장에는 사용할 수 없다.

## V15 교정

V15는 home마다 아직 launch하지 않은 global token을 최대 하나만 허용한다.

```text
prepare 가능 <=> staged 없음 AND offered 없음 AND prepare_pending 아님
```

Inflight physical AI와 다음 staged token 하나의 prefetch는 허용한다. 금지하는 것은 staged
또는 controller에 handed-off된 offered token 뒤의 추가 prepare다. Telemetry는 다음을 직접
기록한다.

- `suppressed_prepare_due_owned_token`
- `maximum_unlaunched_tokens`
- `ownership_invariant_holds`

새 단위시험은 짧은 horizon poll 20회 뒤에도 prepare가 1회이고 broker state가
`[held, ready]`, untracked token이 0임을 확인한다. Offered token을 commit/release하기 전에도
추가 prepare가 억제된다.

## 모델 교정

V14의 16-state model은 한 global request의 control failure만 열거해 두 번째 prepare 분기를
표현하지 못했다. V15 model v2는 기존 16-state/20-edge fault model과 별도로 one-home
single-token ownership 4-state/8-edge model을 추가한다. 두 모델의 invariant violation은 0이다.

Envelope checker v10은 pipelined mode에 다음 두 field를 요구한다.

```text
single_token_ownership_required = true
single_token_ownership_qualified = true
```

V14는 첫 field만 참이므로 QSU에서 UQ로 강등된다. 물리 gate 전 V15의 임시 분류는
`QSU5/QSN0/MI3/UQ13`이었다. C145/C146이 두 독립 A100 node에서 AI45, 세 control fault
point, retained-token branch와 max-unlaunched=1을 모두 통과해 V15를 QSU로 승격했다.
당시 분류는 `QSU6/QSN0/MI3/UQ12`였다. V16 four-point gate의 현재 분류는
`QSU6/QSN0/MI3/UQ13`이다.

## 물리 재자격 gate

각 node에서 다음 네 arm을 실행한다.

1. AI45 conditional exchange 1 arm
2. post-apply prepare fault 1 arm
3. post-apply commit fault 1 arm
4. post-apply complete fault 1 arm

모든 arm은 기존 safety·duplicate·commit7 gate에 더해 다음을 만족해야 한다.

```text
maximum_unlaunched_tokens <= 1
suppressed_prepare_due_owned_token > 0
broker outstanding_tokens = 0  # 정상 AI45 arm
```

첫 실패에서 campaign을 중단하고 bound나 gate를 사후 완화하지 않는다.

C145 `nid001244`와 C146 `nid003417`은 각각 네 arm을 모두 통과했다. 두 node 합계는
12,160 TB, AI45 exchange 12회, control fault 6 arm, fault 뒤 radio 7,463건이다.
Retained-token 억제 분기는 537회 관측됐고 모든 arm의 최대 unlaunched token은 1,
safety·duplicate·source-hash 위반은 0이었다.

## Artifact

- [V14/V15 ownership regression](../../results/softwall_multigpu/softwall_v14_v15_staged_ownership_regression_v1.json)
- [V15 finite model v2](../../results/softwall_multigpu/softwall_pipelined_control_model_v2.json)
- [C145--C146 결과](SOFTWALL_CONFIRM145_146_V15_SINGLE_TOKEN_RESULT_KO.md)
- [V15 qualified envelope](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v15.json)
- [V15 validation](../../results/softwall_multigpu/softwall_envelope_v15_validation_summary.json)
- [V15 immutable manifest](../../results/softwall_multigpu/softwall_v15_single_token_manifest.json)
- [V15 client](../../scripts_for_node/softwall_same_gpu/single_token_pipelined_global_trace_client.py)
- [V15 unit tests](../../scripts_for_node/softwall_same_gpu/test_single_token_pipelined_global_trace_client.py)
