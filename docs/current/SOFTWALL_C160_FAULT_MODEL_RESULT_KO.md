# C160 fault state-model 결과

**판정:** `C160_MODEL_PASS`  
**범위:** epoch/generation/idempotency와 AI physical-fence lifecycle의 유한 상태 감사  
**결과:** `results/softwall_multigpu/c160_fault_state_model_v1.json`

## 검증한 전이

- 이전/현재/미래 epoch의 NeuralRx outcome
- 이전/현재/미래 generation
- common-cutoff success batch의 duplicate transaction
- Qwen lease commit, physical start, latest-start non-launch
- completion fence와 matching/wrong/missing marker
- post-fence 및 pre-fence RPC timeout
- fence-required lease retire
- stale/duplicate radio commit

Unit test 8개와 51개 state-transition product를 실행했다. Reject 경로 state mutation,
fenceless lease retire, duplicate physical launch와 duplicate radio commit은 모두 0이었다.

## 첫 실행 실패와 교정

첫 analyzer는 정상적으로 fence-confirmed retire한 뒤에도 fence 이력을 상태에 보존하지 않아
`unfenced_lease_not_released` 한 건을 잘못 보고했다. Runtime transition 위반은 아니었지만
검사기가 해당 결론을 증명할 정보가 부족했다. 첫 결과를
`c160_fault_state_model_v1_attempt1_audit_failure.json`으로 보존하고 상태에
`physical_fences`, `retired_leases`, `fenceless_retires`를 추가했다.

교정 뒤 모든 lease retire에 대해 다음을 직접 검사한다.

```text
fenceless_retires == 0
retired_leases <= physical_fences
```

## 의미와 한계

이 결과는 C161 물리 fault matrix가 따라야 할 transaction semantics를 고정한다. 특히
pre-fence timeout은 AI를 quarantine하면서 lease를 보유하고, matching completion marker가
있는 post-fence timeout만 lease를 반환한다.

CPU pure-state 결과이므로 CUDA worker crash, MPS client teardown, GPU hang이나 radio
continuation을 입증하지 않는다. 이 항목들은 C161의 실제 P180 경로에서 별도로 검증해야 한다.
