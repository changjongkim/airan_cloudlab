# C161 1단계 실제 GPU fault 결과

**판정:** `C161_PHASE1_TWO_NODE_PASS`  
**범위:** A0 no-fault, A1 correlated all-fail, A4 physical latest-start non-launch  
**결합 결과:** `results/softwall_multigpu/c161_phase1_two_node.json`

## 무엇을 검증했는가

C160은 epoch, generation, transaction id와 completion fence의 상태전이 의미를 CPU 모델로
고정했다. C161 1단계는 그중 두 핵심 분기를 실제 P180/D155 경로에 연결했다.

1. **A1 correlated all-fail:** 네 actual NeuralRx 결과를 모두 물리적으로 실행한 뒤, common
   cutoff에서 관측된 success 공개를 제어된 방식으로 억제했다. Runtime은 네 unresolved debt를
   유지하고 GPU2의 shared cuPHY conventional recovery를 네 건 모두 실행해야 한다.
2. **A4 latest-start non-launch:** certificate와 AI lease를 commit한 뒤 host dispatch를
   `latest_start` 이후까지 의도적으로 지연했다. Qwen worker는 CUDA kernel을 시작하지 않고
   검증된 empty lease만 retire해야 한다.

A1은 NeuralRx 계산 자체를 생략한 합성 실패가 아니다. 네 GPU0/1 입력이 CUDA IPC와 P2P를
통해 GPU3의 persistent actual NeuralRx endpoint에 도달하고 물리 결과가 만들어진 뒤,
controller의 결과 공개 지점에서 fault를 주입했다. 따라서 실제 data path와 correlated outcome
처리를 함께 검사한다.

## 실행 독립성

| Campaign | Job / node | Seed | arm 순서 | 판정 |
|---|---|---:|---|---|
| Development | 58859044 / nid001824 | 36,000,000 | A0 → A1 → A4 | 15/15 gate PASS |
| Holdout | 58859145 / nid001025 | 37,000,000 | A4 → A1 → A0 | 15/15 gate PASS |

Holdout protocol은 development node를 사전 제외했다. 두 실행은 서로 다른 allocation, node,
seed를 사용했고 arm 순서를 반대로 배치했다. 두 작업은 모두 exit code 0으로 끝났으며 allocation은
반환됐다.

## 결합 결과

| 항목 | Development | Holdout | 합계/최대 |
|---|---:|---:|---:|
| Radio epoch | 90 | 90 | 180 |
| Actual NeuralRx request | 360 | 360 | 720 |
| Physical NeuralRx success | 246 | 264 | 510 |
| Fault 적용 뒤 effective success | 169 | 179 | 348 |
| Physical conventional recovery | 191 | 181 | 372 |
| A1 correlated recovery | 120 | 120 | 240 |
| 완료된 Qwen unit | 25 | 27 | 52 |
| A4 guarded physical non-launch | 25 | 27 | 52 |
| Radio single commit | 360 | 360 | 720 |
| Deadline miss | 0 | 0 | 0 |
| NRx host path 최대 | 19.350 ms | 20.091 ms | 20.091 ms |
| Recovery host path 최대 | 12.582 ms | 6.853 ms | 12.582 ms |
| Qwen host path 최대 | 52.837 ms | 52.924 ms | 52.924 ms |
| Radio commit 최대 | 125.205 ms | 123.968 ms | 125.205 ms |

두 campaign에서 source hash, pre-staged input, full recovery preflight, actual NRx/recovery
transport, common-cutoff atomic batch, Qwen bound와 physical start guard, single timely commit,
CUDA IPC lifecycle을 포함한 15개 gate가 모두 통과했다.

## 현재 주장할 수 있는 것

Warm persistent 4×A100 P180 mode에서 네 actual NeuralRx outcome이 상관 실패한 것으로
처리돼도 all-fail recovery certificate가 네 물리 conventional recovery를 D155 전에
실행했다. 또한 lease commit 뒤 제어 tail로 launch window를 잃은 경우에는 worker-side
absolute latest-start 검사가 Qwen kernel을 실제로 시작하지 않고 empty lease를 회수했다.

이 결과는 conditional recovery credit과 AI lease가 단순한 analytic schedule이 아니라,
actual NeuralRx, shared cuPHY, Qwen, CUDA IPC/P2P lifecycle에 연결된 fault-aware substrate라는
주장을 강화한다.

## 아직 닫히지 않은 범위

C161 전체가 끝난 것은 아니다. A2 stale/duplicate NeuralRx outcome, A3 post-fence Qwen reply
delay, A5 pre-fence channel loss, A6 stale/duplicate recovery reply의 실제 GPU arm은 아직
실행하지 않았다. Response loss, worker crash, missing completion fence, GPU/driver hang,
cold/restart lifecycle, production `d_MAC`, 다른 GPU family는 계속 UQ다. 다음 gate는 A2/A3/A5/A6를
동결된 2단계 source로 구현하고 새 development/holdout node에서 반대 순서로 검증하는 것이다.
