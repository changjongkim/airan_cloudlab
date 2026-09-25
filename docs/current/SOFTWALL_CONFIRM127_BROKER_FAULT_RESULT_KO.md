# SoftWall C127: global AI commit 응답 유실의 격리와 RAN 지속

**상태:** 2026-09-24, 사전 고정 2-arm PASS  
**대상:** 2×A100, home당 4셀·2 NeuralRx endpoint·Qwen worker, MIG OFF/MPS ON  
**주장 범위:** global AI ownership 장애를 한 home에 격리하면서 local RAN certificate를 계속 실행

## 1. 왜 이 실험이 필요했는가

C123은 한 BurstGPT queue의 request를 두 RAN home이 중복 없이 나눠 처리할 수 있음을
보였다. C126은 각 home의 ready recovery를 live certificate 순서로 실행해야 함을 보였다.
그러나 global broker가 `commit`을 적용한 직후 응답이 사라지면 home은 commit 성공 여부를
알 수 없다. 같은 요청을 재시도하거나 다른 home에 돌려주면 Qwen 요청이 두 번 실행될 수
있다. 반대로 전체 controller를 종료하면 AI control-plane fault가 RAN availability까지
전파된다.

C127의 질문은 다음과 같다.

> Commit 결과가 애매한 AI request 하나를 희생하더라도, 이를 재시도하지 않고 해당
> home의 신규 global AI만 닫아 두 RAN home과 다른 home의 AI를 계속 실행할 수 있는가?

## 2. fault-contained protocol

정상 경로는 다음 순서를 사용한다.

```text
global prepare(held)
  -> local recovery replan + bounded-AI lease
  -> global commit(inflight)
  -> Qwen GPU 실행
  -> physical completion
  -> local AI lease retire + global complete
```

`commit` RPC가 성공 응답을 주지 않으면 다음 규칙을 적용한다.

1. 같은 global token의 commit을 재시도하지 않는다.
2. 그 request를 ready queue로 돌려놓거나 다른 home에 재할당하지 않는다.
3. Qwen은 commit 성공 응답 뒤에만 launch하므로, 이 fault point에서는 GPU 작업이
   제출되지 않았다. 해당 local AI execution lease는 confirmed-unlaunched 상태로 회수한다.
4. 영향받은 home의 신규 global AI admission만 닫는다.
5. Local all-fail recovery calendar와 certificate-ordered conventional execution은 계속한다.
6. 다른 home은 broker의 다른 연결을 통해 AI request를 계속 처리한다.

이 선택은 exactly-once recovery가 아니라 **at-most-once fail-closed containment**다. Broker에
남는 `inflight` token 하나는 누수가 아니라, 실행 여부가 불명확한 request를 재사용하지
않았다는 격리 표식이다. 이를 자동 회수하려면 durable execution ID와 worker-side 실행
기록을 이용한 별도 reconciliation protocol이 필요하다.

## 3. fault 주입과 사전 판정

두 formal arm 전에 source와 gate를
[protocol](../../results/softwall_multigpu/confirm127_broker_fault_protocol.json)에 고정했다.
각 arm에서 home 0의 30번째 global commit을 broker state에 적용한 뒤 응답을 쓰기 전에
그 client connection만 닫았다. Broker process, home 1 연결, 두 RAN controller와 GPU
worker는 계속 실행했다. 개발 canary는 formal count에서 제외했다.

주요 gate는 다음과 같다.

- 각 home의 340 release·1,360 TB가 마지막 release까지 완료될 것
- deadline, NRx, conventional, AI class bound와 recovery guard 위반이 0일 것
- home 0이 commit ambiguity를 정확히 한 번 기록하고 신규 global AI를 닫을 것
- ambiguous request가 실제 Qwen 기록에 없고 broker의 유일한 outstanding token일 것
- duplicate commit과 duplicate execution이 0일 것
- home 1이 fault 뒤에도 AI를 실행할 것
- 모든 local recovery, endpoint와 AI execution credit이 0으로 drain될 것

## 4. 결과

| 항목 | Seed 1 | Seed 2 | 합계/판정 |
|---|---:|---:|---:|
| RAN TB | 2,720 | 2,720 | **5,440** |
| home 0 fault 뒤 RAN TB | 1,244 | 1,246 | **2,490** |
| home 1 fault 뒤 AI unit | 832 | 836 | **1,668** |
| home 0 실제 AI 완료 | 29 | 29 | 58 |
| home 1 실제 AI 완료 | 862 | 863 | 1,725 |
| deadline miss | 0 | 0 | **0** |
| NRx/conv/AI bound·guard 위반 | 0 | 0 | **0** |
| ambiguous broker token | 1 | 1 | arm당 정확히 1 |
| duplicate commit/execution | 0 | 0 | **0** |
| local recovery/AI lease 잔여 | 0 | 0 | **0** |

Seed 1과 seed 2의 broker는 각각 global request 891개와 892개를 적시에 완료했다. Home 0은
30번째 commit의 응답 유실 뒤 29개 완료 상태에서 AI admission을 중단했다. Home 1은 같은
broker process에서 각각 862개와 863개를 완료했다. 두 ambiguous request는 실제 Qwen
execution record에 없고 broker state에서만 `inflight`로 남았다.

[frozen aggregate](../../results/softwall_multigpu/confirm127_broker_fault_result.json)는 모든
사전 gate를 통과했다. [artifact manifest](../../results/softwall_multigpu/confirm127_artifact_manifest.json)는
28개 formal artifact와 고정 source hash의 일치를 확인한다.
전체 회귀 검증은 runtime 21개를 포함한 단위시험 55개와 fault campaign 10,000회를
통과했다([validation summary](../../results/softwall_multigpu/confirm127_validation_summary.json)).

## 5. 이 결과가 추가하는 설계 근거

C127은 global request uniqueness와 local recovery safety의 failure domain을 분리한다.
Global AI control 경로가 애매해져도 이미 유지 중인 RAN certificate에는 변경을 가하지
않는다. 영향받은 home은 외부 AI 처리량을 포기하고 RAN을 계속하며, 다른 home은 독립 local
certificate 아래 AI를 계속한다. 따라서 current multi-home scheme은 다음 세 조건을 함께
갖는다.

1. Broker의 generation-tagged hold/commit으로 정상 경로의 global request ownership을
   직렬화한다.
2. 각 home은 all-fail recovery certificate와 local execution credit을 독립적으로 유지한다.
3. Ambiguous global transition은 재시도하지 않고 한 home의 AI admission boundary에서
   격리한다.

이 조합은 단순한 multi-GPU 배치나 MPS scheduling 주장보다 좁고 검증 가능하다. 현재
차별점은 **optional per-TB NeuralRx가 만든 conditional recovery debt, bounded external-AI
lease, global request ownership과 ambiguity containment를 물리 completion lifecycle로
연결한 substrate**다.

## 6. 주장할 수 없는 것

- Broker crash 뒤 durable state 복구나 leader failover
- Network partition, Byzantine broker/home 또는 여러 동시 ambiguity
- Ambiguous request를 손실 없이 exactly once로 복구하는 protocol
- GPU/driver hang 뒤 local RAN 지속
- Shared mandatory recovery lane이나 shared NRx endpoint의 multi-home atomic certificate
- Production DU deadline이나 hard WCET

C127은 위 문제를 해결한 결과가 아니다. 검증한 fault는 broker가 살아 있는 상태에서 한
client의 post-apply commit response만 유실된 경우다. 이 범위 안에서는 AI control fault가
전체 RAN controller 중단으로 번지지 않는다는 물리 증거를 제공한다.

## 7. 다음 우선순위

1. Production DU/FAPI에서 `d_MAC`을 계측해 synthetic expiry를 실제 timing contract로 교체한다.
2. Mode qualification을 sample maximum이 아닌 tail model과 violation budget으로 정식화한다.
3. Durable worker execution record를 추가해 broker restart와 ambiguous token reconciliation을
   검증한다.
4. Shared NRx/recovery resource가 필요하다면 multi-home generation과 credit을 포함한
   multi-resource certificate를 새로 설계한다.

Fault class를 mode 축으로 추가한
[sharded-home envelope v5](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v5.json)는
처리 규칙이 없는 reply-loss mode를 UQ, C127의 no-retry quarantine mode를 QSU로 분리한다.
