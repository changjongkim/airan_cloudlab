# Confirm136: V12 AI40 conditional class의 독립 node 재자격

**상태:** 2026-09-24, 실행 전 protocol 동결 후 6/6 arm PASS  
**Allocation:** job 58829672, `nid002817`, A100 4장  
**범위:** 두 RAN home, home당 4셀·NRx endpoint 2개·ring depth 1,
MIG OFF/MPS ON, synthetic context-128 Qwen

## 목적

C135는 V12의 완전한 57 ms transaction이 static all-fail slack 53 ms에는 들어가지 않고
conditional decision window 58 ms에만 들어가는 것을 한 node의 두 arm에서 보였다.
Confirm136은 같은 source와 contract를 유지하면서 C135·C134·검증에 사용한 node를 제외하고,
새 node와 여섯 독립 seed/process lifecycle에서 이 결과를 재자격한다.

고정한 산술은 다음과 같다.

```text
static all-fail slack            155 - 2 - 4*25 = 53 ms
conditional recovery boundary   155 - 2 - 2*25 = 103 ms
observe-all decision bound      45 ms
conditional window              103 - 45 = 58 ms
complete transaction            AI40 + control15 + completion guard2 = 57 ms
static margin                   53 - 57 = -4 ms
conditional margin              58 - 57 = +1 ms
```

## 사전 protocol

[동결 protocol](../../results/softwall_multigpu/confirm136_v12_ai40_requalification_protocol.json)은
다음을 실행 전에 고정했다.

- 6개 arm, arm마다 서로 다른 두 home의 payload/channel seed
- arm마다 160 release와 20 warmup, 총 1,280 TB
- `nid001245`, `nid002688`, `nid001781` 제외
- 각 arm에서 exact `2 admitted/2 success/2 rejected-live` branch와 AI40 exchange 필수
- 57 ms 완전 charge를 기준으로 static 거절·conditional 수락 필수
- 첫 safety·bound·credit·source·artifact 실패에서 즉시 중단
- 실패 뒤 bound를 넓히거나 seed를 교체하지 않음

## 결과

새 node `nid002817`에서 모든 arm이 통과했다.

| Arm | Radio TB | Candidate branch | AI40 candidate exchange | 최소 physical guarded margin |
|---|---:|---:|---:|---:|
| s1 | 1,280 | 153 | 4 | 52.256 ms |
| s2 | 1,280 | 161 | 4 | 50.262 ms |
| s3 | 1,280 | 149 | 6 | 50.845 ms |
| s4 | 1,280 | 137 | 4 | 51.463 ms |
| s5 | 1,280 | 153 | 5 | 48.965 ms |
| s6 | 1,280 | 155 | 5 | 45.970 ms |
| **합계** | **7,680** | **908** | **28** | **45.970 ms** |

Atomic exchange는 총 53회였다. 28개 목표 exchange는 모두 동일 사건 기준으로 static
계약에서 거절되고 conditional 계약에서 수락됐다. Deadline, NRx/conventional/AI bound,
horizon, credit, duplicate와 source/artifact gate 위반은 0이었다.

관측 최대는 NRx response 28.098 ms, conventional host path 6.722 ms, AI host execution
28.453 ms, AI GPU 27.284 ms였다. 이는 선언한 45/25/40 ms 안의 표본 최대이며 WCET로
승격하지 않는다.

## C135와 합친 결과

[결합 감사](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)는
두 node의 8개 arm을 합친다.

| 항목 | 결합 결과 |
|---|---:|
| A100 node | 2 |
| 독립 seed/process arm | 8 |
| Radio TB | 10,240 |
| Home release | 2,560 |
| Atomic exchange | 72 |
| Candidate branch | 1,196 |
| AI40 candidate exchange | 38 |
| 선언 safety 위반 | 0 |

38/38 exchange는 static margin −4 ms, conditional margin +1 ms 조건을 만족했다.

## Zero-failure 상한의 해석

실패 0회인 Bernoulli trial `N`개에 대해 95% one-sided exact upper bound는
`1-0.05^(1/N)`이다. 어떤 단위를 독립 trial로 두는지에 따라 결과가 크게 달라진다.

| 독립성 가정 | N | 95% failure-probability upper bound |
|---|---:|---:|
| TB가 IID | 10,240 | 0.0293% |
| Home release가 IID | 2,560 | 0.1170% |
| Arm이 IID | 8 | 31.23% |
| Node가 IID | 2 | 77.64% |

TB-level 숫자만 보고 신뢰도를 주장하지 않는다. Persistent process 안의 요청은 서로
의존할 수 있고 두 node만으로 hardware population을 일반화할 수 없다. 따라서 논문의
보수적 표현은 **두 A100 node·8 arm의 finite-sample requalification**이다. 이 표는 표본
근거의 범위가 어디까지인지 드러내기 위한 sensitivity이며 hard-real-time 증명이 아니다.

## 논문에 허용되는 주장

허용:

> The complete 57 ms conditional AI transaction was requalified on a second
> A100 node across six new process/seed arms. Combined with C135, all 38
> observed target exchanges across two nodes were infeasible under the static
> 53 ms contract but feasible under the 58 ms conditional contract, with no
> declared safety violation in 10,240 radio TBs.

허용하지 않음:

- 10,240 TB가 WCET 또는 failure probability 0을 증명한다는 주장
- 두 A100 node 결과를 다른 GPU family에 일반화
- synthetic context-128 workload를 BurstGPT 처리량 결과로 해석
- production DU `d_MAC`, broker restart 또는 exactly-once 주장

## Artifact

- [C136 result](../../results/softwall_multigpu/confirm136_v12_ai40_requalification_result.json)
- [C136 manifest](../../results/softwall_multigpu/confirm136_artifact_manifest.json)
- [Allocation environment](../../results/softwall_multigpu/raw/confirm136_environment_job58829672.json)
- [Combined C135/C136 audit](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [V12 cross-node validation summary](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_validation_summary.json)
- [V12 cross-node manifest](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_manifest.json)
- [Analyzer](../../scripts_for_node/softwall_same_gpu/analyze_confirm136_v12_requalification.py)
- [Zero-failure unit tests](../../scripts_for_node/softwall_same_gpu/test_analyze_confirm136_v12_requalification.py)
