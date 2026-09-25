# Confirm118 멀티 GPU component-bound 재자격 결과

**실행일:** 2026-09-24  
**환경:** job `58815435`, `nid001016`, A100-SXM4 40GB 3장, MIG OFF, MPS ON  
**판정:** **PASS — 두 독립 arm의 system-safety 및 사전 고정 component gate 전부 통과**

## 1. C117과의 관계

C117은 전체 시스템 safety를 통과했지만 GPU0 front/back의 2 ms 후보가 각각 한 건씩
깨져 component qualification에 실패했다. 그 결과를 수정하거나 삭제하지 않고, C118은
새 protocol에서 다음 규칙을 실행 전에 고정했다.

- GPU0 front/back만 2 ms에서 **3 ms**로 변경
- P2P forward 250 us, remote NRx 2.5 ms, P2P backward 100 us, remote worker 6 ms,
  end-to-end NRx 25 ms는 그대로 유지
- C117과 다른 payload/channel seed 두 개 사용
- 두 arm 모두 system safety와 모든 component에서 위반 0이어야 통과

[C118 protocol](../../results/softwall_multigpu/confirm118_component_requalification_protocol.json)은
이 규칙과 실행 전 source hash를 보존한다.

## 2. 결과

| 항목 | seed3 | seed4 | Bound | 판정 |
|---|---:|---:|---:|---|
| GPU0 front 최대 | 1.831 ms | **2.732 ms** | 3 ms | PASS |
| GPU0 back 최대 | 1.886 ms | 1.371 ms | 3 ms | PASS |
| End-to-end NRx 최대 | 12.579 ms | 11.565 ms | 25 ms | PASS |
| Remote forward 최대 | 157.408 us | 122.432 us | 250 us | PASS |
| Remote NRx 최대 | 1.702 ms | 1.648 ms | 2.5 ms | PASS |
| Remote backward 최대 | 40.704 us | 34.592 us | 100 us | PASS |
| Remote worker 최대 | 4.209 ms | 2.483 ms | 6 ms | PASS |

Remote worker 수는 controller의 timed endpoint admission 수와 정확히 일치했다. 두 arm의
요약은 다음과 같다.

| 항목 | seed3 | seed4 |
|---|---:|---:|
| Release / TB | 340 / 1,360 | 340 / 1,360 |
| NRx commit | 535 | 515 |
| Conventional commit | 825 | 845 |
| Atomic retime+AI lease | 91 | 96 |
| Timely Qwen unit | 888 | 889 |
| Timely token value | 225,131 | 225,402 |
| Deadline/bound/guard/fault/credit 위반 | 0 | 0 |

Arm wrapper에 보이는 remote worker 최대 44--48 ms는 worker 초기 warmup을 포함한다. Frozen
analyzer는 protocol의 warmup count를 제거한 timed record만 component gate에 사용했다.

## 3. 정확한 의미

C118은 이 exact mode의 finite-sample service vector를 제공한다.

```text
mode = (4 cells, local1+remote2 endpoints, GPU0 Qwen,
        NVLink P2P, MPS cap40/cap40/cap40 + Qwen cap20,
        warm, GC OFF, mixed correlated fault, P180/D155)

qualified candidate vector =
  front 3 ms / forward 250 us / remote NRx 2.5 ms /
  backward 100 us / remote worker 6 ms / back 3 ms /
  end-to-end NRx 25 ms
```

다음 주장은 하지 않는다.

- 3 ms가 WCET라는 주장
- cold start, GC ON, worker restart, 다른 topology에도 같은 bound가 적용된다는 주장
- 3 GPU가 단일 GPU보다 AI 처리량을 높였다는 주장
- 실제 DU `d_MAC` deadline을 만족했다는 주장

특히 seed4 front 최대 2.732 ms는 3 ms 후보와 가깝다. 이 값은 여유 있는 보편 상한이
아니라 현재 mode에서 다음 prospective envelope 계산에 사용할 자격화 후보로만 취급한다.

## 4. 설계와 모델에 반영할 내용

1. Endpoint profile은 평균이나 isolated NRx 시간 하나가 아니라 component-bound vector를
   가진다.
2. GPU0 front/back bound는 remote transport와 Qwen의 co-run class에 조건부다.
3. Endpoint eligibility는 component 합성 bound와 fallback cutoff로 계산한다.
4. C117의 2 ms 거절점과 C118의 3 ms 통과점을 envelope의 관측 경계로 함께 보존한다.
5. 다음 scale 실험은 이 vector로 실행 전에 safe/unsafe 지점을 예측한 뒤 경계 안팎만
   실행한다.

## 5. 권위 artifact

- [C118 aggregate 판정](../../results/softwall_multigpu/confirm118_component_requalification_job58815435.json)
- [C118 사전 protocol](../../results/softwall_multigpu/confirm118_component_requalification_protocol.json)
- [C118 seed3 arm](../../results/softwall_multigpu/c118_components_s3_job58815435_result.json)
- [C118 seed4 arm](../../results/softwall_multigpu/c118_components_s4_job58815435_result.json)
- [C117/C118 artifact manifest](../../results/softwall_multigpu/confirm117_118_artifact_manifest.json)
- [C117 실패 경계 설명](SOFTWALL_CONFIRM117_COMPONENT_BOUND_RESULT_KO.md)
