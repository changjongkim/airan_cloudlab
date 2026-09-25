> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# Confirm113 실제 trace 강한 baseline 최종 판정

**상태:** 2026-09-24 frozen campaign 완료  
**권위 결과:** [confirm113 분석](../../results/softwall_same_gpu/confirm113_strong_baselines_job58815435.json)  
**사전 protocol:** [confirm113 protocol](../../results/softwall_same_gpu/confirm113_strong_baselines_protocol.json)  
**원본 manifest:** [53개 artifact SHA-256](../../results/softwall_same_gpu/confirm113_artifact_manifest.json)

## 결론

SoftWall의 조건부 복구 substrate는 실제 BurstGPT 도착·요청 크기를 Qwen2.5-1.5B
prefill로 매핑한 4셀 MIG-off MPS mode에서 안전하게 동작했다. 두 독립 PHY seed의
SoftWall arm 네 개 모두에서 복수 recovery-credit tail compaction과 AI lease의 원자
transaction이 실행됐고, deadline·NRx/복구/AI bound·guard·credit/fault 위반은 0이었다.

그러나 가장 강한 `safe work-conserving` 기준선보다 적시 token value가 2% 이상 높다는
성능 가설은 기각됐다. 효과는 두 seed에서 `+0.113%`, `+0.099%`였고 paired
source-second bootstrap 95% CI는 모두 0을 포함했다. 따라서 이 결과는 optimizer나
처리량 우위가 아니라 **안전한 조건부 복구 substrate와 no-benefit feasibility boundary**를
지지한다.

## 고정한 구성

| 항목 | 값 |
|---|---|
| GPU | NVIDIA A100 한 장, MIG OFF, MPS ON |
| RAN | 4셀, P180/D155, NRx endpoint 2개 cap40, AI cap20 |
| Bounds | NRx45 ms, conventional host path12 ms, guard2 ms |
| AI | Qwen2.5-1.5B full-transformer prefill, final-token LM head |
| AI bounds | 16/32/64=35 ms, 128=40 ms, 256=65 ms, 512=75 ms |
| Trace | BurstGPT densest 60초, 1,136 request, value 288,591 input token |
| AI deadline | 원 trace에 없으므로 고정 synthetic 1,000 ms |
| Fault | 5 release마다 single/correlated NRx failure 교대 |
| 반복 | seed1 `work→SoftWall→SoftWall→work`, seed2 반대 순서 |

모든 시스템은 같은 observable feature, radio admission, endpoint routing, all-fail
certificate, early conventional, physical completion 확인을 사용했다. 차이는 recovery
credit을 고정하는지, 현재 빈 구간만 쓰는지, 남은 복수 credit과 AI lease를 원자적으로
재배치하는지뿐이다.

## 결과

| Seed/시스템 | 반복별 적시 request | 반복별 적시 token value | Retiming+lease | Safety |
|---|---:|---:|---:|---|
| seed1 static | 293 | 73,841 | 0 | PASS |
| seed1 work-conserving | 887 / 887 | 224,862 / 224,921 | 0 | PASS |
| seed1 SoftWall | 889 / 887 | 225,273 / 225,020 | 81 / 81 | PASS |
| seed2 static | 293 | 73,770 | 0 | PASS |
| seed2 work-conserving | 886 / 885 | 224,723 / 224,493 | 0 | PASS |
| seed2 SoftWall | 889 / 887 | 224,877 / 224,783 | 88 / 88 | PASS |

| 사전 고정 비교 | SoftWall 평균 | Work-conserving 평균 | 차이 | 효과 | Bootstrap 95% CI | 판정 |
|---|---:|---:|---:|---:|---:|---|
| seed1 ABBA | 225,146.5 | 224,891.5 | +255 | +0.113% | [−121.5, 710.0] | FAIL |
| seed2 BAAB | 224,830.0 | 224,608.0 | +222 | +0.099% | [−457.5, 948.5] | FAIL |

Static-safe 대비 work-conserving/SoftWall의 차이는 크다. 이는 고정 all-fail block을
실제 결과와 무관하게 끝까지 보유하는 비용을 보여준다. 하지만 그 이득 대부분은 단순
조기 복구와 work conservation으로 얻어지며, multi-credit atomic exchange의 추가
처리량은 이 mode에서 측정 가능한 크기가 아니다.

Radio의 gate/admission/강제 실패/commit-kind signature는 각 seed의 모든 시스템에서
완전히 같았다. Conventional decoding의 `correct` 합은 seed1에서 636–638, seed2에서
606–608로 작은 실행 간 변동이 있었으므로 PHY payload 정답의 완전 결정성은 주장하지
않는다.

## 개발 실패에서 얻은 경계

- **C107:** 가변 512-token worker가 모든 token의 logits를 보유해 4셀 적재가 OOM이었다.
  Full transformer 뒤 마지막 token에만 LM head를 적용해 불필요한 resident memory를
  제거했다.
- **C107c:** isolated 평균에서 만든 256-token 45 ms bound가 co-run에서 51.8 ms로 깨졌다.
  최종 mode는 65 ms로 재자격화했다.
- **C108:** 125 ms SLO에서는 추가 unit이 아예 들어가지 않았고, 200 ms의 단일-seed
  +6.34%는 새 seed에서 부호가 반전했다.
- **C109–111:** arrival dequantization, 작은 Qwen, 3× load를 검사했지만 이득이 같아지거나
  30 ms bound가 43.6 ms 실행에 깨졌다. 사후 유리한 mode 탐색을 종료했다.
- **C112:** frozen 첫 SoftWall arm에서 retimed credit의 현재 start 대신 immutable NRx
  cutoff를 비교하는 dispatch 오류가 fail-closed로 드러났다. 실패 결과를 보존하고
  회귀 수정 뒤 새 protocol C113을 처음부터 실행했다.

이 실패들은 service bound, resident memory, lifecycle, work-unit granularity가 독립
상수가 아니라 하나의 mode를 이룬다는 feasibility-envelope 주장을 강화한다.

사후 [envelope 감사](../../results/softwall_same_gpu/confirm113_feasibility_envelope_posthoc.json)는
이를 식으로 확인한다. 4셀·12 ms recovery credit에서 compaction의 최대 추가 slack은
36 ms인데, trace의 99.824%는 65/75 ms 계약 unit이다. 기존 slack을 `W`, 회수량을
`Δ`라 하면 SoftWall만 새 unit을 허가할 수 있는 용량 필요조건은
`W < B_eff ≤ W+Δ`다. 최종
mode는 대부분 이 구간 밖이고 1초 SLO가 뒤쪽 실행을 허용하므로, 81/88회의 exchange가
있어도 최종 적시 가치 차이는 작다.

후속 V10 감사는 이 식이 충분조건이 아님을 확인했다. 이후 V11은 실제 IPC
`ring_depth=1`과 executor별 rejected-recovery phase를 반영해 전체 grid의 후보 수를
교정했다. V12는 AI completion guard 2 ms까지 청구했다. 이 교정으로 full-control
두-home mode에는 완전한 AI40+control15+guard2 후보가 유지됐고 C135가
이를 처음 물리 검증했으며 C136이 새 A100 node의 여섯 arm에서 재자격했다. C113의
single-home conv12 mode에서는 모든 평가 AI class가 이미
105 ms static base slack에 들어가므로 exchange-only class가 아니다. 따라서 C113의
처리량 outcome과 해석은 바뀌지 않는다. C113은 실제 trace strong-baseline 결과이고,
C135/C136은 별도 synthetic mechanism boundary 검증이다.

## 논문에 허용되는 주장

> In a qualified four-cell MIG-off MPS mode, SoftWall safely executed 81--88
> atomic multi-credit/AI-lease exchanges per full trace replay with no observed
> deadline, bound, guard, or credit violation. A strong safe work-conserving
> baseline recovered nearly all of the throughput available over static
> reservation, while SoftWall's additional timely-token gain was only
> 0.10--0.11% and statistically inconclusive. The contribution is therefore
> the certified conditional-recovery substrate and its measured feasibility
> envelope, rather than a throughput-superior optimizer.

실제 DU의 MAC expiry, WCET, GPU hang/driver reset, 다른 GPU와 channel model 일반화는
여전히 남는다.
