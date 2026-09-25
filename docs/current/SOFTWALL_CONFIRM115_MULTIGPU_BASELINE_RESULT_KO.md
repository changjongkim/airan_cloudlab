# Confirm115: 같은 2-GPU 예산의 강한 baseline 판정

**상태:** 2026-09-24 formal campaign 완료  
**실행:** job `58815435`, 10개 full arm  
**판정:** provenance·safety·radio structural parity PASS, throughput outcome FAIL

## 질문

Confirm114는 local CUDA-IPC endpoint와 remote P2P endpoint가 섞인 SoftWall의 안전 경로를
통과시켰다. Confirm115는 GPU를 더 준 효과를 SoftWall 효과로 오해하지 않도록 모든 시스템에
정확히 같은 자원과 입력을 주고 다음 하나를 물었다.

> 같은 GPU0+GPU1 placement, PHY input, endpoint engine, Qwen trace, service bound와 fault
> pattern에서 atomic recovery-credit exchange가 safe work-conserving보다 deadline 내
> Qwen token value를 2% 이상 높이는가?

## 고정한 구성

- GPU0: 4-cell Aerial, local CUDA-IPC NeuralRx 1개, Qwen2.5-1.5B
- GPU1: remote NVLink-P2P NeuralRx 1개
- MIG OFF, MPS: local NRx 40, remote NRx 40, Qwen 20
- `P180/D155`, NRx45, conventional12, guard2 ms
- BurstGPT 60초 1,136-request trace, 동일 context별 AI bound
- 5 release마다 single/correlated failure 혼합
- Seed1 `static→work→SoftWall→SoftWall→work`
- Seed2 `static→SoftWall→work→work→SoftWall`
- 각 arm 340 release, 1,360 TB

Static, work-conserving과 SoftWall은 radio 선택, endpoint queue, early fallback, physical
completion과 all-fail safety를 공유한다. SoftWall만 복수 recovery credit의 tail compaction과
AI lease를 하나의 transaction으로 commit한다.

## 안전·출처 판정

10개 arm 모두 다음을 통과했다.

- RAN deadline miss 0
- NRx45, conventional12, context별 AI bound 위반 0
- recovery guard 위반 0
- endpoint/background fault 0
- residual recovery, endpoint, AI lease credit 0
- local worker와 remote worker 실제 사용
- protocol source hash와 실행 artifact hash 일치

같은 seed 안에서 `gate_skipped`, NRx admission, forced failure와 commit kind로 만든 radio
decision signature는 모든 시스템에서 동일했다. Correct-cell 수는 seed1 618–620,
seed2 630–631로 좁게 달랐다. 따라서 exact radio-output equality를 주장하지 않고, 결정 구조
parity와 safety를 통과한 비교로 한정한다.

## 적시 AI 결과

| Seed/order | Static | Work-conserving 평균 | SoftWall 평균 | SoftWall−work | 효과 | Paired source-second bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| seed1 ABBA | 74,766 | 224,602.0 | 224,782.5 | +180.5 | +0.080% | [−478, 834.5] |
| seed2 BAAB | 73,447 | 224,638.5 | 224,949.0 | +310.5 | +0.138% | [−124, 884.5] |

SoftWall은 각 seed의 두 반복에서 recovery retime+AI lease를 각각 84회와 86회 물리적으로
완료했다. 즉 mechanism이 실행되지 않아 차이가 없는 것이 아니다. 그러나 두 효과 모두
사전 최소 유의미 효과 2%보다 작고 CI 하한이 0보다 크지 않다. Primary outcome gate는
실패다.

Static 대비 work-conserving의 큰 차이는 고정 all-fail block을 매 release 유지하는 비용을
보여준다. 이는 SoftWall 고유 이득이 아니라 단순 work conservation의 가치다.

## 해석

C115는 C113의 single-GPU 음성 결과를 같은 2-GPU 예산에서도 재현했다.

1. Remote NeuralRx를 추가해도 현재 trace의 AI-unit granularity와 SLO에서는 atomic
   compaction의 추가 처리량 효과가 유의미하지 않다.
2. GPU 추가가 recovery-credit exchange의 성능 novelty를 되살리지 않는다.
3. Local/remote endpoint, Qwen과 correlated failure가 있는 10개 arm에서 safety가 유지돼
   transport-independent substrate 증거는 강해졌다.
4. 이후 새로운 workload를 사후 탐색해 +2% mode를 찾지 않는다. 모델이 사전에 예측한
   경계점만 prospective gate로 실행한다.

따라서 현재 정확한 논문 주장은 다음이다.

> SoftWall은 optional NeuralRx가 만드는 multi-cell recovery debt를 local/remote endpoint와
> 외부 AI의 physical-credit lifecycle에 연결한다. 이 계약은 single-GPU와 2-GPU의 자격화된
> 4-cell mode에서 동작했지만, 검증한 실제 trace에서 strong safe work-conserving보다 추가
> 처리량 우위는 관측되지 않았다.

## 분석 실행 환경 오류

10번째 arm 뒤 compute-node host Python이 `from __future__ import annotations`를 지원하지
않아 aggregate analyzer가 실행 전 `SyntaxError`로 종료됐다. 10개 arm은 이미 정상 완료됐고
원시 artifact는 보존됐다. 같은 frozen analyzer와 source hash를 project GPU container의
Python에서 실행해 위 결과를 만들었다. 이 사건은 별도 failure record에 남겼다.

## 권위 artifact

- [사전 protocol](../../results/softwall_multigpu/confirm115_multigpu_strong_baselines_protocol.json)
- [최종 결과](../../results/softwall_multigpu/confirm115_multigpu_strong_baselines_job58815435.json)
- [artifact manifest](../../results/softwall_multigpu/confirm115_artifact_manifest.json)
- [host-Python 분석 오류](../../results/softwall_multigpu/confirm115_analyzer_host_python_failure_job58815435.json)

