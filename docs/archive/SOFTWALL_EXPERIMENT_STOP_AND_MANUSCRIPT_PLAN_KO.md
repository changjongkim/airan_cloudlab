> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall 실험 중단선과 원고 전환 결정

**결정일:** 2026-09-25  
**판정:** `SOFTWALL_LIFECYCLE_EXPANSION_STOPPED_FOR_MANUSCRIPT`  
**다음 gate:** audited 영문 초고를 venue 원고로 압축하고 도표·인용·reviewer defense 고정

## 1. 결정

C164 lifecycle matrix는 qualified/partial 5개와 UQ 5개에서 의도적으로 닫는다. 10개 mode는
모두 채워야 하는 checklist가 아니다. 현재 논문이 주장하는 warm qualified operation과 네
claim-scoped lifecycle subset을 입증하는 데 5개면 충분하다. 나머지 mode는 다음과 같이
명시적인 운영 경계로 남긴다.

- process/model cold와 5분·30분 idle: whole-mode physical qualification 없음
- GC ON: C89에서 선언 bound 위반을 관측한 unqualified mode
- worker process replacement: independent holdout과 완전한 crash-window proof 없음
- production `d_MAC`, WCET, cross-family: lifecycle matrix 자체의 범위 밖

새 physical experiment는 원고 내부 검토에서 특정 핵심 주장에 직접 필요한 evidence gap이
발견될 때만 연다. 단순히 matrix의 빈칸을 줄이기 위한 C164/C165 campaign은 열지 않는다.

## 2. 왜 지금 멈추는가

논문의 중심 기여는 모든 lifecycle을 견디는 범용 GPU runtime이 아니다. 중심 기여는
MIG-off MPS AI-RAN에서 optional NeuralRx가 만드는 조건부 recovery debt를 executable
all-fail certificate로 유지하고, recovery-credit 이동과 external-AI lease를 원자적으로
결합하는 substrate다. C162까지 다음 핵심 증거가 닫혔다.

| 주장 | 현재 가장 강한 증거 | 상태 |
|---|---|---|
| 모델–구현 동치 | exact state 16,023개에서 mismatch 0 | 완료 |
| 보지 않은 경계 예측 | two-node physical 88/89 ms boundary, 180 round | 완료 |
| 제어평면 확장성 | 64-debt certified scheduler p99 1.494 ms | 완료 |
| 실제 AI-RAN 경로 | actual TensorRT NRx, shared cuPHY recovery, Qwen, multi-GPU IPC | 완료 |
| fault closure | C161 A0–A6와 stale/duplicate/terminal handling | 완료 |
| lifecycle provenance | warm+4 subset qualified/partial, 나머지 UQ | claim 범위에 충분 |

여기서 process replacement와 durable journal recovery를 주 기여에 넣으면 새 질문이 열린다.
Crash가 journal fsync 전·후 어느 instruction에서 발생하는지, old CUDA context의 종료를 누가
증명하는지, MPS daemon과 GPU/driver failure를 어떻게 구분하는지, replacement 동안 optional
service availability를 보장하는지까지 다뤄야 한다. 이는 현재 substrate의 증거를 보완하는
작은 실험이 아니라 별도 mechanism과 fault model이다.

## 3. 마지막 exploratory canary의 처리

이미 시작했던 single-node development canary는 `nid001061`, job `58862843`에서 끝났다.

| 항목 | 결과 |
|---|---:|
| Replacement episode | 8 |
| MPS `terminate_client` 성공 | 8/8 |
| Prepared / launched branch | 4 / 4 |
| Mandatory release / cell decode | 486 / 1,944 |
| Deadline miss / component-bound 위반 | 0 / 0 |
| Quiescence certificate | 8/8 PASS |

이 결과는 mechanism feasibility를 보여 주는 development observation이다. 독립 node holdout을
열지 않고 lifecycle matrix의 `worker_process_replacement`를 qualified로 바꾸지 않는다.
앞선 prelaunch failure와 pretermination failure도 함께 보존한다. 본문에서는 사용하지 않으며,
필요하면 appendix/future work에서 “single-node exploratory canary”로만 언급한다.

## 4. 첫 원고에 넣을 주장

핵심 문장은 다음으로 고정한다.

> SoftWall은 MIG-off MPS AI-RAN에서 optional NeuralRx의 미해소 conventional recovery
> obligation을 executable certificate로 유지하고, multi-home recovery retiming과 bounded
> external-AI lease를 하나의 generation-safe transaction으로 결합한다. Qualified A100
> mode에서 이 certificate는 fault가 있는 실제 NRx–cuPHY–Qwen 경로의 safety를 보존하며,
> 모델은 보지 않은 feasibility boundary를 정확히 예측한다.

본문의 기여는 네 가지로 제한한다.

1. Conditional-recovery debt와 all-fail executable certificate의 형식화
2. Multi-credit retiming, atomic AI lease, physical fence/IPC lifecycle을 묶은 runtime
3. QSU/QSN/MI/UQ predictive envelope와 exact verifier/certified scheduler
4. Actual NRx–shared cuPHY–Qwen multi-GPU 경로의 fault·boundary 검증

Lifecycle 결과는 네 번째 기여의 mode-provenance 보조 증거다. 10/10 coverage를 독립 기여로
주장하지 않는다. Joint optimizer 우위도 주장하지 않으며, strong baseline과 oracle에서
headroom이 없었던 음성 결과를 그대로 보고한다.

## 5. 첫 원고에서 명시할 한계

- Deadline은 production `d_MAC`이 아니라 synthetic `P180/D155` contract다.
- Qualification은 finite sample이며 WCET 증명이 아니다.
- Hardware claim은 현재 A100 family에 한정한다.
- 다섯 lifecycle mode는 명시적으로 UQ다.
- 350 ms tail은 한 가설만 제거됐고 최종 원인은 미해결이다. 원인을 추정해 덮지 않고
  limitation/open diagnosis로 기록하며, 이를 이유로 journal-recovery 범위를 자동 확장하지 않는다.
- Process-replacement canary는 independent holdout이 없는 exploratory result다.

이 한계를 쓰는 것으로 충분하며, 한계를 모두 제거한 뒤에만 원고를 쓸 필요는 없다.

## 6. 원고 작성 순서

아래 1차 순서는 [영문 원고 초안](SOFTWALL_MANUSCRIPT_DRAFT_EN.md)에 반영됐다. 현재는 이
초안을 학회 분량으로 압축하고 도표 번호·정리 번호·bibliography를 고정하는 단계다.

1. Abstract와 Introduction을 위 핵심 문장과 네 기여로 다시 쓴다.
2. Formal model에서 obligation, certificate, transaction, verifier의 정의와 정리를 본문 크기로
   압축한다.
3. Design은 request lifecycle과 failure transition을 한 장의 도식으로 고정한다.
4. Evaluation은 MPS-only failure → end-to-end mechanism → fault closure → predictive boundary →
   scalability → lifecycle sensitivity 순으로 구성한다.
5. Related work는 GPU sharing 자체가 아니라 conditional same-TB recovery obligation,
   executable certificate, physical lifecycle의 결합 차이를 표로 비교한다.
6. Limitations에서 production timing, WCET, cross-family, UQ lifecycle을 명시한다.

초고가 완성되기 전에는 새 lifecycle campaign을 열지 않는다. 초고 검토에서 핵심 주장과
직접 연결되지 않는 요청은 future work로 유지한다.

## 7. 권위 artifact

- [실험 중단 결정](../../results/softwall_multigpu/softwall_experiment_stop_decision_v1.json)
- [C162 predictive envelope](SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md)
- [C164 claim-scoped lifecycle matrix](../../results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json)
- [C164 lifecycle 결과](SOFTWALL_C164_IDLE30_RESULT_KO.md)
- [논문 구조](SOFTWALL_PAPER_STRUCTURE_KO.md)
