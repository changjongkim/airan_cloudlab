# 현재 권위 문서 인덱스

**기준일:** 2026-10-04

## 슬롯 단위 Antiphase (최신 스킴, 2026-09-29 ~)

지금 구현된 스킴은 아래 문서가 기준이다. 논문 본문과 발표 자료에는 아직 반영하지 않았다.

- **[복구 손실 모델, 스킴 수정, 범위 확장](BACKSTOP_V14_MODEL_AND_EXTENSIONS_KO.md) (2026-10-04, 스킴과 스케줄링 수치의
  기준):** 복구 손실의 해석 모델(사건 모의, 닫힌 형태), 규칙 변경(AI 조각 중단, NRx 옆 금지, 한도 1.2 ms를 넘는 AI
  단위는 기존 수신기 옆 금지), 512 슬롯 반복과 시드 5의 재측정, YinYangRAN 설계를 재구성한 추정기 기준선, 토큰
  생성과 여러 모델, UL goodput, link adaptation, 미래를 아는 스케줄과의 거리(9장), outer loop를 넣은 닫힌 루프 link
  adaptation(10장, 2026-10-05). README의 4장과 6.4–6.13절이 이 문서를 따른다.
- **[기존 GPU 공유 방식과의 sweep 비교](BACKSTOP_V13_SWEEPS_KO.md) (2026-10-02, 이전 규칙의 수치; AI 부하·버스트·셀 수 sweep은 이 문서만 있음):** 낮은 MPS
  우선순위를 기준선에 넣고 Antiphase의 규칙을 고친 뒤(낮은 우선순위 + 빈 NRx 여유), AI 부하·부하 변화·NRx
  수요·셀 수·GPU 수 sweep으로 고정 비율, 부하 따라 비율, 낮은 우선순위, 유휴 시간만 쓰는 방식과 비교했다.
- **[수신기 검증과 다시 한 실험](BACKSTOP_V4_VERIFICATION_KO.md) (2026-10-01, 먼저 읽을 것):** NRx 입력 교정, LDPC 반복
  횟수를 맞춘 수신기 비교, NVlabs 공개 NRx 둘(nrx_rt, nrx_large) 검증, 실험 네 판(v4, v5, v6, v7), 지금 쓸 수 있는
  문장. 주장의 기준은 v7(큰 모델을 rescue로)이다. 아래 결과 문서의 "살린 TB" 수치는 이 문서로 대체된다.
- 논문 초안(LaTeX, 28쪽): `paper/backstop_slot_v14/main.tex`, `main.pdf`. v14의 규칙과 수치로 쓴 초안이다(손실 모델 절,
  추정기 기준선, 여러 AI 작업, goodput, link adaptation, 미래를 아는 스케줄과의 거리, 닫힌 루프 link adaptation). v13까지의 초안은 `paper/backstop_slot_v7`, 이전 스킴의 논문은
  `paper/softwall_sigmetrics27`에 그대로 두었다.
- [논문용 영문 문단 초안](BACKSTOP_V7_PAPER_DRAFT_EN.md): LaTeX 초안을 쓰기 전의 문단 모음.

- [설계](BACKSTOP_SLOT_DESIGN_KO.md): 마감 두 개, NRx 규칙, AI 규칙, 시간 한도, 코드 위치, 실행 방법, 알려진 틈
- [결과](BACKSTOP_SLOT_RESULTS_KO.md): 유효한 실험과 폐기한 실험, 무선 목표를 지키며 낸 AI 처리량, 읽을 때 주의
- [기존 연구와 다른 점](BACKSTOP_SLOT_DIFFERENTIATION_KO.md): 기존 연구가 돌리는 무선 작업, 동작 방식 차이 다섯 가지, 쓸 수 있는 문장

- [스케줄러 발전 계획과 진행 기록](BACKSTOP_SLACK_BUDGET_PLAN_KO.md): 지금까지의 정리, 여유 예산 방식(v3) 계획, 단계별 결과

과정 기록(수치를 인용할 때는 위 결과 문서를 따른다):

- [슬롯 단위 재설계 기록](BACKSTOP_SLOT_SCALE_REDESIGN_KO.md): 8·16셀, 마감 하나(4.0 ms)였던 첫 구성과 ablation
- [연속 슬롯 확장 계획과 실행 기록](BACKSTOP_CONTINUOUS_SLOT_PLAN_KO.md): 계획, 부품 측정, 폐기한 v2 결과, 버그 수정, 최종 비교

## 논문 현재판 (SoftWall/Backstop P180·D155)

**기준일:** 2026-09-25  
**현재 연구축:** MIG-off MPS 기반 certified conditional-recovery substrate와 provenance-qualified feasibility envelope  
**현재 production 판정:** P3의 Sionna CDL-D/E channel mode만 제한적으로 통과했고, P1 live-DU timing과 P2 4.5 ms fast path는 미통과다.

`docs/current`에는 논문 구조와 현재 claim을 직접 판단하는 문서만 둔다. 과거 실험별 결과, 중간 novelty 결정, MIG/GPU0 전용/CloudLab/4-GPU pool 설계 문서는 `docs/archive`에 원본을 보존한다. 그런 과거 배치 가정은 현재 SoftWall 주장의 근거가 아니다.

## 논문과 주장

- [영문 원고 초고](SOFTWALL_MANUSCRIPT_DRAFT_EN.md): 현재 주장과 평가 서술의 Markdown 권위본
- [한국어 논문 구조](SOFTWALL_PAPER_STRUCTURE_KO.md): Introduction–Design–Evaluation 설명
- [SIGMETRICS 제출 계획](SOFTWALL_SIGMETRICS27_SUBMISSION_PLAN_KO.md): Winter 일정, 제출 범위와 reviewer risk
- [노벨티 방어 행렬](SOFTWALL_NOVELTY_DEFENSE_MATRIX_KO.md): closest-work와 claim boundary
- [Related-work 감사](SOFTWALL_RELATED_WORK_AUDIT_KO.md): 원문 기준 선행연구 비교

## 모델과 시스템 계약

- [형식 모델](SOFTWALL_FORMAL_MODEL_KO.md): debt, certificate, atomic refinement와 envelope
- [서비스 상한 자격 검증](SOFTWALL_SERVICE_BOUND_QUALIFICATION_KO.md): mode별 finite-sample bound 규칙
- [강한 baseline 명세](SOFTWALL_STRONG_BASELINE_SPEC_KO.md): 공정 비교 정책
- [연구 운영 지침](SOFTWALL_RESEARCH_GUIDELINE_KO.md): frozen protocol, provenance와 stop rule

## 현재 production exit gate

- [Production exit 계획](SOFTWALL_PRODUCTION_EXIT_PLAN_KO.md): P1–P4 판정과 P2 양 경로 cuPHY fast-path 구현 계획
- [PUSCH timing contract](SOFTWALL_PUSCH_TIMING_CONTRACT_KO.md): live-DU trace 최소 schema와 validator bridge
- [C160 fault model](SOFTWALL_C160_FAULT_MODEL_RESULT_KO.md): fault semantics
- [C161 physical refinement](SOFTWALL_C161_PHASE2_RESULT_KO.md): physical lifecycle 검증
- [C162 predictive envelope](SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md): 모델–물리 경계와 확장성
- [C164 lifecycle envelope](SOFTWALL_C164_IDLE30_RESULT_KO.md): 다섯 qualified/partial subset과 다섯 UQ mode
- [완료도 감사](SOFTWALL_COMPLETION_AUDIT_KO.md): claim-scoped 완료 상태

## 보관 원칙

과거 문서는 삭제하지 않는다. 각 archived Markdown의 첫 줄은 보관 사유와 오래된 배치 가정이 현재 claim basis가 아님을 명시한다. 원고 수치의 권위는 문서 이름이 아니라 machine-readable claim audit의 `(artifact, field, expected value)` 매핑으로 판정한다.
