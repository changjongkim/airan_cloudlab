# SoftWall 현재 권위 문서 인덱스

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
