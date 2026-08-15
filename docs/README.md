# 문서 카탈로그

## Current — 먼저 읽을 문서

- [연구 종합본](current/MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md): 문제, 실측, 설계, novelty, 남은 gate
- [연구 체크포인트](current/MIG_NRX_RESEARCH_CHECKPOINT_KO.md): 완료 실험의 수치와 claim 경계
- [GDR pool 실행 계획](current/MIG_NRX_GDR_POOL_EXECUTION_PLAN_KO.md): 완료된 multi-endpoint campaign 구성
- [통합 구현 계획](current/DART_RX_MULTI_ENDPOINT_INTEGRATION_PLAN_KO.md): actual-radio와 pool 통합 기준
- [현재 연구 인덱스](current/CURRENT_RESEARCH_INDEX_KO.md): 상세 문서 및 결과 링크

## Architecture

- `architecture/DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`
- `architecture/DART_RX_FINAL_ARCHITECTURE_KO.md`
- `architecture/DART_RX_NOVELTY_REALISM_PIVOT_KO.md`
- `architecture/DART_RX_REALISTIC_WORKLOAD_GATE_KO.md`
- `architecture/ISCA_ARCHITECTURE_V2_KO.md`
- `architecture/ISCA_V2_EXECUTIVE_BRIEF_KO.md`

이 디렉터리는 설계 공간과 novelty 논증을 보존한다. 서로 충돌하면 `current/`의 종합본을 따른다.

## Experiments와 setup

- `experiments/`: 정식 matrix, 하루 campaign, 실행·retry 정책
- `setup/`: 새 CloudLab 노드 복구와 과거 설치 이력
- `tables/`: 실행 ledger, matrix, ablation CSV

## Archive

`archive/`는 연구 과정 추적용이다. 파일명이 `FINAL`, `MASTER`, `NEXT`여도 현재 authoritative 문서가 아니며, 현재 claim의 근거로 직접 인용하지 않는다.
