# AI-RAN MIG–NRx 연구 워크스페이스

이 디렉터리의 단일 진입점이다. 새로 작업을 시작할 때는 아래 순서로 읽는다.

1. [현재 연구 종합본](docs/current/MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md)
2. [현재 문서 인덱스](docs/current/CURRENT_RESEARCH_INDEX_KO.md)
3. [실험 데이터 카탈로그](data/README.md)
4. [빈 CloudLab 노드 복구 절차](docs/setup/CLOUDLAB_EMPTY_NODE_RESTORE_RUNBOOK_KO.md)

## 디렉터리 구조

| 경로 | 역할 |
|---|---|
| `docs/current/` | 현재 판단에 사용하는 authoritative 문서 |
| `docs/architecture/` | DART-Rx 구조, novelty, workload 정의 |
| `docs/experiments/` | 실험 계획, campaign, 실행 정책 |
| `docs/setup/` | CloudLab 설치·복구·troubleshooting |
| `docs/tables/` | 실험 matrix, ledger, 요약 CSV |
| `docs/archive/` | 이전 가설·계획·보고서; 현재 결론으로 사용하지 않음 |
| `data/` | 실제 데이터 디렉터리로 연결되는 단일 카탈로그 |
| `results/` | 날짜별 전체 실험 history와 raw/analysis 결과 |
| `task1_final/` | 2026-08-12~14의 최종 chain/GDR/radio 결과 |
| `task1_p2p_fair/` | P2P fairness 실험 |
| `cloudlab_final_snapshot_20260814/` | 22GB의 검증된 최종 원격 snapshot; 수정 금지 |
| `scripts_for_node/`, `scripts_node/` | CloudLab 배포·실행 스크립트 |
| `tools/` | 분석 도구와 과거 plotting utility |
| `archive/` | 과거 루트 로그, 완료 marker, 초기 run 사본 |

## 관리 원칙

- 새 연구 문서는 루트에 만들지 않고 `docs/`의 해당 분류에 둔다.
- 현재 결론은 `docs/current/`만 기준으로 삼는다.
- raw result는 덮어쓰지 않고 새 timestamp 디렉터리에 저장한다.
- `cloudlab_final_snapshot_20260814/`는 checksum이 고정된 복구 기준점이므로 편집하지 않는다.
- 데이터 위치를 찾을 때는 항상 [data/README.md](data/README.md)부터 확인한다.
