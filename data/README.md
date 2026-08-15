# 실험 데이터 카탈로그

대용량 결과의 기존 경로를 바꾸면 runner와 재현 기록이 깨질 수 있어 원본 디렉터리는 이동하지 않았다. 이 디렉터리의 symlink와 아래 표를 데이터 탐색의 단일 진입점으로 사용한다.

가장 최근 결과만 바로 보려면 `data/current/`를 사용한다. 이 안의 `gdr_pool`, `radio_pool`, `background_contention`, `multicell_workloads`, `placement`, `drain_free`가 아래 원본으로 연결된다.

## 현재 핵심 결과

| 데이터 | 실제 경로 | 설명 |
|---|---|---|
| GDR NRx pool | [`task1_final/gdr_pool_20260814T014651Z`](../task1_final/gdr_pool_20260814T014651Z/) | 412 validated runs, 348 full-matrix runs |
| Actual-radio pool | [`task1_final/dart_rx_radio_pool`](../task1_final/dart_rx_radio_pool/) | cuPHY→GDR→NRx→LDPC/CRC vertical slice |
| CUDA-IPC/GDR gate | [`task1_final/gdr_cuda_ipc_gate`](../task1_final/gdr_cuda_ipc_gate/) | caller GPU memory와 endpoint-owned MR 연결 검증 |
| Background contention | [`results/isca_v2/.../06_background_contention`](../results/isca_v2/mig_causal_20260813T1138Z/06_background_contention/) | ResNet/BERT/Whisper/Qwen × naive/adaptive |
| Multi-cell workload | [`results/isca_v2/.../07_multicell_workloads`](../results/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/) | 87-trace queue/fragmentation gate |
| Placement 비교 | [`results/20260813_nrx_placement`](../results/20260813_nrx_placement/) | MIG/MPS/MIG+MPS/P2P/GDR |
| Drain-free 연구 | [`results/20260813_drain_free`](../results/20260813_drain_free/) | fixed-MIG capacity, policy, reclaim 결과 |
| Task 1 chain | [`task1_final/chain`](../task1_final/chain/) | cuPHY/NRx shm, CPU-RDMA, GDR 결과 |
| P2P fairness | [`task1_p2p_fair`](../task1_p2p_fair/) | same-placement/P2P 비교 |

## 전체 history와 복구 데이터

| 경로 | 내용 | 사용 기준 |
|---|---|---|
| [`results/`](../results/) | 2026-05 이후 날짜별 raw 결과와 분석 | 전체 실험 history |
| [`final_snapshot`](final_snapshot/) | 2026-08-14 CloudLab 전체 백업 | 복구용 immutable 기준점 |
| [`final_snapshot/mydata/results`](final_snapshot/mydata/results/) | 백업 시점의 원격 `/mydata/results` | 원격 원본 대조 |
| [`final_snapshot/mydata/datasets`](final_snapshot/mydata/datasets/) | radio/workload datasets | 새 노드 입력 복구 |
| [`final_snapshot/mydata/hf_cache`](final_snapshot/mydata/hf_cache/) | Qwen2.5-7B Hugging Face cache | 새 노드 모델 복구 |
| [`final_snapshot/mydata/torch_cache`](final_snapshot/mydata/torch_cache/) | Torch model/cache | 새 노드 모델 복구 |
| [`export_20260814`](export_20260814/) | 전달용 export bundle | 시점별 사본; current source 아님 |
| [`artifact_bundle`](artifact_bundle/) | 이전 Codex artifact bundle | 중복 보존본 |

## 데이터 읽는 순서

1. 각 campaign의 `analysis/REPORT.md` 또는 `SUMMARY*.md`를 먼저 읽는다.
2. 집계값은 같은 디렉터리의 CSV와 `result.json`으로 대조한다.
3. 원시 timing은 run별 JSON/NPZ/log를 사용한다.
4. `FAILED`, `_attempt*_failed`, smoke 디렉터리는 정식 결과와 섞지 않는다.
5. snapshot 내부 파일은 직접 수정하지 않고 필요한 경우 작업 디렉터리로 복사한다.
