# 현재 MIG–NRx/DART-Rx 연구 문서 인덱스

**Updated:** 2026-08-16 KST
**Workspace entry:** `../../README.md`  
**Data catalog:** `../../data/README.md`

## 먼저 읽을 문서

1. [`RESEARCH_WALKTHROUGH_KO.md`](RESEARCH_WALKTHROUGH_KO.md)
   - 실험 figure를 따라 background/problem → DART-Rx design → setup → evaluation을 읽는 대표 문서
2. [`RESEARCH_WALKTHROUGH_EN.md`](RESEARCH_WALKTHROUGH_EN.md)
   - 동일한 원시 데이터와 별도 영어 figure 19개를 사용하는 영문 대표 보고서
3. [`MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md`](MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md)
   - 현재 문제, 실측, DART-Rx 설계, 효과, novelty, 남은 gate의 단일 종합본
4. [`MIG_NRX_GDR_POOL_EXECUTION_PLAN_KO.md`](MIG_NRX_GDR_POOL_EXECUTION_PLAN_KO.md)
   - 완료된 process-per-endpoint actual GDR campaign의 실행 구성
5. [`MIG_NRX_RESEARCH_CHECKPOINT_KO.md`](MIG_NRX_RESEARCH_CHECKPOINT_KO.md)
   - 완료된 causal/five-way/background/multi-cell 결과와 수치의 상세 기준
6. [`DART_RX_MULTI_ENDPOINT_INTEGRATION_PLAN_KO.md`](DART_RX_MULTI_ENDPOINT_INTEGRATION_PLAN_KO.md)
   - pool 측정기를 실제 cuPHY/conventional/NRx 다중 endpoint 스킴으로 바꾸는 구현 기준

위 여섯 문서가 현재 authoritative set이다. 처음 이해할 때는 walkthrough를 읽고, 수치와
실행 상태가 충돌하면 더 구체적인 result report와 보존된 CSV/JSON을 따른다.

## Architecture와 novelty 상세

- [`DART_RX_FINAL_ARCHITECTURE_KO.md`](../architecture/DART_RX_FINAL_ARCHITECTURE_KO.md)
  - single-endpoint radio vertical slice와 3-part architecture의 이전 상세안
  - multi-endpoint pool 상태는 현재 종합본으로 대체됨
- [`DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`](../architecture/DART_RX_CANONICAL_NOVELTY_THESIS_KO.md)
  - novelty/prior-work/transaction contract의 장문 검토
  - hardware primitive는 아직 확정안이 아님
- [`ISCA_ARCHITECTURE_V2_KO.md`](../architecture/ISCA_ARCHITECTURE_V2_KO.md)
  - DART queue/doorbell/commit hardware 후보
  - Nsight attribution 전의 design space이므로 final proposal로 읽지 않음
- [`DART_RX_NOVELTY_REALISM_PIVOT_KO.md`](../architecture/DART_RX_NOVELTY_REALISM_PIVOT_KO.md)
  - 단순 MIG/MPS/P2P/GDR 비교에서 deadline-safe transaction으로 pivot한 이유
- [`DART_RX_REALISTIC_WORKLOAD_GATE_KO.md`](../architecture/DART_RX_REALISTIC_WORKLOAD_GATE_KO.md)
  - single/multi-cell/selective workload 현실성 gate
- [`ISCA_V2_EXECUTIVE_BRIEF_KO.md`](../architecture/ISCA_V2_EXECUTIVE_BRIEF_KO.md)
  - 짧은 executive-level architecture 설명

## 실험 계획과 재현

- [`CLOUDLAB_EMPTY_NODE_RESTORE_RUNBOOK_KO.md`](../setup/CLOUDLAB_EMPTY_NODE_RESTORE_RUNBOOK_KO.md)
  - 완전히 빈 새 d8545 노드에 최종 코드·image·MIG·RDMA/GDR·radio gate를
    복구하는 canonical runbook
- [`cloudlab_final_snapshot_20260814/`](../../cloudlab_final_snapshot_20260814/)
  - 원격 결과/소스/model cache/dataset과 환경 manifest의 최종 로컬 백업

- [`ISCA_EXPERIMENT_PLAN_V2_KO.md`](../experiments/ISCA_EXPERIMENT_PLAN_V2_KO.md)
  - 전체 실험 matrix와 metric/fairness 원칙
- [`ISCA_FULL_DAY_CAMPAIGN_20260813_KO.md`](../experiments/ISCA_FULL_DAY_CAMPAIGN_20260813_KO.md)
  - 이전 하루치 causal campaign 구성
- [`ISCA_FULL_DAY_RUN_POLICY_KO.md`](../experiments/ISCA_FULL_DAY_RUN_POLICY_KO.md)
  - retry, marker, cleanup, artifact 보존 정책
- [`FRESH_CLOUDLAB_SETUP.md`](../setup/FRESH_CLOUDLAB_SETUP.md)
  - 초기 Task 1/2 설치 이력과 troubleshooting reference; 신규 복구는 위
    canonical runbook을 우선
- [`DRAIN_FREE_NRX_EXPERIMENT_PLAN.md`](../experiments/DRAIN_FREE_NRX_EXPERIMENT_PLAN.md)
  - fixed-MIG/drain-free 연구의 초기 계획; 현재 방향의 역사적 참고

## 완료 결과 보고서

- [`task1_final/gdr_pool_20260814T014651Z/analysis/REPORT.md`](../../task1_final/gdr_pool_20260814T014651Z/analysis/REPORT.md)
  - 348-run full matrix와 64-run gate/replica/representative를 합친 actual GDR pool 최종 결과
- [`task1_final/gdr_cuda_ipc_gate/`](../../task1_final/gdr_cuda_ipc_gate/)
  - endpoint-agent 소유 GPU MR를 L1 process가 CUDA IPC로 직접 접근한 64KiB/1.35MiB gate
- [`task1_final/dart_rx_radio_pool/analysis/REPORT.md`](../../task1_final/dart_rx_radio_pool/analysis/REPORT.md)
  - 실제 cuPHY/conventional/3-endpoint GDR NRx/LDPC/CRC paired correctness 최종 표
- [`task1_final/dart_rx_radio_pool/.../nsys_l1.nsys-rep`](../../task1_final/dart_rx_radio_pool/dart_radio_pool_e3_round_robin_all_t34_20260814T093833Z/nsys_l1.nsys-rep)
  - warm-up 제외 CUDA Profiler API와 stage NVTX가 포함된 Nsight Systems capture

- [`results/isca_v2/.../07_multicell_workloads/analysis/REPORT.md`](../../results/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/analysis/REPORT.md)
  - 87-trace compute/queue problem gate
- [`results/isca_v2/day1_20260813T0523Z/SUMMARY.md`](../../results/isca_v2/day1_20260813T0523Z/SUMMARY.md)
  - direct TRT, P2P/GDR, placement 초기 결과
- [`codex_20260814/.../paired_final.../analysis/REPORT.md`](../../codex_20260814/results/isca_v2/dart_rx_integrated_campaign/paired_final_20260813T104930Z/analysis/REPORT.md)
  - actual radio utility/single-endpoint transaction 결과 snapshot
- [`results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv`](../../results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv)
  - MIG/MPS/MIG+MPS/P2P/GDR placement 수치

## 최근 완료 campaign

CloudLab authoritative result root:

```text
/mydata/results/isca_v2/gdr_pool_20260814T014651Z
```

주요 상태/로그:

```text
/mydata/results/isca_v2/gdr_pool_20260814T014651Z/controller.log
/mydata/results/isca_v2/gdr_pool_20260814T014651Z/01_smoke/
/mydata/results/isca_v2/gdr_pool_20260814T014651Z/02_replica_sweep/
/mydata/results/isca_v2/gdr_pool_20260814T014651Z/03_representative/
/mydata/results/isca_v2/gdr_pool_20260814T014651Z/04_full/
```

상태:

```text
348/348 full runs complete
412 total validated runs
FINAL_COMPLETE
GPU0 4g+3g MIG restored
GPU1/2/3 full-GPU restored
mlx5_0 PORT_ACTIVE restored
```

로컬 전체 raw 결과와 byte-identical 재분석본:

```text
/Users/changjongkim/New_research/cloudlab_results/task1_final/gdr_pool_20260814T014651Z/
```

## 과거 문서

아래 파일은 연구 과정 보존용이며 현재 상태를 판단하는 첫 문서로 사용하지 않는다.

- `DART_RX_SCHEME_DESIGN_V0_KO.md`
- `ISCA_DAY_CAMPAIGN_20260813_KO.md`
- `ISCA_TODAY_ALL_IN_ONE_PLAN_KO.md`
- `MORNING_RECOVERY_PLAN.md`
- `NEXT_EXPERIMENT_PLAN.md`
- `TODAY_PLAN_20260523.md`
- `TOMORROW_5H_PLAN.md`
- `MASTER_SUMMARY.md`
- `REPORT_FINAL_20260702.md`

`codex_20260814/docs/` 아래의 동명 문서는 시점별 snapshot이다. 현재 편집 기준은
`docs/current/`, `docs/architecture/`, `docs/experiments/`, `docs/setup/` 아래의 원본이다.
