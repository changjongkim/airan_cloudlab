# 현재 MIG–NRx/DART-Rx 연구 문서 인덱스

**Updated:** 2026-09-25 UTC
**Workspace entry:** `../../README.md`  
**Data catalog:** `../../data/README.md`

## 현재 방향 (2026-09-25)

- [SoftWall 형식 모델과 입증 의무](SOFTWALL_FORMAL_MODEL_KO.md):
  all-fail dominance, 원자 상태전이 안전성, conditional-slack/no-benefit, transport
  eligibility와 4상태 feasibility envelope를 정의한다. V16은 home-local slack,
  finite IPC ring, executor phase, 완전한 AI admission guard와 observe-all decision-time 충분조건을 연결하고 exact
  oracle, C138 wall-bound 반례, C139--C141 synchronous correction, C142--C144
  pipelined-control, V14 ownership 반례, C145/C146 V15 runtime 재자격과 C147/C148 abort
  fault holdout으로 구현을 감사한다.
- [SoftWall 멀티 GPU·모델링 고도화 로드맵](SOFTWALL_MULTIGPU_MODELING_ROADMAP_KO.md):
  단일 GPU CUDA IPC를 기준점으로 유지하면서 remote P2P NRx endpoint를 같은 all-fail
  certificate와 credit lifecycle에 연결하는 구조, 서비스 모델, 강한 baseline, G0--G8
  gate와 중단 기준을 정리한다. C114에서 G0/G1a/G2a/G3를 통과해 cross-process P2P,
  actual remote NeuralRx와 4셀 SoftWall/Qwen/fault 통합까지 완료했고, C116--120에서
  three/four-endpoint pool과 co-run component/lifecycle 경계까지 검증했다. C121/C122는
  sharded recovery home을 실행했고, C123--148은 global AI ownership, conv12 falsification,
  executor-order 반례, conv25 certificate executor, commit ambiguity와 bounded broker
  fail-stop 격리, timeout/wall-bound 분리, launch-time revalidation과 AI45 conditional class까지 검증했다.
  V17 후보는 shared mandatory-recovery GPU에서 local 판정이 false-safe가 되는 반례와
  global executable calendar를 구현했다. C151/C152는 두 node의 shared cuPHY/P2P data
  path 1,000건을 오류0으로 통과시켰다. C153은 one-node injected-outcome canary에서 global
  fifth reject, Qwen fence와 certificate-ordered cuPHY recovery를 D155 안에 통합했다.
  C154/C155는 네 controlled branch를 서로 다른 두 node에서 반대 순서로 재현해 여덟 arm,
  global reject4, Qwen4, recovery20, correct commit20, miss0을 얻었다. Controlled-outcome
  integration은 통과했다. C156은 fixed-lease launch-control 반례를 V17.1로 교정하고
  C156/C156b 두 node에서 Qwen/recovery GPU kernel2,660, 금지 overlap0 ns, commit4/4와
  miss0을 확인했다. C157A/C157B는 두 node의 실제 TensorRT NeuralRx CRC로 credit을
  전이했다. C158은 이를 두 node·500 epoch에서 반복해 actual NRx2,000, success1,440,
  shared recovery560, Qwen500, commit2,000과 safety/fence violation0을 얻었다. C159-Q1은
  pre-staged GPU bank와 full recovery-path preflight로 P180/D155 context-64 mode를 두 새
  node에 자격화했다. Actual NRx2,000, success1,418, recovery582, Qwen500, commit2,000에서
  위반0이다. C159-Q2는 여섯 variable-context class를 두 node·1,200 epoch, actual NRx4,800,
  recovery1,312, Qwen1,077, commit4,800에서 위반0으로 통과했다. Q3 oracle은
  certificate-preserving recovery-first와 SoftWall이 385,262 token으로 동일해 성능 holdout을
  중단했다. 이 동률은 certificate가 아니라 AI-first retiming의 추가 처리량 주장을 기각한다. C160은
  51개 fault state transition을 통과했다. C161 1단계 A0/A1/A4와 2단계 A2/A3/A5/A6를
  합치면 700 epoch·actual NRx2,800·recovery942·radio commit2,800·deadline miss0이다.
  Stale/duplicate pair40, terminal channel fault4, fault 뒤 continuation260을 qualified node에서
  재현했다. 같은 source에서 NRx45를 깬 `nid001044` lifecycle, process/model cold,
  restart 중 availability와 production expiry는 UQ다.
  C162는 qualified state 16,023개에서 analytic/exact mismatch0, 과거 물리 decision
  1,200/1,200 일치, 새 두-node boundary 180/180 일치를 얻었다. Actual NRx720,
  physical recovery330, Qwen90, commit720, miss0이며 context64의 88/89 ms decision boundary도
  30/30씩 재현했다.
- [C153 integrated shared-recovery 결과](SOFTWALL_CONFIRM153_INTEGRATED_RESULT_KO.md):
  앞선 socket-path 실패와 합성 IQ release 정의 실패를 보존하고 job 58852924의 10/10 gate,
  Qwen23.059 ms, 두 home commit82.412/97.644 ms와 miss0을 정리한다.
- [C154/C155 controlled integrated holdout](SOFTWALL_CONFIRM154_155_CONTROLLED_HOLDOUT_KO.md):
  두 독립 A100 node, 반대 branch 순서, 동일 frozen source의 여덟 arm과 controlled-outcome
  통합 판정 및 actual-NRx/fault UQ 경계를 정리한다.
- [C156 GPU timeline과 V17.1 launch revalidation](SOFTWALL_CONFIRM156_GPU_TIMELINE_RESULT_KO.md):
  Nsight clock 호환 실패, fixed lease의 실제 gate 실패, control5+AI35 교정과 11/11 GPU
  timeline gate를 실패를 포함해 정리한다.
- [C157 actual TensorRT NeuralRx 통합](SOFTWALL_CONFIRM157_ACTUAL_NRX_RESULT_KO.md):
  실제 NRx CRC가 global credit, Qwen lease와 shared cuPHY recovery를 구동한 두-node
  결과와 monitor-JIT 등 여섯 development failure를 함께 정리한다.
- [C158 반복 actual TensorRT NeuralRx 자격](SOFTWALL_CONFIRM158_REPEATED_QUALIFICATION_KO.md):
  실제 NRx→global certificate→Qwen/shared recovery 전이를 두 node의 2,000 request로
  반복하고 mmap control, exact input/output echo와 monitor preflight를 자격화한다.
- [C160--C162 fault/envelope 실행 계획](SOFTWALL_C160_C162_EXPERIMENT_PLAN_KO.md):
  Q2의 P180 actual-NRx/variable-Qwen 경로에 stale·duplicate·late·fence-loss를 직접 주입하는
  두-node matrix와 debt/time/class/lifecycle/memory 예측 grid, false-safe 중단 기준을 고정한다.
- [C160 fault state-model 결과](SOFTWALL_C160_FAULT_MODEL_RESULT_KO.md):
  Unit test8개와 51개 epoch/generation/fence/marker 전이에서 invariant violation과 reject-path
  mutation0. Pre-fence lease 보존과 matching-fence-only retire 의미론을 C161 전에 고정했다.
- [C161 1단계 실제 GPU fault 결과](SOFTWALL_C161_PHASE1_RESULT_KO.md):
  두 독립 A100 node와 반대 arm 순서에서 A0/A1/A4를 실행했다. Actual NeuralRx720,
  correlated recovery240, guarded non-launch52, radio commit720, deadline miss0으로 15/15 gate를
  각각 통과했다. 나머지 fault는 C161 2단계에서 닫았다.
- [C161 2단계 실제 GPU fault 결과](SOFTWALL_C161_PHASE2_RESULT_KO.md):
  두 추가 node의 반대 arm 순서에서 A2/A3/A5/A6를 합계 520 epoch 실행했다. Actual NRx2,080,
  event replay pair40, terminal fault4, post-fault continuation260, commit2,080, miss0이다.
  [full qualification](../../results/softwall_multigpu/c161_full_fault_qualification.json)과
  [756-file manifest](../../results/softwall_multigpu/c161_full_fault_manifest.json)을 고정했다.
- [C162 predictive feasibility envelope 결과](SOFTWALL_C162_PREDICTIVE_ENVELOPE_RESULT_KO.md):
  QSU/QSN/MI/UQ 모델을 16,023-state exact oracle와 대조하고, 두 독립 A100 node의 사전 고정
  여섯 boundary를 각30회 물리 검증했다. Model/lease mismatch0, deadline miss0이며
  certified scheduler는 64-debt decision p991.494ms와 returned-certificate verifier 실패0을 얻었다.
  [결합 판정](../../results/softwall_multigpu/c162_boundary_two_node.json)과
  [83-file manifest](../../results/softwall_multigpu/c162_artifact_manifest.json)를 고정했다.
- [필요성 반론 감사와 추가 실험 결정](SOFTWALL_NECESSITY_GAP_DECISION_KO.md):
  recovery-first가 이미 certificate-preserving 기준선임을 명확히 하고, C162의 사전 고정 E4/E6b가
  debt-blind current-idle admission에 대해 12/1ms contract 초과를 허용하지만 SoftWall은 두 node에서
  합계60/60회 launch 전 거절했음을 감사했다. 이는 observed debt-blind miss가 아닌 contract-level
  witness다. `B_conv` 민감도에서 E6b는 24ms 이하에서, E4는 19ms 이하에서 사라진다. 현재 Q2
  recovery 표본 최대는 13.465ms이며 더 작은 bound는 mode 재자격 전까지 UQ다. Failure-correlation
  sweep은 기준선 의미를 바꾸고 all-fail 보장에 필요하지 않아 열지 않는다.
- [SIGMETRICS 2027 익명 LaTeX 원고](../../paper/softwall_sigmetrics27/main.tex):
  `acmsmall,screen,review,anonymous` 15쪽 원고, 정규화 BibTeX, closest-work/necessity 표,
  reproducibility와 generative-AI disclosure를 포함한다. [컴파일 PDF](../../paper/softwall_sigmetrics27/main.pdf)와
  [26항목 제출 감사](../../results/softwall_multigpu/softwall_sigmetrics_submission_audit_v1.json)가 PASS다.
- [C162 이후 production/cross-family 계획](SOFTWALL_POST_C162_EXPERIMENT_PLAN_KO.md):
  C163 실제 DU/FAPI `d_MAC`, C164 cold/reconnect/long-idle와 restart 재자격, C165 다른 GPU family의 순서와
  표본·baseline·false-safe 중단 규칙을 고정한다. C163은 raw-event builder,
  request-specific exact envelope bridge, content-hash/clock/expiry/mode validator와 단위시험
  24개까지 완료했으나 실제 production trace가 없어 `UQ_NO_PRODUCTION_TRACE`다.
- [Production exit gate와 고도화 계획](SOFTWALL_PRODUCTION_EXIT_PLAN_KO.md):
  Aerial testMAC의 `T0+4.5 ms` 중간 목표와 외부 TDL-A를 별도 exit gate로 검사했다. 시험한
  모든 timing 경로가 threshold tail을 남겼고, 최선의 same-stream raw-IQ two-GPU 경로도
  885/1,000만 적시 완료했다. Stage diagnostic은 남은 tail을 remote cuPHY LS channel
  estimation으로 좁혔다. Aerial TDL-A는 계속 NeuralRx 0이지만, 공개 계약을 복원한 Sionna
  CDL-D/E/100 ns 독립 holdout 500건은 P3를 통과했다. 저 SNR NeuralRx-only31 대
  conventional-only12, exact `p=0.00540`이다. 따라서 전체 판정은 여전히
  `FAIL_CURRENT_DESIGN_NOT_PRODUCTION_QUALIFIED`이며 P1 live-DU clock, P2 persistent fast path,
  P4 integrated requalification이 남는다. P3 PASS는 Sionna D/E mode에만 적용된다.
- [C164 lifecycle-token, idle30, MPS-restart와 Qwen reload 결과](SOFTWALL_C164_IDLE30_RESULT_KO.md):
  node/GPU/software/placement/lifecycle fingerprint에 묶인 qualification token 상태모델이 9개
  lifecycle·8개 restart 전이에서 unsafe optional admission0을 통과했다. Ready persistent
  clients가 schedule 공개 전에 30초 idle한 development/holdout 두 node의 180 boundary round도
  actual NRx720·recovery330·Qwen90·commit720·miss0으로 통과했다. 별도 두 node에서 실제
  4-GPU MPS epoch를 완전히 종료·재생성한 뒤 fresh-client requalification과 같은 180 boundary
  round도 동일 count와 miss0으로 통과했다. 같은 recovery GPU의 Qwen fresh load+6-class
  warmup60회 동안 mandatory release3,334·cell13,336도 miss0·cell25 위반0이었다. Optional
  inference는 reload 중 닫았다. Durable reconnect model은 43-state audit에서 retire2·
  fail-closed41·violation0을 얻었고, same-worker channel reconnect는 두 node·token120·physical
  Qwen60·terminal fence120·mandatory release145에서 miss0으로 통과했다. Restart/reload 중 AI
  availability, full six-class lifecycle, NeuralRx cold, worker process replacement와 5분·30분
  idle은 UQ다. 이 범위는
  [machine-readable lifecycle matrix](../../results/softwall_multigpu/c164_lifecycle_qualification_summary_v1.json)에
  qualified/partial 5개와 UQ 5개로 고정했다. 남은 mode는 논문 완성을 위한 checklist가
  아니다. 단일-node process-replacement development canary는 PASS했지만 holdout 없이
  exploratory evidence로만 보존한다. Claim/evidence freeze와 audited 영문 초고를 완료했고,
  다음 내부 gate는 venue 원고 압축, 도표·bibliography와 reviewer-objection audit이다.
- [실험 중단선과 원고 전환 결정](SOFTWALL_EXPERIMENT_STOP_AND_MANUSCRIPT_PLAN_KO.md):
  lifecycle 5/10을 의도적인 claim boundary로 고정하고, single-node process-replacement
  canary를 exploratory evidence로만 보존하며 새 실험은 원고 검토에서 claim-critical gap이
  확인될 때만 연다는 중단 규칙을 정리한다.
- [SoftWall 영문 원고 초안](SOFTWALL_MANUSCRIPT_DRAFT_EN.md):
  Abstract부터 Conclusion까지 conditional-recovery model, certificate-preserving runtime,
  predictive envelope, physical evaluation, 음성 throughput 결과와 limitation을 하나의
  submission story로 작성한 1차 초고다.
- [Novelty defense matrix](SOFTWALL_NOVELTY_DEFENSE_MATRIX_KO.md):
  N1 recovery debt, N2 physical refinement, N3 predictive envelope를 가까운 top-tier·최신
  AI-RAN 연구와 claim-by-claim으로 비교하고 예상 reviewer objection을 정리한다.
- [SIGMETRICS 2027 제출 전환 계획](SOFTWALL_SIGMETRICS27_SUBMISSION_PLAN_KO.md):
  Systems primary/Measurement secondary track, 20쪽 body budget, 필수 도표·LaTeX·anonymity·
  artifact·AI-use disclosure와 새 실험 중단 규칙을 고정한다.
- [SoftWall 최종 실험 계획](SOFTWALL_FINAL_EXPERIMENT_PLAN_KO.md):
  완료된 C153--C158 뒤 P180 trace-mode 자격 C159-Q, BurstGPT/Qwen 강한 baseline C159-P,
  fault matrix와 feasibility-envelope/scalability의 순서와 중단 기준을 고정한다.
- [C159 actual-NRx trace 사전 명세](SOFTWALL_CONFIRM159_TRACE_PROTOCOL_KO.md):
  BurstGPT source-time development/calibration/holdout 분리, P180 mode 재자격, 세 safe system의
  유일한 차이, 표본 수 결정, paired 통계와 safety/performance gate를 고정한다.
  [machine-readable prespec](../../results/softwall_multigpu/confirm159_experiment_prespec_v1.json)은
  holdout을 만들기 전 dataset·builder·calibration hash와 다음 gate를 봉인한다.
- [C159-Q1 P180 actual-NRx 재자격](SOFTWALL_CONFIRM159_Q1_P180_RESULT_KO.md):
  두 독립 node에서 P180 context-64 mode를 actual NRx2,000·recovery582·Qwen500·commit2,000,
  miss·bound·fence·contract 위반0으로 통과했다. 첫 250-epoch 실행의 full-path cold tail을
  보존하고 whole-path preflight 뒤 warm recovery 최대11.555 ms로 lifecycle을 분리했다.
  [결합 판정](../../results/softwall_multigpu/confirm159_q1_p180_two_node.json)과
  [135-file manifest](../../results/softwall_multigpu/confirm159_q1_p180_manifest.json)을 고정했다.
- [C159-Q2 variable-context 재자격](SOFTWALL_CONFIRM159_Q2_VARIABLE_RESULT_KO.md):
  두 독립 node에서 여섯 Qwen class와 actual NeuralRx/shared cuPHY를 합계 1,200 epoch 실행해
  16/16 gate를 통과했다. Stale launch certificate와 sequential outcome replay 반례를 보존하고
  bounded revalidation, physical latest-start, atomic observed-success batch로 교정했다.
  [결합 판정](../../results/softwall_multigpu/confirm159_q2_variable_two_node.json)을 고정했다.
- [C159-Q3 oracle screen](SOFTWALL_CONFIRM159_Q3_ORACLE_RESULT_KO.md):
  calibration 네 창의 exact offline upper bound가 strong Event-driven과 SoftWall 모두
  385,262 token으로 같아 사전 5% MDE에 실패했다. Confirmatory performance holdout은 열지 않는다.
- [C114 멀티 GPU 결과](SOFTWALL_CONFIRM114_MULTIGPU_RESULT_KO.md):
  4×A100 NVLink topology, 실제 payload 10,000회, remote NeuralRx 1,000회와 두 독립
  local+remote 4셀 통합 arm의 결과·주장 경계를 정리한다. 같은 2-GPU 예산의 처리량
  우위와 production WCET는 아직 주장하지 않는다.
- [C115 같은 2-GPU 예산 baseline](SOFTWALL_CONFIRM115_MULTIGPU_BASELINE_RESULT_KO.md):
  Static/work-conserving/SoftWall 10개 arm의 safety·decision parity PASS와 처리량 outcome
  FAIL을 기록한다. 추가 효과 +0.080%/+0.138%, 두 CI 0 포함으로 멀티 GPU 우위도 기각한다.
- [C116 three-endpoint pool](SOFTWALL_CONFIRM116_THREE_ENDPOINT_RESULT_KO.md):
  Controller의 2-endpoint 가정을 제거하고 GPU0 local 1개+GPU1/2 remote 2개를 같은 4셀
  certificate에 연결했다. 두 seed에서 세 endpoint 실제 사용, exchange90/98, 위반0이다.
- [C117 component-bound 실패 경계](SOFTWALL_CONFIRM117_COMPONENT_BOUND_RESULT_KO.md):
  전체 system safety와 remote component는 통과했지만 GPU0 front/back 2 ms 후보가 각각
  한 건씩 깨진 결과와 co-run timestamp 진단을 보존한다.
- [C118 component-bound 재자격](SOFTWALL_CONFIRM118_COMPONENT_REQUALIFICATION_RESULT_KO.md):
  C117 뒤 새 protocol/seed로 home 3 ms와 원격 component vector를 고정해 두 arm 모두
  zero-violation gate를 통과한 결과를 정리한다. WCET가 아닌 exact warm mode 자격이다.
- [C119/C120 4-GPU·4-endpoint 결과](SOFTWALL_CONFIRM119_120_FOUR_GPU_RESULT_KO.md):
  GPU0 local+GPU1/2/3 remote endpoint의 네 formal arm에서 lifecycle/system safety는
  통과했지만 eager-loading OOM과 두 fine-component gate 실패가 나타난 경계를 정리한다.
- [Endpoint-only multi-GPU envelope v1](SOFTWALL_ENDPOINT_OFFLOAD_ENVELOPE_RESULT_KO.md):
  GPU 1/2/4×cell 4/8/12의 deterministic grid를 계산한다. Single-home 구조에서는 4-cell만
  QSU이고 8/12-cell은 C84 GPU0 receiver-memory 경계 때문에 MI임을 명시한다. V16은
  C124--148을 반영해 bound, executor, control fault, launch-time critical/deferred control,
  single-token ownership과 four-point physical coverage를 mode 축으로 분리하고
  QSU6/QSN0/MI3/UQ13으로 판정한다.
  cross-home, unlimited-ring과 executor-phase 추상화 오류를 교정했다. 39개 endpoint
  pool과 10,920개 bounded state에서 exact oracle과 수락 cardinality 차이는 0이다.
  Full-control mode의 decision 후보 두 개는 C135에서 처음 물리 검증됐고 C136의 새
  A100 node 여섯 arm에서 재자격됐다.
- [C121/C122 sharded-home 결과](SOFTWALL_CONFIRM121_122_SHARDED_HOME_RESULT_KO.md):
  동기 release의 2 GPU/8셀과 4 GPU/12셀에서 총 13,600 TB, atomic exchange, bounded Qwen과
  correlated NRx failure를 실행해 위반 0을 얻었다. C124 후속 tail로 C121 conv12 자격은
  철회했고 C122 3-cell/home exact mode는 유지한다.
- [C123--126 global AI와 executor 결과](SOFTWALL_CONFIRM123_126_GLOBAL_AI_RESULT_KO.md):
  하나의 global BurstGPT queue와 두 local certificate의 합성, C124 conv12 tail, C125
  executor-order 반례, C126 certificate-ordered conv25의 21,760 TB 결과를 정리한다.
  Global routing 처리량 우위와 strict radio parity 실패도 함께 보존한다.
- [C127 global broker fault containment](SOFTWALL_CONFIRM127_BROKER_FAULT_RESULT_KO.md):
  post-apply commit reply loss를 두 독립 2-GPU arm에 주입한다. Ambiguous request를 재시도하지
  않고 affected-home AI만 닫아 총 5,440 RAN TB와 다른 home의 AI를 계속한 결과를 정리한다.
- [C128--132 broker fail-stop/control-budget 판정](SOFTWALL_CONFIRM128_132_BROKER_CRASH_RESULT_KO.md):
  synchronous marker가 만든 C128 miss, 무제한 complete RPC가 D155를 깨뜨린 C129,
  물리 통과했지만 proof가 불완전한 C130/C131과 세 RPC 15 ms를 모두 admission에 반영해
  5,440 TB를 통과한 C132를 하나의 refinement chain으로 정리한다.
- [C128--133 full control-point fault matrix](SOFTWALL_CONFIRM128_133_CONTROL_FAULT_RESULT_KO.md):
  C132 post-commit 두 arm과 C133 post-prepare/post-complete 네 arm을 합쳐 세 synchronous
  transaction point를 각각 두 번 검증한다. 총 16,320 TB, crash 뒤 14,938 TB에서 safety와
  duplicate-execution 위반은 0이다.
- [C134 독립 allocation/node 재자격](SOFTWALL_CONFIRM134_INDEPENDENT_NODE_RESULT_KO.md):
  새 job 58823497과 다른 A100 node nid002688에서 prepare/commit/complete 각 두 arm을
  재실행했다. 총 16,320 TB, crash 뒤 14,926 TB에서 모든 safety·artifact gate가 통과했다.
- [C135 conditional AI40 예측 검증](SOFTWALL_CONFIRM135_V11_AI40_RESULT_KO.md):
  실제 endpoint ring depth, executor phase와 completion guard를 반영한 V12가 static 53 ms 밖의
  `AI40+control15+guard2=57 ms` class를 58 ms conditional window에서 유지했다. 실행 전 동결한
  두 arm에서 후보 분기 288회, AI40 exchange 10회, 2,560 TB 안전 위반 0을 얻었고,
  사건별 반사실 감사도 static margin −4 ms, conditional margin +1 ms를 확인했다.
  별도 allocation의 전체 Aerial155+Qwen3+DART30 시험도 통과했다. 판정과 해시는
  [V12 검증 요약](../../results/softwall_multigpu/softwall_envelope_v12_validation_summary.json)에 고정했다.
- [C136 V12 독립 node 재자격](SOFTWALL_CONFIRM136_V12_REQUALIFICATION_RESULT_KO.md):
  기존 node를 제외한 `nid002817`에서 새 seed/process lifecycle 6개 arm을 실행했다.
  7,680 TB, candidate branch 908회, AI40 exchange 28회가 모두 통과했다. C135와 합친
  두 node·8 arm은 10,240 TB와 38개 목표 exchange에서 선언 위반 0이다. TB/arm/node
  zero-failure 상한을 분리해 두 node만으로 WCET나 hardware 일반화를 주장하지 않는다.
  전체 판정과 파일 해시는 [cross-node validation summary](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_validation_summary.json)와
  [cross-node manifest](../../results/softwall_multigpu/softwall_envelope_v12_cross_node_manifest.json)에 고정한다.
- [C137 service-bound/RPC telemetry](SOFTWALL_CONFIRM137_SERVICE_BOUND_TELEMETRY_RESULT_KO.md):
  새 A100 node `nid001177`의 6개 arm에서 prepare/commit/complete 5,993개 wall time을
  계측했다. 최대 2.593595 ms, frozen 5 ms 초과0이었고 7,680 TB·AI40 exchange28의
  safety gate도 통과했다. C135/C136과 instrumentation mode를 분리하며 WCET는 주장하지 않는다.
  [C137 manifest](../../results/softwall_multigpu/confirm137_artifact_manifest.json)와
  [service-bound v2 manifest](../../results/softwall_multigpu/softwall_service_bound_v2_manifest.json)에
  source·raw result·문서 hash를 고정한다.
- [C138–C140 control-bound correction](SOFTWALL_CONFIRM138_140_CONTROL_BOUND_CORRECTION_KO.md):
  C138이 faulting prepare/complete 5.205/5.831 ms로 기존 5 ms wall-bound를 기각했다.
  C139는 5 ms socket timeout과 7 ms admission wall bound를 분리해 6 arm·16,320 TB와
  RPC 2,020회에서 7 ms 초과0을 얻었다. V13은 과거 AI40을 제거하고
  `AI35+control21+guard2=58 ms`를 예측했으며, C140 두 arm은 static margin−5 ms인
  target exchange 8회를 conditional window에서 위반 없이 실행했다.
- [C141 corrected V13 독립-node 재자격](SOFTWALL_CONFIRM141_V13_INDEPENDENT_RESULT_KO.md):
  기존 corrected node를 제외한 `nid001372`에서 control fault 6 arm과 AI35 두 arm을 모두
  통과했다. C139/C140과 합치면 두 A100 node, control 12 arm·RPC4,030회·7 ms 초과0,
  AI35 4 arm·exchange15·safety 위반0이다. 현재 service-bound 권위 판정은
  [qualification v4](../../results/softwall_multigpu/softwall_service_bound_qualification_v4.json)다.
- [V14 pipelined global control 설계](SOFTWALL_PIPELINED_CONTROL_DESIGN_KO.md):
  prepare/abort/complete를 전용 비동기 worker로 옮기고, staged token을 launch 직전 현재
  certificate로 재검증한 뒤 bounded commit ACK가 있을 때만 Qwen을 제출하는 상태 기계와
  `B_AI+B_commit+guard` 계약을 정의한다.
- [C142--C144 V14 결과](SOFTWALL_CONFIRM142_144_V14_PIPELINED_RESULT_KO.md):
  두 A100 node의 AI45 exchange11, control fault9 arm·21,120 TB·post-fault17,914 TB,
  deferred RPC 7 ms 초과50과 safety/duplicate 위반0을 정리한다. Finite model은
  16 state·20 transition에서 invariant 위반0이다.
- [V14 ownership 반례와 V15 교정](SOFTWALL_V14_OWNERSHIP_CORRECTION_KO.md):
  V14의 retained-token 뒤 두 번째 prepare가 untracked broker-held token을 만드는 반례와
  V15의 home별 single-token 불변식, model v2, 물리 branch gate를 정리한다.
- [C145--C146 V15 결과](SOFTWALL_CONFIRM145_146_V15_SINGLE_TOKEN_RESULT_KO.md):
  두 새 A100 node의 8 arm·12,160 TB·AI45 exchange12·fault 뒤7,463 TB,
  prepare 억제537회·maximum unlaunched token1·safety/duplicate 위반0을 정리한다.
- [C147--C148 V16 four-point 결과](SOFTWALL_CONFIRM147_148_V16_FOUR_POINT_RESULT_KO.md):
  두 추가 A100 node의 abort fault2 arm·3,200 TB·post-fault2,662·target 실행0을 정리하고,
  C145--C148 operation별 두 arm·radio15,360·post-fault10,125의 V16 QSU 판정을 고정한다.
- [V17 shared-recovery control-plane 후보](SOFTWALL_SHARED_RECOVERY_V17_MODEL_KO.md):
  두 home이 한 mandatory recovery lane을 공유할 때 local-safe/global-unsafe가 되는 10개
  반례와 global calendar·AI lease의 generation-safe 원자 transaction을 정리한다.
  Equal grid 25개와 capacity 1/2 variable timing/blackout 3,400개에서 oracle 차이0이다.
- [C151/C152 shared conventional path](SOFTWALL_CONFIRM151_152_SHARED_RECOVERY_PATH_KO.md):
  GPU0/1의 두 home PUSCH를 GPU2의 한 persistent cuPHY worker로 P2P 전송해 두 node 합계
  1,000/1,000 decoded TB를 맞춘 결과를 정리한다. Global order와 IPC lifecycle 오류는0이며,
  certificate-driven deadline/Qwen 통합 전까지 V17 전체 mode는 UQ다.
- [Service-bound qualification 방법론](SOFTWALL_SERVICE_BOUND_QUALIFICATION_KO.md):
  mode fingerprint, 최초 실현/holdout 분리, event·arm·node evidence 계층, 초과 시 UQ 강등,
  C138 반증과 C139/C140 corrected-control chain, production/WCET 승격 조건을 정리한다.
- [논문 구조 검토본](SOFTWALL_PAPER_STRUCTURE_KO.md): 지금까지 완료된 결과를
  Introduction, Background, Design, Implementation, Evaluation, limitations와 남은
  gate 순서로 통합한 검토용 초안. 새 작업은 이 문서 검토 뒤 진행한다.
- [Substrate + feasibility envelope 논문 전환안](SOFTWALL_SUBSTRATE_ENVELOPE_PAPER_PLAN_KO.md):
  C102 outcome 실패를 최종 optimizer 중단 gate로 적용한다. max-radio controller를
  공통으로 고정하고 conditional-recovery contract, atomic credit substrate,
  feasibility envelope, 음성 정책 결과를 논문 기여로 재편한다.
- [현재 SoftWall 스킴과 노벨리티 판정](SOFTWALL_CURRENT_SCHEME_AND_EVIDENCE_KO.md):
  C148/V16까지의 양성·음성 증거를 반영한 권위 요약. all-fail certificate, 조건부
  recovery credit, 원자적 retiming+AI lease, 물리 fence lifecycle을 현재 시스템
  기여로 정의하고, optimizer·추가 처리량 우위와 NRx-count scalar 비용은 기각된 주장으로 분리한다.
- [C113 실제 trace 강한 baseline 최종 판정](SOFTWALL_CONFIRM113_STRONG_BASELINE_RESULT_KO.md):
  BurstGPT 60초→Qwen2.5-1.5B, static/work-conserving/SoftWall 10개 full arm의
  safety·radio parity PASS와 처리량 outcome FAIL을 정리한다. SoftWall 원자 exchange는
  81/88회 실행됐지만 strong baseline 대비 +0.113%/+0.099%이고 두 CI가 0을 포함한다.
- [후속 스킴·강한 baseline 검증 기록](SOFTWALL_VALIDATION_PROGRESS_KO.md): Confirm56–95의 채널/queue/복구 baseline, 독립 PHY 가치 보정, 3·4셀 원자 AI lease·동시 all-fail 복구, AI 응답 지연 fault, mode별 bound 감사를 공동 정책 검증과 구분한다.
- [9월 22일 의사결정 gate](SOFTWALL_SEP22_DECISION_GATE_KO.md): 현재 물리 부품·PHY 가치 입력·strong greedy 대조의 판정과 **서비스 bound 자격을 경합 trace보다 먼저 확인하는** 실험 순서, 성능 노벨리티 중단 기준.
- [SoftWall bound-first 의사결정](SOFTWALL_BOUND_FIRST_DECISION_KO.md): Confirm82–91의 mode별 서비스 상한, 실제 25/12 ms 달력 실패, 실제 30/12 ms 달력의 두 seed 통과, 8셀 메모리 경계와 host tail 진단을 한곳에 모은다.
- [공동 정책의 다음 다중 사건 gate](SOFTWALL_MULTI_EVENT_POLICY_GATE_KO.md): Confirm92/93의 강한 공동 greedy와 exact 일치, Confirm94의 새 PHY seed 단계별 대조, C91 실제 AI7 용량 대조를 토대로 **AI bound 자격을 물리 경합 주장보다 먼저** 확인한다. 무선 품질 최대화 기준선과의 비교와 온라인 paired 중단 기준도 정의한다.
- [Confirm102 독립 다중 사건 판정](../../results/softwall_same_gpu/confirm102_independent_multi_event_policy.json): 새 PHY에서 calibration·구조 gate는 통과했으나 guarded joint가 max-radio+exact recourse와 39/39 같아 성능 gate 실패. staged/low gate 우위만으로 노벨리티를 주장하지 않는다. [C104 장시간 ABBA](../../results/softwall_same_gpu/confirm104_nrx_count_long_abba_job58788756.json)의 순서별 count 비용 부호 반전에 이어 [C105 persistent-process 동일 PHY interleave](../../results/softwall_same_gpu/confirm105_interleaved_count_job58789084.json)도 네 strata가 `양수/불확실/0/음수`로 갈려 실패했다. 안정적인 NRx-count AI slowdown scalar 가설은 폐기한다.
- [C96 이후 노벨리티·다음 gate](SOFTWALL_POST_C96_NOVELTY_GATE_KO.md): AI8 물리 자격 뒤 7-unit 단일 사건의 정책 우위 0, Qwen C97 실패·C98 좁은 통과·[C99 관측 순서 때문에 NRx45 실패](../../results/softwall_same_gpu/confirm99_qwen_fault_abba_job58747070.json)·[C100 관측 우선 mode 두 seed×1,000 통과](../../results/softwall_same_gpu/confirm100_observe_first_qwen_job58747620.json)·[C101 안전한 고장 ABBA의 적시 AI −7/−14](../../results/softwall_same_gpu/confirm101_observe_first_fault_abba_job58747620.json)를 기록한다. 실제 AI-RAN 작업 수요→다른 안전한 결정 증인→강한 기준선→GPU paired 성능의 시작·중단 조건.
- [AI-RAN 작업 자격 gate](SOFTWALL_AI_RAN_WORKLOAD_GATE_KO.md): Qwen 물리 경합 canary와 최종 도메인 작업을 구분하고, O-RAN SC QoE Predictor xApp·공개 KPI 데이터의 도착·마감·예측 가치·실제 서비스 상한을 확보하는 순서를 정한다.
- [동일 GPU 실험 gate ledger](../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md): Confirm96까지의 frozen PASS/FAIL과 현재 novelty 판정. GC OFF 예열 4셀 실제 30/12 ms 달력과 AI8 예산은 각각 두 seed의 표본 gate를 통과했지만, 25/12 ms는 첫 seed의 NRx 상한 초과로 실패했다. Confirm94의 최소-NRx 대비 AI 가치 이득은 주로 남는 두 번째 endpoint 사용이며 max-radio 기준선 대비 무선 손실이 있다. [AI8 모델 재검사 C96](../../results/softwall_same_gpu/confirm96_ai8_capacity_screen.json)은 70/70 상태에서 모든 정책이 AI 7개를 수용해 현 단일 사건 성능 가설을 기각했다.
- [채널 feature calibration 무효 판정](../../results/softwall_same_gpu/channel_feature_gate_job58693851.md): train/test 채널 seed 499/500 중복으로 gate 검증 무효. 잡음 전력 기준 변경에 따른 NeuralRx 정답 급변은 별도 paired 관측으로 보존하며, field 일반화는 하지 않음.
- [Aerial TDL-A 호환성 canary 실패](../../results/softwall_same_gpu/aerial_tdl_compatibility_job58694167.md): 잡음 없는 10 TTI에서 conventional 10/10, NeuralRx 0/10. 원인 분리 전 외부 채널 SNR/AI/deadline 실험 중단.
- [Aerial TDL-A 1×4 기하 정정 canary 실패](../../results/softwall_same_gpu/aerial_tdl_1x4_compatibility_job58694384.md): 단일 송신 스트림을 정확히 역산해 예제와 같은 1×4 채널에 넣어도 conventional 10/10, NeuralRx 0/10. 기하 차이만으로 설명 불가.
- [SoftWall 노벨리티 판정과 종료 기준](SOFTWALL_NOVELTY_DECISION_KO.md): Confirm55의 여유 부하와 현재 직렬 제어 루프가 공동 정책의 독립 기여를 검증하지 못하는 이유, 다음 단일 paired 비교의 중단 기준.
- [SoftWall 공동 스킴 후보](SOFTWALL_JOINT_SCHEME_PROPOSAL_KO.md): 같은 GPU MPS에서 NeuralRx 수락·동시 실패 복구·허용 AI unit을 함께 계획하는 정책 가설, 안전 불변식, 강한 baseline과 시작·중단 gate. 아직 노벨리티·보장 입증 전.
- [SoftWall 노벨리티·구체 스킴 판정](SOFTWALL_NOVELTY_SCHEME_DECISION_KO.md): NeuralRx의 conventional 대비 조건부 추가 무선 가치, 다중 셀 복구 의무, 허용 AI 실행권을 한 결정으로 묶는 연구 가설과 강한 baseline 대비 입증 조건.
- [PUSCH 타이밍 계약 감사](SOFTWALL_PUSCH_TIMING_CONTRACT_KO.md): slot 주기와 실제 MAC CRC 소비 기한을 분리. K2/N2로 gNB expiry를 직접 유도할 수 없으며 대상 DU 설정은 아직 없다. G7 same-clock trace validator와 9개 단위시험은 완료했고 실제 trace 부재로 UQ다.
- [SoftWall 노벨리티 실행안](SOFTWALL_NOVELTY_ACTION_PLAN_KO.md): 다음 공동 제어·강한 baseline·실제 PHY deadline gate와 중단 기준.

## 이전 단계 문서와 실험 기록

아래 항목의 ‘현재’ 표기는 각 문서 작성 시점의 상태다. 위 2026-09-24 방향과 gate ledger가 우선한다.

- [`../../results/softwall_same_gpu/INITIAL_FEASIBILITY_REPORT_KO.md`](../../results/softwall_same_gpu/INITIAL_FEASIBILITY_REPORT_KO.md) ← **현재 실험 결과**
  - 동일 A100 / MIG OFF / MPS에서 valid paired PUSCH, 실제 NeuralRx, GEMM, HBM 실행
  - S0 quiet-window, S1 cap80, S2 optional NeuralRx/recovery의 deadline/처리량 결과와 실패한 계약을 함께 기록
  - 별도-process NRx overrun과 timeout lifecycle 진단 완료; unconditional hard deadline과 production slot deadline은 아직 미완료

- [`../../results/softwall_same_gpu/S2_NATURAL_CHANNEL_REPORT_KO.md`](../../results/softwall_same_gpu/S2_NATURAL_CHANNEL_REPORT_KO.md) ← **현재 S2 authoritative report**
  - 같은 noisy PUSCH의 conventional/NeuralRx 상보성, 원자 recovery 예약, single commit, fail-closed admission
  - 10,000 one-cell paired noninferiority와 background +3.17%
  - 3,000 two-cell host-tail-robust paired 실험에서 모든 frozen gate 통과, background +5.53%
  - 실패한 confirm19 timing gate와 confirm23 baseline AI violation도 보존·분석
  - 별도 MPS client physical overrun에서 D25 실패를 확인해 cap-only 완전 격리를 반증
  - D35에서 admission quarantine만으로 cap20 1/5,000 miss, 즉시 process retirement로 cap100 6/5,000·cap20 11/5,000 miss를 확인; teardown은 maintenance window로 분리
  - 오염된 초기 maintenance 자료는 제외. 독립 Confirm39 재실행에서 RAN-free maintenance 뒤 6,000회 miss 0, 반면 ready 전 GPU warmup 뒤 worker를 첫 release 전에 종료한 sham은 8/10,000 miss; 전체 gate 실패. Confirm46의 CPU socket sham 대 GPU/MPS client 비교는 CPU 0/10,000 대 GPU/MPS 12/10,000 miss로 모든 frozen gate를 통과했다. CUDA warmup·context 생성·MPS 등록·종료 중 단일 원인은 아직 분리하지 못했다.
  - 별도 persistent cap80 MPS/CUDA-IPC same-request endpoint는 clean 10,000 releases correct 10,000/10,000, miss/timeout 0으로 통과
  - 같은 외부 endpoint의 −8.5 dB 10,000 releases에서 자연 fallback 221건 중 47건 recovery, miss/timeout 0으로 통과
  - Confirm36 동일 node/job/trace 비교에서 external S2 9,840, eager 9,835, conventional 3,688/10,000 correct; 모두 miss 0, external−eager paired 95% CI [−0.0088,+0.1088] pp, 불필요한 conventional 실행 9,797회 생략
  - Confirm37의 6 ms AI RPC 계약 실패를 보존하고, 독립 seed·40 ms 계약의 Confirm37b를 통과
  - Confirm38 동일 node/job/trace 각 10,000회: external/local S2/eager 모두 RAN miss·AI 계약 위반 0, 외부 AI 완료량 130,177회로 eager보다 2.93% 높음. 역순 독립 seed Confirm41도 세 방식 correct 9,817·miss 0, 외부 AI 130,254회로 eager보다 3.06% 높음. 두 외부 경로는 아직 원자 예약이 없는 조건부 복구 구현
  - Confirm42는 별도 CUDA-IPC endpoint에 실제 원자 fallback 예약·single commit을 연결하고 500회에서 miss 0, 자연 fallback 18건 중 4건 복구, 복구 전 AI 52 units를 완료. Confirm43 gap lease OFF→ON→ON→OFF는 각 10,000회에서 RAN miss 0, ON−OFF AI +1,069 units를 기록했으나 NRx 30 ms 상한 위반 3건으로 전체 strict gate 실패. 역순·독립 seed Confirm44도 각 10,000회에서 RAN miss 0, ON−OFF AI +3,317(+1.304%)이지만 같은 상한 위반 3건으로 전체 gate 실패.
  - Confirm40 cold-start ABBA는 OFF 첫 conventional 복구 22.76/35.30 ms 대 ON 2.28/2.29 ms로 모든 frozen gate 통과. 실제 transaction 대 eager의 Confirm45는 독립 seed·역순에서 완료 AI +2.95%/+2.41%, 새 50 ms NRx admission bound·35 ms AI socket timeout의 Confirm47은 +1.38%/+1.11%로 모두 frozen PASS. 같은 새 계약의 Confirm50 gap ON/OFF도 독립 seed·역순에서 +0.468%/+0.535%, 40,000건 RAN miss·NRx bound 위반 0 및 38개 frozen gate PASS. Confirm43/44의 별도 30 ms 상한 실패는 유지한다.
- [`../../results/softwall_same_gpu/SOFTWALL_COMPONENT_ABLATION_KO.md`](../../results/softwall_same_gpu/SOFTWALL_COMPONENT_ABLATION_KO.md) ← **MPS와 결합할 요소별 근거**
  - MPS/cap, bounded admission, recovery transaction, commit fence, endpoint lifecycle의 직접 근거와 남은 동일조건 ablation을 구분

- [`../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md`](../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md)
  - confirm7–55의 PASS/FAIL/diagnostic/prepared 상태를 한 표에 고정. Confirm49는 준비만 했고 실행하지 않았다.
  - 성공 follow-up이 앞선 실패 protocol을 지우지 않도록 claim 범위를 관리

- [`../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md`](../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md) ← **최신 N1 진단 FAIL**
  - 1셀 P90/45/25/12·D80에서 eager는 전부 miss 0. External은 P45부터 무선 정답 감소, P25/12에서 AI 완료 0. 전체 다중 셀 N1과 강한 결합 baseline 비교는 미완료.

- [`SOFTWALL_MPS_ISOLATION_FEASIBILITY_KO.md`](SOFTWALL_MPS_ISOLATION_FEASIBILITY_KO.md) ← **현재 실행 우선순위**
  - 사용자 확정 목표: 허용한 AI 작업 안에서 RAN deadline 보장, 남는 자원으로 AI 처리량 확보
  - **같은 GPU / MIG OFF / MPS**의 전체 PHY → 시간 분리 → 제한 overlap → NRx/recovery vertical slice 완료
  - External transaction의 eager 대비 이득을 독립 seed·역순으로 Confirm45/47에서 재현했고, gap lease ablation은 Confirm43/44에서 순증을 보였으나 30 ms bound gate에 실패했다. 남은 범위는 강한 결합 baseline·다중 셀 및 hard-real-time 보장 증거다.
  - MPS 기능·현재 A100/driver 확인, 조건부 보장과 관측 결과의 구분, 최소 비교 matrix
  - 아래 문서의 L1 전용 GPU·2-GPU 우선 제안보다 이 결정을 우선한다

- [`SOFTWALL_NOVELTY_THESIS_KO.md`](SOFTWALL_NOVELTY_THESIS_KO.md)
  - 현재 방어할 기여 C1–C3: 무선 이득·복구 비용 결합, 복구 가능 상태를 유지하는 공동 제어, 실제 PHY 검증
  - primary–backup·optional/mandatory·Neural Simplex 비교, 강한 결합 baseline과 novelty 확인 순서

- [`SOFTWALL_COMPLETION_AUDIT_KO.md`](SOFTWALL_COMPLETION_AUDIT_KO.md)
  - C148/V16 기준 현재 목표를 재감사한다. Bounded volatile substrate, launch-time
    revalidation, single-token AI45 class와 pipelined fault telemetry까지 완료했다. Production d_MAC,
    deterministic/WCET bound와 cross-family hardware는 제출 gate로 분리하며 C91 시점
    원문은 archive에 보존한다.

- [`SOFTWALL_STRONG_BASELINE_SPEC_KO.md`](SOFTWALL_STRONG_BASELINE_SPEC_KO.md)
  - Confirm50의 단일 메커니즘 비교와 전체 channel-gate·queue-aware·고정 복구·bounded AI 결합 baseline을 구분하고, 동일 정보·안전장치·GPU 예산의 비교 조건을 명시

- [`SOFTWALL_RELATED_WORK_AUDIT_KO.md`](SOFTWALL_RELATED_WORK_AUDIT_KO.md)
  - AI-RAN 선행 연구 점검: YinYangRAN·CloudRIC·Concordia, ARCHES·OCUDU 및 최신 공개 자료
  - 본논문/short/preprint 구분, 중복되는 주장, joint recovery/admission/lease 가설과 필수 비교 실험

- [`SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md`](SOFTWALL_SCHEME_AND_NEXT_STEPS_KO.md)
  - **후속 다중 GPU 확장 참고안**: mandatory/recovery·NRx 후처리 예약, endpoint 배치,
    background work-unit lease, commit·buffer 수명, v0/v1 비교와 실행 순서
  - 이전 2-GPU 우선 순서는 변경됨. 같은 GPU 검증을 먼저 통과한 후 이 문서의 pool 확장을 검토
  - 성능 검증 전 제안이며, 근거·수치의 정정 사항은
    [`RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md`](RESEARCH_PLAN_SOFTWALL_REVIEW_KO.md) 참고

- [`RESEARCH_PLAN_SOFTWALL_KO.md`](RESEARCH_PLAN_SOFTWALL_KO.md) ← **원 계획 초안**
  - MIG를 쓰지 않는 배치에서 소프트웨어 deadline 계약을 검증하려는 SoftWall 초안과
    SIGMETRICS/NSDI 목표 연구 계획 (문제 난이도, 배포 요구사항, 기여 구조 C1–C5,
    단계별 게이트 P0–P7, 평가 설계, 리스크 레지스터)
  - 4-GPU 노드: L1 전용 GPU + NVLink P2P NRx pool + bounded lease background
  - 아래 `RESEARCH_DIRECTION_MPS_KO.md`와 `results/20260803/MIG_MPS_COMBINED_REPORT.md`의
    충돌 해소 입장은 이 문서 §2.3, 정정 항목은 §1.3

- [`RESEARCH_DIRECTION_MPS_KO.md`](RESEARCH_DIRECTION_MPS_KO.md)
  - MIG는 제약 조건, 최적화는 MPS 위에서. 단일 GPU MPS 스케줄링 → Perlmutter MPS + RDMA 분산
  - 근거 수치, Future Direction 문구, Perlmutter 확인 항목, NVIDIA 미팅 질문, 개념 정리
  - §4의 "Perlmutter MPS daemon 확인 필요"는 2026-06-05에 이미 해결됨 (계획서 §1.3 참조)

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

위 여섯 문서는 기존 DART 실험의 authoritative set이다. 새 SoftWall 실행 우선순위는 위의 MPS 타당성 문서를 따른다. 처음 이해할 때는 walkthrough를 읽고, 수치와
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
- `cloudlab_final_snapshot_20260814/` *(현재 workspace에는 archive 미복원)*
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

- `task1_final/gdr_pool_20260814T014651Z/analysis/REPORT.md` *(현재 workspace에는 archive 미복원)*
  - 348-run full matrix와 64-run gate/replica/representative를 합친 actual GDR pool 최종 결과
- `task1_final/gdr_cuda_ipc_gate/` *(현재 workspace에는 archive 미복원)*
  - endpoint-agent 소유 GPU MR를 L1 process가 CUDA IPC로 직접 접근한 64KiB/1.35MiB gate
- `task1_final/dart_rx_radio_pool/analysis/REPORT.md` *(현재 workspace에는 archive 미복원)*
  - 실제 cuPHY/conventional/3-endpoint GDR NRx/LDPC/CRC paired correctness 최종 표
- `task1_final/dart_rx_radio_pool/.../nsys_l1.nsys-rep` *(현재 workspace에는 archive 미복원)*
  - warm-up 제외 CUDA Profiler API와 stage NVTX가 포함된 Nsight Systems capture

- `results/isca_v2/.../07_multicell_workloads/analysis/REPORT.md` *(현재 workspace에는 archive 미복원)*
  - 87-trace compute/queue problem gate
- `results/isca_v2/day1_20260813T0523Z/SUMMARY.md` *(현재 workspace에는 archive 미복원)*
  - direct TRT, P2P/GDR, placement 초기 결과
- `codex_20260814/.../paired_final.../analysis/REPORT.md` *(현재 workspace에는 archive 미복원)*
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
