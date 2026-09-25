# SoftWall 목표 완료 여부 점검 — 2026-09-21 UTC

이 문서는 [원 계획서](RESEARCH_PLAN_SOFTWALL_KO.md)의 P0–P7과 사용자 목표인 **같은 GPU / MIG OFF / MPS에서 허용된 AI 작업의 RAN deadline 보호와 남는 자원의 AI 처리량 회수**를 현재 결과와 대조한다. [실험 gate ledger](../../results/softwall_same_gpu/EXPERIMENT_GATE_LEDGER_KO.md)의 판정을 우선한다. 아래 초기 표는 Confirm47–55까지의 범위를 설명하며, 후속 Confirm56–81의 현재 판정은 [검증 기록](SOFTWALL_VALIDATION_PROGRESS_KO.md)을 따른다.

**2026-09-22 현재 판정:** [공동 스킴 명세](SOFTWALL_NOVELTY_SCHEME_DECISION_KO.md)에 맞춘
실험은 계속 중이다. 독립 채널 가치 입력은 Confirm74/77, 3셀 동시 복구는 Confirm75,
원자 AI lease의 3셀 성공 경로는 Confirm76, 4셀 경로는 Confirm80에서 통과했고,
Confirm81은 AI GPU 완료 뒤 RPC 응답 지연에 한해 RAN 지속을 확인했다. Confirm78의 실제 PHY feature를
쓴 3셀 CPU 결정 스크린과 Confirm79b의 4셀 경합 스크린에서는 강한 greedy와 exact가 동일했다. 따라서 **같은 정보·안전장치를
가진 AI 인지 greedy보다 PHY 가치 기반 공동 정책이 낫다는 GPU paired 결과가 없어
노벨리티는 미확보**다. 실제 DU `d_MAC`, AI timeout 뒤 물리 작업의 fault continuation,
mode별 방어 가능한 실행 상한도 없고 GPU hang·marker 없는 worker crash 지속 경로도 없다. 따라서 원 계획 P0–P7이나 사용자 목표를 완료로
판정하지 않는다. 아래 오래된 "다음" 문단은 해당 단계 당시의 기록이고 이 최신 판정을
덮어쓰지 않는다.

**2026-09-22 bound-first 추가 판정:** 현재 4셀 D155에서 선언값 `B_NRx50+
4×B_conv25=150 ms`가 경합 실험의 선택 영역을 좁힌다. [기존 raw 재감사](../../results/softwall_same_gpu/pre_confirm82_bound_mode_audit.json)는
짧은 예열 표본의 낮은 최대치와 달리 구 mode 장시간 run에서 NRx 30 ms 초과
6건, 미예열 conventional 첫 GPU 최대 35.303 ms를 확인했다. 따라서
[Confirm82](../../results/softwall_same_gpu/confirm82_bound_campaign_job58738952.json)에서
현재 예약을 그대로 둔 독립 4셀×3,000-release 두 run을 검사했다. 기존
50/25 ms 계약 gate는 통과했고 NRx 25/30 ms 후보는 최대 44.589 ms로
탈락했다. 예열 4셀 conventional 8 ms는 표본상 통과했지만 최대 7.978 ms로
여유가 0.022 ms다. [Confirm83](../../results/softwall_same_gpu/confirm83_conv8_fourcell_job58738952.json)은
실제 8 ms 예약표에서 conventional 초과 0·deadline miss 0이었지만 NRx
50 ms 초과 2건으로 전체 gate가 실패했다. 즉 서비스 상한은 각각 따로
줄여도 되는 상수가 아니라 공동 실행 mode의 계약이다. [Confirm84](../../results/softwall_same_gpu/confirm84_eightcell_conv12_protocol.json)의
새 8셀·12 ms 탐색은 [독립 PHY receiver 생성 중 GPU OOM](../../results/softwall_same_gpu/confirm84_pretraffic_oom_failure.json)으로
timed release 전에 실패했다. [Confirm85](../../results/softwall_same_gpu/confirm85_tail_probe_protocol.json)의
NRx pre-dispatch tail 원인 분리와 receiver 메모리 구조 변경이 다음 단계다.
현재 결과는 hard WCET나 12셀 가능성 증명이 아니다. 경합 trace·정책 비교는
이 자격 gate 다음에 둔다.

**후속 bound 자격 갱신:** [Confirm88](../../results/softwall_same_gpu/confirm88_gc_off_long_bounds_job58743005.json)은
GC OFF 예열 4셀의 두 5,000-release 반복에서 기존 50/25 ms 계약을 통과했고,
NRx25·복구12 ms를 표본 후보로 남겼으나 복구8 ms는 한 건 초과로 탈락했다.
[Confirm90](../../results/softwall_same_gpu/confirm90_tight_calendar_gc_off_job58743005.json)은
실제 25/12 ms 달력의 새 두 3,000-release 반복 중 첫 arm에서 NRx 응답
26.875 ms 한 건으로 **전체 실패**했다. worker 결과가 먼저 게시되어도 AI 반환과
세 필수 conventional commit 뒤 controller가 관측하면 전체 NRx 경로 상한이
깨진다. [Confirm91](../../results/softwall_same_gpu/confirm91_conv12_nrx30_gc_off_job58743005.json)은
별도 사전 protocol의 실제 30/12 ms 달력에서 두 새 seed의 frozen gate를
모두 통과했다. 이는 그 예열·GC OFF·4셀 mode의 유한 표본 자격이며, WCET·
실제 DU·다셀 메모리·AI 처리량 우위·공동 정책 노벨리티의 공백은 그대로다.
[Confirm89의 사전 ON/OFF 독립 재현](../../results/softwall_same_gpu/confirm89_gc_causal_replication_job58743005.json)은
전체 실패다. 첫 pair에서는 같은 요청의 GC ON−OFF NRx 응답이 +38.797 ms여서
메커니즘 gate를 통과했지만 ON 50 ms 상한 위반 2건으로 안전 gate가 실패했다.
둘째 역순 pair의 +8.069 ms는 사전 +10 ms 기준에 못 미쳤다. 따라서 GC ON의
기존 50 ms bound를 유효한 보증으로 간주하지 않고, C91의 GC OFF 자격과 분리한다.

## 현재 확정된 결과

| 주장 | 근거 | 판정 범위 |
|---|---|---|
| 실제 원자 fallback 예약·single commit·복구 전 AI lease를 같은 GPU의 별도 MPS endpoint에 연결 | Confirm42 통합, Confirm45 독립 seed·역순 eager 비교 | Confirm45 전체 frozen gate 통과, AI 완료량 +2.95%/+2.41%, RAN miss 0 |
| 보수적인 새 admission 계약에서도 같은 방향의 이득 | [Confirm47](../../results/softwall_same_gpu/confirm47_conservative_bound.md) | NRx 50 ms·AI socket timeout 35 ms에서 전체 gate 통과, +1.38%/+1.11%, RAN miss 0; 외부 RAN p99 약 55.8 ms 대 eager 약 16 ms, D80 |
| 복구 전 gap lease의 추가 처리량 | Confirm43/44 ON/OFF | AI 순증 +0.419%/+1.304%; 두 실험 모두 30 ms NRx bound 위반 각 3건으로 전체 gate 실패 |
| GPU/MPS client 수명주기는 deadline 위험 요소 | [Confirm46](../../results/softwall_same_gpu/confirm46_retirement_cpu_control.md) | CPU sham 0/10,000 대 GPU/MPS 12/10,000 miss; 어떤 내부 단계가 원인인지는 미분리 |
| 새 50 ms 계약에서 gap lease가 고정 복구+bounded AI보다 추가 AI를 완료 | [Confirm50](../../results/softwall_same_gpu/confirm50_requalified_gap_abba.md) | 독립 seed·역순 ON−OFF +0.468%/+0.535%; 네 조건 40,000건 RAN miss·NRx/conv bound·AI 위반 0, 38개 frozen gate PASS. 한 셀·한 endpoint의 작은 구성요소 이득으로 해석 |
| 1셀 부하 증가 시 시스템 우월성 진단 | [Confirm51](../../results/softwall_same_gpu/confirm51_n1_single_cell_diagnostic.md) | P90/45/25/12·D80 첫 라운드에서 eager miss 0/0/0/0. External miss도 0이지만 correct TB는 P45부터 eager 대비 하락, P25/12 AI 완료 0. 두 라운드 모두 교차점을 요구하는 gate가 불가능해져 조기 중단. 전체 다중 셀 N1은 미완료 |

**환경 구성 판정:** Confirm47의 Slurm job `58688541`이 `nid001084`에서 MPS daemon/client, shifter image, TensorRT engine, cuPHY, CUDA IPC endpoint와 결과 분석기를 실제 실행하고 정상 종료했으므로 **이번 같은 GPU 실험 환경은 동작 검증됐다.** 원 계획서의 4-GPU pool 및 노드 간 Slingshot 환경까지 구성·검증됐다는 뜻은 아니다. 모든 새 프로젝트 파일과 실행 결과는 지정된 `airan_cloudlab` 루트 안에 둔다.

**예약 시각 계측 공백:** Confirm45/47/50의 일반 controller 원자료에서 `fallback_start_ns`는 예약한 시각이며 실제 시작 시각이 아니다. `DartTransaction` 내부의 실제 시작 값은 당시 결과에 기록되지 않았다. 따라서 이 세 실험의 RAN deadline miss와 bound gate는 실제 완료·상한 위반 여부를 보여도, 모든 fallback이 예약 시각에 시작했는지는 직접 검증하지 못한다. Confirm50이 종료된 뒤 일반 controller에 `fallback_actual_start_ns`, `fallback_start_lateness_ms`, `fallback_started` 계측을 추가했다. 이는 **후속 실행에만 적용되며 과거 결과를 소급 보완하지 않는다.** 별도 fault probe도 실제 시작 시각을 기록하도록 준비되어 있다. Confirm51의 새 계측에서 실제 fallback 시작 시각과 1 ms 초과 지연 0건을 확인했지만 이는 1셀 진단의 관측 결과일 뿐이며, 과거 세 실험을 소급 보완하거나 최악 경우의 예약 시각 보장을 증명하지 않는다.

**시각 필드의 정확한 뜻:** `fallback_actual_start_ns`는 명칭과 달리 host가
`start_fallback`/`run_conventional`을 호출하기 직전에 기록한 값이다. GPU kernel의
물리 시작 시각은 아니다. `completed_ns`도 `complete_conventional` 호출이 **끝나기 전**의
시각이다. Confirm51의 “1 ms 초과 지연 0건”은 예약 시각 대비 **host 결정 지연**에
한정한다. Confirm55의 48개 fallback에서 host 결정부터 commit **호출 전**까지의
경과 시간에서 GPU event duration을 뺀 잔여가 최소 0.110 ms, 중앙값 0.117 ms,
최대 0.201 ms였다([원자료 재계산](../../results/softwall_same_gpu/confirm55_novelty_gap.json)).
이 잔여에는 launch/queue/host 후처리가 함께 들어 있으므로 kernel 시작 지연이나
commit 완료 지연으로 해석할 수 없다. [후속 계측판](../../scripts_for_node/softwall_same_gpu/two_endpoint_bringup_controller_v2.py)은 commit 함수 반환 시각을
별도로 기록하고 이를 보수적인 deadline 판정에 사용하도록 수정했다. 기존 raw에는 소급되지 않으며 후속 계측판의 GPU 실행은 아직 없다.

**Confirm55 소스 보존 한계:** 실행 당시 [frozen protocol](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup_protocol_retry1.json)은 controller SHA-256 `05aac537…`를 고정했고, [그때의 판정](../../results/softwall_same_gpu/confirm55_two_endpoint_bringup.json)은 소스 해시 gate 통과를 기록했다. 이후 controller에 계측 수정이 들어갔으며 현재 프로젝트 파일은 그 해시와 일치하지 않는다. 이번에 후속 계측판을 별도 파일로 분리했으나, 실행 당시의 정확한 원본 소스를 프로젝트에서 복원하지 못했다. 따라서 현재 파일로 frozen analyzer를 재실행해 같은 소스 해시 PASS를 재현할 수는 없다. 보존된 원자료와 당시 판정은 사후 소스 수정과 구분해서 인용한다.

**timeout 정리 경계:** 두-endpoint controller는 `wait_backward` timeout에서 conventional
복구를 시작하지 않고 실행을 중단한다. 그러나 예외 정리에서는 worker 종료 신호를 쓰고
0.2초 뒤 IPC owner를 닫으며, 그 전에 미완료 GPU 작업의 종료를 확인하지 않는다.
Confirm55에는 timeout이 0건이므로 성공 경로의 결과는 그대로 유효하지만, 이 코드로
늦거나 멈춘 worker까지 포함하는 버퍼 수명 보장을 주장할 수 없다. 후속 정책의 fault
시험에는 supervisor와 물리 완료/격리 확인 뒤 버퍼 해제하는 절차가 필요하다.

**정책 범위:** 현재 한 셀·한 endpoint controller는 `DartRuntime.submit()`의 고정 50 ms service profile과 25 ms fallback calendar를 사용하고, background는 남은 시간에 40 ms budget+2 ms guard가 들어갈 때 반복 허가한다. Endpoint 선택은 후보가 하나라 의미가 없고, 채널 feature에 따른 NRx gate나 radio utility·AI 처리량을 함께 최적화하는 정책은 구현하지 않았다. 그러므로 Confirm45/47/50은 **안전한 구성의 시스템 실현과 비용·효과**에 대한 근거이지 원 계획 P2/P4의 새 공동 최적화 알고리즘 검증이 아니다. Confirm51에서는 직렬 fallback 대기가 다음 release의 NRx 수락을 막아 무선 정답률이 감소했다. 강한 결합 baseline과 차이가 작거나 없을 가능성을 열어 둔다.

**N1 다중 셀 계약의 선행 제약:** D80·단일 conventional lane·복구 상한 25 ms·guard 2 ms에서는 같은 시각에 4/8셀이 모두 실패할 때 필요한 복구 시간이 각각 102/202 ms이므로 현 계약에서 불가능하다. 2셀은 `[28,53]`, `[53,78]` ms 복구 구간을 예약할 수 있지만 첫 셀의 NRx를 50 ms까지 기다리지 못하고 28 ms에 포기해야 한다. 이는 결과 품질·미완료 GPU 작업 격리 비용을 새로 검증해야 하는 **다른 제어 스킴**이다. Confirm24의 2셀 PASS는 P150/D130 계약이므로 D80 다중 셀 가능성 증명이 아니다. all-fail 지속 부하에서 1셀 P12는 `25/12 > 1`로 단일 복구 lane의 처리 능력을 넘는다. 이 조건들을 만족하지 않는 원래 N1 격자는 그대로 실행할 수 없다.

## 원 계획서 P0–P7의 실제 상태

| 단계 | 원 계획의 필수 범위 | 현재 증거와 남은 공백 | 상태 |
|---|---|---|---|
| P0 | 0.5/1 ms slot-paced L1, 실제 absolute deadline, 프레임당 malloc/free 제거 후 MPS 결론 재검증 | 현재 가장 깨끗한 same-GPU 계약은 P90/D80이며 원 계획의 0.5/1 ms production-slot 증명과 메모리 할당 제거 게이트를 대체하지 못함 | 미완료 |
| P1 | 플랫폼 불일치, compute/memory 온라인 판별, bistability, fp16 convert 판정 | MPS cap/HBM 반례와 단일 노드 경향은 측정했으나 μs–ms 판별자·플랫폼 차이·convert 결론은 미확정 | 일부 |
| P2 | co-tenant-aware 완료시간 모델과 queue-only 대비 held-out miss 검증 | 현재 실험은 보수적 고정 admission bound; 계획서의 모델 비교는 수행하지 않음 | 미완료 |
| P3 | 다중 GPU P2P push/pull, copy 발행 위치, NRx 내부 배치·batch sweep | 기존 P2P 기초 측정은 있으나 이번 same-GPU transaction과 연결한 E3-a/b/c 전체 gate는 미완료 | 일부 |
| P4 | 3개 제어 루프, GPU0 회수 사다리, priority/static pct 대비, fault injection | 원자 복구 calendar+AI lease의 GPU 성공 경로 Confirm73/76과 3셀·2 endpoint 동시 all-fail 분기 Confirm75/76은 통과. GPU0 회수 사다리, full lifecycle/AI timeout 뒤 물리 완료 추적, 세 제어 루프의 독립 비교는 미완료 | 일부 |
| P5 | §8의 workload·cell·arrival·budget 전반 평가와 강한 결합 baseline | 채널 gate+queue-aware+조기 복구 결합 Confirm62, AI 인지 greedy retime Confirm71/72, 독립 PHY 가치 보정 Confirm74까지 확보. **같은 조건의 완전한 AI 인지 greedy 대 PHY 가치 공동 정책 paired GPU 비교**와 여러 AI 마감/도착·실제 DU trace는 미완료 | 일부 |
| P6 | 노드 간 Slingshot/libfabric preview | 이번 라운드에서 실행하지 않음 | 미완료 |
| P7 | 논문 집필과 선행 연구 대비 | [선행연구 감사](SOFTWALL_RELATED_WORK_AUDIT_KO.md)와 [기여 논지](SOFTWALL_NOVELTY_THESIS_KO.md)는 작성. 원 계획의 완성 원고·강한 baseline 결과는 없음 | 일부 |

P0의 간극은 단순 표기 문제가 아니다. Confirm47 r1 외부 경로의 10,000건 원자료에서 front GPU 시간 중앙값 0.934 ms, post GPU 시간 중앙값 0.753 ms, conventional fallback GPU 시간 중앙값 2.642 ms, 전체 RAN 응답 중앙값 3.076 ms였다. 이 수치만으로 최적화된 생산용 cuPHY의 처리율 한계를 단정할 수는 없지만, **현재 Python/CUDA-IPC full-path harness의 P90/D80 통과를 0.5 ms TTI 보장으로 환산할 근거는 없다.**

## 노벨리티 판정과 다음 의사결정

**조건부 same-GPU 시스템 기여의 반복 가능한 실측 근거는 있다. 원 계획서 전체나 탑컨퍼런스급 독창성, hard-real-time 보장은 아직 완료가 아니다.** 공개 선행 연구에서 같은 전체 범위를 확인하지 못했지만, 유사 연구의 부재는 증명할 수 없다. 가장 큰 논문 심사 위험은 강한 기존 기법 결합 baseline을 아직 동일 자원·정보·계약에서 이기지 못했다는 점이다. 단순 MPS 공유·fallback·짧은 AI unit 각각은 선행 연구와 겹친다.

Confirm51에서 1셀 P12까지 eager miss가 없고 external의 무선 효용과 AI 처리량이 높은 부하에서 감소했다. 현재 구조로 고부하 우월성을 주장하지 않는다. 다음 연구 판단은 무한한 seed 추가가 아니다. 새 50 ms 계약에서 고정 복구+bounded AI와 gap lease ON/OFF를 비교한 Confirm50은 job `58689254`에서 두 독립 seed·역순 모두 PASS했지만 이득은 +0.47%/+0.54%로 작다. 추가 seed를 붙이지 않는다. 이것은 전체 채널 gate+queue-aware 결합 baseline과 같지 않으며, 그 baseline의 동일 입력·GPU 예산·radio 비열등성 비교는 별도 공백으로 남는다. 운영 보증을 주장하려면 NRx/mandatory service의 방어 가능한 worst-case bound, 실제 fallback 시작 시각 계측과 AI timeout 고장 주입이 필요하다. 원 계획서의 P0–P7 전체 완료를 요구한다면 위 표의 미완료 항목은 계속 남는다.
