# SoftWall PUSCH 타이밍 계약의 출처와 공백 — 2026-09-25

**판정:** 현재 P25/D21, P90/D80, P150/D130은 실험에서 정한 `(도착 주기 P, release-to-expiry D)` 쌍이다. 프로젝트 안에서 대상 DU/MAC가 uplink TB의 CRC 결과를 실제로 언제까지 소비해야 하는지 정한 설정이나 timestamp trace는 확인되지 않았다. 따라서 이 수치들을 production HARQ 기한이라고 부르지 않는다.

| 시각·설정 | 뜻 | 현재 확인 상태 |
|---|---|---|
| `t_IQ_ready` | gNB에서 해당 slot의 IQ/PUSCH 입력이 수신 경로에 준비됨 | timed harness의 명시적 계측 없음 |
| `t_release` | cuPHY/NeuralRx 요청을 제출할 수 있음 | synthetic harness에 있음; `t_IQ_ready`와의 차이 미계측 |
| `t_branch_ready` | DMRS/채널 feature를 이용해 optional NRx를 선택할 수 있음 | feature 계산 비용 일부 측정; 실제 DU 경로의 ready 시각 미계측 |
| `t_CRC_visible` | LDPC/CRC 결과가 MAC에 안전하게 공개됨 | 현재 harness의 완료 시각·single commit은 계측; 실제 FAPI/MAC 소비 시각 미계측 |
| `d_MAC` | 이 TB의 CRC 결과가 상위 scheduler/HARQ 결정에 유효한 마지막 시각 | **미정**. 대상 DU 설정·MAC/FAPI trace 또는 명시적인 계약 필요 |
| `P` | 같은 셀의 slot/request 도착 간격 | μ=1이면 0.5 ms slot이지만 `d_MAC−t_release`와 동일한 값이 아님 |

[ETSI TS 38.214 §6.1.2.1](https://www.etsi.org/deliver/etsi_ts/138200_138299/138214/19.02.00_60/ts_138214v190200p.pdf)의 `K2`는 scheduling DCI에서 UE가 PUSCH를 전송할 슬롯까지의 오프셋이다. 같은 규격 §6.4의 `N2`는 **UE의 PUSCH 준비 시간**이다. 둘은 gNB가 수신 PUSCH를 복호한 뒤 MAC에 CRC 결과를 전달해야 하는 `d_MAC`를 직접 정의하지 않는다. 프로젝트의 종전 “N2/K2에서 primary expiry를 유도” 문구는 이 구분 없이 사용하면 틀리므로 교정했다.

**P0 진입 조건:** 대상 DU의 설정 또는 FAPI/MAC trace에서 `t_IQ_ready`, `t_release`, `d_MAC`와 slot 번호를 같은 시계 기준으로 얻고, 요청별 `D=d_MAC−t_release`를 고정한다. 그 다음 AI 없는 필수 경로가 해당 `P,D`와 all-fail capacity 조건을 만족하는지 검증한다. 대상 DU 계약이 없으면 `D21/D80/D130`은 연구용 가정으로만 유지한다. 이 문서는 특정 3GPP 릴리스가 보편적인 gNB 처리 마감값을 준다고 주장하지 않는다.

## Production trace 최소 schema와 수용 규칙

새 schema를 만들지 않는다. 현재 구현된
[raw trace template](../../results/softwall_multigpu/c163_raw_du_trace_template_v1.json),
[clock calibration](../../results/softwall_multigpu/c163_clock_calibration_template_v1.json),
[expiry contract](../../results/softwall_multigpu/c163_expiry_contract_template_v1.json),
[mode qualification](../../results/softwall_multigpu/c163_mode_qualification_template_v1.json) 네 파일을
[v2 contract](../../results/softwall_multigpu/softwall_du_timing_contract_template_v2.json)로 조립한다.

| 계층 | 최소 필드 | 수용 규칙 |
|---|---|---|
| Trace provenance | `source_type`, `identifier`, 원본 `path`, `sha256` | `du-fapi-trace` 또는 `du-mac-trace`; 원본 hash가 일치해야 함 |
| Clock | `domain`, `timestamp_unit=ns`, `synchronized`, `method`, `max_error_ns`, calibration artifact | same-host monotonic-raw, PTP/PHC, TAI calibration 중 하나; 모든 record가 같은 domain |
| TB identity | `request_id`, `batch_id`, `home_id`, `cell_id`, `slot_id`와 가능하면 SFN | request ID 유일; raw event를 조립한 뒤 identity가 바뀌면 거부 |
| TB timestamps | `iq_ready_ns`, `release_ns`, `phy_submit_ns`, `crc_visible_ns`, `fapi_publish_ns`, `mac_consume_ns`, `expiry_ns` | 모두 비음수 integer이고 같은 clock; 앞의 여섯 시각은 단조 증가 |
| Radio outcome | `radio_commit_count`, `nrx_admitted`, `nrx_outcome_at_decision`, `nrx_outcome_observed_ns` | commit 정확히 1회; decision 이후에 관측된 NRx outcome을 미래 정보로 사용하면 거부 |
| Decision | `batch_id`, `decision_ns`, `mandatory_request_ids`, `unresolved_request_ids`, `ai_context_length`, `ai_deadline_ns`, `ai_lease_accepted`, `ai_complete_ns` | batch의 mandatory/unresolved 집합과 raw record가 일치; 허가된 AI는 class bound와 deadline을 만족 |
| Expiry provenance | `source_type`, `source_reference`, 원본 `path`, `sha256`, `synthetic=false`, `derived_from_ue_k2_n2=false` | DU configuration, MAC scheduler trace, FAPI contract만 허용; UE K2/N2나 사후 선택 D는 거부 |
| Mode fingerprint | lifecycle, node qualification, recovery capacity, `B_NRx`, `B_recovery`, `B_control`, guard, AI class bounds, qualification artifact/hash | 하나라도 바뀌면 다른 mode이며 기존 bound 재사용 금지 |

추가 hard rule은 `mac_consume_ns + max_error_ns <= expiry_ns`다. 관측 miss 한 건, duplicate
commit 한 건, artifact reconstruction 불일치 한 건도 mode를 fail-closed한다. 유한 trace가
통과해도 WCET로 승격하지 않는다.

## 세 가지 획득 경로와 필드 매핑

| 획득 경로 | 직접 얻는 필드 | 결합해야 하는 필드 | 판정 범위 |
|---|---|---|---|
| **A. Target DU 직접 계측** | slot clock에서 IQ ready/release/PHY submit, cuPHY/FAPI에서 CRC visible/publish, MAC에서 consume/expiry | request ID와 SFN/slot로 세 계층 join, clock calibration | P1을 닫을 수 있는 우선 경로 |
| **B. FAPI trace + MAC scheduler log join** | FAPI capture가 PHY submit/CRC visible/publish, MAC log가 consume과 explicit expiry 제공 | IQ ready/release hook, 공통 TAI/PTP 변환, expiry configuration hash | 두 artifact가 같은 TB identity와 clock으로 재조립될 때 P1 가능 |
| **C. Aerial testMAC controlled-L2** | slot `T0`, handler arrival, `T0+4.5 ms` configured threshold | 실제 target MAC consume/expiry는 없음 | fast-path integration 진단만 가능; production `d_MAC` 대체 불가 |

경로 A의 현재 Aerial hook 후보는 `nv_tick_generator`의 slot indication,
`phy::send_crc_indication`, 그 호출부와 target MAC consume 지점이다. 경로 B는 source artifact를
두 개 이상 사용해도 최종 builder가 content hash와 request identity로 하나의 contract를
재구성해야 한다. 경로 C의 4.5 ms는 구현 최적화용 보수적 `D` 후보이며 실제 DU trace를
얻기 전에는 P1 evidence로 승격하지 않는다.

## Trace에서 SoftWall parameter로의 결정적 변환

요청 `i`와 clock 최대 오차 `epsilon`에 대해 validator/bridge는 다음 값만 사용한다.

```text
D_i                    = expiry_i - release_i - epsilon
release_i^recovery     = release_i + B_NRx    (NRx admitted)
                       = release_i            (NRx not admitted)
deadline_i^recovery    = expiry_i - guard - epsilon
service_i^recovery     = B_recovery
AI blackout            = [decision,
                          decision + B_control + B_AI(context_class)]
```

따라서 slot duration이나 평균 arrival period `P`를 `D_i`로 치환하지 않는다. 초기 all-fail
집합과 각 decision 시각의 unresolved 집합에 certificate를 각각 만들고, AI blackout을 넣은
뒤에도 certificate가 남을 때만 lease를 허가한다. 관측 결과와 predicted QSU/QSN/MI/UQ가
다르면 해당 mode는 UQ다. 이 mapping은 production trace가 들어왔을 때 그대로 실행되지만,
trace 부재 자체를 해결하지는 않는다.

## 로컬 Aerial source audit

현재 checkout에서 CRC/FAPI publication과 slot clock을 삽입할 위치는 확인했다.

- `third_party/aerial-cuda-accelerated-ran/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi_phy.cpp:4528`의
  `phy::send_crc_indication`은 `t_CRC_visible`/FAPI publish 계측 후보이고, 호출부는 같은
  파일 4409행에 있다.
- `third_party/aerial-cuda-accelerated-ran/cuPHY-CP/cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`의
  slot indication generator는 slot ID와 공통 clock을 연결할 후보 지점이다.
- Checkout의 `T1a` 필드는 `cuphydriver/include/cell.hpp`와
  `cuphydriver/include/dl_validation_params.hpp`의 설명대로 O-RAN fronthaul packet의
  송수신 advance/window 값이다. PUSCH decode 뒤 MAC이 CRC를 소비하는 expiry가 아니다.

추가로 Aerial testMAC의 실행 계약을 확인했다. 기본 설정은 early HARQ를 `T0+2.0 ms`,
CRC를 포함한 UL indication을 `T0+4.5 ms`로 두며,
`scf_fapi_handler::validate_indication_timing`은 SFN/slot에서 복원한 `T0`와 indication handler
진입시각을 직접 비교한다. 이는 현재 synthetic `D155`보다 강한 vendor integration target이다.
그러나 testMAC README가 이를 developer용 controlled L2라고 명시하므로 target production
MAC의 `d_MAC`을 대신하지 않는다.

4.5 ms를 진단 threshold로 사용한 결과, local clean wait-then-recover와 same-GPU speculative
path는 각각 0/1,000이었다. GPU0 conventional과 GPU1 full NeuralRx를 raw-IQ P2P로 겹친
최선의 prototype은 852/1,000까지 개선됐지만 148개 tail이 남았다. 따라서 코드 위치와
vendor threshold를 찾은 것만으로 production timing gate를 통과하지 못하며, 현재 판정은
`FAIL_CURRENT_DESIGN_NOT_PRODUCTION_QUALIFIED`다. 상세 경로와 후속 gate는
[production exit plan](SOFTWALL_PRODUCTION_EXIT_PLAN_KO.md)에 기록했다.

## G7 validator v1 구현 상태

[DU timing validator](../../scripts_for_node/softwall_same_gpu/du_timing_contract_validator.py)와
[입력 template](../../results/softwall_multigpu/softwall_du_timing_contract_template_v1.json)을
구현했다. Validator는 다음을 모두 검사한다.

1. 입력이 synthetic harness가 아니라 식별자와 hash가 있는 DU/FAPI 또는 DU/MAC trace인가
2. 모든 timestamp가 하나의 synchronized nanosecond clock domain에 있는가
3. expiry가 명시적 gNB DU/MAC/FAPI 계약에서 왔고 UE-side K2/N2에서 유도되지 않았는가
4. IQ-ready→release→PHY submit→CRC visible→FAPI publish→MAC consume 순서가 맞는가
5. request ID가 유일하고 radio commit이 정확히 한 번인가
6. MAC consume이 request expiry 전인가
7. 선언한 home별 recovery bound와 guard의 all-fail demand가 최소 request deadline에 들어가는가

단위시험 9개가 정상 입력과 synthetic expiry, K2/N2 유도, clock mismatch, duplicate request,
timestamp 역전, duplicate commit, observed miss와 mandatory-capacity 실패를 각각 판정한다.
[Readiness artifact](../../results/softwall_multigpu/softwall_du_timing_readiness_v1.json)의 상태는
`UQ_NO_PRODUCTION_TRACE`다. Validator 준비는 실제 trace 확보를 대체하지 않으며 unit-test
fixture도 production evidence로 사용하지 않는다.

## C163 validator v2와 production envelope bridge

V1의 `cell_count × recovery_bound <= min(D)` 검사는 요청별 release/expiry, 병렬 recovery
capacity, decision 시각과 AI blackout을 표현하지 못한다. 따라서 C163은 다음 네 원본
artifact를 content hash로 묶는 v2 경로를 사용한다.

| Artifact | 필수 내용 | 거부 조건 |
|---|---|---|
| Raw DU trace | TB identity, 동일 clock의 여섯 timestamp, NRx outcome, SoftWall decision, single commit | event 누락·중복, request identity 변화 |
| Clock calibration | clock domain, 동기화 방법, 최대 변환 오차 | 미지원 clock 또는 오차 미제시 |
| Expiry contract | TB별 절대 `d_MAC`, DU/MAC/FAPI 출처 | synthetic 또는 UE K2/N2 유도 |
| Mode qualification | topology/lifecycle과 NRx·recovery·control·AI class whole-path bound | hash 불일치 또는 unqualified mode |

Validator는 hash만 대조하지 않는다. 네 artifact로 contract 전체를 다시 조립해 원본
expiry·bound·decision과 동일한지 확인하며, NRx outcome 관측시각이 SoftWall decision보다
늦으면 미래 정보 누출로 거절한다.

[C163 contract builder](../../scripts_for_node/softwall_same_gpu/c163_build_du_contract_v2.py)는
원시 event를 request별 record로 조립한다. [V2 validator](../../scripts_for_node/softwall_same_gpu/du_timing_contract_v2.py)는
clock 최대 오차를 expiry에서 차감하고, [production bridge](../../scripts_for_node/softwall_same_gpu/c163_production_envelope_bridge.py)는
각 request `i`에 다음 absolute recovery obligation을 만든다.

```text
release_i^rec = release_i + B_NRx     (NRx admitted)
              = release_i             (NRx not admitted)
deadline_i^rec = d_MAC,i - guard - clock_error
service_i^rec  = B_recovery
```

초기 all-fail 집합과 decision 시각의 unresolved 집합에 exact certificate를 각각 생성한다.
AI가 있으면 `[decision, decision+B_control+B_AI(class)]`를 full-GPU blackout으로 삽입해 다시
검사한다. 이 예측 QSU/QSN/MI/UQ와 물리 `ai_lease_accepted`가 다르면 mode를 통과시키지
않는다. 실제 `mac_consume + clock_error > d_MAC`도 한 건이면 실패다.

[C163 readiness v2](../../results/softwall_multigpu/c163_du_timing_readiness_v2.json)는 builder,
bridge와 validator 단위시험 24개를 통과했다. 작은 batch는 exact solver, 10개를 넘는
debt는 independent verifier를 포함한 C162 certified scheduler를 사용한다. 큰 상태에서
certificate를 찾지 못하면 MI로 단정하지 않고 UQ로 남긴다. 현재 상태는 계속
`UQ_NO_PRODUCTION_TRACE`다. 이는 C163 실험을 받을 분석 경로가 준비됐다는 뜻이며 실제 DU
deadline 보장은 아니다.
