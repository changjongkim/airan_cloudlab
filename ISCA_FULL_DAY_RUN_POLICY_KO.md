# ISCA 정식 실험 실행 정책

## 상태 정의

- `VALIDATED_QUICK`: 기능 또는 원인 확인에는 충분하지만, 논문의 정식 성능 수치는 아니다.
- `PARTIAL_QUICK`: 축약 반복, 짧은 trace, 불완전한 접근법 집합 중 하나 이상에 해당한다.
- `NOT_RUN`: 구현 또는 정식 하드웨어 실행이 아직 없다.
- `BLOCKED_DATA`: 입력 데이터와 provenance가 없으므로 synthetic 결과로 대체해 주장하지 않는다.
- `VALIDATED_FULL`: 계획된 반복 수, 실행 시간, correctness gate, provenance를 모두 충족한 경우에만 부여한다.

## 결과 디렉터리 분리

기존 `day1_20260813T0523Z`는 quick campaign으로 동결한다. 정식 실행은 새로운
`/mydata/results/isca_v2/full_day_<UTC timestamp>`에만 기록한다. Quick 결과를
정식 결과에 복사하거나 결측 trial을 채우는 용도로 사용하지 않는다.

각 job은 다음을 남겨야 한다.

1. immutable manifest: source SHA-256, image ID, GPU/MIG UUID, driver, engine SHA-256
2. exact command and environment
3. per-trial raw samples, stdout/stderr, 시작·종료 시각과 exit code
4. correctness result and invalidation reason
5. `COMPLETE` 또는 `FAILED`, 전체 캠페인의 `STATUS.tsv`

## 공정 비교 규칙

MPS, MIG, MIG+MPS, P2P, GDR은 같은 TensorRT engine, input tensors, arrival
trace, warm-up, sample count, latency boundary를 사용한다. 접근법마다 wrapper 또는
serialization 경계가 다르면 component latency와 end-to-end latency를 함께 기록하고,
서로 다른 경계를 하나의 bar chart에서 직접 비교하지 않는다.

Background workload는 synthetic tensor만으로 `realistic`이라고 부르지 않는다. Text,
image, audio의 데이터 provenance와 preprocessing 포함 여부를 명시한다. Video와 speech는
실제 입력이 준비될 때까지 `BLOCKED_DATA`로 둔다.

## 장시간 실행 순서

1. NRx wrapper/raw/binding/graph 정식 trial과 Nsight/NCU 원인 분석
2. 4g/3g/full GPU의 replica capacity 및 queue stability
3. 실제 text prompt와 real-dataset training workload qualification
4. 동일 workload/arrival trace의 MPS·MIG·MIG+MPS·P2P·GDR 비교
5. GDR reservation ring, epoch commit, deadline fallback이 통합된 DART hardware run
6. killer trace 및 mechanism-by-mechanism cumulative ablation

한 단계가 실패해도 독립 job은 계속하되, dependency가 깨진 downstream job은
`SKIPPED_DEPENDENCY`로 기록한다. MIG topology는 캠페인 중 재구성하지 않는다.
