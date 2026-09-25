# C151/C152: shared conventional-recovery 물리 data path 결과

**판정:** two-node physical path PASS / integrated V17 mode UQ  
**실행일:** 2026-09-24 UTC  
**노드:** `nid001253`, `nid001280`  
**GPU:** 노드별 NVIDIA A100-SXM4-40GB 4장, MIG OFF, MPS ON

## 목적

V16까지 여러 RAN home의 mandatory conventional recovery는 home별 GPU에서 실행됐다.
V17 control-plane 모델은 여러 home의 recovery debt를 한 global calendar로 검사하지만,
실제 cuPHY conventional 작업을 한 shared GPU로 보내는 경로가 없었다.

C151/C152는 다음 물리 경로의 첫 자격 시험이다.

```text
home 0 / GPU0 PUSCH window --CUDA IPC + NVLink P2P--+
                                                       GPU2 persistent cuPHY
home 1 / GPU1 PUSCH window --CUDA IPC + NVLink P2P--+  conventional worker
                                                       |
home별 decoded TB + CRC <-----------P2P---------------+
```

Shared worker는 각 round를 전역 key `(sequence, home_id)` 순서로 실행한다. Input과 decoded
payload는 GPU memory에 유지하며 host는 generation doorbell만 사용한다. Worker는 두 home의
receiver seed와 PHY config를 따로 보유하고, decoded payload와 CRC를 원 home buffer에
반환한다.

## 실행 전 고정

각 protocol은 formal arm 전에 다음을 고정했다.

- owner/worker/P2P/PHY/IPC source SHA-256
- 반복 수 250, worker warmup 10
- 두 home의 receiver seed와 channel seed base
- SNR 20 dB, pre-fading noise convention
- GPU0/1 source, GPU2 shared worker 배치
- `(sequence, home_id)` global order
- 처리량·deadline/WCET가 아니라 physical data-path gate라는 claim scope

C149 첫 bring-up은 current-device 오류로 CuPy peer-access 경고와 279 ms cold outlier가
있었다. GPU2 context 안에서 input window를 조립하도록 수정한 C149 attempt2와 C150은
경고 없이 통과했지만, protocol에 seed가 사전 기록되지 않은 것을 감사에서 발견했다.
이 둘은 development evidence로 보존하고 C151/C152를 새 frozen protocol로 다시 실행했다.

## 결과

| 항목 | C151 | C152 |
|---|---:|---:|
| Job / node | `58851949` / `nid001253` | `58852015` / `nid001280` |
| Home당 request | 250 + 250 | 250 + 250 |
| Shared-worker request | 500 | 500 |
| Correct decoded payload | 500/500 | 500/500 |
| Decode/CRC error | 0 | 0 |
| Global-order mismatch | 0 | 0 |
| Source/config mismatch | 0 | 0 |
| P2P/IPC lifecycle error | 0 | 0 |
| Conventional GPU p99 / max | 2.561 / 6.613 ms | 2.555 / 5.458 ms |
| Worker path p99 / max | 3.146 / 11.871 ms | 2.918 / 10.909 ms |
| Forward P2P p99 | 53.696 us | 52.576 us |
| Backward P2P p99 | 23.264 us | 23.072 us |

두 노드를 합치면 home request 1,000건과 worker request 1,000건이 모두 일치하며 decode
error는 0이다. 각 arm의 9개 gate와 combined 10개 gate가 모두 통과했다.

권위 artifact:

- [C151 protocol](../../results/softwall_multigpu/confirm151_shared_conv_qualified_job58851949_protocol.json)
- [C151 result](../../results/softwall_multigpu/confirm151_shared_conv_qualified_job58851949_result.json)
- [C152 protocol](../../results/softwall_multigpu/confirm152_shared_conv_holdout_job58852015_protocol.json)
- [C152 result](../../results/softwall_multigpu/confirm152_shared_conv_holdout_job58852015_result.json)
- [Combined qualification](../../results/softwall_multigpu/confirm151_152_shared_conventional_qualification.json)
- [24-file transitive manifest](../../results/softwall_multigpu/confirm151_152_shared_conventional_manifest.json)

## 무엇을 입증했는가

1. 두 독립 RAN home의 실제 PUSCH input을 한 remote GPU의 persistent cuPHY conventional
   worker가 처리할 수 있다.
2. Cross-process CUDA IPC handle과 NVLink P2P를 거쳐도 decoded TB/CRC가 원 home의 truth와
   1,000/1,000 일치한다.
3. 하나의 worker가 두 home을 전역 순서로 직렬화하고, sequence skip·duplicate·IPC handle
   조기 해제 없이 종료할 수 있다.
4. 같은 source와 서로 다른 frozen seed를 두 A100 node에서 재현했다.

## 아직 입증하지 않은 것

- V17 exact global certificate가 이 worker의 dispatch를 실시간으로 구동하는 통합 경로
- Optional NeuralRx 성공/실패에 따른 debt 생성·삭제와 shared worker의 conditional 실행
- Shared recovery calendar 변경과 external Qwen lease의 한 원자 transaction
- 실제 release/deadline 아래의 all-fail miss 0
- Worker path 11.871 ms보다 큰 자격 service bound 또는 WCET
- Worker crash, response loss, stale generation의 integrated fault containment
- Static shared calendar/safe work-conserving 대비 AI utility

따라서 현재 정확한 판정은 **global certificate model PASS + shared cuPHY data path PASS +
integrated V17 mode UQ**다. C151/C152의 최대값은 다음 bound 후보를 설계하는 입력일 뿐,
deadline 보장으로 사용하지 않는다.

## 다음 gate

다음 구현은 두 home controller가 `reserve_mandatory/resolve_success`를 global coordinator에
보내고, coordinator가 발급한 generation과 placement만 shared worker가 실행하게 만드는
통합 경로다. 최소 물리 반례는 local calendar에서는 각각 가능한 3+2 recovery를 한 shared
lane의 D100에 넣는 경우다. Local-only arm은 over-admit하거나 late가 되고, global arm은
다섯 번째 obligation을 GPU launch 전에 거절하면서 기존 네 recovery를 완료해야 한다.
그 뒤에 NRx success로 debt 하나가 해제될 때만 25 ms Qwen lease가 열리는 조건부 arm을
추가한다.
