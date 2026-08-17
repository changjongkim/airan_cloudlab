#!/usr/bin/env python3
"""Generate English versions of the research walkthrough figures.

The plotting logic and every numeric value come from
``generate_research_walkthrough_figures.py``.  This wrapper only translates text
objects immediately before saving, which keeps the Korean and English figures
on one measurement/code path.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator

import generate_research_walkthrough_figures as base


OUT = base.ROOT / "docs" / "current" / "figures_en"

EN = {
    "보호된 L1과 요청 생성 (Blue)": "Protected L1 & Request Construction (Blue)",
    "Deadline 기반 endpoint 선택 (Beige)": "Deadline-Aware Endpoint Selection (Beige)",
    "유효기간을 지키는 결과 commit (Blue)": "Expiry-Safe Result Commit (Blue)",
    "상주 NRx service pool (Green)": "Resident NRx Service Pool (Green)",
    "기존 수신기\n항상 준비": "Conventional receiver\nalways ready",
    "L1 등록\nGPU buffer": "L1 registered\nGPU buffer",
    "Slot 요청 descriptor · cell · slot · epoch · expiry · radio utility":
        "Slot request descriptor · cell · slot · epoch · expiry · radio utility",
    "전용 4g MIG · 반드시 끝나야 하는 L1 경로":
        "Dedicated 4g MIG · mandatory L1 critical path",
    "submit +1 · completion -1 · 원격 GPU queue scan 없음":
        "submit +1 · completion -1 · no remote GPU queue scan",
    "NRx 0   pending 3   ·   끝날 시각 늦음": "NRx 0   pending 3   ·   finishes late",
    "NRx 1   pending 0   ·   지금 가능   ← 선택":
        "NRx 1   pending 0   ·   available now   ← SELECT",
    "NRx 2   pending 1   ·   곧 가능": "NRx 2   pending 1   ·   available soon",
    "예측 완료 = max(현재시각, 예약 tail) + service bound":
        "predicted finish = max(now, reserved tail) + service bound",
    "Queue credit 예약\n및 dispatch": "Reserve queue credit\n& dispatch",
    "HOST CONTROL PLANE · counter · health · deadline 판단":
        "HOST CONTROL PLANE · counters · health · deadline decision",
    "기존 수신기\n결과": "Conventional\nresult",
    "NRx completion\n결과 + epoch": "NRx completion\nresult + epoch",
    "검증\nslot · epoch · expiry\nhealth · CRC":
        "Validate\nslot · epoch · expiry\nhealth · CRC",
    "정확히 하나의\nLDPC/CRC\n결과": "Exactly one\nLDPC/CRC\noutcome",
    "늦거나 · 오래됐거나 · 고장났거나 · 불가능한 NRx → 기존 수신 결과 사용":
        "Late · stale · unhealthy · infeasible NRx → use conventional fallback",
    "GPU payload fabric · 지원될 때 P2P  |  ConnectX-6 Dx GPUDirect RDMA":
        "GPU payload fabric · P2P when supported  |  ConnectX-6 Dx GPUDirect RDMA",
    "고정 MIG 격리벽 · 상주 model · fast path에서 MIG 재구성 없음":
        "Fixed MIG walls · resident models · no fast-path MIG reconfiguration",
    "작은 descriptor": "small descriptor",
    "endpoint id\n+ queue credit": "endpoint id\n+ queue credit",
    "NRx 결과": "NRx result",
    "작은 control metadata": "Small control metadata",
    "GPU payload / 결과": "GPU payload / result",
    "Queue 길이 = dispatcher가 추적하는 미완료 요청 수 · 원격 CUDA queue를 읽는 값이 아님":
        "Queue length = dispatcher-tracked outstanding work · not a remote CUDA queue scan",
    "DART-Rx 전체 구조: L1을 보호하면서 queue 상태로 NRx endpoint를 선택":
        "DART-Rx Architecture: Protect L1 and Select NRx Endpoints by Queue State",
    "그림: L1은 기존 수신 fallback을 유지하고, DART-Rx는 고정 MIG 경계를 넘어 제시간에 끝날 상주 NRx를 예약한다.":
        "Figure: L1 retains a conventional fallback while DART-Rx reserves a timely resident NRx across fixed MIG boundaries.",
    "(a) MPS: 하나의 GPU를 모두 공유": "(a) MPS: all workloads share one GPU",
    "공유 GPU 자원\n(SM · HBM · 작업 대기열)": "Shared GPU resources\n(SMs · HBM · work queues)",
    "하드웨어 벽 없음 → 높은 활용률, 약한 L1 보호": "No hardware wall: high utilization, weak L1 protection",
    "(b) MIG: L1과 NRx는 같은 4g": "(b) MIG: L1 and NRx share one 4g",
    "같은 방의 경합": "Same-partition contention",
    "3g MIG\n\nQwen /\nBackground": "3g MIG\n\nQwen /\nbackground",
    "굵은 선 = MIG 하드웨어 격리벽": "Thick line = MIG hardware isolation wall",
    "(c) MIG+MPS: 같은 4g 안의 몫 조절": "(c) MIG+MPS: tune shares inside one 4g",
    "점선은 몫 제어이지 격리벽이 아님": "Dashed boxes control shares; they are not isolation walls",
    "Sibling은 격리 · L1/NRx 경합은 남음": "Sibling isolated; L1/NRx contention remains",
    "(d) MIG+P2P*: 분리된 GPU 공간을 직접 연결": "(d) MIG+P2P*: directly connect separate GPU spaces",
    "2g MIG\n\nL1\nGPU buffer": "2g MIG\n\nL1\nGPU buffer",
    "2g MIG\n\nNRx\nGPU buffer": "2g MIG\n\nNRx\nGPU buffer",
    "3g MIG\n\nBackground": "3g MIG\n\nBackground",
    "* 이번 gate: 한 process가 두 MIG CUDA context를 소유": "* This gate: one process owns both MIG CUDA contexts",
    "peer access가 실제로 열리는 topology에서만 사용": "Use only on a topology where peer access succeeds",
    "(e) MIG+GDR: NIC loopback으로 GPU 메모리 연결": "(e) MIG+GDR: connect GPU memory through NIC loopback",
    "2g MIG\nL1 process\nGPU MR": "2g MIG\nL1 process\nGPU MR",
    "2g MIG\nNRx process\nGPU MR": "2g MIG\nNRx process\nGPU MR",
    "3g MIG\nBackground": "3g MIG\nBackground",
    "payload: GPU → NIC → GPU": "Payload: GPU → NIC → GPU",
    "CPU DRAM을 payload가 통과하지 않음": "Payload does not traverse CPU DRAM",
    "(f) 그림을 읽는 기준": "(f) How to read the diagrams",
    "L1: 반드시 끝나야 하는 PHY 경로": "L1: mandatory PHY critical path",
    "NRx: 선택적으로 호출하는 AI 수신기": "NRx: optional neural receiver",
    "Background: 남는 자원에서 도는 AI": "Background: AI using spare capacity",
    "지원되는 GPU peer data path": "Supported GPU peer data path",
    "NRx 요청 1개 = CE가 끝난 뒤\nL1 측 scheduler가 보낸 cell-slot 추론 1개":
        "One NRx request = one cell-slot inference\nissued by the L1-side scheduler after CE",
    "다섯 배치의 실제 차이: L1–NRx 사이의 벽과 데이터 이동 경로":
        "The actual difference among five placements: the L1–NRx wall and data path",
    "배치 설명도(성능 그래프 아님): MPS와 local MIG/MPS는 L1·NRx가 같은 실행 공간, P2P/GDR는 둘을 분리":
        "Placement schematic (not a performance chart): MPS and local MIG/MPS colocate L1/NRx; P2P/GDR separate them",
    "1단계 · 단일 GPU cross-MIG GDR baseline": "Stage 1 · Single-GPU cross-MIG GDR baseline",
    "물리 GPU 1개 · NRx replica 1개": "1 physical GPU · 1 NRx replica",
    "증명: 격리된 두 MIG 사이 GPU-memory GDR data path":
        "Proves: GPU-memory GDR data path between two isolated MIGs",
    "2단계 · Fixed-MIG NRx 3-replica GDR pool": "Stage 2 · Fixed-MIG 3-replica GDR pool",
    "4g source /\nL1-side GPU MR\n(radio 미실행)": "4g source MR\n(no radio)",
    "물리 GPU 3개 · 실제 resident NRx replica 3개 · GPU3 unused":
        "3 GPUs · 3 resident NRx replicas · GPU3 unused",
    "증명: full-size request/result와 request-level scale-out":
        "Proves: full-size transfers and request-level scale-out",
    "3단계 · Actual-radio 3-endpoint correctness gate":
        "Stage 3 · Actual-radio 3-endpoint correctness gate",
    "물리 GPU 4개 관여 · NRx는 GPU1/2/3 full GPU":
        "4 physical GPUs involved · NRx uses full GPUs 1/2/3",
    "증명: CE→NRx→LDPC/CRC·utility·epoch/expiry correctness":
        "Proves: CE→NRx→LDPC/CRC, utility, and epoch/expiry correctness",
    "주의: MIG 자원 효율 비교나 concurrent replica-capacity 실험이 아님":
        "Caution: not a MIG-efficiency or concurrent replica-capacity test",
    "4단계 · 최종 목표: 고정 MIG 위의 resident NRx service pool":
        "Stage 4 · Target: resident NRx service pool over fixed MIG",
    "DART-Rx: utility · deadline · queue 상태로 endpoint 선택":
        "DART-Rx: select endpoint by utility, deadline, and queue state",
    "MIG를 합치는 것이 아니라, slot 요청을 상주 replica 사이에 분산":
        "Distribute slot requests among resident replicas; do not merge MIGs",
    "남은 gate: actual radio + concurrent burst + MIG NRx pool + background":
        "Remaining gate: actual radio + concurrent burst + MIG NRx pool + background",
    "임의의 빈 GPU를 즉시 NRx로 바꾸는 구조가 아님 · model/context는 미리 resident":
        "Does not instantly convert any idle GPU into NRx; model/context is pre-resident",
    "GDR 실험을 단계별로 구분해야 하는 이유: 같은 이름, 다른 물리 topology와 claim":
        "Why the GDR experiments must be separated: one name, different physical topologies and claims",
    "1–3단계는 실제 실행한 topology, 4단계는 아직 하나의 동시 실험으로 닫지 못한 목표 구조":
        "Stages 1–3 are measured topologies; Stage 4 is the target not yet closed in one concurrent experiment",
    "CONTROL PLANE · 작은 descriptor와 completion만 CPU가 관리":
        "CONTROL PLANE · CPU manages only small descriptors and completions",
    "DATA PLANE · 큰 tensor payload는 CPU DRAM을 거치지 않고 GPU → P2P/NIC → GPU":
        "DATA PLANE · Large tensor payloads bypass CPU DRAM: GPU → P2P/NIC → GPU",
    "보호된 L1 · 4g MIG": "Protected L1 · 4g MIG",
    "기존 수신기\n항상 실행 가능한 fallback": "Conventional receiver\nalways-ready fallback",
    "요청 descriptor\ncell · slot · epoch · expiry · utility":
        "Request descriptor\ncell · slot · epoch · expiry · utility",
    "L1-side DART-Rx scheduler · host control plane":
        "L1-side DART-Rx scheduler · host control plane",
    "1 · Admission\nradio utility + deadline": "1 · Admission\nradio utility + deadline",
    "4 · 단일 commit\nslot · epoch · expiry · CRC":
        "4 · Single commit\nslot · epoch · expiry · CRC",
    "2 · Endpoint shadow state": "2 · Endpoint shadow state",
    "GPU queue를 원격으로 스캔하지 않고 completion으로 갱신 · 아래 값은 동작 예시":
        "No remote GPU-queue scan; completions update this table · values illustrate operation",
    "NRx 0  |  pending=3  |  예약 tail=늦음  |  healthy":
        "NRx 0  |  pending=3  |  reserved tail=late  |  healthy",
    "NRx 1  |  pending=0  |  예약 tail=현재  |  healthy  ← 선택":
        "NRx 1  |  pending=0  |  reserved tail=now  |  healthy  ← selected",
    "NRx 2  |  pending=1  |  예약 tail=곧  |  healthy":
        "NRx 2  |  pending=1  |  reserved tail=soon  |  healthy",
    "예측 완료[e] = max(현재시각, 예약 tail[e]) + 보수적 service bound[e]":
        "predicted finish[e] = max(now, reserved tail[e]) + conservative service bound[e]",
    "3 · Queue credit 예약\nsubmit: pending++ · tail 예약     |     completion: pending-- · bound 갱신":
        "3 · Reserve queue credit\nsubmit: pending++ · reserve tail     |     completion: pending-- · update bound",
    "고정 MIG 위의 resident NRx service fabric":
        "Resident NRx service fabric over fixed MIGs",
    "모델 · TensorRT context · CUDA Graph · GPU MR은 미리 상주":
        "Models · TensorRT contexts · CUDA Graphs · GPU MRs stay resident",
    "빈 queue": "empty queue",
    "대기 ● ● ●": "waiting ● ● ●",
    "대기 ●": "waiting ●",
    "예약됨": "reserved",
    "Sibling 4g background lease · Qwen / BERT / Whisper / vision · work-unit 경계에서 양보":
        "Sibling-4g background lease · Qwen / BERT / Whisper / vision · yield at work-unit boundaries",
    "도착": "arrival",
    "credit 예약": "reserve credit",
    "P2P · topology가 지원할 때": "P2P · when topology permits",
    "NIC · GPUDirect RDMA": "NIC · GPUDirect RDMA",
    "선택된 endpoint의 GPU MR\nrequest / result payload":
        "Selected endpoint's GPU MR\nrequest / result payload",
    "DART-Rx 전체 구조: queue 상태를 추적해 가장 빨리 끝날 NRx를 선택":
        "DART-Rx architecture: track queue state and select the earliest-finishing NRx",
    "핵심: '줄이 짧은 곳'은 GPU 내부를 매번 읽어서 찾는 것이 아니라, L1 측 scheduler가 submit/completion으로 유지하는 shadow queue state로 판단":
        "Key: the scheduler does not reread GPU queues; it selects using shadow queue state maintained from submits and completions",
    "4g MIG 안의 MPS": "MPS inside a 4g MIG",
    "NRx client 증가 시 붕괴 구간": "Breakdown region as NRx clients increase",
    "동시에 실행한 독립 NRx process 수": "Concurrent independent NRx processes",
    "(a) Full GPU도 NRx client가 늘면 무너짐": "(a) Even a full GPU degrades as NRx clients increase",
    "중앙값": "Median",
    "L1 kernel 사이 대기시간(us, 로그)": "Wait between L1 kernels (us, log scale)",
    "(b) N=6부터 kernel 사이 빈 시간이 급증": "(b) Inter-kernel gaps jump at N=6",
    "L1 GPU duty cycle": "L1 GPU duty cycle",
    "L1 kernel 길이": "L1 kernel duration",
    "L1 kernel 중앙값(us)": "Median L1 kernel duration (us)",
    "(c) L1이 GPU를 쓰는 비율도 절반 이하로 감소": "(c) L1 GPU duty falls by more than half",
    "MPS의 숨은 scaling 문제: 독립 NRx client가 늘면 L1 scheduling이 급격히 붕괴":
        "MPS's hidden scaling problem: independent NRx clients trigger an L1 scheduling collapse",
    "실험 범위: 과거 20-cell real-cuPHY causal campaign, max-rate NRx process 1/2/3/4/6/8개, MPS on, L1 p99는 3회 중앙값. 현재 최적화 chain과 절대시간 직접 비교 금지":
        "Scope: prior 20-cell real-cuPHY causal campaign; 1/2/3/4/6/8 max-rate NRx processes; MPS on; median L1 p99 over 3 trials; absolute time is not directly comparable to the current optimized chain",
    "Qwen에 허용한 GPU 몫(%)": "GPU share allowed for Qwen (%)",
    "요청 전체 처리시간(ms)": "End-to-end request time (ms)",
    "(a) MPS: Qwen을 더 쓰면 무선 처리가 느려짐": "(a) MPS: More Qwen slows the radio path",
    "평균 처리시간": "Mean",
    "느린 1% 경계(p99)": "Slowest-1% boundary (p99)",
    "옆 파티션 간섭은 차단": "Sibling interference is isolated",
    "같은 파티션의 경합은 남음": "Contention remains inside one partition",
    "4g NRx 처리량\n+ 옆 3g에 Qwen": "4g NRx throughput\n+ Qwen on sibling 3g",
    "L1 실행시간\n+ 같은 4g에 NRx": "L1 execution time\n+ NRx in the same 4g",
    "단독 실행 대비 배율": "Ratio to running alone",
    "(b) MIG: 옆 파티션만 격리됨": "(b) MIG isolates sibling partitions only",
    "단독 실행": "Run alone",
    "다른 작업과 동시 실행": "Run with another workload",
    "(c) MIG+MPS: 같은 파티션의 몫만 재분배": "(c) MIG+MPS only reallocates shares inside one partition",
    "Qwen 처리량(iter/s)": "Qwen throughput (iter/s)",
    "출발점: 세 가지 내부 배치 모두 '강한 격리'와 '필요할 때 NRx 확장'을 함께 제공하지 못함":
        "Starting point: no local placement provides both strong isolation and on-demand NRx scaling",
    "서로 다른 하드웨어 실험 요약: MPS 몫 변화, MIG 인접 파티션 격리, 같은 MIG 안의 L1·NRx 몫 변화(각 3회 중앙값)":
        "Summary of separate hardware experiments: MPS share sweep, sibling-MIG isolation, and L1/NRx shares within one MIG (median of 3 trials)",
    "MIG 안에\nL1+NRx": "L1+NRx\ninside one MIG",
    "MIG+MPS 안에\nL1+NRx": "L1+NRx\ninside MIG+MPS",
    "L1/NRx 분리\n+ P2P": "Separate L1/NRx\n+ P2P",
    "L1 실행시간 증가 배율": "L1 execution-time ratio",
    "(a) NRx를 분리하자 L1 성능이 회복됨": "(a) Separating NRx restores L1 performance",
    "L1 단독 실행": "L1 alone",
    "GPU 사이 직접 전송\n(P2P)": "Direct GPU-to-GPU transfer\n(P2P)",
    "NIC를 거친 GPU 직접 전송\n(GDR)": "Direct GPU transfer through NIC\n(GDR)",
    "(b) NIC로 범위를 넓힌 비용은 평균 0.438 ms": "(b) NIC extension adds 0.438 ms on average",
    "NRx 처리기 3개 중 2개 이상이\n놀고 있는데도 대기 폭증": "The queue explodes even while\nat least 2 of 3 NRx workers are idle",
    "NRx 한 곳에\n계속 고정": "Always use\none NRx",
    "NRx 3곳 중\n빨리 끝날 곳 선택": "Choose the fastest-finishing\nof 3 NRx workers",
    "NRx 응답의 느린 1% 경계(ms, 로그)": "NRx response p99 (ms, log scale)",
    "(c) 고정 배치는 GPU가 놀아도 한 줄만 폭증": "(c) Static pinning overloads one queue",
    "P2P/GDR를 고려한 이유: L1을 NRx와 분리한 채, 떨어진 NRx 처리기까지 사용하기 위해":
        "Why P2P/GDR: separate L1 from NRx and reach remote NRx workers",
    "서로 다른 3개 실험의 연결: 배치별 L1 격리, 같은 대기열 깊이의 전송 비교, 매 1 ms 요청에서 NRx 3개 선택":
        "Evidence from three experiments: L1 isolation by placement, equal-depth transport, and selecting among 3 NRx workers at 1 request/ms",
    "처리량 변화: -0.11%": "Throughput change: -0.11%",
    "4g NRx만 실행": "4g NRx only",
    "4g NRx +\n옆 3g에 Qwen": "4g NRx +\nQwen on sibling 3g",
    "초당 지속 처리 가능한 NRx 요청 수": "Sustainable NRx requests/s",
    "(a) 옆 MIG의 Qwen은 NRx 처리량에 거의 영향 없음": "(a) Qwen on a sibling MIG barely affects NRx throughput",
    "초당 들어오는 NRx 요청 수": "Incoming NRx requests/s",
    "대기시간의 느린 1% 경계(ms, 로그)": "Queueing-time p99 (ms, log scale)",
    "(b) 처리 한계를 넘으면 격리돼도 대기시간은 폭증": "(b) Isolation cannot prevent queue explosion beyond capacity",
    "4g NRx만": "4g NRx only",
    "4g NRx + 옆 3g Qwen": "4g NRx + sibling-3g Qwen",
    "실험 기준선 5 ms": "5 ms experimental threshold",
    "MIG가 해결하는 것과 못 하는 것: 옆 작업의 간섭은 막지만, NRx 한 대의 처리 한계는 남음":
        "What MIG does and does not solve: it blocks sibling interference, but one NRx still has a fixed capacity",
    "실험 범위: A100의 4g MIG 한 개에서 최적화한 TensorRT NRx 실행, 초당 요청 수를 단계적으로 증가":
        "Scope: optimized TensorRT NRx on one A100 4g MIG; incoming request rate is swept upward",
    "기존 Python\n실행 경로": "Existing Python\nexecution path",
    "TensorRT 엔진\n직접 호출": "Direct TensorRT\nengine call",
    "직접 호출\n+ CUDA Graph": "Direct call\n+ CUDA Graph",
    "NRx GPU 실행시간(ms, 로그)": "NRx GPU execution time (ms, log scale)",
    "(a) 105 ms의 대부분은 AI 계산 자체가 아니었음": "(a) Most of the 105 ms was not neural computation",
    "기존 실행 경로와 결과 완전 일치: 최대 차이 0 (통과)": "Bit-exact with the existing path: max difference 0 (pass)",
    "CPU가 GPU에 작업을 넣는 시간(us, 로그)": "CPU submission time (us, log scale)",
    "(b) CUDA Graph로 반복 제출 비용까지 제거": "(b) CUDA Graph also removes repeated submission cost",
    "구현 경로를 바로잡자 NRx가 105.15 ms에서 1.34 ms로 단축(약 74배)":
        "Fixing the execution path cuts NRx from 105.15 ms to 1.34 ms (about 74x)",
    "실험 범위: A100 4g MIG 한 개, 기존 Python 경로 30회, 직접 호출과 CUDA Graph는 준비 실행 뒤 1,000회":
        "Scope: one A100 4g MIG; 30 existing-path runs and 1,000 direct/CUDA-Graph runs after warm-up",
    "셀 1개\n매 1 ms, 모두 NRx": "1 cell\n1 ms period\n100% NRx",
    "셀 4개\n매 1 ms, 몰림 10%": "4 cells\n1 ms period\n10% bursts",
    "셀 4개\n매 0.5 ms, 몰림 10%": "4 cells\n0.5 ms period\n10% bursts",
    "(a) 느린 요청의 대기시간": "(a) Tail response time",
    "한 NRx에 고정": "Pin to one NRx",
    "빨리 끝날 NRx 선택": "Choose earliest finish",
    "5 ms 안에 쓸 NRx 결과가 없는 비율": "Fraction with no usable NRx result within 5 ms",
    "(b) 제시간 결과를 얻지 못한 요청": "(b) Requests without a timely result",
    "놀고 있는 NRx 처리기 비율": "Idle NRx-worker fraction",
    "(c) 다른 NRx 처리기는 얼마나 놀았나": "(c) How much were other NRx workers idle?",
    "문제의 직접 증거: 한 NRx 대기열은 무너지는데 다른 NRx 처리기는 동시에 놀고 있음":
        "Direct evidence: one NRx queue collapses while other NRx workers remain idle",
    "실험 범위: 독립적으로 상주한 TensorRT NRx 3개, 3회 중앙값, 5 ms는 비교용 실험 기준선":
        "Scope: 3 independently resident TensorRT NRx workers; median of 3 trials; 5 ms is an experimental comparison threshold",
    "L1·NRx 분리\nGPU 직접(P2P)": "Separate L1/NRx\nGPU-direct (P2P)",
    "L1·NRx 분리\nNIC 직접(GDR)": "Separate L1/NRx\nNIC-direct (GDR)",
    "(a) 배치와 전송 방식별 처리시간": "(a) Processing time by placement and transport",
    "평균": "Mean",
    "요청 평균 처리시간(ms)": "Mean request time (ms)",
    "(b) 무선 처리시간과 남는 GPU로 돌린 Qwen 처리량": "(b) Radio-path time vs. Qwen throughput on spare GPU capacity",
    "배치 방식 비교: P2P/GDR로 분리는 가능하지만 전체 시간은 NRx 계산과 대기열이 좌우":
        "Placement comparison: P2P/GDR enable separation, but NRx compute and queueing dominate total time",
    "실험 범위: 최적화한 TensorRT 경로. GDR 대기열 깊이는 1, 주 P2P 결과는 2이므로 처리량 수치는 직접 비교하지 않음":
        "Scope: optimized TensorRT path. GDR queue depth is 1 and the main P2P result uses depth 2, so their throughput is not directly compared",
    "모든 방식에 똑같이 넣은 초당 요청 수": "Identical incoming request rate (/s)",
    "도착부터 완료까지 느린 1% 시간(ms, 로그)": "Arrival-to-completion p99 (ms, log scale)",
    "(a) 실제 package별 raw queue 한계 (GPU 자원량 다름)": "(a) Raw queue limit by package (unequal GPU allocations)",
    "MPS · full A100": "MPS · full A100",
    "MIG local · 4g": "MIG local · 4g",
    "MIG+MPS · 4g": "MIG+MPS · 4g",
    "Cross P2P · 2g+2g": "Cross P2P · 2g+2g",
    "Cross GDR · 2g+2g": "Cross GDR · 2g+2g",
    "100 ms 이상: 대기열 붕괴 표시": ">=100 ms: queue-collapse marker",
    "100 ms 이상으로 폭증하기 전 마지막 측정 요청률": "Last measured rate before p99 exceeds 100 ms",
    "(b) Background 없는 raw 안정 범위 · L1 보호 지표 아님": "(b) Background-free raw range · not an L1-protection metric",
    "Raw capacity gate: full-A100 MPS와 MIG slice의 종합 우승 비교가 아님":
        "Raw-capacity gate: not an overall contest between full-A100 MPS and MIG slices",
    "Background 없음 · optimized NRx 경로 1개 · 할당 GPU 자원 불균등. 10초 x 3회 x 5개 배치 = 120회; 100 ms는 진단선":
        "No background · one optimized NRx path · unequal GPU allocations. 10 s x 3 trials x 5 placements = 120 runs; 100 ms is diagnostic",
    "(a) 같은 4g MIG 안의 L1/NRx 몫을 변경": "(a) Vary L1/NRx shares inside one 4g MIG",
    "옆 3g의 Qwen은 계속 약 10.21~10.22 iter/s": "Qwen on the sibling 3g remains at about 10.21-10.22 iter/s",
    "NRx에 허용한 실행 몫(%)": "Execution share allowed for NRx (%)",
    "(b) L1 몫을 늘려도 전체 처리는 더 느려질 수 있음": "(b) More L1 share can still make the whole chain slower",
    "MIG+MPS의 한계: 실행 몫은 한 파티션 안에서 나뉠 뿐, 새 격리 공간은 생기지 않음":
        "Limit of MIG+MPS: shares move within one partition; no new isolation boundary appears",
    "실험 범위: 고정 4g MIG 안에서 L1과 NRx를 별도 MPS 작업으로 실행, 각 몫마다 슬롯 1,000개 x 3회, 두 프로세스는 GDR로 연결":
        "Scope: separate L1 and NRx MPS clients inside one fixed 4g MIG; 1,000 slots x 3 trials per share; processes connected with GDR",
    "L1만 실행": "L1 only",
    "L1+NRx 동시": "L1 + NRx",
    "비동기 free로 변경": "Async free",
    "CUDA 메모리 풀": "CUDA memory pool",
    "CPU가 CUDA API 안에서 기다린 누적시간(ms/30초)": "Cumulative CPU time inside CUDA APIs (ms/30 s)",
    "(a) 셀 40개: API를 바꿔도 CPU 대기 위치만 이동": "(a) 40 cells: changing the API only moves the CPU wait",
    "메모리 해제(cudaFree)": "Free memory (cudaFree)",
    "비동기 해제(cudaFreeAsync)": "Async free (cudaFreeAsync)",
    "메모리 풀 할당": "Memory-pool allocation",
    "메모리 할당(cudaMalloc)": "Allocate memory (cudaMalloc)",
    "비동기 메모리 복사": "Async memory copy",
    "Stream 완료 기다리기": "Wait for stream completion",
    "셀 40개: CPU 대기 15.1배 증가": "40 cells: CPU wait increases 15.1x",
    "동시에 처리하도록 설정한 셀 수": "Configured concurrent cell count",
    "(b) 비동기 메모리 API도 쌓인 GPU 작업은 없애지 못함": "(b) Async memory APIs cannot remove queued GPU work",
    "Nsight 원인 분석: 같은 MIG의 NRx가 L1의 CPU 실행 흐름을 CUDA 호출 안에서 오래 막음":
        "Nsight root cause: NRx in the same MIG blocks the L1 CPU thread inside CUDA calls",
    "실험 범위: 같은 MIG에 cuPHY L1+NRx 배치. 30초 Nsight 구간의 주요 CUDA API 6개 누적값이며 슬롯 하나의 지연시간이 아님":
        "Scope: cuPHY L1+NRx in one MIG; cumulative time for six CUDA APIs over a 30 s Nsight window, not per-slot latency",
    "Qwen-7B\n생성": "Qwen-7B\ndecode",
    "요청 몰림 중 NRx의 느린 1% 시간(ms, 로그)": "NRx p99 during burst (ms, log scale)",
    "(a) 다른 AI 작업을 그대로 두면 NRx 대기 폭증": "(a) Unchanged background causes NRx collapse",
    "계속 같이 실행": "Keep running together",
    "요청이 몰리면 NRx에 양보": "Yield to NRx during bursts",
    "5 ms를 넘긴 NRx 요청 비율": "Fraction of NRx requests over 5 ms",
    "(b) 실험 기준시간을 넘긴 요청": "(b) Requests over 5 ms",
    "계속 처리한 다른 AI 작업 비율(%)": "Background work retained (%)",
    "(c) 다른 AI 작업 대부분을 유지하면서 NRx에 양보": "(c) Yield while retaining background work",
    "NRx에 양보하기까지 걸린 시간(ms)": "Time to yield capacity to NRx (ms)",
    "다른 AI 작업을 짧게 나누면 처리량 대부분을 유지하며 NRx 요청 몰림을 흡수할 수 있음":
        "Short background work units absorb NRx bursts while retaining most background throughput",
    "실험 범위: 최적화한 TensorRT NRx 대기열과 상주 AI 모델만 사용. 이 실험에는 cuPHY와 전송 경로가 없음":
        "Scope: optimized TensorRT NRx queue and resident AI models only; cuPHY and transport are not included",
    "낮은 부하\n<=1,000/s": "Low load\n<=1,000/s",
    "중간 부하\n1,000~1,500/s": "Medium load\n1,000-1,500/s",
    "높은 부하\n>1,500/s": "High load\n>1,500/s",
    "(a) 실제 크기 GPU 데이터를 처리하는 GDR NRx 3개": "(a) Three GDR NRx workers with full-size GPU payloads",
    "한 NRx에 계속 고정": "Always pin to one NRx",
    "셀마다 지정한 NRx에 고정": "Pin each cell to one NRx",
    "예상 완료가 가장 빠른 NRx": "Choose earliest predicted finish",
    "완료 예측 + 느린 경우 억제": "Predicted finish + tail penalty",
    "제시간 결과가 없는 비율 감소(%p, 클수록 좋음)": "Reduction in no-timely-result rate (percentage points; higher is better)",
    "예상 완료가 가장 빠른 NRx\n대비 한 NRx에 계속 고정": "Earliest predicted finish\nvs. always pin to one NRx",
    "예상 완료가 가장 빠른 NRx\n대비 셀마다 지정한 NRx에 고정": "Earliest predicted finish\nvs. per-cell pinning",
    "완료 예측 + 느린 경우 억제\n대비 한 NRx에 계속 고정": "Predicted finish + tail penalty\nvs. always pin to one NRx",
    "완료 예측 + 느린 경우 억제\n대비 셀마다 지정한 NRx에 고정": "Predicted finish + tail penalty\nvs. per-cell pinning",
    "(b) 똑같은 요청 흐름으로 정책끼리 직접 비교": "(b) Pairwise policies on identical request traces",
    "NRx 선택 정책 결과: 완료시간을 예측하면 실패할 전송은 줄지만, 요청 수락 판단은 아직 보수적":
        "NRx policy result: finish prediction reduces futile transfers, but admission remains conservative",
    "Stage 2 정책 결론: 정상 부하는 round-robin, overload에서만 deadline admission이 필요":
        "Stage 2 policy: round-robin within capacity; deadline admission only under overload",
    "실험 범위: 요청 패턴 29개 x 3회 x 정책 4개 = 348회, 5 ms 비교선. '결과 없음'에는 처음부터 기존 수신기를 고른 경우도 포함":
        "Scope: 29 request patterns x 3 trials x 4 policies = 348 runs; 5 ms threshold; 'no result' includes choosing the conventional receiver up front",
    "실험 범위: 요청 패턴 29개 x 3회 x 정책 4개 = 348회, 5 ms 비교선. 대부분 overload stress이며 predicted 정책의 사전 fallback도 '결과 없음'에 포함":
        "Scope: 29 patterns x 3 trials x 4 policies = 348 runs; most are overload stress; predicted-policy fallback counts as no result",
    "같은 4g\nL1+NRx": "Same 4g\nL1+NRx",
    "분리된 2g+2g\nGPU P2P": "Separate 2g+2g\nGPU P2P",
    "분리된 2g+2g\nNIC GDR": "Separate 2g+2g\nNIC GDR",
    "(a) 직렬 E2E: 빠른 same-4g와 격리된 cross 배치":
        "(a) Serial E2E: fast same-4g versus isolated cross placement",
    "L1 단독 실행": "L1 alone",
    "NRx 동시 실행 시 L1 active-time 증가 배율": "L1 active-time multiplier with concurrent NRx",
    "(b) 측정된 isolation: same-4g 대 cross P2P":
        "(b) Measured isolation: same-4g versus cross P2P",
    "NIC GDR\nE2E 측정 완료\nL1 isolation 미측정\nQwen 10.24 it/s":
        "NIC GDR\nE2E measured\nL1 isolation not measured\nQwen 10.24 it/s",
    "Stage 1: same-4g는 빠르지만 L1 경합, cross 배치는 L1 보호 대신 작은 slice 비용":
        "Stage 1: same-4g is fast but contended; cross placement protects L1 at a slice-capacity cost",
    "모든 구성에서 Qwen은 별도 3g에 상주. (a)는 depth=1, (b)는 별도 ring-depth=2 isolation gate; GDR의 L1-active 값은 미측정. P2P↔GDR만 동일 2g+2g transport 비교":
        "Qwen runs on a separate 3g in every configuration. (a) uses depth 1; (b) is a separate ring-depth-2 isolation gate. GDR L1-active time was not measured. Only P2P vs. GDR is an equal 2g+2g transport comparison.",
    "순서대로 분배": "Round-robin",
    "예상 완료가 가장 빠른 곳": "Earliest predicted finish",
    "Round-robin · 모든 요청 수락": "Round-robin · admit every request",
    "Deadline gate · 늦을 요청은 fallback": "Deadline gate · fallback if predicted late",
    "완료 예측 + tail guard": "Predicted finish + tail guard",
    "(a) 셀 1개 · 1 ms마다 NRx": "(a) One cell · NRx every 1 ms",
    "(b) 셀 2개 · 같은 시각에 NRx": "(b) Two cells · synchronized NRx",
    "(c) 셀 4개 · 10% burst 선택": "(c) Four cells · selective 10% bursts",
    "(a) 셀 1개 · 1 ms마다 NRx\n1,000 requests/s":
        "(a) One cell · NRx every 1 ms\n1,000 requests/s",
    "(b) 셀 2개 · 같은 시각에 NRx\n2,000 requests/s":
        "(b) Two cells · synchronized NRx\n2,000 requests/s",
    "(c) 셀 4개 · 10% burst 선택\n평균 385 requests/s":
        "(c) Four cells · selective 10% bursts\nMean 385 requests/s",
    "평균 385 requests/s": "Mean 385 requests/s",
    "동시에 상주한 NRx replica 수": "Concurrent resident NRx replicas",
    "5 ms 안에 도착한 NRx 결과 비율(높을수록 좋음)": "NRx results arriving within 5 ms (higher is better)",
    "95% timely": "95% timely",
    "Stage 2 핵심: NRx 3개는 1,000/s periodic을 처리하지만 2,000/s와 burst는 아직 못 버팀":
        "Stage 2: three NRx workers sustain periodic 1,000/s, but not 2,000/s or bursts",
    "실험 범위: 실제 1/2/3개의 resident 3g-MIG GDR endpoint, 각 점은 같은 representative trace 1회. 예상-완료 정책의 사전 fallback도 제시간 NRx 결과 없음으로 계산":
        "Scope: 1/2/3 real resident 3g-MIG GDR endpoints; one replay per representative trace; predicted-finish fallback counts as no timely NRx result",
    "Stage 2 replica sweep: NRx를 늘리면 capacity는 늘지만, 부하와 정책에 따라 효과가 달라짐":
        "Stage 2 replica sweep: more NRx capacity helps, but the gain depends on workload and policy",
    "실험 범위: 실제 1/2/3개의 resident 3g-MIG GDR endpoint, 각 점은 같은 representative trace 1회. 낮을수록 좋으며 full-matrix 통계는 별도 정책 그림에 제시":
        "Scope: 1/2/3 real resident 3g-MIG GDR endpoints; each point is one replay of the same representative trace; lower is better; full-matrix statistics appear in the separate policy figure",
    "기존 수신기만": "Conventional receiver only",
    "모든 슬롯에 NRx": "NRx for every slot",
    "어려운 슬롯만 NRx": "NRx only for difficult slots",
    "정상적으로 복호한 전송 블록 비율": "Correctly decoded transport-block ratio",
    "(a) 실제 무선 복호 성공률": "(a) Measured radio decoding success",
    "슬롯 100개 중 NRx를 호출한 수": "NRx calls per 100 slots",
    "(b) 실제로 실행한 AI 작업량": "(b) Neural work actually executed",
    "최종 결과를 선택하기까지 걸린 시간(ms)": "Time until final-result selection (ms)",
    "(c) 실제 슬롯 처리시간": "(c) Measured slot decision time",
    "12 ms 뒤에는 결과 폐기": "Discard results after 12 ms",
    "중간값": "Median",
    "실제 무선 결과: 어려운 슬롯에만 NRx를 써도 성공률은 같고 AI 호출은 25% 감소":
        "Measured radio result: selective NRx preserves success while cutting neural calls by 25%",
    "실험 범위: 실제 cuPHY CE -> GDR NRx -> LDPC/CRC, NRx 3개, 요청 100개 x 3회, 결과 유효시간 12 ms":
        "Scope: real cuPHY CE -> GDR NRx -> LDPC/CRC; 3 NRx workers; 100 requests x 3 trials; 12 ms result validity",
    "GPU 작업 흐름\n완료 기다리기": "Wait for GPU stream\ncompletion",
    "비동기\n메모리 복사": "Async\nmemory copy",
    "GPU 메모리\n해제": "Free GPU\nmemory",
    "GPU 메모리\n할당": "Allocate GPU\nmemory",
    "GPU 전체 완료\n기다리기": "Wait for all GPU\nwork",
    "GDR 쓰기\n확인": "Confirm GDR\nwrite visibility",
    "CPU가 CUDA API 안에서 보낸 누적시간(ms, 로그)": "Cumulative CPU time inside CUDA APIs (ms, log scale)",
    "(a) 실제 무선 처리 중 CPU가 기다린 위치": "(a) Where the CPU waits in the real radio path",
    "FP32/FP16\n데이터 형식 변환": "FP32/FP16\ndata conversion",
    "주요 LDPC\n복호 계산": "Main LDPC\ndecoding kernel",
    "기타 cuPHY /\nTensorFlow / 복사": "Other cuPHY /\nTensorFlow / copies",
    "(b) GPU에서는 전송보다 데이터 형식 변환 비중이 큼": "(b) GPU data conversion outweighs transport overhead",
    "실제 cuPHY-GDR-NRx 경로: GDR 확인보다 동기화와 데이터 형식 변환 비용이 더 큼":
        "Real cuPHY-GDR-NRx path: synchronization and data conversion cost more than GDR visibility checks",
    "실험 범위: 실제 CE -> GDR NRx 3개 -> LDPC/CRC 경로에서 요청 12개를 Nsight로 추적. 모든 배치 조건을 대표하지는 않음":
        "Scope: Nsight trace of 12 requests through real CE -> 3 GDR NRx -> LDPC/CRC; not representative of every placement",
    "방식": "Approach",
    "실제 배치": "Physical placement",
    "L1–NRx\n하드웨어 벽": "L1–NRx\nhardware wall",
    "직접 실측 증거\n(실험 조건은 서로 다름)": "Direct evidence\n(conditions differ)",
    "NRx 도달 범위": "Reachable NRx domain",
    "이 연구에서의 역할": "Role in this study",
    "full A100\nL1 · NRx · BG 공유": "full A100\nL1 · NRx · BG share",
    "없음": "None",
    "같은 GPU\nMPS share": "Same GPU\nMPS share",
    "NRx process 1→8\nL1 p99 4.5×": "NRx processes 1→8\nL1 p99 4.5×",
    "한 GPU의\n공유 domain": "Shared domain\non one GPU",
    "높은 활용률\n보호 없는 baseline": "High utilization\nunprotected baseline",
    "4g: L1+NRx\n3g: BG": "4g: L1+NRx\n3g: BG",
    "L1↔NRx 없음\nsibling에는 있음": "None for L1↔NRx\npresent for sibling",
    "sibling 3g\n격리": "Isolated\nsibling 3g",
    "NRx overlap\nL1 active 1.621×": "NRx overlap\nL1 active 1.621×",
    "같은 4g\n안에서만": "Only inside\nthe same 4g",
    "강한 sibling 격리\nlocality baseline": "Strong sibling isolation\nlocality baseline",
    "4g 안 MPS\n3g: BG": "MPS inside 4g\n3g: BG",
    "L1↔NRx 없음\nshare만 조절": "None for L1↔NRx\nshare control only",
    "NRx overlap\nL1 active 1.702×": "NRx overlap\nL1 active 1.702×",
    "GI 내부 share\n조절 baseline": "Intra-GI share-control\nbaseline",
    "2g L1 | 2g NRx\n3g: BG": "2g L1 | 2g NRx\n3g: BG",
    "있음": "Present",
    "NRx overlap\nL1 active 1.043×": "NRx overlap\nL1 active 1.043×",
    "같은 physical GPU의\npeer-capable MIG": "Peer-capable MIGs\non one physical GPU",
    "cross-MIG\n저비용 fast path": "Low-cost cross-MIG\nfast path",
    "L1/NRx 분리\nGPU MR + NIC": "Separate L1/NRx\nGPU MR + NIC",
    "sibling 또는\nremote domain": "Sibling or\nremote domain",
    "P2P 대비 E2E\n+0.438 ms": "E2E vs. P2P\n+0.438 ms",
    "다른 GPU · process\nRDMA endpoint": "Other GPU · process\nRDMA endpoint",
    "DART-Rx의\ncross-GPU fabric": "DART-Rx\ncross-GPU fabric",
    "다섯 방식 한눈에 보기: capacity 순위가 아니라 L1 보호 · 도달 범위 · 비용의 차이":
        "Five Ways at a Glance: L1 Protection, Reach, and Cost—not One Capacity Ranking",
    "핵심: MPS는 자원을 잘 쓰지만 L1 보호벽이 없고, MIG/MIG+MPS는 같은 GI 경합과 고정 capacity가 남는다. P2P는 같은 GPU의 빠른 격리 경로, GDR는 다른 GPU까지 NRx pool을 확장하는 경로다.":
        "Key: MPS uses capacity well but has no L1 wall; MIG/MIG+MPS retain same-GI contention and fixed capacity. P2P is the fast isolated path within one GPU; GDR extends the NRx pool across GPUs.",
    "아직 없는 최종 공정 비교: 동일 물리 A100 budget · 동일 Qwen 처리량 · 동일 NRx burst에서 다섯 방식의 L1 p99 / timely-result / background utility":
        "Missing final matched gate: same physical-A100 budget · same Qwen throughput · same NRx burst; compare L1 p99 / timely results / background utility across all five",
    "NRx 동시 실행 시 L1 active-time 배율": "L1 active-time ratio with concurrent NRx",
    "(a) L1 보호: 낮을수록 좋음": "(a) L1 protection: lower is better",
    "동등한 L1-active\ngate 미측정": "Matched L1-active\ngate not measured",
    "* GDR 1.103× = 추정값": "* GDR 1.103× = estimate",
    "(b) 낮은 부하의 slot tail: 모두 6–7 ms대": "(b) Low-load slot tail: all around 6–7 ms",
    "나머지 depth=2": "Others: depth=2",
    "격리 배치 약 10.2 it/s": "Isolated placements: about 10.2 it/s",
    "(c) Background utility: MPS는 가장 가까운 50% cap 점":
        "(c) Background utility: nearest MPS point is the 50% cap",
    "동시에 실행한 독립 NRx process 수": "Concurrent independent NRx processes",
    "20-cell L1 p99(ms)": "20-cell L1 p99 (ms)",
    "(d) 별도 stress gate: MPS의 multi-NRx scaling 붕괴":
        "(d) Separate stress gate: MPS multi-NRx scaling collapse",
    "4g MIG 안의 MPS": "MPS inside a 4g MIG",
    "다섯 방식의 직접 실측: 낮은 부하 E2E는 비슷해도 L1 보호와 scaling은 다르다":
        "Direct Five-Way Evidence: Similar Low-Load E2E, Different L1 Protection and Scaling",
    "다섯 방식의 실측 비교: 낮은 부하 E2E는 비슷해도 L1 보호와 scaling은 다르다":
        "Five-Way Evidence: Similar Low-Load E2E, Different L1 Protection and Scaling",
    "(a–c) placement campaign, 3회 집계; Full MPS=Qwen 50% cap(11.14 it/s), GDR는 2회·depth=1이고 L1-active 미수집. (d)는 별도 20-cell causal campaign의 3회 중앙값이므로 절대 ms를 (a–c)와 직접 비교하지 않음.":
        "(a–c) placement campaign, three-run aggregate; Full MPS uses the Qwen 50% cap (11.14 it/s); GDR has two runs at depth=1 and no L1-active measurement. (d) is a separate 20-cell causal campaign using three-run medians; do not compare its absolute milliseconds with (a–c).",
    "(a–c) placement campaign, 3회 집계; Full MPS=Qwen 50% cap(11.14 it/s). GDR E2E는 2회·depth=1 실측, (a)의 1.103×만 동등 trace가 없는 추정값(*). (d)는 별도 20-cell causal campaign의 3회 중앙값이므로 절대 ms를 (a–c)와 직접 비교하지 않음.":
        "(a–c) placement campaign, three-run aggregate; Full MPS uses the Qwen 50% cap (11.14 it/s). GDR E2E is measured from two depth=1 runs; only the 1.103× value in (a) is an estimate (*) without a matched trace. (d) is a separate 20-cell causal campaign using three-run medians; do not compare its absolute milliseconds with (a–c).",
    "(a–c) placement campaign, 3회 집계; Full MPS=Qwen 50% cap(11.14 it/s). GDR E2E는 2회·depth=1 실측이고, 동등한 L1-active 값은 수집하지 않음. (d)는 별도 20-cell causal campaign의 3회 중앙값이므로 절대 ms를 (a–c)와 직접 비교하지 않음.":
        "(a–c) placement campaign, three-run aggregate; Full MPS uses the Qwen 50% cap (11.14 it/s). GDR E2E is measured from two depth=1 runs; no matched L1-active value was collected. (d) is a separate 20-cell causal campaign using three-run medians; do not compare its absolute milliseconds with (a–c).",
    "(a–c) placement campaign; Full MPS=Qwen 50% cap(11.14 it/s), GDR E2E는 2회·depth=1. (d)는 별도 20-cell causal campaign의 3회 중앙값이므로 절대 ms를 (a–c)와 직접 비교하지 않음.":
        "(a–c) placement campaign; Full MPS uses the Qwen 50% cap (11.14 it/s), and GDR E2E uses two depth=1 runs. (d) is a separate 20-cell causal campaign using three-run medians; do not compare its absolute milliseconds with (a–c).",
}


def translate(text: str) -> str:
    if text in EN:
        return EN[text]
    match = re.fullmatch(r"더 좋음 (\d+/\d+)", text)
    if match:
        return f"better in {match.group(1)}"
    match = re.fullmatch(r"(\d+)회", text)
    if match:
        return f"{match.group(1)} runs"
    match = re.fullmatch(r"([0-9.]+ ms)\n(\d+)회", text)
    if match:
        return f"{match.group(1)}\n{match.group(2)} calls"
    return text


def save_english(fig, name: str) -> None:
    for axis in fig.axes:
        # set_xticks(..., labels=...) uses an internal FuncFormatter that can
        # recreate the original Korean labels on the next draw.  Freeze the
        # currently laid-out labels after translation so savefig cannot restore
        # source-language strings.
        x_positions = axis.get_xticks()
        x_labels = [translate(item.get_text()) for item in axis.get_xticklabels()]
        axis.xaxis.set_major_locator(FixedLocator(x_positions))
        axis.xaxis.set_major_formatter(FixedFormatter(x_labels))
        y_positions = axis.get_yticks()
        y_labels = [translate(item.get_text()) for item in axis.get_yticklabels()]
        axis.yaxis.set_major_locator(FixedLocator(y_positions))
        axis.yaxis.set_major_formatter(FixedFormatter(y_labels))
    for item in fig.findobj(match=Text):
        item.set_text(translate(item.get_text()))
        item.set_fontfamily("DejaVu Sans")
    untranslated = sorted(
        {
            item.get_text()
            for item in fig.findobj(match=Text)
            if re.search(r"[가-힣]", item.get_text())
        }
    )
    if untranslated:
        raise RuntimeError(f"Untranslated figure strings in {name}: {untranslated}")
    untranslated_formatters = []
    for axis in fig.axes:
        for ticker_axis in (axis.xaxis, axis.yaxis):
            for formatter in (
                ticker_axis.get_major_formatter(),
                ticker_axis.get_minor_formatter(),
            ):
                if isinstance(formatter, FixedFormatter):
                    untranslated_formatters.extend(
                        value for value in formatter.seq if re.search(r"[가-힣]", str(value))
                    )
    if untranslated_formatters:
        raise RuntimeError(
            f"Untranslated tick labels in {name}: {sorted(set(untranslated_formatters))}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    # Figure functions call tight_layout() before save().  Use the Korean-capable
    # source font while those source labels still exist; save_english() switches
    # every translated text object to DejaVu Sans before rendering the PNG.
    font_manager.fontManager.addfont(str(base.KOREAN_FONT))
    source_family = font_manager.FontProperties(fname=base.KOREAN_FONT).get_name()
    plt.rcParams.update(
        {
            "font.family": source_family,
            "font.size": 10,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "axes.unicode_minus": False,
        }
    )
    base.save = save_english
    # The author-supplied architecture map is intentionally shared by both
    # reports.  Point the base module at this wrapper's output directory before
    # invoking its copy-preserving figure function.
    base.OUT = OUT
    functions = [
        base.figure_00_architecture_map,
        base.figure_00a_gdr_evolution,
        base.figure_00d_dart_rx_overall_architecture,
        base.figure_00_three_local_baselines,
        base.figure_00c_mps_multi_nrx_breakdown,
        base.figure_00b_why_cross_endpoints,
        base.figure_01_isolation_and_queue_cliff,
        base.figure_01b_nrx_wrapper_optimization,
        base.figure_02_fragmentation,
        base.figure_03_placement_and_transport,
        base.figure_03e_stage1_equal_depth,
        base.figure_03f_fiveway_evidence_scorecard,
        base.figure_03g_fiveway_measured_evidence,
        base.figure_03b_fiveway_absolute_rate,
        base.figure_03c_mig_mps_quota,
        base.figure_03d_cuda_host_blocking,
        base.figure_04_background_reclaim,
        base.figure_05_gdr_pool_policy,
        base.figure_05b_gdr_replica_sweep,
        base.figure_06_radio_utility,
        base.figure_06b_radio_cuda_calls,
    ]
    for function in functions:
        function()
    print(f"wrote {len(functions)} English figures to {OUT}")


if __name__ == "__main__":
    main()
