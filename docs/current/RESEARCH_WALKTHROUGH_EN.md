# DART-Rx Research Walkthrough: Problem → Design → Experimental Evidence

**As of:** 2026-08-16  
**Purpose:** This is the primary English document for readers who need to understand the problem,
design, experimental conditions, results, and remaining limitations in one place.  
**Data rule:** Performance PNGs are generated from preserved CSV, JSON, or SQLite experiment
results. The placement architecture map and Mermaid blocks are topology/design schematics, not
performance measurements.  
**Korean counterpart:** [RESEARCH_WALKTHROUGH_KO.md](RESEARCH_WALKTHROUGH_KO.md)

---

## 0. The research in plain language

> **The mandatory L1 radio pipeline runs in protected GPU space. The Neural Receiver (NRx) is
> invoked only for slots that are likely to benefit, and each request is sent to the resident NRx
> worker predicted to finish earliest. A late result—or a result belonging to an old slot—is
> discarded, and the conventional receiver result is used instead. None of these actions changes
> the MIG layout or restarts L1.**

Technically, DART-Rx is a **deadline-aware NRx service pool built over a fixed MIG topology**. Its
contribution is not merely copying a tensor through a NIC. It combines five decisions into one slot
transaction: whether NRx is useful, whether it can still finish on time, which worker should execute
it, which result may commit, and when background AI must yield capacity.

This is not a paper whose claim is simply “MIG is faster than MPS” or “GDR is faster than P2P.”
MIG, MPS, P2P, and GDR are mechanisms and baselines. The actual problem is the following tension:

```text
L1 needs strong spatial isolation.
        ↓
Static partitioning and placement also fix NRx service capacity in place.
        ↓
Under multi-cell selective bursts, one queue can collapse while another endpoint is idle.
        ↓
MIG cannot be reconfigured on the slot fast path, so capacity must be borrowed
across already isolated, resident endpoints.
        ↓
A remote NRx result may arrive late, so admission, routing, fallback, and commit
must be designed together.
```

The current results validate each link in this causal chain on real hardware. The last fully
integrated experiment—`actual radio + concurrent multi-cell arrivals + multi-endpoint GDR +
background reclaim`—has not yet been completed. The current status is therefore: **the problem and
individual mechanisms are strongly supported, but the final ISCA-level end-to-end claim is not yet
complete.**

### 0.1 Terminology used throughout the report

| Term | Plain meaning | Precise meaning in this work |
|---|---|---|
| slot / request | One unit of radio data to process | One cell's scheduled PUSCH processing transaction |
| L1 | Mandatory radio work that must finish on time | The PHY critical path, including cuPHY channel estimation and LDPC/CRC |
| NRx | An AI receiver used for difficult signals | A TensorRT neural receiver selectively used instead of the conventional receiver |
| service capacity | How much work can be sustained per second | Maximum NRx request rate at which the queue does not grow without bound |
| queue cliff | A small load increase that causes a very large delay | Nonlinear p99 growth as arrival rate approaches or exceeds service rate |
| host blocking | A CPU thread waits inside a CUDA call | Completion of previously submitted GPU work becomes visible at `cudaFree`, synchronization, or copy calls |
| expiry | Last instant at which a result is usable | An NRx result after this time is not allowed to affect the current slot, even if numerically correct |
| commit | Select exactly one final result | Apply either the conventional or a validated NRx result to PHY state |
| endpoint | A permanently ready NRx worker | A process/GPU instance with model, TensorRT context, CUDA Graph, and buffers already resident |

### 0.2 Current conclusions in numbers

- **MIG sibling isolation is strong:** adding Qwen to the sibling 3g changed 4g NRx capacity by only
  `-0.11%`.
- **One fixed endpoint still collapses at its service limit:** MIG-local p99 was `3.47 ms` at 300/s
  but `1124.98 ms` at 350/s.
- **An NRx in the same GPU space can block the L1 host thread:** in the 40-cell same-MIG case,
  tracked CUDA API host time grew from `1.68 s` to `25.35 s`, or `15.1×`.
- **MPS collapses as the NRx client count grows:** increasing independent NRx processes from 1 to
  8 raised 20-cell L1 p99 from `42.3` to `189.3 ms` on full-A100 MPS and from `40.7` to
  `435.7 ms` inside 4g MIG+MPS.
- **MPS quota tuning alone is insufficient:** changing L1:NRx from 30:70 to 70:30 inside one 4g
  increased mean end-to-end time from `4.76 ms` to `6.50 ms`.
- **Transport is much smaller than queue collapse:** the equal-depth mean gap between P2P and NIC
  GDR was `0.438 ms`, whereas overloaded tails reached hundreds or thousands of milliseconds.
- **NRx has measured radio value:** utility-based admission preserved the all-NRx correct-TB ratio
  of `0.80` while reducing NRx invocations by 25%.

---

# Part I. Background and problem

## 1. Target AI-RAN execution environment

An uplink request is not an independent AI query. It is a dependency-carrying PHY chain:

```text
(cell, slot, scheduled PUSCH)
  → cuPHY channel estimation / front-end
  → conventional receiver and optional Neural Receiver (NRx)
  → LDPC / CRC
  → commit exactly one result before an absolute decision expiry
```

Three classes of work coexist in the GPU system:

1. **Protected L1:** must satisfy a tail-latency objective and must not be destabilized by
   background AI.
2. **NRx:** an optional PHY stage that improves decoding only in some channel regimes.
3. **Background AI:** Qwen, BERT, Whisper, vision, and RAN-AI workloads that use headroom which
   cannot economically remain empty merely to cover the NRx peak.

NRx is not necessary for every slot. The conventional receiver is sufficient under good channel
conditions, while NRx can improve actual decoding under difficult conditions. NRx demand becomes
bursty when periodic slots from multiple cells align or difficult channels occur in succession.
The important quantity is therefore not day-long average utilization. It is how much queue builds
over a few milliseconds and whether a late result can be rejected safely.

### 1.1 MPS, MIG, and MIG+MPS under one set of criteria

These mechanisms are not progressively better versions of the same abstraction. They solve
different problems.

The five deployments become much easier to distinguish when they use one visual grammar. A
**thick black line is a hardware isolation wall created by MIG**. A dashed MPS-client box controls
an execution share; it is not an isolation wall. P2P and GDR are two data paths that move a tensor
after L1 and NRx have been placed on opposite sides of a wall.

![Physical placement and data paths for MPS, MIG, MIG+MPS, MIG+P2P, and MIG+GDR](figures_en/00_architecture_map.png)

The P2P panel deliberately carries an asterisk. The measured P2P gate used **one process owning two
MIG CUDA contexts on a topology where peer access actually succeeded**. It must not be generalized
to every cross-process MIG combination. The GDR experiment instead separated L1 and NRx processes,
registered their GPU memories, and carried payload through NIC loopback without staging it in CPU
DRAM.

**P2P can connect multiple ordinary full GPUs, but the A100 MIG target in this study has a narrower
support domain.** NVIDIA's current MIG guide states that, with the R570 driver generation, P2P is
supported only between MIG instances on the same physical GPU; P2P between MIG instances on
different physical GPUs and between a MIG device and a non-MIG GPU is unsupported. CUDA IPC across
GPU Instances is also unsupported. Our R580 P2P gate succeeded precisely in the supported case: one
process owned a `2g↔2g` pair on the same A100.

| Path | Reachability | Role in the current experiments |
|---|---|---|
| Local | Same MIG/CUDA device | Shortest path, but L1 and NRx share one execution domain |
| CUDA P2P | **Peer-capable MIG pair on the same physical A100** | Stage 1 single-process `2g↔2g` fast-path/lower-bound baseline |
| NIC GDR | Different MIGs on one GPU, **MIGs on different physical GPUs**, and separate processes/containers; potentially another host over RDMA | Stage 1 loopback and the primary fabric for the Stage 2/3 multi-physical-GPU resident pool |

P2P and GDR are therefore both multi-device tensor paths, but P2P cannot replace GDR in the target
pool topology. P2P is the short path within one physical GPU where supported; GDR reaches MIGs on
other physical GPUs. [NVIDIA MIG Deployment Considerations](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)
specifies the P2P and IPC boundary, while the [CUDA multi-GPU guide](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html)
defines general P2P capability per device pair through `cudaDeviceCanAccessPeer()`.

| Deployment | Plain description | What it does well | What it does not solve |
|---|---|---|---|
| Full GPU + MPS | Multiple processes share one full GPU | High utilization and work conservation; spare SM capacity is immediately usable | No physical boundary; long AI work and a shared GPU queue can perturb L1 tail latency |
| MIG local | A physical wall is created, but L1 and NRx share one room | Protects the 4g from Qwen or other work on the sibling MIG | L1 and NRx inside the same 4g still share SMs, memory resources, and GPU work queues |
| MIG + MPS local | MPS reallocates shares between processes inside one MIG | Preserves sibling isolation and tunes average shares inside one GI | Creates no new hardware isolation; a quota is not a deadline or preemption guarantee |
| Cross P2P | L1 and NRx occupy peer-capable MIGs on the same physical A100 and use a CUDA peer path | Separates L1 and NRx compute queues with low transport overhead | Cannot form the cross-process GI pool across physical GPUs; remote NRx capacity is still finite |
| Cross NIC GDR | GPU memories are NIC-registered and connected directly | Reaches isolated MIG/process/GPU endpoints without a CPU-DRAM bounce | Does not make NRx computation faster; admission and queueing remain separate problems |

The relevant axes are:

1. **Isolation:** does hardware stop another AI kernel from delaying L1?
2. **Elasticity:** when one NRx queue is busy, can idle capacity elsewhere serve it?
3. **Wait location:** does work wait in a GPU queue, inside a CPU CUDA call, or at an explicit
   transport completion?

MPS provides good local work conservation at small client counts, but it is weak on physical
isolation and that advantage disappears as independent clients and kernel-launch pressure grow.
MIG is strong on isolation but fixes capacity spatially. MIG+MPS tunes shares within the MIG wall;
it does not remove this contradiction. P2P and GDR do not remove the wall. They are data paths that allow another room's
NRx to be invoked while preserving the wall. DART-Rx is needed because the system must decide which
request to send where and how to handle a late result over those paths.

### 1.2 The three GDR experiments and the target topology must be separated

In the five-placement figure, `MIG+GDR` is the **placement baseline that connects a 2g L1 and 2g
NRx inside one physical GPU through NIC loopback**. The later `three-NRx` result is not merely that
same drawing repeated three times. It is a separate request-level pool of three resident TensorRT
workers spread across multiple physical GPUs.

The three experiments use GDR but have different topologies and validation targets. Part IV §15
therefore places the diagram beside the physical setup, measured results, and claim boundary in
**Stage 1 → Stage 2 → Stage 3 order**. Stage 4 is kept separate as the remaining integrated gate so
that it cannot be mistaken for a completed result.

“Extending MIG” does not mean physically merging `4g+3g` into one CUDA device. Each NRx request
finishes on one worker; different replicas process different cell-slot requests in parallel. The
precise description is **request-level scale-out of NRx service capacity over fixed MIG**.

Nor does the design instantly convert any arbitrary idle GPU into an NRx worker. Every endpoint on
the slot fast path must already hold its model, TensorRT context, CUDA Graph, and registered buffers.
Background AI can occupy remaining 4g/full-GPU domains, but turning one into an NRx endpoint requires
advance provisioning or an activation step outside the slot fast path.

### 1.3 What fails in the three local placements?

The exact placements used in this study are:

| Name | Physical placement | Intended benefit |
|---|---|---|
| Full MPS | L1+NRx client and Qwen client share one full A100 | Use all spare GPU capacity and control workload shares |
| MIG local | L1+NRx in a 4g MIG; Qwen in sibling 3g | Isolate Qwen while keeping L1 and NRx close |
| MIG+MPS local | Separate L1 and NRx MPS clients inside the 4g; Qwen in sibling 3g | Combine sibling isolation with share control inside the 4g |

![Local MPS, MIG, and MIG+MPS each leave a different problem](figures_en/00_three_local_baselines.png)

#### A. Full MPS: efficient, but without a protection boundary

Full-GPU MPS had the largest raw capacity among the measured deployments. Without a background
tenant, it remained stable through the 350 request/s limit of the absolute-rate sweep, with p99
`2.551 ms`. Calling MPS intrinsically slow would therefore be incorrect.

The problem is that increasing the background share also changes the RAN path:

| Qwen MPS cap | Slot E2E mean / p99 | Qwen throughput |
|---:|---:|---:|
| 30% | 5.865 / 6.307 ms | 7.92 it/s |
| 50% | 6.226 / 6.782 ms | 11.14 it/s |
| 70% | 6.656 / 8.066 ms | 17.24 it/s |
| 100% | **8.569 / 11.180 ms** | **21.11 it/s** |

MPS offers high aggregate throughput and background utility, but the cost appears directly in the
L1/NRx latency. Active-thread percentage controls an average resource share; it does not bound the
time to preempt a long kernel or guarantee L1 p99. The issue is not low performance. It is that
**real-time L1 protection and background throughput are coupled on a load-dependent Pareto curve.**

There is a more important limitation. The stable 350 requests/s result above uses **one optimized
NRx execution path with no background tenant**. It does not characterize MPS scaling when independent
processes from multiple cells or services submit NRx work concurrently. A separate causal campaign
produced a very different curve.

![MPS L1-p99 and kernel-gap collapse as independent NRx processes increase](figures_en/00c_mps_multi_nrx_breakdown.png)

- On full-A100 MPS, increasing NRx processes from `1→8` raised L1 p99 from `42.3→189.3 ms`, or **4.5×**.
- Inside a 4g MIG, MPS produced `40.7→331.4→435.7 ms` at `N=1→6→8`, ending at **10.7×**.
- In the 4g case, median L1 inter-kernel gap changed from `1.15 us(N=1)` to `119.7 us(N=6)`
  and `379.1 us(N=8)`, while L1 GPU duty fell from `31.6%` to `13.8%`.

The correct conclusion is not “MPS is good.” **MPS is efficient at low client count and spare load,
but without a hardware wall, scheduler/launch-queue/implicit-synchronization tails propagate directly
into L1 as independent NRx contexts and aggregate launch pressure increase.** `N=6` is the measured
knee for this max-rate NRx workload, not a universal constant. Because this earlier causal campaign
used a 20-cell L1 path, its absolute milliseconds are not compared directly with the current optimized
chain; it is used as causal scaling evidence.

#### B. MIG local: isolates the next room, not L1 from NRx in the same room

Sibling isolation was strong. NRx capacity was `745.1 request/s` alone and `744.2 request/s` with
Qwen running in the sibling 3g, a difference of only `-0.11%`. The statement “MIG failed to isolate
Qwen” is therefore false.

However, placing L1 and NRx in the same 4g leaves them in one GPU instance. L1 active time became
`1.621×` its standalone value, and slot E2E mean was `6.191 ms`. Qwen in the sibling 3g was isolated;
NRx kernels, memory traffic, and queued work inside the same 4g were not isolated from L1.

The NRx capacity of that 4g is also fixed. Its sojourn p99 was `3.470 ms` at 300/s and
`1124.981 ms` at 350/s. Capacity in another idle MIG does not move to this queue automatically.
MIG's limitation is not weak isolation: **isolation also pins service capacity to each room.**

#### C. MIG+MPS local: changes shares but not room size or walls

MIG+MPS remains useful. It can vary L1/NRx shares while isolating sibling-3g Qwen at approximately
`10.21–10.22 it/s`. In the proper two-client quota experiment, L1 and NRx ran as separate MPS
clients inside one 4g and exchanged the same full-size GDR payload between processes.

| L1:NRx share | E2E mean / p99 |
|---:|---:|
| 30:70 | **4.757 / 5.087 ms** |
| 50:50 | 4.971 / 5.099 ms |
| 70:30 | **6.499 / 6.747 ms** |

Reducing the share of the longer NRx stage from 70% to 30% increased the L1 share but made the
whole chain `36.6%` slower. The uncapped two-client absolute-rate run also collapsed between 250/s
(p99 `3.437 ms`) and 300/s (`259.106 ms`). MPS inside one 4g cannot create:

1. a new physical boundary between L1 and NRx;
2. a path to idle capacity outside the 4g; or
3. a hard bound on kernel completion or host-side CUDA blocking.

The local-placement conclusion is not that every approach is bad:

| Placement | Benefit | Remaining problem |
|---|---|---|
| Full MPS | Highest raw capacity and work conservation | L1 tail is coupled to co-tenant load |
| MIG local | Strong sibling isolation | Same-GI L1–NRx contention and fixed capacity |
| MIG+MPS local | Sibling isolation plus intra-GI share control | Same-GI contention, fixed capacity, and quota trade-off |

None provides both **physical L1–NRx separation and the ability to borrow NRx capacity elsewhere on
demand**. This is the motivation for P2P and GDR.



### 1.4 Why consider P2P and GPUDirect RDMA?

The first question was whether moving L1 and NRx to different GPU execution domains actually
restores L1. The second was whether moving their tensors costs more than the isolation is worth.

![Why P2P and GDR are needed to separate L1 and NRx](figures_en/00b_why_p2p_gdr.png)

#### Design motivation A: separate compute queues with P2P

L1 slowdown was `1.621×` in the 4g L1+NRx placement and `1.702×` under MIG+MPS local. Splitting the
GPU into `2g L1 | 2g NRx` and using a real `cudaMemcpyPeerAsync` path reduced L1 slowdown to
**`1.043×`**. Forward-plus-backward P2P copying averaged `76.84 us` in the ring-depth-2 run.

This establishes an important causal link: much of the L1 slowdown came from sharing an execution
domain with NRx rather than from Qwen in the sibling MIG. The smaller 2g NRx was slower than the 4g
NRx, however, so slot E2E increased slightly from `6.191` to `6.383 ms`. Separation restored L1
protection but could not compensate for reduced NRx compute capacity.

#### Design motivation B: use NIC GDR where P2P cannot reach

P2P is the shortest native GPU path and is an important lower-bound baseline, but peer access is not
available across every MIG/GPU/process combination. NIC GDR registers process-owned GPU memory with
the NIC and exchanges requests and results across isolated endpoints without CPU DRAM staging.

On this CloudLab A100/ConnectX-6 Dx node, the experiment used real GPU MRs and physical NIC
loopback. The optimized direct-TensorRT request was `1,415,232 bytes`; the result was
`314,496 bytes`; neither used CPU staging. At equal queue depth 1:

| Path | Slot E2E mean / p99 | Qwen | Interpretation |
|---|---:|---:|---|
| Cross P2P | 5.888 / 6.224 ms | 10.22 it/s | Native lower bound for peer-accessible topology |
| Cross NIC GDR | 6.326 / 6.846 ms | 10.24 it/s | Zero-CPU-bounce path across process/GPU isolation |

NIC GDR was `0.438 ms`, or about 7.4%, slower on mean in this implementation. The anticipated
5–10 us single-slot improvement was not observed. The reason to retain GDR is not that it beats P2P
on latency. It is that **a resident NRx outside the peer-accessible topology can still join the pool.**

#### Design motivation C: connect multiple endpoints as a pool

A single remote endpoint will still collapse when overloaded. At 1 request/ms, pinning all work to
one NRx produced p99 `3293.25 ms` while at least 66.7% of endpoints were idle during part of the
trace. Selecting the endpoint with the earliest predicted finish reduced p99 to `1.63 ms` and the
no-timely-result ratio from `99.97%` to `0.13%`.

The design progression is therefore:

```text
MPS: high throughput, but L1 and co-tenant behavior are coupled
MIG: strong L1 protection, but capacity is static and same-GI contention remains
MIG+MPS: share control inside one GI, but the isolation–elasticity tension remains
        ↓
P2P: demonstrates separation of L1 and NRx compute queues
GDR: reaches isolated NRx endpoints beyond the P2P domain without CPU staging
        ↓
Compose multiple resident NRx endpoints into one pool
        ↓
Use deadline and radio utility to select an endpoint;
recover to the conventional path when the NRx result is late
```

The DART-Rx design may use local within one MIG and P2P for a supported MIG pair on the same physical
GPU. However, **the currently implemented multi-physical-GPU NRx pool uses GDR as its primary path**;
a P2P pool backend has not been implemented or evaluated. The core contribution is not path
selection itself, but the admission, reservation, expiry, and commit contract that makes
heterogeneous endpoints one deadline-safe receiver service.

### 1.5 How to read all five approaches in one figure

The figure below combines the **comparison values** for all five approaches into four panels.
Panels (a)–(c) use the same placement campaign. Full MPS uses the 50% Qwen-cap point (`11.14 it/s`),
the nearest measured point to the `10.22–10.24 it/s` isolated placements. Panel (d) is a separate
causal stress campaign with multiple independent NRx processes.

![Measured L1 protection, E2E, background throughput, and scaling for MPS, MIG, MIG+MPS, P2P, and NIC GDR](figures_en/03g_fiveway_measured_evidence.png)

- **(a) L1 protection:** Full MPS, MIG local, and MIG+MPS increased L1 active time by `1.601×`,
  `1.621×`, and `1.702×` even in the low-load placement gate. Separating L1 and NRx into different
  MIGs with P2P reduced this to `1.043×`. NIC GDR is plotted at `1.103×`. A separate matched
  L1-active trace was not preserved for this value; it is the working estimate used from the current
  implementation and observed P2P/GDR difference. This provenance note remains in the text rather
  than cluttering the graph.

The magnitude of `1.103×` is not arbitrary. In the equal-depth transport gate, the measured GDR/P2P
E2E-p99 ratio is `6.846 / 6.224 = 1.100×`; comparing each of the two GDR repeats against the
aggregate P2P value gives `1.072–1.128×`. Thus `1.103×` is reasonable as a representative value at
approximately **1.10×**. It is not, however, proof of the same L1-active ratio. `l1_active` covers
CE/front plus LDPC/CRC/back, while most RDMA waiting lies outside those CUDA intervals. A
conservative bound assigns none or all of the `0.438 ms` mean GDR–P2P E2E delta to the `2.429 ms`
P2P 2g-L1 baseline, yielding `1.043–1.223×`. The plotted value lies inside this physically plausible
range, but the paper should treat it as an **approximately 1.10× provisional value**, not a
three-decimal measurement. A matched gate with CUDA-event timing around GDR CE/front and
LDPC/CRC/back is required to make it definitive.
- **(b) Low-load slot p99:** all five lie between `6.56` and `7.26 ms`. A single request therefore
  makes local and cross placements look similar. GDR is depth=1 while the preserved results for the
  others use depth=2; this panel bounds implementation cost rather than declaring a final winner.
- **(c) Background utility:** the isolated placements sustain `10.22–10.24 it/s`; the selected Full
  MPS point sustains `11.14 it/s`. It is not perfectly matched, but it is closer than the measured
  30% or 100% MPS-cap points.
- **(d) Scaling:** the limitation hidden by low-load E2E appears in the NRx-process sweep. Full-A100
  MPS increased L1 p99 from `42.3→189.3 ms` (`4.5×`), while MPS inside a 4g increased it from
  `40.7→435.7 ms` (`10.7×`).

The direct conclusion is: **low-load E2E is similar across the five approaches, but MPS-family L1
protection collapses as co-tenant count grows. P2P experimentally restores protection on the
same-GPU path; GDR pays a small additional E2E cost to extend reach across GPUs and processes.** A
final winner still requires one matched gate with the same physical-GPU budget, Qwen throughput,
and NRx burst.



## 2. First misconception: MIG isolation did not fail

We measured resident NRx on a 4g MIG, both alone and while Qwen-7B ran on the sibling 3g.

![MIG isolation and the NRx queue cliff](figures_en/01_mig_isolation_queue_cliff.png)

The result rules out both “MIG did nothing” and “MIG solves everything.”

- **MIG isolation works:** capacity was `745.1 request/s` alone and `744.2 request/s` with sibling
  Qwen, a `-0.11%` difference.
- **Isolation does not add capacity:** p99 was about `1.39 ms` at 700/s, rose to
  `15.58–18.78 ms` at 750/s, and reached `214.29–217.79 ms` at 800/s.

Soundproofing the room prevents the neighbor from slowing the clerk. It does not stop the queue from
growing when 750–800 customers per second arrive at a clerk who can serve about 745. The large tail
was a **queueing collapse near fixed NRx service capacity**, not a failure of sibling MIG isolation.

## 3. Second misconception: same-partition L1+NRx should match L1 alone

MIG isolates **different GPU instances**. L1 and NRx inside one 4g share SMs, memory resources,
launch resources, and queued GPU work. Two metrics must therefore be separated:

- **L1 active time:** whether L1 kernels are protected from NRx/background interference.
- **Dependency-carrying slot E2E:** total CE → NRx → LDPC/CRC latency.

Cross P2P improved L1 slowdown from `1.621×` to `1.043×`. Yet splitting 4g into 2g L1 + 2g NRx
slowed NRx compute, so slot E2E slightly increased from `6.191` to `6.383 ms`. **Isolation is visible
in the L1 metric; smaller-slice NRx compute can still prevent total slot latency from falling.**

This does not make cross-partition execution useless:

- Same-partition execution may minimize one isolated request by avoiding transport and using a
  larger compute slice.
- Cross-partition execution separates the L1 execution queue from NRx and exposes other resident
  capacity. Its advantage grows under simultaneous multi-cell arrivals and tail pressure.
- A fair comparison asks both which path minimizes one request and where each queue collapses under
  an identical arrival rate.

### 3.1 Why the original 105 ms NRx became 1.34 ms

The original 105 ms was not pure TensorRT compute. It included generic layout conversion in the
public Aerial `pycuphy` wrapper. Nsight analysis and caller-owned bindings exposed the actual path.

![NRx execution-path optimization](figures_en/01b_nrx_wrapper_optimization.png)

- Public wrapper GPU mean: `105.15 ms`
- Caller-owned TensorRT binding: `1.413 ms`
- Direct binding + CUDA Graph: `1.340 ms`; host enqueue `2.50 us`
- Maximum absolute output difference from the wrapper: `0`

The CUDA API profile agrees. The wrapper capture spent `1.232 s` across 15
`cudaEventSynchronize` calls and `103.95 ms` across nine `cudaStreamSynchronize` calls. The direct
path reduced those totals to `32.87 ms` across 20 calls and `1.89 ms` across ten calls. One 393 ms
cold cleanup appears in the direct path's `cudaFree` total and must not be mistaken for steady-state
inference. Asynchronous launch alone is insufficient; persistent caller-owned buffers and explicit
synchronization placement are also necessary.

The initial 106–112 ms chain results remain valid measurements of the wrapper-based implementation.
Placement and queue-capacity conclusions use the optimized direct-TensorRT path. Mixing these two
execution paths in one performance table would misattribute wrapper overhead to MIG or transport.

### 3.2 Host blocking is a distinct bottleneck

CUDA kernel launch is usually asynchronous, but the CPU eventually reaches an operation that needs
completion: freeing memory, reusing a buffer, or consuming a result. A CPU thread can then remain
inside `cudaFree`, `cudaStreamSynchronize`, or even a subsequent `cudaMemcpyAsync`. We call this
**host blocking**.

It is different from a `GPU → CPU DRAM → GPU` data path. Even with no CPU-DRAM bounce, a long NRx
work queue in the same scheduling domain can force L1's host thread to absorb the wait.

| Placement | Expected host-blocking path |
|---|---|
| Full MPS | L1, NRx, and background share a full-GPU scheduling domain; AI work may surface at L1 synchronization points |
| MIG local | The sibling 3g is isolated, but outstanding L1 and NRx work inside the 4g is not separated |
| MIG+MPS local | Separate clients and quotas still share physical resources and completion conditions inside one 4g; no hard preemption bound |
| Cross P2P/GDR | The L1 CUDA queue is separated from remote NRx compute; explicit publish, completion, and commit waits replace implicit co-location waits |

The Nsight result below directly validates this hypothesis for same-MIG co-location. The current
evidence also includes a real cuPHY–GDR–NRx vertical slice, but not yet a paired Nsight matrix for
every MPS/MIG/P2P/GDR point under an identical trace.

#### Direct Nsight evidence

![CUDA-call analysis of host blocking under same-MIG L1+NRx](figures_en/03d_cuda_host_blocking.png)

We measured cumulative host time in six major cuPHY CUDA runtime APIs over a 30 s Nsight window. At
40 cells:

| Condition | Total tracked host CUDA API time | Dominant observed wait |
|---|---:|---|
| L1 alone | 1.681 s | `cudaFree` 1.361 s |
| L1 + NRx | **25.348 s** | `cudaFree` 18.076 s; `cudaMemcpyAsync` 7.034 s |
| `cudaFreeAsync` shim | 25.570 s | `cudaMemcpyAsync` **25.221 s** |
| Stream-ordered memory pool | 25.649 s | `cudaMemcpyAsync` **25.539 s** |

Co-location increased tracked host time by `15.1×`. Replacing `cudaFree` with an asynchronous API
did not remove the total wait; it moved the wait to the next copy or synchronization boundary. The
root problem is not the name `cudaFree`. It is **outstanding GPU work in one scheduling domain and a
dependency boundary that must eventually observe its completion.**

This is why a cross-partition design cannot be evaluated from the 0.4 ms mean transport cost alone.
P2P/GDR separates L1's CUDA queue from NRx compute. It introduces explicit publish and completion
events instead, which DART-Rx manages with credit, expiry, and commit rules. The Chain-8 Nsight
numbers are cumulative over 30 s rather than per-slot latency and do not use the exact direct-TensorRT
five-way configuration; they must not be added to the five-way latency numbers.

Current CUDA-level evidence coverage is:

| Placement | Latency/rate sweep | Direct CUDA-call evidence | Current interpretation |
|---|---|---|---|
| Full MPS | Cap sweep + absolute-rate complete | No identical-trace Nsight | Strong throughput; background-tail cause still needs decomposition |
| MIG local | Placement + absolute-rate complete | Same-MIG causal profile available | Strong sibling isolation; intra-GI L1–NRx wait exists |
| MIG+MPS | Proper two-client quota + absolute-rate complete | No identical-trace Nsight | Quota trade-off measured; wait location needs capture |
| Cross P2P | Placement + absolute-rate complete | No identical-trace Nsight | L1 active isolation restored; copy/synchronization decomposition remains |
| Cross NIC GDR | Placement + absolute-rate + radio complete | Actual-radio GDR vertical-slice Nsight | Local synchronization/conversion exceeds GDR flush time |

Performance differences are measured for every approach. Exact CUDA-call causality is directly
shown only for same-MIG co-location and the GDR vertical slice. Paired Nsight captures remain a
required follow-up for the other three conditions.

#### Required five-way paired host-blocking gate

The final mechanism analysis needs **one matched CUDA-call figure for Full MPS, MIG local,
MIG+MPS, Cross P2P, and Cross NIC GDR.** Existing 30-second same-MIG captures and the actual-radio
GDR capture must not be placed in one bar chart: their code path, cell count, process boundary, and
capture duration give them different denominators.

A fair paired gate must fix the following conditions.

| Controlled item | Required condition |
|---|---|
| L1/NRx code | The same optimized direct-TensorRT, caller-owned-buffer, CUDA-Graph path |
| Requests | One preserved arrival trace and identical request/result tensor sizes |
| Load | A common stable point at `180 requests/s` and a pressure point at `250 requests/s` |
| Background | Paired off/on runs; on-runs target approximately `10.2 Qwen it/s` |
| Repetition | 30 seconds × 3 trials per condition; exclude cold initialization and shutdown |
| Baselines | Separate L1-alone captures for full GPU, 4g L1, and 2g L1; normalize within geometry |
| Nsight scope | `cuda,nvtx,osrt`; NVTX ranges for L1 front, transport publish/wait, NRx, L1 back, and commit |

One aggregate API total would still be ambiguous. The final analysis should contain three panels:

1. **L1-coordinator host-blocked time per slot:** divide time inside tracked CUDA APIs by completed
   slots and also report slowdown relative to the matching-geometry L1-alone baseline.
2. **Which API absorbed the wait:** stacked decomposition of `cudaFree`, `cudaMemcpyAsync`,
   `cudaStream/Event/DeviceSynchronize`, `cudaDeviceFlushGPUDirectRDMAWrites`, launch, and other.
   Preserve call count and call-duration p50/p95/p99 in the companion CSV.
3. **Where the wait occurred in the pipeline:** attribute API intervals to L1 front, local NRx, P2P
   copy, GDR publish/completion, and L1 back/commit using NVTX timestamps; measure temporal overlap
   between L1 GPU idle gaps and host API intervals.

Process structure must be handled explicitly. MPS and GDR use separate L1-producer and NRx-consumer
captures, with the **L1 coordinator as the primary metric** and NRx-side APIs in a separate
supporting panel. MIG local and the current P2P path execute in one process, so NVTX range and
thread/timestamp attribution must separate L1 and NRx phases. Summing two processes' cumulative
times would be invalid.

The hypotheses to test are listed below; they are not conclusions until this gate runs.

| Placement | Host-blocking hypothesis to test |
|---|---|
| Full MPS | Co-tenant work surfaces at L1 synchronization/free/copy dependency points |
| MIG local | Sibling Qwen is isolated, but outstanding NRx work in the same 4g increases L1 API wait |
| MIG+MPS | Client quotas either reduce total wait or merely move it between APIs |
| Cross P2P | L1 API wait approaches the 2g L1-alone baseline, leaving explicit peer-copy completion |
| Cross NIC GDR | Remote NRx compute leaves the L1 CUDA queue; GDR flush/completion and local conversion waits remain |

The current `03d_cuda_host_blocking` figure is therefore **causal evidence that host blocking exists
and migrates between API boundaries**, not a five-way winner chart. Only the matched capture can
quantify, on one denominator, which L1 host waits P2P/GDR separation removes.

Up to this point, the evidence explains **why NRx should be separated from L1 and how an isolated
endpoint can be reached.** A data path does not automatically make multiple NRx workers useful.
The next section is the bridge from transport to scheduling: it tests whether the single-endpoint
queue cliff appears as a busy queue and an idle worker in a real multi-endpoint execution.

## 4. Why select among multiple NRx workers? One queue collapses while another worker is idle

The preceding experiments established two facts.

1. **The single-endpoint gate in §2:** MIG correctly isolates Qwen, but one NRx has finite service
   rate. The 4g NRx sustained approximately `745 requests/s`; increasing arrival to 750–800/s made
   p99 rise to `15.58–217.79 ms`. The queue cliff is real.
2. **The placement and host analysis in §3:** placing L1 and NRx in one GI can slow L1 as well, so
   protected L1 and the NRx compute queue should be separated. P2P/GDR provides a data path to an
   isolated NRx.

Those facts alone do not yet justify selecting among a pool of NRx workers. Being able to reach a
remote NRx is different from proving that its capacity is needed. One question remains:

> **Under time-varying slot arrivals, does the currently assigned NRx miss its timeliness target
> while another already-resident NRx is idle at the same time?**

Only a `yes` establishes static-placement fragmentation and motivates request-level endpoint
selection. The experiment chain therefore separates three gates:

```text
[Earlier Gate A] Measure one NRx service curve
                 → its queue collapses beyond capacity

[Earlier Gate B] Measure P2P/GDR data paths
                 → isolated NRx GPU memory is reachable

[Gate C here]    Replay one arrival trace against three NRx workers
                 → does fixed binding create a busy queue plus idle workers?
                 → does per-request selection reduce the problem?
```

The figure in this section is therefore **Gate C, a compute/queue problem-existence experiment, not
a transport benchmark.** One real TensorRT NRx worker resides in a `3g.20gb MIG` on each of three
physical GPUs. It does not reuse the 4g capacity number from §2; each 3g endpoint is calibrated near
`1.6 ms/request` for this experiment.

The same deterministic input tensor is preloaded at every endpoint to remove P2P/GDR performance
from the comparison and isolate the queueing effect of static binding versus request routing. This
gate excludes the cuPHY front/back path, P2P/GDR tensor movement, and conventional fallback
execution. Stage 2 later repeats policy evaluation with full-size GDR request/result transport.

```text
slot / NRx-required trace
        ↓
host scheduler selects one endpoint
        ↓
real TensorRT + CUDA Graph executes on the selected 3g MIG

Excluded: L1 tensor transport, cuPHY CE/LDPC, and conventional fallback execution
```

The two plotted policies are:

- **Red, `static-one`:** pin every request to NRx 0, even as its queue grows. NRx 1 and NRx 2
  receive no work.
- **Blue, `predicted-finish`:** actually select the endpoint among NRx 0/1/2 whose already-reserved
  work is predicted to finish first. If no endpoint can finish before the 5 ms gate, record
  `fallback required` instead of extending the queue. Blue therefore combines **distributed
  selection and deadline admission**.

![One NRx queue collapses while other workers remain idle](figures_en/02_fixed_placement_fragmentation.png)

The three panels ask different questions about the same runs.

| Panel | Plain-language question | Exact denominator and meaning |
|---|---|---|
| **(a) p99 request latency** | How late were the NRx requests that finished? | The slowest-1% boundary of `arrival → queue wait → TensorRT completion` among completed NRx requests. A request rejected early by blue admission is absent from this latency set, so panel (a) must be read with (b). |
| **(b) no result within 5 ms** | For all NRx requests, how often was no result usable in time? | All NRx-required requests are the denominator. Completion after 5 ms, overflow, and predicted-finish pre-fallback all count as failures. This is the more comprehensive usable-result coverage metric. |
| **(c) endpoint idle fraction** | How much of the three NRx workers sat idle meanwhile? | Fraction of `3 endpoints × trace duration` without TensorRT execution. **Both red and blue bars in (c) are idle fractions; blue is not the endpoint-selection rate.** |

| Workload | Pin one NRx: p99 / no timely result | Earliest predicted finish: p99 / no timely result |
|---|---:|---:|
| 1 cell, 1 ms, NRx 100% | 3293.25 ms / 99.97% | 1.63 ms / 0.13% |
| 4 cells, 1 ms, bursty 10% | 51.39 ms / 64.85% | 5.50 ms / 1.61% |
| 4 cells, 0.5 ms, bursty 10% | 3332.84 ms / 99.90% | 5.06 ms / 7.89% |

In the first workload, one NRx has an observed service time near `1.6 ms`, or roughly `625
requests/s`, while NRx 0 alone receives `1,000 requests/s`. Its queue must therefore grow. NRx 1 and
NRx 2 receive no work, leaving approximately `2/3`, or `66.7%`, of aggregate endpoint-time idle.
Blue uses primarily two endpoints and changes p99 from `3293.25` to `1.63 ms`, while no-timely
coverage failures fall from `99.97%` to `0.13%`.

The middle `4 cells · 1 ms · bursty 10%` case is subtler and more important. Both policies are idle
roughly `78–79%` on average because mean demand is low. Yet a short burst forms a long queue at NRx
0 under static-one, producing `51.39 ms` p99. Predicted-finish spreads that burst and reduces p99 to
`5.50 ms`. **Low average GPU utilization can therefore coexist with poor tail latency.** This is
temporal service-capacity fragmentation.

Blue still has a `7.89%` no-timely ratio in the third workload. Routing can recover stranded
capacity, but it cannot make three replicas or the 5 ms gate sufficient for every burst. The result
therefore does not say that three NRx workers solve every load. It says that **static binding wastes
capacity that already exists, and overload still requires admission and fallback after routing.**

`No timely result` means no usable NRx result within the 5 ms experimental gate; it also includes
requests for which the scheduler chose conventional fallback. This compute/queue metric is not the
same as a radio deadline miss because this gate does not execute the full PHY fallback path.

The claim boundary is equally important:

- **Established:** the same three real NRx workers can exhibit a busy queue and idle capacity at
  once because of static binding; request-level routing/admission greatly reduces the damage.
- **Not established here:** whether P2P or GDR is faster, end-to-end behavior with real L1 tensors,
  or universal superiority of predicted-finish over round robin for every trace.
- **Follow-on link:** the P2P/GDR gates test whether tensors can reach the selected endpoint; the
  Stage-2 GDR-pool gate repeats policy evaluation with full-size request/result transport.

## 5. The complete problem: each mechanism solves only one axis

> **When a burst of useful NRx requests exceeds the capacity of the current MIG, use NRx capacity in
> other isolated GPU spaces without stopping L1 or reconfiguring MIG, while preventing late or stale
> AI results from affecting the final radio outcome.**

This is more than load balancing. It combines the following constraints:

| Constraint | Why it matters |
|---|---|
| Static spatial isolation | Protect L1 p99 from co-tenant interference |
| Dynamic service demand | NRx requests burst with cell count and channel conditions |
| Dependency transport | A remote NRx must receive L1 tensors and return LLRs |
| Absolute expiry | Even a correct result is unusable for that slot after expiry |
| Conventional baseline | Failure of an optional NRx must leave a valid recovery path |
| Background utility | Spare accelerators cannot remain empty for peak demand alone |

### 5.1 Where each existing approach stops

| Approach | What it actually solves | Measured remaining limitation | Required next capability |
|---|---|---|---|
| **Full MPS** | Work-conserving sharing of one full GPU across processes | L1 p99 rises `42.3→189.3 ms` as NRx processes grow `1→8`; there is no physical L1 wall | Separate mandatory L1 and optional NRx compute domains |
| **MIG local** | Strongly protects a 4g from Qwen in a sibling 3g | L1 and NRx in the same 4g remain unisolated and L1 active time is `1.621×`; endpoint capacity is fixed | Dedicated L1 and NRx MIGs plus access to external capacity |
| **MIG+MPS local** | Adjusts average shares within one 4g while retaining sibling-MIG isolation | Quotas create neither a new hardware wall nor capacity outside the 4g | Scale out to another resident endpoint rather than only tune shares |
| **Cross P2P** | Separates compute queues for a peer-capable MIG pair and restores L1 slowdown to `1.043×` | Topology/process scope is limited; one remote NRx still has finite capacity | A common path to endpoints outside P2P reach |
| **Cross NIC GDR** | Reaches memory in another MIG/process/GPU without CPU-DRAM bounce | Costs `0.438 ms` mean over P2P here and does not choose an endpoint or handle late results | Deadline-aware routing, bounded credit, and expiry-safe commit |
| **Multiple NRx + static binding** | Pre-provisions more endpoints | One queue reaches seconds while `66.7%` of endpoint-time is idle | A request-level service pool and finish-aware admission |

No mechanism in this table is simply “bad.” **MPS provides utilization, MIG provides isolation, and
P2P/GDR provide reachability, but none completes dynamic NRx service under isolation together with
deadline correctness.**

### 5.2 Architectural requirements derived from the measurements

| Observed problem | Required system property | DART-Rx block |
|---|---|---|
| MPS co-tenant pressure propagates into L1 tail latency | A fixed physical protection boundary for mandatory PHY | **Protected L1 MIG** |
| MIG reconfiguration is unavailable on the fast path and endpoint startup is expensive | Pre-resident model, context, graph, and buffers | **Resident NRx fabric** |
| Some MIG/GPU pairs are outside P2P reach | A remote GPU data path without CPU-DRAM staging | **P2P/GDR tensor plane** |
| A busy queue and idle NRx coexist | Select and reserve one feasible endpoint per request | **Predicted-finish scheduler + credit** |
| A burst violates expiry even at low average load | Absolute-deadline admission that avoids extending a doomed queue | **Utility/deadline admission** |
| A remote NRx can be late, fail, or return an old-slot result | Preserve the baseline and authorize exactly one result | **Conventional fallback + versioned exactly-one commit** |
| Keeping spare endpoints empty wastes value | Background work reclaimable within a bounded interval | **Bounded background lease** |

### 5.3 The causal path into the design

```text
Use only MPS work conservation
    → cannot protect L1 from co-tenant tails

Add fixed MIG isolation
    → protects L1, but pins NRx capacity to rooms

Open GPU-memory paths with P2P/GDR
    → reaches remote NRx, but does not choose queues or validate late results

Select a resident endpoint with a shadow reservation ledger
    → uses idle capacity, but overload and late completion remain possible

Add deadline admission, conventional fallback, and versioned commit
    → use only timely NRx results and block stale results from PHY state

Add bounded background leases
    → avoid leaving peak capacity empty while retaining bounded burst reclaim
```

DART-Rx is therefore neither “fast MIG reconfiguration” nor a claim that GDR is faster than P2P.
It is an **architecture that preserves fixed-MIG L1 isolation, borrows service capacity from
pre-resident NRx workers per request, and commits the result into PHY state only while it remains
valid.** The four blocks in Part II directly implement these measured requirements.

---
# Part II. DART-Rx design

## 6. Overall architecture

DART-Rx is not a collection of unrelated techniques. It is a **slot-processing pipeline that keeps
L1 in dedicated protected space and exposes several resident NRx workers as one selectable pool.**

![DART-Rx control plane, endpoint shadow queue state, fixed-MIG NRx fabric, and P2P/GDR data plane](figures_en/00d_dart_rx_overall_architecture.png)

*The `REQ`/`RES` ports and green lines summarize the logical GDR request/result channel. The
example selects NRx 1 only; they do not indicate that NRx 2 executes or that a CQ carries tensor
payload. The following sections separate the control, payload, and completion paths.*

### 6.1 How does the scheduler know which NRx queue is currently shorter?

The queue is not a remotely inspected GPU hardware queue. It is **dispatcher-side accounting of
outstanding requests at each endpoint: shadow queue state**. `pending=3` means that the endpoint has
accepted three requests for which it has not yet returned completions, including the request
currently executing. The dispatcher therefore does not query a remote GPU over the NIC for every
placement decision.

The current prototype keeps the following state per endpoint.

| Scheduler state | Meaning | Update point |
|---|---|---|
| `pending` | Submitted requests not yet completed | `+1` on submit; `-1` on result completion |
| `predicted_tail` | Reserved completion time of already accepted work | Advanced by a service bound on submit; reset to `now` when empty |
| `service_bound` | Conservative per-request service/exchange time | Calibrated at startup and updated from recent completions |
| `available` | Whether the endpoint may be selected | Cleared on error or a long-stalled in-flight request |
| queue credit | Whether the endpoint control queue can accept another request | Determined by bounded `put_nowait` success or failure |

Each arriving request triggers the following sequence.

1. Drain every endpoint's `started/result/error` events and refresh the local ledger.
2. Predict the next completion at each endpoint as
   `max(now, predicted_tail) + service_bound`.
3. `shortest_queue` selects the smallest `pending`, breaking ties by predicted finish.
   DART-Rx `predicted_finish/tail_aware` directly selects the healthy endpoint with the earliest
   predicted finish.
4. If that prediction is later than slot expiry, do not invoke NRx; retain the conventional result.
5. An accepted request consumes one local control-queue credit and reserves the tail while
   incrementing `pending`. A real GDR result completion decrements `pending` and corrects the
   service bound.

Thus, “the short queue” is neither a guess nor a centralized scan of GPU internals. It is a
**reservation ledger continuously reconciled against real completions by the dispatcher that
issued the work**. Payloads travel between registered GPU memories over P2P/GDR, whereas these
small counters and events remain in the host control plane in the current implementation. It would
therefore be inaccurate to claim that the CPU has no role at all. The precise claim is that **large
tensors avoid CPU-DRAM staging; the CPU performs scheduling and completion bookkeeping only**.
Moving this ledger into NIC/DPU doorbells or an accelerator runtime is a follow-on architectural
optimization for control-plane overhead, not a prerequisite for identifying endpoint state.

The implementation is preserved in `poll_completions`, `snapshot`, `submit`, and `choose_endpoint`
of [`EndpointProcessProxy`](../../../cloudlab_aerial/task1/dart_rx_gdr_pool.py).

One request follows six steps:

1. The conventional receiver remains an executable candidate while L1 prepares the input.
2. Admission checks whether NRx is useful for this channel and whether it can still finish before
   expiry.
3. If feasible, the scheduler atomically reserves buffer and queue credit at the endpoint with the
   earliest predicted finish.
4. The tensor and result move over the endpoint's local, P2P, or GDR path.
5. Exactly one result commits after passing slot identity, deadline, endpoint health, and CRC checks.
6. If NRx is late or fails, the conventional result commits.

```mermaid
flowchart LR
    SLOT[Cell/slot request<br/>absolute expiry] --> L1[Protected L1 MIG<br/>cuPHY CE/front-end]
    L1 --> CONV[Conventional receiver<br/>always-valid baseline]
    L1 --> ADMIT[Utility + deadline<br/>admission]
    ADMIT -->|not useful / infeasible| COMMIT[Epoch + expiry<br/>single commit]
    ADMIT -->|reserve credit| PLACE[Predicted-finish<br/>endpoint selection]

    subgraph FABRIC[Fixed resident receiver fabric; no MIG reconfiguration]
        LOCAL[Local NRx<br/>when feasible]
        E0[Isolated NRx endpoint 0]
        E1[Isolated NRx endpoint 1]
        E2[Isolated NRx endpoint 2]
        BG[Bounded background leases<br/>Qwen / BERT / Whisper / vision]
    end

    PLACE --> LOCAL
    PLACE -->|P2P or GPUDirect RDMA| E0
    PLACE -->|P2P or GPUDirect RDMA| E1
    PLACE -->|P2P or GPUDirect RDMA| E2
    LOCAL --> COMMIT
    E0 --> COMMIT
    E1 --> COMMIT
    E2 --> COMMIT
    CONV --> COMMIT
    BG -. reclaimed at work-unit boundary .-> E0
    BG -. reclaimed at work-unit boundary .-> E1
    BG -. reclaimed at work-unit boundary .-> E2
    COMMIT --> OUT[LDPC/CRC result<br/>exactly one outcome]
```

MIG remains central to the design. It creates non-interfering rooms for the `protected L1` and
`isolated resident NRx endpoints`. DART-Rx does not remove those walls. It adds a safe protocol for
borrowing service across fixed rooms.

## 7. Design block 1: joint radio-utility and deadline admission

NRx is not invoked for every slot. A request carries:

```text
(cell_id, slot_id, epoch, channel_features, release_time, absolute_expiry)
```

Admission reserves NRx credit only when both questions have a positive answer:

1. **Radio utility:** under this channel condition, is NRx likely to improve success over the
   conventional receiver?
2. **Timing feasibility:** is the conservative predicted finish of a healthy endpoint earlier than
   `expiry - commit_guard`?

A request that is unhelpful or unlikely to finish in time uses the conventional path without
polluting a remote queue. The current prototype uses measured SNR bins as its utility gate. A final
system should use signals already present in the gNB, such as CQI, DMRS quality, and decoder/HARQ
history.

## 8. Design block 2: a fixed resident receiver fabric

The fast path never changes MIG geometry. NRx models, TensorRT contexts, CUDA Graphs, and registered
buffers remain resident at each endpoint.

Endpoint selection considers more than round robin or instantaneous queue length. It combines the
time at which a queue becomes available, a conservative service-time tail, and transport/commit
slack:

```text
predicted_finish[e]
  = max(now, endpoint_available[e])
  + conservative_service_tail[e]
  + transport_and_commit_guard[e]
```

The scheduler atomically reserves tensor/ring credit at the fastest feasible endpoint. Credit is a
bounded request position: a worker cannot accept unbounded in-flight work. Direct/local or P2P is
used where the GPU topology permits it; registered GPUDirect RDMA crosses an isolation boundary or
reaches another GPU. GDR's role is not to accelerate NRx compute. Its role is to include **another
isolated endpoint in the service pool without CPU staging.**

The publish order on one RC QP is:

```text
request:   GPU payload WRITE → descriptor WRITE → doorbell WRITE
response:  GPU result WRITE  → completion WRITE → doorbell WRITE
```

NIC completion alone never authorizes a PHY commit.

## 9. Design block 3: baseline-preserving, expiry-safe commit

NRx produces an **expiring alternative result**. The conventional receiver remains the always-valid
baseline.

```mermaid
sequenceDiagram
    participant L1 as Protected L1
    participant C as Conventional RX
    participant S as DART-Rx scheduler
    participant N as Resident NRx endpoint
    participant K as Commit table

    L1->>C: start baseline decode
    L1->>S: request(slot, epoch, expiry, utility)
    alt useful and deadline-feasible
        S->>N: reserve + publish GPU payload
        N-->>K: result(slot, epoch, health_epoch, CRC)
    else not useful or infeasible
        S-->>K: conventional-only
    end
    C-->>K: baseline result
    K->>K: check slot/epoch/health/visibility/expiry/CRC
    K-->>L1: exactly one committed result
    Note over K,N: A late or stale NRx may finish,<br/>but cannot alter committed PHY state.
```

An NRx result may commit only if slot/epoch identity, endpoint health epoch, payload visibility,
completion status, expiry, LDPC/CRC, and transaction-open state all pass. This prevents a late result
from corrupting a reused buffer or the state of a later slot, even when the remote pool is used
aggressively.

## 10. Design block 4: bounded background leases

Keeping every spare NRx endpoint empty is uneconomical. Allowing an unbounded background kernel to
run prevents timely reclaim during a burst. DART-Rx keeps models resident but stops new background
submissions at a **cooperative work-unit boundary**:

```text
low NRx load  : issue bounded leases to resident background models
burst detected: stop issuing new work units
boundary drain: activate the spare endpoint for the NRx pool
load recovers : reduce NRx credit and resume background leases
```

This is not a claim of arbitrary CUDA-kernel preemption. The background work-unit quantum—prefill
chunk, decode step, or batch size—bounds reclaim delay and must therefore be controlled.

## 11. Mapping measured pain points to mechanisms

| Measured problem | DART-Rx mechanism |
|---|---|
| MIG isolates, but endpoint capacity remains fixed | Multi-endpoint service pool over a fixed topology |
| One static queue collapses while another endpoint is idle | Predicted-finish placement and atomic credit |
| NRx utility varies by channel | Radio-utility admission |
| Same-GI NRx work blocks L1 CUDA calls | L1/NRx compute-queue separation plus persistent registered buffers |
| Wrapper conversion and synchronization hide neural compute | Caller-owned TensorRT bindings plus CUDA Graph |
| A remote result can arrive after deadline | Absolute expiry and a commit guard |
| A stale result can target a reused buffer | Slot epoch plus endpoint-health epoch |
| Remote NRx can fail or overload | Always-valid conventional baseline |
| Leaving spare endpoints empty wastes GPU value | Bounded, cooperatively reclaimable background leases |

---

# Part III. Experimental setup

## 12. Hardware and software

| Component | Configuration |
|---|---|
| CloudLab | Wisconsin d8545 bare metal, AIRANSLICING project |
| GPU | 4 × NVIDIA A100-SXM4-40GB |
| MIG | 4g.20gb + 3g.20gb or 3g/2g layouts depending on the experiment; remaining A100s used as full-GPU endpoints |
| NIC | Mellanox ConnectX-6 Dx 200 Gb/s; physical internal loopback; 100 Gb/s Ethernet active |
| Host OS | Ubuntu 22.04.2 |
| NVIDIA stack | Driver 580.173.02, CUDA 13.0, `nvidia_peermem` |
| RDMA stack | MOFED 24.10-3.2.5.0, rdma-core/Pyverbs 57, RoCE v2 GID index 3 |
| AI-RAN | NVIDIA Aerial 25.3.2 with real cuPHY CE and LDPC/CRC |
| NRx | NVIDIA pretrained `neural_rx.onnx`, TensorRT FP16, caller-owned bindings/CUDA Graph |
| Background | Qwen2.5-7B decode plus ResNet-50, BERT-base, and Whisper-base execution workloads |

GDR containers require both the RDMA character device and visibility of the RoCE backing netdev,
so they use `--network=host --device=/dev/infiniband --cap-add=IPC_LOCK`. GPU MRs register CuPy
allocation addresses directly; the implementation never calls Pyverbs `MR.read/write` on a GPU MR.

## 13. Evaluation topologies

```mermaid
flowchart TB
    subgraph SAME[Same-partition baselines]
        MPS[Full GPU MPS<br/>L1 + NRx + Qwen]
        MIGL[4g MIG: L1 + NRx<br/>sibling 3g: Qwen]
        MIGMPS[4g MIG + MPS: L1 + NRx<br/>sibling 3g: Qwen]
    end

    subgraph CROSS[Cross-partition baselines]
        P2P[2g L1 ↔ 2g NRx via P2P<br/>3g Qwen]
        GDR[2g L1 ↔ 2g NRx via NIC GDR<br/>3g Qwen]
    end

    subgraph POOL[DART-Rx pool evaluation]
        SRC[Protected 4g source/L1]
        R0[Resident endpoint 0]
        R1[Resident endpoint 1]
        R2[Resident endpoint 2]
        SRC <--> R0
        SRC <--> R1
        SRC <--> R2
    end
```

Every MPS/MIG/MIG+MPS/P2P/GDR baseline is necessary, but each answers a different question:

| Comparison | Question answered |
|---|---|
| Full-MPS cap sweep | What RAN-latency/background-throughput Pareto appears when spatial isolation is removed? |
| MIG local | How much L1–NRx contention remains inside one GI despite sibling isolation? |
| MIG+MPS local | Does splitting clients/quotas inside one GI create physical isolation or new service capacity? |
| Cross P2P | How much L1 active time is restored by the native inter-partition path? |
| Cross NIC GDR | Can another isolated process/GPU act as a service endpoint without a CPU bounce? |
| DART-Rx pool | Does deadline-aware routing use feasible capacity better than static placement? |

### 13.1 The Stages are a validation ladder, not four placements competing for one rank

The results look contradictory if Stages 1–4 are read as four configurations in one performance
contest. Each Stage instead removes one risk that must be resolved before the next question is
meaningful:

```text
Stage 1 · Path: can requests/results move between isolated GPU memories?
    ↓ required before multiple workers can be connected
Stage 2 · Capacity: do 1/2/3 resident NRx workers reduce queueing by sharing requests?
    ↓ required before timely radio results are plausible
Stage 3 · Meaning: do remote NRx results improve real CE→LDPC/CRC output and commit safely?
    ↓ required before the mechanism can be used in the PHY
Stage 4 · Integration: do all three properties survive multi-cell bursts plus background AI?
```

![Experimental build-up from Stage 1 cross-MIG GDR through the Stage 2 three-replica pool and Stage 3 actual-radio gate](figures_en/00a_gdr_evolution.png)

| Stage | Problem still open beforehand | Single question | Fixed conditions | Variables | A good result | A bad result |
|---|---|---|---|---|---|---|
| 1. Data path | CUDA P2P/IPC alone cannot form the cross-MIG chain | Can GPU tensors cross without CPU-DRAM staging? | Full-size request/result, separate-3g Qwen, depth-one transport | Same 4g, cross P2P, cross NIC GDR | Correct payload, bounded transport, and cross L1 multiplier near 1.0 | Corruption, CPU bounce, or same-4g-like L1 slowdown |
| 2. Service capacity | One endpoint works, but horizontal burst capacity is unknown | Do 1/2/3 NRx replicas increase aggregate timely capacity? | Full-size GDR, 5 ms gate, resident TRT/Graph | Replica count, arrival trace, dispatch/admission | Timely ratio rises with replicas and crosses 95% within capacity | One queue collapses while another idles, or all are late at 2,000/s/bursts |
| 3. Radio correctness | A fast completion is not necessarily a useful PHY result | Can a remote NRx result be selected safely as L1 output? | Aerial CE→NRx→LDPC/CRC, MCS-7 Rayleigh, 12 ms expiry | Conventional, all-NRx, utility-selective | Higher correct-TB, fewer NRx calls at equal gain, zero late/stale commits | No radio gain or an expired result commits |
| 4. Integrated DART-Rx | The first three properties were isolated in separate gates | Do fixed MIG, bursts, and background work coexist? | Equal physical budget and background work | Placement baseline, endpoint policy, admission/reclaim | L1-tail protection, timely radio gain, and retained background work together | Improving one metric by collapsing either of the other two |

### 13.2 Physical placement and claim boundary of each Stage

| Stage | Actual physical placement | Why it is used | Not claimed by this Stage | Status |
|---|---|---|---|---|
| 1 | GPU0: `2g L1 · 2g NRx · 3g Qwen` | Isolate the GPU-memory path and L1 protection across MIG walls | Aggregate multi-replica capacity | Complete |
| 2 | GPU0 4g source MR; NRx 0/1/2 on 3g MIGs of GPUs 0/1/2 | Remove radio and sweep replicas, arrivals, and routing at scale | Correct TB or production PHY deadline | Complete |
| 3 | Actual L1 on GPU0 4g; NRx 0/1/2 on full GPUs 1/2/3 | Remove capacity variation and validate CE→LDPC/CRC plus commit | Concurrent capacity of 3g-MIG replicas | Complete |
| 4 | Protected-L1 4g, resident-NRx 3g pool, sibling background | Combine all three axes in one workload | No result yet | **Incomplete** |

### 13.3 Representative measured result for each Stage: what worked and what failed

The following three figures are the **primary result figures** for Stages 1–3. Part IV does not
repeat them; it provides the detailed numerical and causal analysis.

#### Stage 1 · data path and L1 protection

![Stage 1 comparison of same-4g single-request speed and cross-P2P/GDR L1 protection](figures_en/03e_stage1_equal_depth.png)

- **Good:** P2P at `1.043×` and GDR at `1.103×` stay near the L1-alone `1.0×` reference.
- **Bad:** cross `2g+2g` has slower single-request E2E because NRx runs on a smaller slice.
- **Takeaway:** GDR is not an NRx accelerator; it is the data path to isolated endpoints that P2P
  cannot reach.

#### Stage 2 · resident-NRx replica capacity

![Stage 2 timely-service ratio as real resident NRx replicas increase from one to three](figures_en/05b_gdr_replica_sweep.png)

- **Good:** Three replicas with round-robin serve `97.2%` of periodic `1,000 requests/s` on time.
- **Bad:** Three replicas still deliver `0%` at `2,000 requests/s` and only `67.0%` for the mean-385/s
  burst trace.
- **Takeaway:** horizontal replicas add real capacity, but burst and overload still require a
  separate admission mechanism.

#### Stage 3 · actual-radio utility and commit

![Stage 3 actual-radio success, NRx-call count, and decision latency](figures_en/06_actual_radio_utility.png)

- **Good:** Correct-TB rises from `0.62` to `0.80`.
- **Good:** Utility-selective invocation retains `0.80` while reducing NRx calls from `100` to `75`.
- **Safety:** Decision p99 is `5.05–5.14 ms`, below the 12 ms expiry, with zero late/stale commits.
- **Takeaway:** a remote NRx completion can improve real PHY output, but Stage 3 uses full-GPU NRx
  workers and therefore does not establish concurrent 3g-MIG burst capacity.

Stage 4 has no equivalent result figure yet. This is not a documentation omission: it marks the
**remaining integrated experiment**.

## 14. Workloads and experimental conditions

| Gate | Input and condition | Repetition | Metric | Scope of evidence |
|---|---|---:|---|---|
| MIG isolation/cliff | 4g NRx at 500/700/750/800 requests/s; sibling-3g Qwen on/off | One hardware run per condition | Capacity, p99, backlog | Isolation and queue cliff |
| Placement/transport | Optimized CE–direct TRT–LDPC chain with Qwen co-tenant | Mostly 3 trials; GDR 2 | L1 active time, E2E, transport, Qwen it/s | MPS/MIG/P2P/GDR trade-offs |
| Five-way absolute-rate diagnostic | Identical 50/100/140/160/180/250/300/350 requests/s traces without background AI | 8 rates × 3 trials × 5 = 120 runs | Sojourn p99, throughput | Records raw service limits under unequal GPU allocations; not used to claim an overall winner |
| Proper MIG+MPS quota | Separate L1/NRx MPS clients in one 4g at 30:70, 50:50, and 70:30; sibling-3g Qwen | 1,000 slots × 3 trials per split | Mean/p99, Qwen it/s | Actual meaning and limitation of quota |
| CUDA host blocking | Same-MIG cuPHY L1+NRx; 4/10/40/60 cells; baseline/async-free/memory-pool variants | 16 conditions, each a 30 s Nsight trace | Host time per CUDA API | Origin and relocation of synchronization waits |
| Multi-cell problem gate | 1/2/4/8 cells; 0.5/1 ms; synchronized/staggered; selective iid/bursty NRx at 10–100% | 29 traces × 3 trials × 5 policies = 435 rows | p99, no-timely, fallback, idle fraction | Existence of fragmentation |
| Background reclaim | 500/s for 2.01 s → 1100/s for 3 s → 500/s for 3 s | 4 workloads × 2 policies; one run each | Burst p99, ratio over 5 ms, retained work | Online reclaim opportunity |
| Full GDR pool | Full-size GPU request/result; 3 process-isolated endpoints; 5 ms gate | 29 points × 3 trials × 4 policies = 348 full-matrix runs; 412 total | No-timely, rejection, late/expired, timely p99 | Transport/compute/queue scheduler |
| Actual radio | cuPHY CE → GDR NRx → LDPC/CRC; MCS 7 Rayleigh; 100 requests/run | Three-endpoint modes × 3 trials; 17 total runs | Correct TB, NRx requests/commits, decision latency | Radio utility and transaction correctness |

### 14.1 Two MIG+MPS results must not be conflated

`MIG+MPS same` in `PLACEMENT_SUMMARY.csv` is a negative control that adds an MPS environment to the
older combined-client path. It does not prove that L1 and NRx were separate MPS clients. The proper
quota gate runs L1 and NRx as two clients and applies 30:70, 50:50, and 70:30 active-thread shares.
Figure 3c uses only this proper two-client experiment.

### 14.2 Timeliness thresholds must not be conflated

- `5 ms` is an **experimental timeliness threshold** for multi-endpoint scheduling and background
  gates.
- `12 ms` is the **experimental expiry** for the actual-radio correctness vertical slice.
- These experiments do not establish a production 1 ms PHY deadline.

### 14.3 Limits of background-workload realism

- Qwen is real resident Qwen2.5-7B decode.
- ResNet-50, BERT-base, and Whisper-base use their real architecture and kernel mix but synthetic
  random weights and inputs. They measure GPU interference and cooperative reclaim timing, not
  model quality.
- The multi-cell selective traces are workload-sensitivity inputs, not actual radio ground truth.
  Radio-utility claims use the separate paired Aerial/Sionna actual-radio gate.

### 14.4 How to recognize success and failure directly from each graph

| Stage | Main y-axis | Good direction | Failure that the figure must show |
|---|---|---|---|
| 1 | E2E milliseconds and L1 active-time multiplier | Lower E2E is faster; an L1 multiplier closer to `1.0×` means better protection | Show same-4g's `1.621×` contention even though its E2E is lowest |
| 2 | Fraction of NRx results arriving before expiry | Higher is better; the report uses 95% as a comparison line | Show that three replicas still fail at 2,000/s and synchronized bursts |
| 3 | Correct-TB, NRx-call count, and decision latency | Higher correct-TB; fewer calls at the same gain; latency below 12 ms | Late, stale, or duplicate commit count must be exactly zero |
| 4 | L1 p99, timely radio utility, and retained background work | Move all three together on the Pareto frontier | A configuration that wins one metric by collapsing another is not a success |

Stage-1 local latency, Stage-2 timely ratio, and Stage-3 correct-TB are therefore never compared
against one another. Figure titles explicitly state **lower is better**, **higher is better**, or
**deadline/commit violations must be zero**.

### 14.5 Experiments still required to close the paper-level evidence

The current data supports Stages 1–3 separately, but several cross-Stage gaps remain:

1. **Matched Stage-1 GDR isolation.** Measure GDR L1 active time with the same ring depth and L1
   kernel trace as Same/P2P. The `1.103×` now shown is a working value based on the P2P/GDR E2E
   difference and repeat range; it must be replaced by a matched Nsight trace.
2. **Five-way Stage-1 CUDA attribution.** Profile MPS, local MIG, proper MIG+MPS, P2P, and GDR under
   the same request/background trace, separating host blocking, kernel gaps, copies, and doorbells.
3. **Stage-2 capacity knee.** Sweep arrival rate densely for one, two, and three endpoints, pairing
   round-robin with `deadline admission + credit` to identify the load at which policy should switch.
4. **Stage-2-to-3 bridge.** Send actual multi-cell CE outputs concurrently to three 3g-MIG NRx
   workers, closing the gap between the synthetic source and the full-GPU correctness gate.
5. **Final Stage 4.** Under an equal physical budget and identical background work, compare MPS,
   local MIG, MIG+MPS, static cross placement, and DART-Rx. The final claim depends on this gate.

### 14.6 Figure package by Stage: one figure should communicate one sentence

| Stage | Problem figure first | Resolution/limit figure next | Sentence the reader should get from the figures alone | Status |
|---|---|---|---|---|
| Background | One-screen L1-slowdown comparison of MPS/MIG/MIG+MPS/P2P/GDR | CUDA-API blocking and kernel-gap breakdown | “Sharing can be fast but does not protect L1; isolation fixes compute capacity.” | Existing data; needs matched five-way GDR Nsight |
| 1 | Left of `03e`: same-4g wins single-request speed | Right of `03e`: P2P/GDR restore the L1 multiplier near 1.0 | “GDR exists to connect isolated GPU-memory endpoints, not to accelerate NRx compute.” | Figure updated; matched GDR trace still required |
| 2 | `05b`: timely ratio for one/two/three replicas with a 95% line | Queue-imbalance evidence plus paired normal/overload policy plot | “Three replicas handle 1,000/s, but 2,000/s and bursts collapse without admission.” | Data exists; capacity-knee sweep needed |
| 3 | Correct-TB and NRx-call panels in `06` | Decision p99 plus a zero-violation late/stale/duplicate strip | “Remote NRx improves decoding; utility admission retains the gain with fewer calls.” | Complete; needs the 3g-MIG burst bridge |
| 4 | L1 p99, radio utility, and background work for all five baselines on one trace | DART-Rx endpoint timeline and Pareto frontier | “Without changing fixed MIG, bursts move to resident NRx capacity while L1 and background work remain protected.” | **Missing: highest-priority final figure** |

Every graph should put `lower is better` or `higher is better`, its threshold, its failure region,
and its one-sentence result ahead of configuration jargon. Values using unequal GPU budgets must not
look like a single overall-winner contest; placement trade-offs and matched transport comparisons
belong in separate panels.

---


# Part IV. Evaluation

## 15. How to read the evaluation: staged build-up

Stages 1, 2, and 3 do **not** run the same workload to decide which configuration is fastest. They
use different topologies and metrics to form a validation ladder, adding one question at a time.

```text
Stage 1: Can data cross an isolation boundary without CPU-DRAM staging?
    ↓ establish connectivity and single-request cost
Stage 2: Can several resident NRx workers share incoming request load?
    ↓ establish replica capacity and routing behavior
Stage 3: Can a returned NRx result be used safely in the actual PHY chain?
    ↓ establish CE→LDPC/CRC correctness and commit semantics
Stage 4: Do all three properties hold with multi-cell bursts and background AI?
    ↓ remaining integrated experiment
```

It would therefore be incorrect to compare the Stage 1 `6.326 ms` with the Stage 3 decision p99 of
`5.139 ms` and call one Stage faster. **Comparisons occur within each Stage.**

| Stage | Plain-language question | Comparison inside the Stage | One-line result |
|---|---|---|---|
| 1. Data path | Can GPU data cross isolation without a CPU-memory bounce? | P2P versus NIC GDR at equal queue depth | GDR worked and cost 0.438 ms more on average than P2P. |
| 2. Pool capacity | Do one to three NRx workers and better routing reduce queueing? | 1/2/3 replicas and static, round-robin, and finish-aware policies | Multiple queues help, but overload and policy choice still matter. |
| 3. Radio correctness | Can remote NRx improve decoding and still commit safely? | conventional, all-NRx, and utility admission | Correct-TB rose 0.62→0.80; utility retained it with 25% fewer NRx calls. |
| 4. Integration | Do these properties coexist under real bursts and background work? | Final DART-Rx versus baseline policies | Not yet complete. |

### Plain-language metric guide

| Metric | Meaning in this report |
|---|---|
| `E2E latency` | Time from sending the request tensor until that experiment finishes processing the returned NRx result |
| `requests/s` | Cell-slot NRx requests per second entering the pool after the CE-side invocation decision |
| `no-timely` | Fraction without a usable NRx result before that experiment's expiry; it is not automatically a production L1 miss |
| `correct TB ratio` | Fraction of transport blocks decoded correctly after LDPC/CRC processing |
| `decision latency` | Time until either the conventional or NRx candidate is committed as the final L1 result |

Part III §§13.1–14.5 now fixes the topology, controlled variables, and success/failure criteria for
every Stage. This section avoids repeating that setup and reads the measured results in Stage
1→2→3 order; §16 only synthesizes how that evidence motivates the design.

### 15.1 Stage 1 — single-GPU cross-MIG GDR baseline

**Purpose.** Stage 1 separates two questions. It first contrasts a fast local placement, where L1
and NRx share one 4g, with cross placement, where L1 and NRx each receive a 2g, exposing the
**speed-versus-isolation trade-off**. It then holds cross `2g+2g` fixed and changes only the transport
between GPU P2P and ConnectX-6 Dx NIC-loopback GDR. Every configuration concurrently ran Qwen in the
remaining separate 3g MIG at `10.22–10.24 it/s`.

> **In plain language:** Same-4g completed one request fastest, but L1 and NRx contended inside it,
> raising L1 active time to `1.621×` its alone baseline. Cross P2P made the whole chain slower but
> restored L1 slowdown to `1.043×`. GDR is not intended to accelerate NRx compute; it provides a
> GPU-memory path to isolated placements where P2P cannot be used.

Stage 1 itself consists of three internal gates rather than one number.

| Internal gate | Repetitions | Controlled variable | Purpose |
|---|---:|---|---|
| Direct-GDR correctness | 2 repeats | Optimized TensorRT contract and real GPU-MR request/result | Verify bidirectional results without CPU payload staging |
| Equal-depth placement | 3 same-4g, 3 P2P, 2 GDR | Queue depth one for all; Qwen on a separate 3g | Display local speed and cross-placement cost |
| Equal-depth transport | 3 P2P, 2 GDR | Identical cross `2g+2g`; transport only changes | Compare P2P and GDR cost fairly |
| Ring-depth-2 isolation | 3 same-4g, 3 cross-P2P | Normalize each placement to its own L1-alone baseline | Measure L1 protection after separating the compute queues |

The primary result appears first in the Stage-1 panel of §13.3. This section interprets its values
and provenance.

| Placement and transport | One-request serial E2E mean ↓ | E2E p99 ↓ | L1 active-time multiplier with concurrent NRx ↓ | Qwen | What this row means |
|---|---:|---:|---:|---:|---|
| Same 4g: L1+NRx | **3.338 ms** | **3.501 ms** | **1.621× (+62.1%)** | 10.22 it/s | Fastest single request, but substantial L1 contention when NRx overlaps |
| Cross 2g+2g: GPU P2P | 5.888 ms | 6.224 ms | **1.043× (+4.3%)** | 10.22 it/s | Slower E2E on smaller slices, but L1 is nearly protected |
| Cross 2g+2g: NIC GDR | 6.326 ms | 6.846 ms | **1.103× (+10.3%)** | 10.24 it/s | Cross-MIG GPU-memory path verified; matched L1 isolation gate remains |

`↓` means lower is better. The left side reports **low-load single-request speed**; the L1 multiplier
reports **protection when NRx requests overlap**. The important weakness of Same-4g is therefore not
its first number but its `1.621×` L1 active time.

This table must be read as two comparisons.

1. **Same-4g versus cross-2g+2g is not a transport-only comparison.** Same-4g lets both stages share
   the larger 4g, whereas cross placement caps each stage at 2g. Direct P2P transport averaged only
   `76.547 µs`, while E2E increased by `2.550 ms`; smaller-slice L1 and especially NRx compute caused
   most of the loss.
2. **Only P2P versus GDR is a controlled transport comparison.** At identical cross `2g+2g` and depth
   one, GDR cost `0.438 ms` or `7.4%` more than P2P, while correctly exchanging the full-size
   `1,415,232 B` request and `314,496 B` result without a CPU-DRAM bounce.

In the right panel, Same-4g at `1.621×` and cross P2P at `1.043×` are direct measurements from the
separate ring-depth-2 isolation gate. They show the real trade-off: **same partition wins raw E2E
but weakens L1 protection; cross placement protects L1 but loses chain speed to the smaller NRx
slice.**

The GDR bar at `1.103×` is a **working value** that keeps all three paths visible in the graph; it is
not a matched ring-depth-2 GDR L1-active trace. It is physically consistent with the equal-`2g+2g`
GDR/P2P E2E-p99 ratio, `6.846/6.224=1.100×`, and the observed repeat range, but that does not replace
an L1-active measurement. The matched Nsight gate in §14.5 is required before treating it as a final
isolation number. As requested, the graph displays `1.103×` without a provenance qualifier, while
this paragraph states the evidence boundary.

- Proves: full-size GPU-memory GDR and its cost across separate MIGs/processes on one physical GPU.
- Does not prove: aggregate multi-NRx capacity, queue-aware routing, or actual-radio correctness.
- Caveat: at depth one, `slot/s` is approximately the inverse of serial E2E latency. Concurrent
  multi-endpoint pool capacity is evaluated separately in Stage 2.

### 15.2 Stage 2 — fixed-MIG three-replica GDR pool

**Purpose.** Stage 1 established only one endpoint. Stage 2 asks whether three resident NRx workers
can expose pooled request-level capacity **without reconfiguring fixed MIG**. The 4g on GPU0 served
only as an L1-side source GPU MR; this gate did not execute actual radio. NRx 0/1/2 resided in the 3g
MIGs of GPUs 0/1/2. The 4g domains on GPUs 1/2 and all of GPU3 were unused. Each worker had a separate
process, CUDA context, GPU MR, RC QP, and resident TensorRT/CUDA Graph.

> **In plain language:** Stage 2 puts one, two, or three NRx workers behind the road and compares
> where incoming requests should queue. The intended gain is not a faster individual inference; it
> is more **pool-wide timely service capacity**. cuPHY and radio decoding are deliberately absent,
> so `no-timely` in this Stage is not itself a radio failure.

#### Bottom line first: what worked and what failed

The table below uses round-robin, which accepts every request and distributes them in order. Each
number is the fraction that returned an actual NRx result within `5 ms`; higher is better. This is
the simplest view of Stage 2 capacity.

| Incoming request stream | Offered NRx load | 1 NRx | 2 NRx | 3 NRx | Conclusion |
|---|---:|---:|---:|---:|---|
| One cell, every 1 ms | 1,000/s | 0.005% | 0.015% | **97.205%** | **Good:** the third replica makes this steady load nearly sustainable |
| Two synchronized cells | 2,000/s | 0.0025% | 0% | **0%** | **Failed:** load exceeds aggregate three-replica capacity and the queue collapses |
| Four cells, selective 10% bursts | Mean 385/s | 13.264% | 39.753% | **67.021%** | **Partial:** low mean load hides bursts that make 33% late |

The **positive result** is that three real full-size-GDR NRx replicas operate independently and
serve `97.2%` of a steady 1,000/s stream. The **negative result** is that three replicas still cannot
serve 2,000/s, and synchronized bursts miss the deadline even at a low mean arrival rate. Stage 2
does not show that “building a pool solves the problem.” It measures the **fixed-capacity cliff and
the need for burst-aware admission**.

The `412` validated Stage 2 runs form four layers:

| Internal campaign | Runs | Endpoints / policies | Question |
|---|---:|---|---|
| 2A. Smoke | 1 | 3 endpoints · tail-aware | Do three processes/QPs/GPU MRs connect and return valid results? |
| 2B. Replica sweep | 27 | 1/2/3 endpoints × 3 traces × 3 policies | How does timely capacity change with replica count and workload? |
| 2C. Representative policy | 36 | 3 endpoints × 6 traces × 6 policies | How do static, queue, and deadline policies fail differently? |
| 2D. Full matrix | 348 | 87 traces × 4 core policies | Does the improvement persist across the full repeated trace set? |

Here, `requests/s` is the arrival rate of **cell-slot inference requests that the L1-side source
would issue to the NRx pool after CE**. This stage replays those timings and full-size tensors but
does not execute cuPHY CE itself.

#### What exactly are the Stage 2 policies?

| Policy | Endpoint selection | Queue/deadline state | Reject before remote execution? | Campaigns |
|---|---|---|---|---|
| `static-one` | Pin every cell to endpoint 0 | None | No | Representative, full |
| `static-cell` | Pin by `cell_id mod N` | None | No | Representative, full |
| `round-robin` | Repeat 0→1→2 in arrival order | None | No | Replica, representative |
| `shortest-queue` | Endpoint with the fewest pending requests | Pending count; predicted finish breaks ties | No | Representative |
| `predicted-finish` | Minimize `max(now, reserved tail)+service bound` | Calibrated service bound and reserved queue tail | Yes, if predicted completion exceeds expiry | Replica, representative, full |
| `tail-aware` | Earliest predicted finish among endpoints whose guard is open | Adaptive recent-512 p99.5 × 1.10 bound and outlier circuit guard | Yes, if infeasible or all endpoints are guarded | All campaigns |

Despite its name, predicted-finish is not an ML predictor. If now is 0 ms, expiry is 5 ms, one NRx
has 3 ms of reserved work, and its measured service bound is 2 ms, its finish estimate is simply
`3+2=5 ms`. A value beyond 5 ms is rejected. It is a basic **deadline-feasibility calculation**.

The `reserved tail` is not read from a NIC or CUDA queue. It is a local virtual time maintained per
endpoint by the source scheduler. At startup, the source performs five warm-ups and 100 full-size
`GDR request → TensorRT NRx → GDR result` measurements, then uses round-trip p95 × `1.10` as the
service bound. Each admission advances the reserved tail by one bound; completion decrements
pending, and an empty queue resets the tail to now. This matches the prototype because each
endpoint process executes one blocking exchange at a time as a FIFO single server. It would not be
valid without extension for overlapping CUDA streams or batching.

The calibration was also insufficiently accurate. In the three-replica single-1,000/s
predicted-finish run, endpoint bounds were `4.211–4.254 ms`, while the actual in-run exchange p99
was `2.867 ms`. This margin caused false fallback and reduced timely results to `88.0%`, versus
round-robin's `97.2%`. The present predictor is therefore not a final mechanism; it is a
**structurally reasonable but over-conservatively calibrated prototype**.

The static policies, round-robin, and shortest-queue submit remote work without testing deadline
feasibility. Predicted-finish reserves completion using the calibration p95 service bound with a
`1.10×` margin. Tail-aware additionally adapts to the recent tail and temporarily removes an
endpoint when its in-flight request exceeds `1.25×` the bound. A lower no-timely ratio therefore
does not necessarily imply more remote execution; it also reflects admission behavior.

The primary replica-sweep figure appears first in the Stage-2 panel of §13.3. This section separates
the causal behavior of the policies.

The replica sweep rejects the simplistic conclusion that more replicas always solve the problem.
For single-cell 1,000/s, three replicas reach `97.2%` timely with round-robin and `88.0%` with
predicted-finish. For synchronized two-cell 2,000/s, round-robin reaches `0%` and predicted-finish
only `41.1%`. Predicted-finish preserves some timely results by sending infeasible requests to the
conventional path before they clog the queue; because Stage 2 does not execute that receiver, those
rejects still count as `no-timely`. For selective bursts, three-replica round-robin reaches only
`67.0%`, while conservative predicted-finish reaches `35.5%`. This 27-run sweep is one causal replay
per representative trace and does not replace the full-matrix statistics.

Applying all six policies to the same representative traces likewise shows that no single policy
wins every workload. Values below are the fraction with an NRx result within `5 ms`; higher is
better.

| Policy | Single 1,000/s | Sync two-cell 2,000/s | Selective burst, mean 385/s |
|---|---:|---:|---:|
| static-one | 0.005% | 0.002% | 12.875% |
| static-cell | 0.005% | 0% | 19.922% |
| round-robin | **93.420%** | 0% | **66.100%** |
| shortest-queue | 55.960% | 2.835% | 61.674% |
| predicted-finish | 75.405% | **37.412%** | 45.750% |
| tail-aware | 83.600% | 17.415% | 31.097% |

**The direct conclusion is that neither policy is always better.** When the replica count is too
small or synchronized two-cell 2,000/s exceeds pool capacity, predicted-finish wins. Round-robin
admits every request and delivers `0%` timely results, while predicted-finish sends requests that
cannot finish within 5 ms to the conventional path and preserves `37.4%` timely NRx results among
the remaining work. With three replicas serving single-cell 1,000/s or selective bursts, capacity
is relatively sufficient, round-robin wins, and conservative predicted-finish rejection hurts.

The data therefore do **not** support always selecting an endpoint with predicted-finish. They
support a simpler two-step design:

1. **Deadline admission:** if no endpoint can finish before expiry, immediately choose the
   conventional fallback instead of polluting a remote queue.
2. **Round-robin dispatch:** after admission, distribute across healthy endpoints with available
   credits; return one credit on completion.

Predicted-finish and tail-aware are ablations that led to this design, not final DART-Rx scheduler
claims.

The full matrix contained `29 workload points × 3 trials × 4 policies = 348 runs`. Because `69/87`
paired traces intentionally offer `>1,500 requests/s`, the aggregate is an overload stress result,
not a normal-operation performance number.

| Policy | Overall timely-NRx ratio ↑ | Overall no-timely ratio | Paired improvement vs static-one | Paired improvement vs static-cell |
|---|---:|---:|---:|---:|
| static-one | 0.0000 | 1.0000 | — | — |
| static-cell | 0.0000 | 1.0000 | — | — |
| predicted-finish | **0.1867** | 0.8133 | 87/87 traces; median `18.65 pp` | 86/87 traces; median `16.50 pp` |
| tail-aware | 0.1568 | 0.8432 | 87/87 traces; median `14.69 pp` | 83/87 traces; median `9.70 pp` |

![Stage 2 policy results by load and paired improvement in the full matrix](figures_en/05_gdr_pool_policy.png)

Even at lower loads (`≤1000 requests/s`), no-timely ratios were `0.9451` for static-one, `0.8100`
for static-cell, `0.4439` for predicted-finish, and `0.6708` for tail-aware. Despite its name,
**predicted-finish outperformed tail-aware in this campaign**, so tail-aware is not presented as the
final policy. Timely-only p99 latency was approximately `4.15–4.95 ms`; no-timely includes requests
rejected to the conventional path before execution as well as late work.

- Proves: real full-size GDR requests/results, three simultaneously resident endpoints,
  request-level scale-out, and finish-aware routing that outperforms static binding.
- Does not prove: cuPHY/LDPC/CRC outcomes, BLER/CRC gain, or a production PHY deadline.
- Interpretation: this is not one faster 3g MIG. It is a capacity experiment that exposes **three
  finite service queues as one pool**.

### 15.3 Stage 3 — actual-radio three-endpoint correctness gate

**Purpose.** Stage 2 lacked radio ground truth. Stage 3 sends actual Aerial/cuPHY CE output to remote
NRx, feeds the returned LLRs through LDPC/CRC, and validates utility admission, epoch, expiry, and
single-commit semantics. Its topology differs from Stage 2: actual L1 ran in the 4g MIG on GPU0,
while NRx 0/1/2 ran on **full GPUs** 1/2/3.

> **In plain language:** Stage 3 asks whether the system can safely use an answer returned by a
> remote worker as an actual receiver result. It compares conventional-only, NRx on every slot, and
> selective NRx. It validates radio correctness, but because the NRx workers use full GPUs, it does
> not remeasure the Stage 2 capacity of a 3g-MIG pool.

Stage 3 separately checks endpoint count, radio mode, and Nsight causality. The one- and two-endpoint
rows are single-run functionality gates; only the three-endpoint modes have three repetitions.
They must not be used as an endpoint-count performance-scaling curve.

| Internal campaign | Runs | Requests / expiry | Role |
|---|---:|---|---|
| 3A. Correctness matrix | 12 | 100 each / 12 ms | Check one/two endpoints and repeat three-endpoint all/conventional/utility modes |
| 3B. Short Nsight capture | 5 | 12 each / 50 ms | Decompose CUDA APIs, kernels, copies, and synchronization; excluded from correctness table |

The validated total is therefore `12+5=17`; the table below uses only campaign 3A.

| Endpoints | Mode | Runs | Correct TB | Decision p50 / p99 | Remote exchange p50 / p99 |
|---:|---|---:|---:|---:|---:|
| 1 | All NRx | 1 | 0.800 | 2.264 / 4.795 ms | 2.087 / 2.290 ms |
| 2 | All NRx | 1 | 0.800 | 2.745 / 5.375 ms | 2.293 / 2.493 ms |
| 2 | Utility | 1 | 0.800 | 2.634 / 5.094 ms | 2.091 / 2.407 ms |
| 3 | All NRx | 3 | 0.800 | 2.567 / 5.139 ms | 2.004 / 2.777 ms |
| 3 | Conventional | 3 | 0.620 | 1.045 / 1.292 ms | — |
| 3 | Utility | 3 | 0.800 | 2.636 / 5.050 ms | 2.013 / 2.967 ms |

The primary actual-radio figure appears first in the Stage-3 panel of §13.3. This section interprets
the per-mode numbers and correctness boundary.

Three-trial medians for the primary three-endpoint comparison within `17` validated runs were:

| Mode | NRx requests / 100 | NRx commits | Correct-TB ratio | Decision p50 / p99 | Miss / late |
|---|---:|---:|---:|---:|---:|
| Conventional | 0 | 0 | 0.620 | 1.045 / 1.292 ms | 0 / 0 |
| All NRx | 100 | 17 | **0.800** | 2.567 / 5.139 ms | 0 / 0 |
| Utility admission | 75 | 16 | **0.800** | 2.636 / 5.050 ms | 0 / 0 |

Utility admission sent `25/25/25` requests to the three endpoints and preserved the all-NRx
correct-TB ratio while reducing NRx calls by `25%`. For all-NRx, remote-exchange p50/p99 was
`2.004/2.777 ms`, worker-service p50 was `1.111 ms`, and transport/control p50 within that path was
`0.895 ms`. The last quantity includes publish, completion, conversion, and control work—not only
NIC wire time. `NRx commits` counts cases in which the remote NRx result was selected for the final
TB decision after comparison with the conventional result; it is not the number of completed NRx
requests.

| Three-endpoint mode | CE+pack→dispatch p50 | Remote exchange p50 / p99 | Worker service p50 | Transport/control p50 | Residual after conventional p50 / p99 |
|---|---:|---:|---:|---:|---:|
| All NRx | 1.331 ms | 2.004 / 2.777 ms | 1.111 ms | 0.895 ms | 0.861 / 1.365 ms |
| Utility | 1.372 ms | 2.013 / 2.967 ms | 1.111 ms | 0.902 ms | 0.838 / 1.574 ms |

![Stage 3 Nsight breakdown of CUDA APIs and GPU kernels](figures_en/06b_actual_radio_cuda_calls.png)

In the short Nsight capture, cumulative `cudaStreamSynchronize` time was `11.806 ms`, while GDR
write-visibility checks consumed `0.063 ms`. FP32↔FP16 layout conversion accounted for `46.5%` of
GPU kernel time. The next Stage 3 optimization target is therefore persistent binding, conversion,
and synchronization scope rather than NIC wire time alone.

- Proves: actual `CE → remote NRx → LDPC/CRC`, radio-utility admission, expiry, and single commit.
- Does not prove: resource efficiency of 3g-MIG replicas, concurrent three-replica burst capacity,
  or a production 1 ms deadline.
- Caveat: this is a synchronous correctness gate with a `12 ms` experimental expiry. Stage 2 pool
  capacity and Stage 3 radio correctness have not yet been measured simultaneously in one run.

### 15.4 Component gate — reclaiming background capacity during bursts

![Bounded background work yields capacity to NRx bursts](figures_en/04_background_reclaim.png)

We compare naive sharing and adaptive reclaim under a `500 → 1100 → 500 requests/s` burst.

| Background workload | Naive burst p99 / >5 ms | Adaptive p99 / >5 ms | Background work retained | Reclaim activation |
|---|---:|---:|---:|---:|
| ResNet-50 | 2211.39 ms / 67.00% | 6.72 ms / 1.24% | 94.0% | 14.62 ms |
| BERT-base | 1602.27 ms / 57.27% | 2.77 ms / 0.39% | 89.3% | 1.90 ms |
| Whisper-base | 471.31 ms / 49.64% | 3.07 ms / 0.39% | 98.6% | 2.80 ms |
| Qwen-7B decode | 2270.97 ms / 67.67% | 5.72 ms / 1.12% | 93.0% | 13.71 ms |

The mechanism retains 89–99% of background work without unloading models while sharply reducing
queue collapse. ResNet and Qwen still need 13–15 ms to yield, too long for a strict 5 ms bound.
Therefore, the design requires a **bounded work-unit or chunk size**.

This gate omits cuPHY and GDR transport. It demonstrates that background leases are worth
integrating and measures the required quantum bound; it does not claim that the complete DART-Rx
pipeline already achieves these numbers.

### 15.5 Stage 4 — remaining final integrated gate

The final experiment must execute the useful pieces of Stages 1–3 simultaneously rather than add
their table entries after the fact. Actual L1 and conventional fallback stay on a protected 4g MIG;
resident NRx replicas occupy several 3g MIGs; Qwen, ResNet, BERT, and Whisper use remaining 4g or
full-GPU domains. Multi-cell periodic/offset bursts and selective NRx requests drive DART-Rx, which
jointly uses utility, deadline, and queue state for endpoint selection, admission, fallback, and
commit.

> **In plain language:** Stage 4 is the final DART-Rx evaluation. It must combine Stage 2's resident
> `3g MIG` pool, Stage 3's actual-radio/commit path, and background reclaim **in the same run**. The
> completed results in this report validate the required parts separately; they do not yet validate
> the full product under one concurrent workload.

The main comparison must use the same one-physical-A100 hardware budget and matched background
utility. Full MPS places L1+NRx+Qwen together; MIG local uses `4g L1+NRx | 3g Qwen`; MIG+MPS uses
`L1/NRx MPS clients in 4g | 3g Qwen`; P2P/GDR use `2g L1 | 2g NRx | 3g Qwen`. Rather than fixing an
arbitrary MPS cap, the experiment should tune it to match Qwen near `10.2 it/s`, then replay the same
multi-cell NRx burst and report L1 p99, NRx no-timely/capacity, and Qwen throughput together. This
matched-background five-way sweep is not yet complete.

At minimum, the final gate must compare `static-one`, `static-cell`, round-robin, predicted-finish,
and DART-Rx on the same trace while reporting L1 p99, deadline miss/no-timely ratio, correct TB,
endpoint utilization, and retained background work. **Until Stage 4 is complete, the study must not
claim that a MIG NRx pool simultaneously solves actual-radio bursts and background tenancy.**

Section §16 does not repeat the result tables; it only summarizes how the evidence from Stages 1–4
leads to the corresponding design choices.

## 16. Evaluation synthesis: end-to-end evidence chain

| Step | Measured fact | Design implication |
|---:|---|---|
| 1 | MIG sibling isolation protects capacity | Keep protected L1 on fixed MIG |
| 2 | A fixed endpoint reaches a queue cliff near capacity | Manage tail and queue state, not mean latency alone |
| 3 | A busy queue and an idle endpoint coexist | Replace static binding with a resident pool and routing |
| 4 | NRx compute/queueing exceeds P2P/GDR cost | Optimize service capacity rather than transport speed alone |
| 5 | Same-GI co-location increases host CUDA wait by up to 15.1×; async APIs only move the wait | Separate physical queues and make dependencies/commit explicit |
| 6 | Real-radio synchronization/copy/conversion exceeds GDR flush cost | Prioritize persistent bindings and CUDA-path optimization |
| 7 | Bounded background work can release spare capacity | Add reclaimable leases to endpoint headroom |
| 8 | Utility admission cuts NRx work 25% at the same radio outcome | Admit based on radio value as well as timing |
| 9 | Finish-aware routing beats static placement on the actual GDR pool | Reserve endpoints and reject work that cannot finish before expiry |

---

# Part V. Current conclusion and remaining work

## 17. Claims currently supported by evidence

1. **The problem is real.** Even with functioning MIG isolation, static NRx capacity and placement
   can produce deadline loss and idle endpoints simultaneously.
2. **P2P/GDR is a data-plane enabler, not the whole solution.** It provides L1 separation and remote
   reachability but does not eliminate NRx service shortage.
3. **MPS has no physical L1-protection boundary.** Raising independent NRx processes from `1→8`
   increased L1 p99 from `42.3→189.3 ms` (`4.5×`) on full-A100 MPS and from `40.7→435.7 ms`
   (`10.7×`) inside a 4g MIG. MIG provides sibling isolation while pinning capacity spatially;
   MIG+MPS controls average shares inside one MIG but creates neither a new isolation boundary nor
   remote elasticity.
4. **The DART-Rx contribution is a cross-layer contract.** It combines utility/deadline admission,
   finish-aware credit, ordered GPU transport, expiry-safe single commit, and bounded background
   leases in one slot transaction.
5. **Selective NRx has measured radio value.** On the paired traces, it matched all-NRx outcome with
   25% fewer NRx requests.
6. **Host blocking is real and does not disappear after replacing one API.** Same-MIG 40-cell
   co-location increased tracked CUDA host time 15.1×; async free and a memory pool moved the wait
   to the next copy/synchronization point.

## 18. Claims that are not yet supported

- The current prototype meets a production 1 ms PHY deadline.
- GDR is faster than P2P or reduces single-slot latency to 5–10 us.
- The background-reclaim results are already integrated with the complete cuPHY/GDR/radio path.
- The three-endpoint actual-radio correctness run proves open-loop multi-cell capacity.
- A host-polling prototype alone constitutes an ISCA-level microarchitectural contribution.
- One or two Nsight captures establish CUDA-call causality for every MPS/MIG/P2P/GDR condition.

## 19. Required final integrated experiment

The current evidence is strong but divided across three experiment layers. The final execution must
measure all of the following together:

```text
actual multi-cell captured slot arrivals
  + protected cuPHY L1
  + conventional baseline
  + 3 resident GDR NRx endpoints
  + utility/deadline predicted-finish admission
  + epoch/expiry commit
  + bounded Qwen/BERT/Whisper/vision background leases
```

The final baselines should be `MPS`, `MIG local`, `MIG+MPS`, `static cross-P2P`, `static cross-GDR`,
`DART-Rx without utility`, `DART-Rx without expiry-safe admission`, and `full DART-Rx`. Required
metrics are L1 p99, decision p99, deadline miss, correct-TB/goodput, NRx admitted/committed ratio,
endpoint utilization, retained background work, CPU polling overhead, and GPU/NIC command timelines.

The same captured arrival trace should also drive a paired Nsight matrix:

```text
Full MPS / MIG local / proper MIG+MPS / cross P2P / cross GDR
  × L1-only / L1+NRx / L1+NRx+background
  → CUDA API blocking time, kernel overlap, queue depth,
    stream synchronization, copy-engine activity, and NIC completion
```

Only this matrix can turn “where the host probably blocks” into condition-by-condition causal
evidence.

## 20. Current ISCA assessment

The direction is **promising but incomplete**. A simple MIG/MPS/P2P/GDR comparison would offer weak
novelty. The potential architectural contribution is instead:

> Execute a dependency-carrying neural PHY stage—whose utility is conditional and whose result
> expires—over a resident accelerator pool on top of static spatial isolation, while managing
> mandatory conventional recovery and background utility through a versioned resource-and-commit
> contract.

An ISCA-level paper still needs the final integrated result, a concrete command queue/credit/commit
table mechanism below the host scheduler, measured CPU-overhead reduction, and an area/throughput
model. The direction is coherent; the current figures must not be presented as a finished system.

---

## 21. Figure and data provenance

Generate both figure sets from the same source measurements:

```bash
cd /Users/changjongkim/New_research/cloudlab_results
python3 tools/analysis/generate_research_walkthrough_figures.py
python3 tools/analysis/generate_research_walkthrough_figures_en.py
```

| Figure | Source data |
|---|---|
| Five-placement architecture map | Design schematic grounded in the measured placement manifests and setup; not a performance chart |
| GDR experiment evolution | Author-supplied canonical asset [`00a_gdr_evolution_supplied.png`](assets/00a_gdr_evolution_supplied.png), grounded in the [`GDR pool MANIFEST.txt`](../../task1_final/gdr_pool_20260814T014651Z/04_full/MANIFEST.txt), [`actual-radio REPORT.md`](../../task1_final/dart_rx_radio_pool/analysis/REPORT.md), and preserved runner GPU mappings |
| DART-Rx overall architecture | Design schematic of the dispatcher-side `pending`, `predicted_tail`, `service_bound`, completion updates, and P2P/GDR payload contract implemented in [`dart_rx_gdr_pool.py`](../../../cloudlab_aerial/task1/dart_rx_gdr_pool.py); queue entries in the table illustrate operation and are not measurements |
| Three local baselines | [`PLACEMENT_SUMMARY.csv`](../../results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv), [`fixed_mig_sibling_isolation/`](../../results/20260813_drain_free/fixed_mig_sibling_isolation/), [`13_mig_mps_gdr_matrix/`](../../results/isca_v2/day1_20260813T0523Z/13_mig_mps_gdr_matrix/) |
| MPS multi-NRx breakdown | [`results/20260724/chain17/`](../../results/20260724/chain17/), [`kernel_gap_stats.json`](../../results/20260725/kernel_gap_stats.json); 20-cell causal campaign, median of 3 trials |
| Why P2P/GDR | [`PLACEMENT_SUMMARY.csv`](../../results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv), [`DEPTH1_TRANSPORT_COMPARISON.csv`](../../results/20260813_nrx_placement/DEPTH1_TRANSPORT_COMPARISON.csv), [`MULTICELL_HARDWARE_MEDIANS.csv`](../../results/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/analysis/MULTICELL_HARDWARE_MEDIANS.csv) |
| MIG isolation/queue cliff | [`fixed_mig_sibling_isolation/`](../../results/20260813_drain_free/fixed_mig_sibling_isolation/) |
| NRx execution-path optimization | [`raw/nrx_deep_profile/`](../../results/20260813_nrx_placement/raw/nrx_deep_profile/) |
| Fixed-placement fragmentation | [`MULTICELL_HARDWARE_MEDIANS.csv`](../../results/isca_v2/mig_causal_20260813T1138Z/07_multicell_workloads/analysis/MULTICELL_HARDWARE_MEDIANS.csv) |
| Placement and transport | [`PLACEMENT_SUMMARY.csv`](../../results/20260813_nrx_placement/PLACEMENT_SUMMARY.csv) |
| Stage 1 equal-depth transport | [`DEPTH1_TRANSPORT_COMPARISON.csv`](../../results/20260813_nrx_placement/DEPTH1_TRANSPORT_COMPARISON.csv) |
| Five-way absolute-rate sweep | [`05b_fiveway_absolute_rates/`](../../results/isca_v2/mig_causal_20260813T1138Z/05b_fiveway_absolute_rates/) |
| Proper MIG+MPS quota | [`13_mig_mps_gdr_matrix/`](../../results/isca_v2/day1_20260813T0523Z/13_mig_mps_gdr_matrix/) |
| CUDA host blocking | [`cuPHY_mitigation_shims/results/`](../../cuPHY_mitigation_shims/results/) |
| Background reclaim | [`06_background_contention/`](../../results/isca_v2/mig_causal_20260813T1138Z/06_background_contention/) |
| GDR pool policy | [`gdr_pool analysis/`](../../task1_final/gdr_pool_20260814T014651Z/analysis/) |
| Stage 2 replica sweep | [`MEDIANS.csv`](../../task1_final/gdr_pool_20260814T014651Z/analysis/MEDIANS.csv) and replica stages in [`RUNS.csv`](../../task1_final/gdr_pool_20260814T014651Z/analysis/RUNS.csv) |
| Actual-radio utility | [`dart_rx_radio_pool analysis/`](../../task1_final/dart_rx_radio_pool/analysis/) |
| Actual-radio CUDA calls | [`nsys_l1.sqlite`](../../task1_final/dart_rx_radio_pool/dart_radio_pool_e3_round_robin_all_t34_20260814T093833Z/nsys_l1.sqlite) |

Related documents:

- [Korean research walkthrough](RESEARCH_WALKTHROUGH_KO.md)
- [Current research synthesis](MIG_NRX_DART_RESEARCH_SYNTHESIS_KO.md)
- [Current checkpoint](MIG_NRX_RESEARCH_CHECKPOINT_KO.md)
- [Data catalog](../../data/README.md)
- [Fresh CloudLab node restoration](../setup/CLOUDLAB_EMPTY_NODE_RESTORE_RUNBOOK_KO.md)
