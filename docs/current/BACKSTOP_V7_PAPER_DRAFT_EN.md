# Backstop: draft text for the paper (v7 results)

Status: draft, 2026-10-01. Not merged into `paper/softwall_sigmetrics27`. Every number comes from
[BACKSTOP_V4_VERIFICATION_KO.md](BACKSTOP_V4_VERIFICATION_KO.md) (sections in brackets). Citation keys in
brackets are placeholders; add a reference only after checking the source.

## Abstract

AI-RAN servers run the uplink physical layer and AI inference on the same GPUs. Neural receivers decode
transport blocks (TBs) that the conventional receiver fails, yet their value and their cost on a shared GPU
are not established. We compare public neural receivers with the cuPHY receiver at the same LDPC iteration
count and find that the real-time model helps only when the channels of paired users are correlated, while a
larger model decodes 15-67% of the failed TBs on every channel we test. The larger model takes 5.3 ms per
slot, twice the uplink period. This paper presents Backstop, which runs the larger model only for failed TBs
under a later deadline and grants AI inference in short pieces that keep both the layer-1 deadline and the
recovery deadline. On four A100 GPUs with 16 cells, Backstop recovers 63-66% of the failed TBs. Under a load
that changes every two seconds, Backstop serves 27.4k tokens/s while keeping 98.6% of the recoveries, 5.0x a
fixed GPU share and 1.17x a share that follows the load, at the same recovery level.

## Introduction, fourth paragraph

In this paper, we present Backstop, a GPU sharing scheme for AI-RAN servers that uses a neural receiver as a
second receiver for the TBs the conventional receiver fails. Specifically, Backstop introduces 1) a recovery
path that sends a failed TB to the neural receiver only when the number of failed code blocks predicts a
successful decode, 2) two deadlines per TB, the layer-1 deadline for the conventional receiver and a later
recovery deadline before the retransmission is scheduled, and 3) AI admission in bounded pieces, where the
controller allows a piece only if every running and waiting radio job still meets its deadline. Our
evaluation on four A100 GPUs with 16-32 cells, the cuPHY receiver, the public NVlabs neural receivers and
Qwen2.5 prefill requests shows that Backstop recovers 63-66% of the TBs the conventional receiver fails.
With mixed channels and a load that alternates between full and half every two seconds, Backstop serves
27.4k tokens/s within the 200 ms limit and keeps 98.6% of the recoveries. A fixed 10% GPU share keeps the
same recoveries at 5.5k tokens/s, and a share that switches between 10% and 50% with the load, with no
switching cost, reaches 23.5k tokens/s.

## Section: When does a neural receiver help? [2장, 3장]

**Same LDPC effort.** A receiver comparison is valid only when both receivers run the same number of LDPC
iterations. cuPHY uses 10 iterations for the TB sizes in our setup, and the evaluation code of the public
neural receivers uses 20 [NVlabs neural_rx]. With 10 against 20 iterations, the real-time model appears to
decode 25-35% of the TBs that cuPHY fails on the low-correlation channel. With 20 iterations on both sides,
the figure is 2%. Raising cuPHY from 10 to 20 iterations alone recovers 152 of 158 failed single-user TBs,
at 15-19% more receiver time for TBs of 13-20 code blocks. Therefore, every result below uses 20 iterations
in both receivers.

| Channel (two users, 16QAM, 273 PRBs) | TBs | cuPHY | Real-time model (1.35 ms) | Larger model (4.29 ms) |
|---|---|---|---|---|
| Low correlation, 2-3 dB | 600 | 515 | 484, decodes 2% of cuPHY failures | 572, decodes 67% |
| UMi, 1-6 dB | 600 | 425 | 420, 2% | 451, 15% |
| Medium correlation, 8-18 dB | 600 | 321 | 372, 21% | not measured |
| High correlation, 10-16 dB | 1,400 | 946 | 1,099, 38% | 1,159, 51% |

**The model that helps is too slow for every slot.** The larger model decodes more TBs than cuPHY on all
three channels and loses none of the TBs cuPHY decodes on two of them. It takes 5.3 ms per slot alone and
6.3 ms next to the cell workload, against an uplink period of 2.5 ms. Running it for every slot of four
cells leaves 61% of the slots unprocessed and decodes 89.4% of the TBs. Running it only for the TBs cuPHY
fails needs 25% of the runs, gives the same decode result offline, and decodes 94.5-95.1% in the timed
system [6.1, 6.8]. This motivates Backstop's recovery path.

**Failed code blocks predict the outcome.** On the UMi channel, 87% of the failed TBs have every code block
failed and the neural receiver recovers 2% of them. TBs with at most nine failed code blocks are recovered
in 93% of the cases. Sending only those TBs removes 85% of the neural receiver runs and loses 5% of the
recoveries [6.7].

## Section: Evaluation, main comparison [6.12-6.14]

**Setup.** One node with four A100 80 GB GPUs under MPS. Sixteen cells, four per GPU. One cell per GPU
carries two users on the same PRBs, drawn from three channels (low correlation, high correlation, UMi); the
other cells carry one strong rank-2 user. Both receivers run 20 LDPC iterations and the neural receiver is the
larger NVlabs model. Uplink period 2.5 ms, layer-1 deadline 4.0 ms, recovery deadline
11.5 ms. AI requests are Qwen2.5-1.5B prefill requests with a 200 ms limit, placed on the GPUs by one
dispatcher. The radio load alternates every two seconds between full load and each cell active with
probability 0.5. Runs last 20 s; values are means of five seeds, ranges in parentheses.

**Baselines.** *Fixed share* gives the AI worker a fixed MPS share. *Share that follows the load* keeps one
AI worker per share loaded on every GPU and activates the low share during full load and the high share
otherwise. It has no switching cost and hands waiting requests over at a switch; we run it with the load
known at once and known one second late, the minimum reconfiguration interval of [YinYangRAN]. All policies
use the same neural receiver rules, the same AI requests and the same admission of AI requests.

| Policy | AI tokens/s within 200 ms | Recoveries kept (of the run without AI) |
|---|---|---|
| Backstop | 27.4k (26.0-28.5k) | 98.6% (97.6-99.3%) |
| Fixed 10% | 5.5k (5.3-5.6k) | 98.8% (97.4-99.8%) |
| Fixed 30% | 22.8k (21.9-23.5k) | 96.7% (95.5-98.6%) |
| Share follows load, 10/30%, load known at once | 13.5k (12.6-14.6k) | 98.7% (97.7-99.6%) |
| Share follows load, 10/50%, load known at once | 23.5k (22.9-24.6k) | 98.4% (97.3-99.3%) |
| Share follows load, 10/30%, load known 1 s late | 11.7k (11.5-11.8k) | 98.0% (96.6-99.2%) |

Figure X shows AI served against recoveries kept. Backstop serves 27.4k tokens/s and keeps 98.6% of the
recoveries. The fixed 10% share keeps 98.8% and serves 5.5k tokens/s, 5.0x less. The fixed 30% share serves
22.8k tokens/s and loses 3.3% of the recoveries, 2.4x more than Backstop. The share that follows the load
serves 13.5-23.5k tokens/s at the same recovery level, 1.17-2.0x less than Backstop, and 11.7k tokens/s when
it sees the load one second late. Backstop achieves this by granting AI during full load whenever no
neural receiver job runs on that GPU: it serves 18.8k tokens/s in the full-load phases, while the share
that follows the load drops to its 10% share there and serves 2.4-2.7k tokens/s.

**Steady full load and 32 cells.** At a steady full load with the same mixed channels, Backstop serves 22.7k
tokens/s and keeps 97.9% of the recoveries. A fixed 10% share keeps 98.3% at 5.0k tokens/s, and a fixed 30%
share serves 20.9k tokens/s and keeps 95.1% (five seeds) [6.14]. With 32 cells and a load that alternates
between half and a quarter, Backstop serves 26.4k tokens/s at 99.4%, against 5.3k for the fixed 10% share and
22.7k for the share that follows the load (two seeds) [6.14].

**Where sharing by a fixed share is enough.** With the real-time model (2.7 ms per recovery) and the 11.5 ms
recovery deadline, a fixed 70% share loses 1.5% of the recoveries and serves more AI than Backstop [5.7].
With the UMi channel and the failed-code-block rule, 5% of the slots need the neural receiver and a fixed
50% share serves 46.0k tokens/s against 33.6k for Backstop at the same 99.8% [6.7]. Backstop's advantage
needs a neural receiver load that fills a large part of the GPU.

## Limitations (for the discussion section)

- Channels are the synthetic channels of the public NVlabs configuration. A larger model at full load
  allows four to five cells per GPU; 32 cells run when half of the cells are idle.
- The recovery deadline follows from the TDD pattern and the PUSCH preparation time of TS 38.214; it is not
  validated with a layer-2 stack. An 11.5 ms deadline delays the retransmission of an unrecovered TB by 5 ms.
- The share that follows the load is our implementation with two shares and no load predictor.
- Layer-1 misses are 0.02-0.05% of the TBs for every policy at these settings, a property of the Python
  prototype.
