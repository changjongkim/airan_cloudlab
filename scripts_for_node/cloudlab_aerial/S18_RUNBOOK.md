# §18 AI-side NSYS decomposition — CloudLab runbook

Goal: produce AI-process NSYS traces that mirror §17 (which only had cuPHY L1).
Two AI workloads × three placements = six conditions, each producing one AI-side
sqlite + one L1-side sqlite for cross-check.

Cluster: CloudLab d8545 (A100 ×4), AIRANSLICING project (Wisconsin).

---

## 0. Preflight (5 min)

On d8545 head node:

```bash
# Confirm container image present
docker images | grep airan:25-3-final            # expect 1 row

# Confirm cuBB SDK mount
ls /mydata/aerial-cuda-accelerated-ran/pyaerial/models | head   # expect *.onnx

# Confirm GPU 0 is idle and not stuck in MIG mode that blocks reset
nvidia-smi -i 0 --query-gpu=mig.mode.current,memory.used --format=csv
# Expect: Enabled,0 MiB  (any nonzero used means leftover container — clean it)

# If leftover containers exist, kill them first
docker ps -q | xargs -r docker kill

# Pull latest from repo (this script + analyzer):
cd ~/cloudlab_aerial && git pull
ls -la s18_dual_capture.sh                       # must be executable
```

If anything fails, **stop here** and resolve before continuing.

---

## 1. Sanity check — single dual capture (5 min)

Before running all 6 scenarios, run one alone to confirm the dual-nsys flow works:

```bash
DATE_DIR="$(date +%Y%m%d)" \
WARMUP_S=15 CAPTURE_S=30 \
bash s18_dual_capture.sh 2>&1 | tee ~/cloudlab_aerial/results/$(date +%Y%m%d)/s18_dryrun.log
```

Hit **Ctrl-C after X1 completes** (first scenario, ~1.5 min). Verify:

```bash
DATE_DIR=$(date +%Y%m%d)
ls -lh ~/cloudlab_aerial/results/$DATE_DIR/s18_ai_nsys/X1_*
# Expect: X1_neuralrx_alone_3g_AI.nsys-rep AND .sqlite (both nonzero)

# Confirm AI-side has AI kernels (not cuPHY)
sqlite3 ~/cloudlab_aerial/results/$DATE_DIR/s18_ai_nsys/X1_neuralrx_alone_3g_AI.sqlite \
  "SELECT s.value, COUNT(*) c
     FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
     GROUP BY s.value ORDER BY c DESC LIMIT 8;"
```

**Verification gate**: top-8 kernel names must NOT include `convert_kernel`,
`chEstFilterNoDftSOfdmDispatchKernel`, `noiseIntfEstNoDftSOfdmKernel`
(those are cuPHY L1, which would mean nsys attached to the wrong process).
For NeuralRx expect TRT engine kernels (`trt_*`, `gemm_*`, `cudnn_*`,
`elementwise_*`). For chanpred expect cupy/numpy GPU kernel names.

If gate fails → check `--gpus` UUID assignment; the AI container may be
inheriting all-GPU mode.

---

## 2. Full run (≈50 min)

```bash
DATE_DIR="$(date +%Y%m%d)" \
bash s18_dual_capture.sh 2>&1 \
  | tee ~/cloudlab_aerial/results/$(date +%Y%m%d)/s18_full.log
```

Expected per-scenario timing:

| Scenario              | Stage           | ≈ wall time |
|-----------------------|------------------|-------------|
| X1 NeuralRx alone     | warmup+capture  | 60 s |
| X2 NeuralRx cross     | warmup+L1+capture | 75 s |
| X3 NeuralRx coloc     | warmup+L1+capture | 75 s |
| X4 chanpred alone     | warmup+capture  | 60 s |
| X5 chanpred cross     | warmup+L1+capture | 75 s |
| X6 chanpred coloc     | warmup+L1+capture | 75 s |
| MIG reconfig (×2)     | nvidia-smi mig  | 2 × 15 s |
| Total                 |                  | ≈ 7.5 min profile + overhead |

In practice overhead from container start, sqlite export, and warmup pushes
total to ~45–55 min. Budget 60 min.

---

## 3. Verification gates after full run

For each AI sqlite, check the basics:

```bash
DATE_DIR=$(date +%Y%m%d)
OUT=~/cloudlab_aerial/results/$DATE_DIR/s18_ai_nsys
for f in $OUT/X*_AI.sqlite; do
  echo "--- $(basename $f) ---"
  sqlite3 "$f" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
  sqlite3 "$f" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY"
  sqlite3 "$f" "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
done
```

**Gate A — every AI sqlite has > 0 kernels.** Empty = nsys missed the process.
The fix is to re-run with longer `WARMUP_S` (process didn't reach the capture window).

**Gate B — coloc cudaFree > alone cudaFree (NeuralRx).** If our queue hypothesis
holds AI-side too:

```bash
for cond in X1_neuralrx_alone_3g X3_neuralrx_L1_coloc_3g; do
  echo "--- $cond ---"
  sqlite3 $OUT/${cond}_AI.sqlite "
    SELECT s.value, COUNT(*), SUM(r.end-r.start)/1e6 ms
      FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
      WHERE s.value='cudaFree' GROUP BY s.value"
done
```

Expected: X3 (coloc) cudaFree total ms >> X1 (alone) cudaFree total ms.
If X3 ≤ X1 the AI-side queue hypothesis is wrong (still publishable — it would
mean L1 is the asymmetric victim and AI is shielded by L1's host-blocking).

**Gate C — chanpred shows smaller delta than NeuralRx.** Compare cudaFree fold-
change for NeuralRx (X1 vs X3) versus chanpred (X4 vs X6). If chanpred shows the
same inflation, then PHY-AI specificity is wrong (also publishable, but reshapes
the §11 / §2.2 story).

Document the actual numbers in `s18_observations.md` even if hypotheses fail.

---

## 4. Rsync to laptop and render figures

From laptop:

```bash
DATE_DIR=YYYYMMDD            # the same date used on CloudLab
rsync -avz d8545:cloudlab_aerial/results/$DATE_DIR/s18_ai_nsys/ \
  /Users/changjongkim/New_research/cloudlab_results/results/$DATE_DIR/s18_ai_nsys/

cd /Users/changjongkim/New_research/cloudlab_results/results/visual_evidence
S18_DATE_DIR=$DATE_DIR python3 build_time_breakdown_ai.py
```

Produces four figures in `figures/`:

| File | Mirrors §17 figure | Question it answers |
|---|---|---|
| `fig_s18_01_ai_gpu_activity_decomposition.png` | §17.1 | Does AI also show idle-gap dominance under coloc? |
| `fig_s18_02_ai_kernel_stages.png` | §17.2 | Which AI model component grows? |
| `fig_s18_03_ai_runtime_api.png` | §17.3 | Does AI-side cudaFree explode like L1's? |
| `fig_s18_04_ai_normalized_wallclock.png` | §17.4 | % shift per condition |

---

## 5. Deck integration (after figures look right)

Add four slides to `build_progress_slides_20260620.py` after the existing §17
block (current slides 3–6), so the §18 block sits at positions 7–10 and the
mechanism/verification/MPS slides shift down. Or build a separate v6 deck.

Talking points for §18:
- §18.1 vs §17.1: AI's idle gap pattern (symmetric victim or asymmetric story)
- §18.3: cudaFree fold-change AI-side vs L1-side — if both grow, mechanism is
  bidirectional (queue affects both); if only L1 grows, queue is asymmetric
- §18.2: why NeuralRx specifically — kernel-class composition vs chanpred
- §18.4: normalized view — how time ratios shift across placements

---

## Failure modes and fallbacks

- **Coloc UUID rejected by docker** (`X3`, `X6`): two containers attached to the
  same MIG UUID may fail in some driver configs. Fallback: enable MPS for the
  coloc scenarios. Edit X3/X6 in the script to set `CUDA_MPS_PIPE_DIRECTORY` and
  start MPS daemon before the dual capture.

- **nsys version doesn't know `--delay/--duration`**: use older syntax
  `--capture-range=cudaProfilerApi` and wrap AI/L1 work with cudaProfilerStart/Stop.
  Realistically Aerial 25-3 nsys supports `--delay/--duration` — verify with
  `docker run --rm $IMAGE nsys --version` (expect 2024.x+).

- **AI process dies during nsys**: NeuralRx TRT engine build can take >15s on
  first invocation. Increase `WARMUP_S=30` and `AI_TOTAL_S` accordingly.

- **MIG reconfig blocked by running container**: kill all containers before each
  `mig_create` call. The script already does `kill_all_bg` between scenarios via
  the AI container `--rm` flag, but a hung L1 container can block. If stuck:
  `docker ps -q | xargs -r docker kill && sleep 2 && bash s18_dual_capture.sh`.
