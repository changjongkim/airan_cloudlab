# Backstop (slot-scale scheme, v13 results): paper draft

Draft of the paper for the slot-scale Backstop scheme with the larger NVlabs neural receiver as the
recovery receiver. It does not replace `paper/softwall_sigmetrics27`, which describes the earlier scheme.

- `main.tex`: 14-page `acmart` draft (abstract, introduction, receiver comparison, design, evaluation,
  related work, limitations, conclusion). Since 2026-10-02 the AI admission rule is the v13 rule (low-priority
  AI, free-receiver rule) and the evaluation is the v13 sweeps. `main_v12_backup.tex` and
  `figures/architecture_v12_backup.tex` keep the previous text.
- `references.bib`: the bibliography of `paper/softwall_sigmetrics27` plus `nrxrt`, `neuralrxsw`
  (from the README of `third_party/neural_rx`) and `orion` (checked against Crossref)
- `figures/architecture.tex`: overall architecture (TikZ, style in `figures/tikz_style.tex`)
- `figures/timeline_server.pdf`: measured timeline of the four GPUs, drawn by
  `scripts_for_node/backstop_slot/plot_reserve_timeline.py`
- `figures/ai_load.pdf`, `figures/changing_load.pdf`, `figures/step_load.pdf`: drawn by
  `scripts_for_node/backstop_slot/plot_v13.py` (`aiload`, `frontier`, `series`) with `SCHEME_NAME=Backstop`
- `figures/timeline_one_gpu.pdf`, `changing_load_16cells.pdf`, `changing_load_32cells.pdf`,
  `steady_low_correlation.pdf`: figures of the v12 text, not used now

Build:

```bash
module load texlive/2024
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The receiver numbers (Section 2, "Choosing the TBs to recover") are taken from
`docs/current/BACKSTOP_V4_VERIFICATION_KO.md` (sections 2, 6.1-6.8). The scheduling numbers (Sections 3.3,
3.4, 4) are taken from `docs/current/BACKSTOP_V13_SWEEPS_KO.md`. The submission audits of
`paper/softwall_sigmetrics27` do not apply to this draft.
