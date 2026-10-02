# Backstop (slot-scale scheme, v7 results): paper draft

Draft of the paper for the slot-scale Backstop scheme with the larger NVlabs neural receiver as the
recovery receiver. It does not replace `paper/softwall_sigmetrics27`, which describes the earlier scheme.

- `main.tex`: 10-page `acmart` draft (abstract, introduction, receiver comparison, design, evaluation,
  related work, limitations, conclusion)
- `references.bib`: the bibliography of `paper/softwall_sigmetrics27` plus `nrxrt` and `neuralrxsw`
  (taken from the README of `third_party/neural_rx`)
- `figures/architecture.tex`: overall architecture (TikZ, style in `figures/tikz_style.tex`), with the values
  of the run shown in the timeline figure
- `figures/timeline_one_gpu.pdf`: measured timeline of one GPU, drawn by
  `scripts_for_node/backstop_slot/plot_design_timeline.py`
- other `figures/*.pdf`: drawn by `scripts_for_node/backstop_slot/plot_v11.py` and `plot_v7.py` with
  `SCHEME_NAME=Backstop`

Build:

```bash
module load texlive/2024
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Every number is taken from `docs/current/BACKSTOP_V4_VERIFICATION_KO.md` (sections 2, 5.7, 6.1-6.14).
The submission audits of `paper/softwall_sigmetrics27` do not apply to
this draft.
