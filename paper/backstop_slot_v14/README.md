# Antiphase (slot-scale scheme, v14 results): paper draft

Draft of the paper for the slot-scale Antiphase scheme with the v14 rule (stoppable low-priority AI pieces,
no AI on a GPU whose neural receiver runs, AI unit sizes next to the conventional receivers by measured
bounds) and the recovery-loss model. `paper/backstop_slot_v7` keeps the v13 draft.

- `main.tex`: `acmart` (acmsmall) draft. Sections: introduction, background (receiver comparison), recovery
  path, recovery-loss model, AI admission, evaluation (model validation, full load, changing load, rules,
  distance to a clairvoyant schedule, server sizes, kinds of AI work, goodput, link adaptation at a fixed MCS
  and with an outer loop), related work, limitations, conclusion.
- `references.bib`: the bibliography of `paper/backstop_slot_v7` plus `bge`, `vit`, `harcholbalter`
  (checked against Crossref, arXiv, and DataCite on 2026-10-04) and `eolla` (outer-loop link adaptation;
  checked against the publisher page on 2026-10-04).
- `figures/architecture.tex`: overall architecture (TikZ, style in `figures/tikz_style.tex`).
- `figures/*.pdf`: drawn by `scripts_for_node/backstop_slot/plot_v14.py` with `SCHEME_NAME=Antiphase`
  (`bash figures.sh`).

Build:

```bash
bash figures.sh
module load texlive/2024
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The numbers are taken from `README.md` (sections 6.1-6.13) and
`docs/current/BACKSTOP_V14_MODEL_AND_EXTENSIONS_KO.md`.
