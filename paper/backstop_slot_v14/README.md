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
- `figures/*.tex`: design figures in TikZ (style in `figures/tikz_style.tex`): `architecture.tex` (overall
  architecture, all four GPUs, with a small content in every component: one TB State row, the assigner
  check, one grant per GPU, the request queue, and per GPU the AI request, the TB on the neural receiver, and
  the cells), `recovery_path.tex` (the recovery path as a six-step procedure, Receive to Report, with three
  failed TBs of cell 0, slots 2731-2733; the copy over CUDA IPC runs below the steps, and TB State and the
  two deadlines are at the bottom), `hold.tex` (four neural receivers on a time axis, alone and next to AI,
  with the start window of every failed TB and the measured loss), `admission.tex` (the decision of AI
  Admission, and per GPU the grant and the 30 units of the AI request at the instant of that recovery).
  Every TB keeps one color in all figures (`tba`, `gold`, `recc`, `tbd` in `tikz_style.tex`), values come
  from one measured run, and every arrow runs in a gap between boxes. Text inside the figures is at least
  7 pt at the text width. Heights at the text width: 8.5, 5.5, 4.9, and 6.2 cm.
- Typography of the design figures (stated at the top of `figures/tikz_style.tex`): bold only for the names
  of boxes (layers, components, GPUs, steps) and for the one or two key values of a figure; everything inside
  a box is regular; labels of arrows and notes are italic and never bold. Arrows are at least 0.35 cm long
  (0.25 cm with the small head inside a box), and no text or arrow crosses the border of a box it does not
  belong to. The README images are drawn at four times the resolution and scaled down (300 dpi); the
  alpha-bits options of Ghostscript break the hatched fills.
  `standalone.tex` compiles one of them alone; `figures.sh` exports them as PNG for the README.
- `figures/eval_*.pdf`: evaluation figures drawn by `scripts_for_node/backstop_slot/plot_eval.py` (one metric
  per panel, one color per policy; the number on a baseline is the factor of Antiphase over it). The other
  PDF figures come from `plot_paper.py` and `plot_v14.py` (`bash figures.sh`).

Build:

```bash
bash figures.sh
module load texlive/2024
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The numbers are taken from `README.md` (sections 6.1-6.13) and
`docs/current/BACKSTOP_V14_MODEL_AND_EXTENSIONS_KO.md`.
