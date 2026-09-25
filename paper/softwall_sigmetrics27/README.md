# SoftWall SIGMETRICS 2027 review manuscript

This directory contains the anonymous `acmart` review manuscript.

- `main.tex`: paper source
- `references.bib`: normalized bibliography
- `main.pdf`: compiled 15-page review PDF, including references

Build from this directory on the project environment:

```bash
module load texlive/2024
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Run the claim and submission audits from the repository root:

```bash
python3 scripts_for_node/softwall_same_gpu/analyze_softwall_necessity_witness.py
/usr/bin/python3.11 scripts_for_node/softwall_same_gpu/audit_softwall_manuscript_claims.py
python3 scripts_for_node/softwall_same_gpu/audit_sigmetrics_submission.py
```

The PDF is claim-scoped: five lifecycle subsets are qualified or partially qualified and five remain
unqualified. It makes no WCET, production-HARQ, cross-family, or throughput-superior optimizer claim.
