#!/bin/bash
# Builds the Antiphase design figures of the second style: src/design2_<name>.tex (with src/tikz_style2.tex, the
# look of the ScaleQsim paper) -> <name>.png here.   Usage: build_v2.sh [NAME ...]   (no name: all)
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:$PATH
if [ $# -gt 0 ]; then list=$(for n in "$@"; do echo "src/design2_$n.tex"; done); else list=$(ls src/design2_*.tex); fi
for t in $list; do
  n=$(basename "$t" .tex); out=${n#design2_}
  ( cd src && pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone2.tex}" > "$n.build.log" 2>&1 ) || { grep -A6 '^!' "src/$n.build.log" | head -20; exit 1; }
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile="$out.png" "src/$n.pdf"
  rm -f "src/$n.aux" "src/$n.log" "src/$n.build.log"
  echo "built $out.png"
done
