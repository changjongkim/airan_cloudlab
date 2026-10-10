#!/bin/bash
# Builds the slide-only figures (TikZ, the style of the design figures).  usage: bash build.sh [NAME ...]
cd "$(dirname "$0")"
export PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:$PATH
list="$@"; [ -z "$list" ] && list=$(ls n*.tex | sed 's/\.tex$//')
for n in $list; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone.tex}" > "$n.build.log" 2>&1 || { grep -A6 '^!' "$n.build.log" | head -14; echo "FAILED $n"; continue; }
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile="../$n.png" "$n.pdf"
  rm -f "$n.aux" "$n.log" "$n.build.log"
  echo "built $n.png $(file ../$n.png | sed -E 's/.*PNG image data, ([0-9]+ x [0-9]+).*/\1/')"
done
