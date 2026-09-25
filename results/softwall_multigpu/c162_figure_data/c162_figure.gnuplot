if (term eq 'svg') set terminal svg size 1250,500 enhanced font 'Arial,12'
if (term eq 'pdf') set terminal pdfcairo size 12.5in,5in enhanced font 'Arial,12'
if (term eq 'png') set terminal pngcairo size 1875,750 enhanced font 'Arial,16'
set output out
set multiplot layout 1,2 margins 0.07,0.98,0.14,0.91 spacing 0.11,0.03
set key top right opaque
set grid ytics xtics lc rgb '#dddddd'
set title '(a) Certified AI-admission frontier (accepted debt = 4)'
set xlabel 'Decision time after release (ms)'
set ylabel 'Unresolved recovery debts'
set xrange [40:120]
set yrange [-0.25:4.25]
set ytics 0,1,4
plot 'results/softwall_multigpu/c162_figure_data/frontier.tsv' using (int($1)==64?$3:1/0):2 with linespoints lw 2 pt 7 ps 0.7 lc rgb '#0072B2' title 'context 64', \
+'results/softwall_multigpu/c162_figure_data/frontier.tsv' using (int($1)==128?$3:1/0):2 with linespoints lw 2 pt 7 ps 0.7 lc rgb '#009E73' title 'context 128', \
+'results/softwall_multigpu/c162_figure_data/frontier.tsv' using (int($1)==256?$3:1/0):2 with linespoints lw 2 pt 7 ps 0.7 lc rgb '#E69F00' title 'context 256', \
+'results/softwall_multigpu/c162_figure_data/frontier.tsv' using (int($1)==512?$3:1/0):2 with linespoints lw 2 pt 7 ps 0.7 lc rgb '#CC79A7' title 'context 512', \
     'results/softwall_multigpu/c162_figure_data/physical_safe.tsv' using ($1-0.2):2 with points pt 7 ps 1.3 lc rgb '#0072B2' title 'physical admit', \
     'results/softwall_multigpu/c162_figure_data/physical_reject.tsv' using ($1+0.2):2 with points pt 5 ps 1.5 lw 2 lc rgb '#D55E00' title 'physical reject', \
     'results/softwall_multigpu/c162_figure_data/physical_safe.tsv' using ($1-0.2):2:3 with labels offset char 0.6,0.7 tc rgb '#333333' notitle, \
     'results/softwall_multigpu/c162_figure_data/physical_reject.tsv' using ($1+0.2):2:3 with labels offset char 0.6,-0.8 tc rgb '#333333' notitle
unset logscale x
set title '(b) Certified scheduler and verifier latency'
set xlabel 'Recovery debts'
set ylabel 'Latency (ms)'
set xrange [0.8:80]
set yrange [0:5.25]
set logscale x 2
set xtics ('1' 1,'2' 2,'4' 4,'8' 8,'16' 16,'32' 32,'64' 64)
set key top left opaque
plot 'results/softwall_multigpu/c162_figure_data/latency.tsv' using 1:2 with linespoints lw 2 pt 7 lc rgb '#0072B2' title 'scheduler p99', \
     'results/softwall_multigpu/c162_figure_data/latency.tsv' using 1:3 with linespoints lw 2 pt 5 lc rgb '#D55E00' title 'scheduler max', \
     'results/softwall_multigpu/c162_figure_data/latency.tsv' using 1:4 with linespoints lw 2 pt 9 lc rgb '#009E73' title 'verifier p99', \
     5 with lines dt 2 lw 2 lc rgb '#555555' title '5 ms control budget'
unset multiplot
