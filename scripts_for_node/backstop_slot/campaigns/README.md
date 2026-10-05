# Campaign scripts

Copies of the campaign scripts in `run_state/backstop_slot/` (that directory is not tracked). The documents
in `docs/current/` name them by their `run_state/backstop_slot/` path; run them from there, inside an
allocation, as `srun --jobid=<job> --overlap -N1 -n1 --gpus-per-node=4 bash run_state/backstop_slot/<script>`.

v14 (2026-10-02 to 04):

| script | what it runs |
|---|---|
| `v14.sh`, `v14b.sh` | rule comparison with the 128-slot pool; estimator calibration (superseded by `v14c.sh`) |
| `v14c.sh` | 512-slot pool: estimator calibration (`yyrcal`), baselines of the three load patterns (`full`, `alt`, `steps`) |
| `v14d.sh`, `v14e.sh` | other server sizes (`w8`, `w26`, `gpu1`, `gpu2`); rule variants (`conv`: unit sizes next to the conventional receiver) |
| `v14_headline1.sh` | driver for the baselines of the three load patterns (sets the estimator files) |
| `v14_final.sh` | the final rule (policy code `wm`) in every condition |
| `v15.sh`, `v15_samejob.sh`, `classes5/` | several kinds of AI work: unit bounds (`bench`), mixes, every policy in one job |
| `la.sh`, `la2.sh`, `la_more.sh` | link adaptation: both receivers at MCS 10-16 and fixed Es/No |
| `la_cl_gen.sh` | closed-loop link adaptation: slots of MCS 10-16 for one channel and Es/No (one ring per MCS level) |
| `la_cl.sh`, `la_cl2.sh` | closed-loop link adaptation: one target, seeds, and policies per call (`la_cl2.sh` adds `xpP`, `wrR`, and the `EXTRA` options) |
| `la_cl_chain.sh` ... `la_cl_chain4.sh`, `la_cl_chain6.sh` | the closed-loop runs of jobs 59345188 and 59362400 (targets 10 / 3 / 1%, two GPUs, one GPU, eight two-user cells, outer-loop step 0.5) |
| `v14g.sh`, `v14g_chain.sh` | conditions that had only v13 results: AI load, AI bursts, one GPU seeds 3-5, 32 / 48 / 20 cells (`wm` there picks the unit classes by active cells) |
| `v14h.sh`, `v14h_chain.sh`, `v14h_chain_b.sh` | rule variants by active cells (`wa`, `wb`, `we`; `wd` = the final rule for any number of cells per GPU), calibration with two and three cells per GPU (`actcal`), 20 cells with the 7.6 ms bound (`dense76`), alternation intervals and cell bursts |
| `v14_report.sh` | tables and figures from the raw runs (login node) |
| `v14_chain2.sh`, `v14_chain3.sh` | step dispatchers used during the runs |

