# Campaign scripts

Copies of the campaign scripts in `run_state/backstop_slot/` (that directory is not tracked). The documents
in `docs/current/` name them by their `run_state/backstop_slot/` path; run them from there, inside an
allocation, as `srun --jobid=<job> --overlap -N1 -n1 --gpus-per-node=4 bash run_state/backstop_slot/<script>`.
