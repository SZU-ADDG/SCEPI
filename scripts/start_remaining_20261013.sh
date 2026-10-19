set -euo pipefail
cd /data3/doomx/RECOMB-3-Epistasis
source scripts/runtime.sh
TASK_PYTHON=/data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python
test ! -e runs/mean_path_completion_20261013.log
test ! -e runs/kermut_Call_20261013.log
test ! -e runs/mavenn_nojit_20261013.log
nohup "$TASK_PYTHON" scripts/complete_mean_evaluations.py > runs/mean_path_completion_20261013.log 2>&1 < /dev/null &
echo "mean_evaluations_pid=$!"
nohup "$TASK_PYTHON" scripts/run_independent_workers.py --name kermut_Call_20261013 --gpu-memory-gib 6 scripts/run_kermut_targets.py --mode C --conditions Call > runs/kermut_Call_20261013.log 2>&1 < /dev/null &
echo "kermut_Call_pid=$!"
nohup "$TASK_PYTHON" scripts/run_independent_workers.py --name mavenn_nojit_20261013 --gpu-memory-gib 12 scripts/run_mochi_targets.py --method mavenn --dataset functional_doubles --mode C --order 2 --device cuda > runs/mavenn_nojit_20261013.log 2>&1 < /dev/null &
echo "mavenn_pid=$!"
