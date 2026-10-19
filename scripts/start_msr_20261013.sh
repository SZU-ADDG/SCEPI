set -euo pipefail
cd /data3/doomx/RECOMB-3-Epistasis
source scripts/runtime.sh
test ! -e runs/msr_source_nonoverlap_20261013.log
nohup /data1/doomx/anaconda3/envs/recomb-3-epistasis/bin/python scripts/complete_msr_source.py > runs/msr_source_nonoverlap_20261013.log 2>&1 < /dev/null &
echo "msr_source_controller_pid=$!"
