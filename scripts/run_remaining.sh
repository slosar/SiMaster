#!/bin/bash
# Continuation chain: val3 (dev+flat variants), val4, SHT scaling, MC
# comparison, report.  val1/val2 results already on disk.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/anze/anaconda3/envs/simaster/bin/python
LOG=../report/data
mkdir -p "$LOG"

run() {
  name=$1; shift
  echo "=== $name start $(date) ==="
  stdbuf -oL -eL $PY "$@" > "$LOG/$name.log" 2>&1
  rc=$?
  echo "=== $name done rc=$rc $(date) ==="
  tail -3 "$LOG/$name.log"
}

run val3      val3_lss.py --lmax 64
run val4      val4_iterate.py
run bench_sht bench_sht_scaling.py
run val1_mc   val1_cmb.py --lmax 64 --sigma-T 50 --fisher mc --nsims 8192 --tag val1_mc
run report    build_report.py
echo "ALL DONE $(date)"
