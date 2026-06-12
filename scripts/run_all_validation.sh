#!/bin/bash
# Full validation chain at nside=32 (hours; run in background).
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

run bench    bench_feasibility.py
run val1     val1_cmb.py
run val2     val1_cmb.py --ivar-maker val2_ivar.py --tag val2
run val3     val3_lss.py
run val4     val4_iterate.py
run val1_mc  val1_cmb.py --fisher mc --nsims 8192 --tag val1_mc
run report   build_report.py
echo "ALL DONE $(date)"
