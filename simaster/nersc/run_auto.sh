#!/bin/bash
# Reference NERSC sbatch wrapper: run the whole pilot->MC->combine MC-Fisher
# workflow as ONE multi-node job (run_auto fans each phase out with srun).
# Allocate >= max(len(k_grid), n_seeds) nodes.  Env: PROBLEM NSIDE OUTDIR [NSIMS NSEEDS K OMP FEXACT]
#SBATCH -A m4895
#SBATCH -C cpu
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -J mcfisher
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate simaster
python -m simaster.nersc.run_auto --problem "$PROBLEM" --nside "$NSIDE" \
   --outdir "$OUTDIR" --nsims "${NSIMS:-512}" --n-seeds "${NSEEDS:-6}" \
   --k "${K:-auto}" --omp "${OMP:-128}" ${FEXACT:+--f-exact "$FEXACT"}
