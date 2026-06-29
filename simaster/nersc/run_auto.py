"""Driver: run the semi-automatic MC-Fisher workflow as ONE NERSC SLURM job.

Invoked *inside* an sbatch allocation (see ``run_auto.sh``); uses
:class:`SlurmScheduler` to ``srun``-fan each phase across the allocated nodes::

    python -m simaster.nersc.run_auto --problem mymod --nside 128 --outdir DIR
"""
import argparse

from ..fisher_auto import run_auto
from .slurm import SlurmScheduler


def main(argv=None):
    P = argparse.ArgumentParser()
    P.add_argument("--problem", required=True, help="module with build_workspace()")
    P.add_argument("--nside", type=int, required=True)
    P.add_argument("--outdir", required=True)
    P.add_argument("--nsims", type=int, default=512)
    P.add_argument("--n-seeds", type=int, default=6)
    P.add_argument("--k", default="auto", help="'auto' or an integer deflation k")
    P.add_argument("--k-grid", default="200,400,800,1600")
    P.add_argument("--n-holdout", type=int, default=100)
    P.add_argument("--cg-tol", type=float, default=1e-2)
    P.add_argument("--omp", type=int, default=128)
    P.add_argument("--f-exact", default=None)
    a = P.parse_args(argv)
    k = a.k if a.k == "auto" else int(a.k)
    sched = SlurmScheduler(omp_threads=a.omp)
    run_auto(sched, a.problem, a.nside, a.outdir, nsims=a.nsims, n_seeds=a.n_seeds,
             k=k, k_grid=tuple(int(x) for x in a.k_grid.split(",")),
             n_holdout=a.n_holdout, cg_tol=a.cg_tol, F_exact_path=a.f_exact)


if __name__ == "__main__":
    main()
