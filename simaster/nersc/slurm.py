"""SLURM Scheduler for NERSC Perlmutter -- the only HPC-specific piece.

Fans ranks out with ``srun`` *inside the current allocation*, so an entire
pilot -> MC -> combine workflow runs as ONE SLURM job (the user sbatch's a thin
script -- see ``run_auto.sh`` -- that allocates N nodes and calls
``simaster.fisher_auto.run_auto`` with this scheduler).  Each phase uses as many
of the allocated nodes as it has ranks; ranks index work via ``$SLURM_PROCID``.

This matches the project rule: trivially-parallel work is a single multi-node
job, never a job array.
"""
import os, subprocess

from ..fisher_auto import Scheduler


class SlurmScheduler(Scheduler):
    """Run worker ranks via ``srun`` within the current SLURM allocation.

    Parameters
    ----------
    cpus_per_task : cores per rank (256 = full Perlmutter CPU node).
    omp_threads : OMP_NUM_THREADS exported to each rank (ducc SHT threads).
        Scale with nside (few at small nside; ~128 at nside>=128).
    python : interpreter to invoke the worker with.
    """
    def __init__(self, cpus_per_task=256, omp_threads=128, python="python"):
        self.cpus_per_task = cpus_per_task
        self.omp_threads = omp_threads
        self.python = python

    def map(self, n_tasks, worker_argv):
        env = dict(os.environ, OMP_NUM_THREADS=str(self.omp_threads),
                   NUMEXPR_NUM_THREADS=str(self.omp_threads),
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cpu")
        inner = (f"{self.python} -m simaster.fisher_worker "
                 + " ".join(worker_argv) + " --rank $SLURM_PROCID")
        cmd = ["srun", "-N", str(n_tasks), "-n", str(n_tasks),
               "--ntasks-per-node=1", "-c", str(self.cpus_per_task),
               "bash", "-c", inner]
        subprocess.run(cmd, check=True, env=env)
