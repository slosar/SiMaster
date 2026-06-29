"""NERSC (Perlmutter) adapter for simaster's portable harnesses.

Everything SLURM/NERSC-specific lives here; the rest of simaster is portable.
"""
from .slurm import SlurmScheduler

__all__ = ["SlurmScheduler"]
