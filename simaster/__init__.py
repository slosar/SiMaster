"""SiMaster: GPU-accelerated QML power spectrum estimation on the sphere.

A quadratic-maximum-likelihood (optimal) bandpower estimator for spin-0 and
spin-2 fields on cut skies, with a NaMaster-like interface.  All linear
algebra is matrix-free (preconditioned conjugate gradients; no dense
covariance inversion) and batched on GPU through JAX.
"""

import jax as _jax

# QML needs double precision: quadratic estimators difference large numbers.
_jax.config.update("jax_enable_x64", True)

from .field import Field
from .bins import Bins
from .noise import PixelNoiseCov, iqu_from_cov
from .qml import QMLWorkspace, BandpowerResult, LikelihoodExpansion
from .compat import compute_full_master
from .utils import cl_matrix, RealAlmIndex
from .radical import compress, CompressedLikelihood
from .deflation import DeflationSpace, build_deflation, harvest_ritz
from .subsample import SubsampleStore, SubsampleError
from .mc_fisher import (MCFisherStore, compute_mc_error, BandedFisher,
                        band_fisher, banded_fisher, banded_index_map)
from .fisher_auto import Scheduler, LocalScheduler, run_auto
from . import score  # field-level likelihood score / autodiff (advanced)

__version__ = "0.1.0"
__all__ = ["Field", "Bins", "PixelNoiseCov", "iqu_from_cov",
           "QMLWorkspace", "BandpowerResult",
           "LikelihoodExpansion", "compute_full_master", "cl_matrix",
           "RealAlmIndex", "compress", "CompressedLikelihood", "score",
           "DeflationSpace", "build_deflation", "harvest_ritz",
           "SubsampleStore", "SubsampleError",
           "MCFisherStore", "compute_mc_error", "BandedFisher",
           "band_fisher", "banded_fisher", "banded_index_map",
           "Scheduler", "LocalScheduler", "run_auto"]
