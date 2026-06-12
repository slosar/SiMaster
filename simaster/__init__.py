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
from .qml import QMLWorkspace, BandpowerResult
from .compat import compute_full_master
from .utils import cl_matrix, RealAlmIndex
from .radical import compress, CompressedLikelihood

__version__ = "0.1.0"
__all__ = ["Field", "Bins", "QMLWorkspace", "BandpowerResult",
           "compute_full_master", "cl_matrix", "RealAlmIndex",
           "compress", "CompressedLikelihood"]
