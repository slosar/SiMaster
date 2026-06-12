"""Spherical-harmonic transform layer.

Two exact backends:

* ``dense`` -- the real-basis synthesis matrix Y (restricted to observed
  pixels) is precomputed once and stored on the accelerator.  Every SHT then
  becomes a GEMM, which is ideal for the batched conjugate-gradient solves
  that dominate QML estimation.  Memory scales as N_obs * (lmax+1)^2, so this
  is the backend of choice for nside <= 64 on small GPUs (and <= 256 on large
  ones).

* ``ducc`` -- matrix-free transforms through ducc0 (the engine behind
  healpy), exact in double precision and multithreaded on the CPU.  Memory is
  O(N_obs), so this scales to nside = 1024 and beyond; the linear algebra of
  the solver still runs on the GPU via ``jax.pure_callback``.

s2fft (native-JAX GPU SHTs) was evaluated and rejected for now: its HEALPix
spin-2 synthesis carries ~5-13% pointwise error in v1.4.0 (spin-0 is exact),
which is fatal for a QML estimator that relies on exact adjoint pairs.  The
backend interface below is deliberately minimal so a future GPU SHT can be
dropped in.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import healpy as hp
import ducc0

from .utils import RealAlmIndex

_NTHREADS = max(1, (os.cpu_count() or 2) - 1)


def _healpix_geom(nside: int) -> dict:
    """Ring geometry arrays of the RING-ordered HEALPix grid for ducc0."""
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    return base.sht_info()


def _mstart(lmax: int) -> np.ndarray:
    m = np.arange(lmax + 1)
    return (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)


def synthesis(alm: np.ndarray, nside: int, lmax: int, spin: int,
              nthreads: int = _NTHREADS) -> np.ndarray:
    """Exact synthesis (alm -> map), healpy-compatible conventions.

    ``alm``: (ncomp, nalm) complex, healpy triangular layout; ncomp is 1 for
    spin 0 and 2 (E, B) for spin 2.  Returns (ncomp, npix) maps (T or Q, U).
    """
    geom = _healpix_geom(nside)
    return ducc0.sht.experimental.synthesis(
        alm=np.ascontiguousarray(alm), lmax=lmax, spin=spin,
        mstart=_mstart(lmax), nthreads=nthreads, **geom)


def adjoint_synthesis(maps: np.ndarray, nside: int, lmax: int, spin: int,
                      nthreads: int = _NTHREADS) -> np.ndarray:
    """Exact adjoint of :func:`synthesis` (map -> alm), Y^dagger m."""
    geom = _healpix_geom(nside)
    return ducc0.sht.experimental.adjoint_synthesis(
        map=np.ascontiguousarray(maps), lmax=lmax, spin=spin,
        mstart=_mstart(lmax), nthreads=nthreads, **geom)


# --------------------------------------------------------------------------
# Matrix-free real-basis operators (ducc backend)
# --------------------------------------------------------------------------

class RealSHT:
    """Real-basis synthesis Y and its transpose for one spin component.

    Y maps a real coefficient vector (nmodes(*2 for spin 2),) to maps on the
    observed pixels (nobs(*2 for spin 2),).  Batched over a trailing axis.
    """

    def __init__(self, nside: int, index: RealAlmIndex, spin: int, obs_pix: np.ndarray):
        self.nside, self.index, self.spin = nside, index, int(spin)
        self.obs_pix = np.asarray(obs_pix)
        self.npix = hp.nside2npix(nside)
        self.nobs = self.obs_pix.size
        self.ncomp = 1 if spin == 0 else 2
        self.nrow = self.ncomp * self.nobs
        self.ncol = self.ncomp * index.nmodes

    def _to_healpy(self, a):  # (ncomp, nmodes, B) real -> (ncomp, nalm, B) complex
        return np.stack([self.index.real_to_healpy(np.moveaxis(ai, -1, 0))
                         for ai in a])  # (ncomp, B, nalm)

    def synth(self, a: np.ndarray) -> np.ndarray:
        """a: (ncol, B) -> maps (nrow, B)."""
        B = a.shape[-1]
        a = a.reshape(self.ncomp, self.index.nmodes, B)
        alm = self._to_healpy(a)  # (ncomp, B, nalm)
        out = np.empty((self.ncomp, self.nobs, B))

        def run(b):
            m = synthesis(np.ascontiguousarray(alm[:, b]), self.nside,
                          self.index.lmax, self.spin, nthreads=2)
            out[:, :, b] = m[:, self.obs_pix]

        with ThreadPoolExecutor(max_workers=_NTHREADS // 2 + 1) as ex:
            list(ex.map(run, range(B)))
        return out.reshape(self.nrow, B)

    def adjoint(self, m: np.ndarray) -> np.ndarray:
        """m: (nrow, B) -> coefficients (ncol, B);  exact transpose of synth."""
        B = m.shape[-1]
        m = m.reshape(self.ncomp, self.nobs, B)
        out = np.empty((self.ncomp, self.index.nmodes, B))

        def run(b):
            fullb = np.zeros((self.ncomp, self.npix))
            fullb[:, self.obs_pix] = m[:, :, b]
            alm = adjoint_synthesis(fullb, self.nside, self.index.lmax,
                                    self.spin, nthreads=2)
            for c in range(self.ncomp):
                out[c, :, b] = self.index.healpy_to_real(alm[c])

        with ThreadPoolExecutor(max_workers=_NTHREADS // 2 + 1) as ex:
            list(ex.map(run, range(B)))
        return out.reshape(self.ncol, B)


# --------------------------------------------------------------------------
# Dense synthesis matrices (dense backend)
# --------------------------------------------------------------------------

def _cache_path(cachedir, nside, lmin, lmax, spin, obs_pix):
    h = hashlib.sha1(np.ascontiguousarray(obs_pix).tobytes()).hexdigest()[:12]
    return os.path.join(cachedir, f"Y_n{nside}_l{lmin}-{lmax}_s{spin}_{h}.npy")


def build_dense_Y(nside: int, index: RealAlmIndex, spin: int,
                  obs_pix: np.ndarray, cachedir: str | None = None,
                  dtype=np.float64) -> np.ndarray:
    """Dense real-basis synthesis matrix restricted to observed pixels.

    Returns (nrow, ncol) with nrow = ncomp*nobs, ncol = ncomp*nmodes.
    Row blocks are (T) for spin 0 and (Q, U) for spin 2; column blocks are
    (T) or (E, B).  Built column-by-column (exact), threaded, and cached on
    disk keyed by (nside, l-range, spin, footprint hash).
    """
    obs_pix = np.asarray(obs_pix)
    if cachedir:
        os.makedirs(cachedir, exist_ok=True)
        path = _cache_path(cachedir, nside, index.lmin, index.lmax, spin, obs_pix)
        if os.path.exists(path):
            return np.load(path, mmap_mode=None).astype(dtype, copy=False)

    ncomp = 1 if spin == 0 else 2
    nobs, K = obs_pix.size, index.nmodes
    Y = np.empty((ncomp * nobs, ncomp * K), dtype=dtype)
    lmax = index.lmax
    nalm = hp.Alm.getsize(lmax)
    # healpy alm value encoding one real-basis mode k
    sgn = (-1.0) ** np.abs(index.m)
    idx_h = hp.Alm.getidx(lmax, index.l, np.abs(index.m))
    val = np.where(index.m == 0, 1.0 + 0j,
                   np.where(index.m > 0, sgn / np.sqrt(2.0) + 0j,
                            1j * sgn / np.sqrt(2.0)))

    def run(k):
        alm = np.zeros((ncomp, nalm), dtype=np.complex128)
        for cc in range(ncomp):  # unit E mode, then unit B mode
            alm[:] = 0
            alm[cc, idx_h[k]] = val[k]
            mp = synthesis(alm, nside, lmax, spin, nthreads=1)
            Y[:, cc * K + k] = mp[:, obs_pix].reshape(-1)

    with ThreadPoolExecutor(max_workers=_NTHREADS) as ex:
        list(ex.map(run, range(K)))

    if cachedir:
        np.save(path, Y)
    return Y
