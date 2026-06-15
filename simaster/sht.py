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

* ``s2fft`` -- native-JAX matrix-free transforms (:class:`S2fftSHT`).  The
  synthesis runs entirely on the accelerator inside XLA (no host round-trip,
  unlike the ``ducc`` ``pure_callback``), and its exact transpose Y^T is
  obtained from ``jax.linear_transpose`` of the differentiable synthesis --
  exactly the adjoint a QML estimator needs.  This is the path to a GPU SHT
  at nside >~ 256 and to an end-to-end differentiable pipeline.

  CAVEAT: requires an s2fft build in which HEALPix spin-2 synthesis is exact.
  Released s2fft (<= 1.4.0) has a Wigner-d recursion-node bug that makes
  HEALPix spin!=0 transforms carry ~5-13% pointwise error (spin-0 is exact),
  which is fatal for QML; it is fixed on the ``fix/healpix-spin-recursion-node``
  branch (PR upstream).  :class:`S2fftSHT` self-checks against ducc0 at
  construction and refuses to run if the installed s2fft fails the check.
  See ``docs/method.md`` (s2fft backend) for details.
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


# --------------------------------------------------------------------------
# Native-JAX matrix-free operators (s2fft backend)
# --------------------------------------------------------------------------

def _flm_scatter(index: RealAlmIndex, spin: int, L: int):
    """Linear map (real coeff vector) -> flat s2fft 2D flm of the spin field.

    Returns integer row/col index arrays and complex weights such that

        flm_flat = zeros(L*(2L-1)).at[col].add(weight * a[row])

    builds, for spin 0, the conjugate-symmetric scalar flm and, for spin 2,
    the spin-(+2) coefficients ``-(E + iB)`` (with the E/B reality symmetry
    filling the negative-m half), matching healpy/ducc synthesis exactly.
    The real coefficient vector is component-major: spin-0 is ``[T]`` and
    spin-2 is ``[E modes, B modes]`` (the layout used by :class:`RealSHT`).
    """
    ncomp = 1 if spin == 0 else 2
    K = index.nmodes
    W = 2 * L - 1
    rows, cols, wts = [], [], []
    for c in range(ncomp):
        for k in range(K):
            l, m = int(index.l[k]), int(index.m[k])
            mu = abs(m)
            sgn = (-1.0) ** mu
            # real mode -> complex weight on healpy a_{l,mu}
            if m == 0:
                wc = 1.0 + 0j
            elif m > 0:
                wc = sgn / np.sqrt(2.0) + 0j
            else:
                wc = 1j * sgn / np.sqrt(2.0)
            row = c * K + k
            pp = l * W + (L - 1 + mu)        # flm[l, +mu]
            pm = l * W + (L - 1 - mu)        # flm[l, -mu]
            if spin == 0:
                rows.append(row); cols.append(pp); wts.append(wc)
                if mu > 0:
                    rows.append(row); cols.append(pm); wts.append(sgn * np.conj(wc))
            elif c == 0:                     # E -> -(E)
                rows.append(row); cols.append(pp); wts.append(-wc)
                if mu > 0:
                    rows.append(row); cols.append(pm); wts.append(-sgn * np.conj(wc))
            else:                            # B -> -i(B)
                rows.append(row); cols.append(pp); wts.append(-1j * wc)
                if mu > 0:
                    rows.append(row); cols.append(pm); wts.append(-1j * sgn * np.conj(wc))
    return (np.asarray(rows, np.int64), np.asarray(cols, np.int64),
            np.asarray(wts, np.complex128))


class S2fftSHT:
    """Real-basis synthesis Y and its exact transpose via s2fft (native JAX).

    Same interface and conventions as :class:`RealSHT` (``synth``/``adjoint``
    map ``(ncol, B) <-> (nrow, B)``), but everything is a JAX op: the
    synthesis is :func:`s2fft.inverse` on the device, and the adjoint Y^T is
    its ``jax.linear_transpose`` -- an exact transpose, not the quadrature
    ``s2fft.forward``.  Construction self-checks against ducc0 synthesis and
    raises if the installed s2fft is inexact (the spin-2 HEALPix bug).
    """

    def __init__(self, nside: int, index: RealAlmIndex, spin: int,
                 obs_pix: np.ndarray, check: bool = True, atol: float = 1e-8):
        import jax
        import jax.numpy as jnp
        import s2fft

        self.nside, self.index, self.spin = nside, index, int(spin)
        self.obs_pix = np.asarray(obs_pix)
        self.npix = hp.nside2npix(nside)
        self.nobs = self.obs_pix.size
        self.ncomp = 1 if spin == 0 else 2
        self.K = index.nmodes
        self.ncol = self.ncomp * self.K
        self.nrow = self.ncomp * self.nobs
        L = index.lmax + 1
        self.L = L
        W = 2 * L - 1

        row, col, wt = _flm_scatter(index, self.spin, L)
        row, col = jnp.asarray(row), jnp.asarray(col)
        wt = jnp.asarray(wt)
        obs = jnp.asarray(self.obs_pix)
        sp = self.spin
        zeros_flm = jnp.zeros((L * W,), jnp.complex128)

        def synth1(a):                       # (ncol,) real -> (nrow,) real
            flm = zeros_flm.at[col].add(wt * a[row]).reshape(L, W)
            m = s2fft.inverse(flm, L, spin=sp, nside=nside,
                              sampling="healpix", method="jax", reality=False)
            if sp == 0:
                return jnp.real(m)[obs]
            return jnp.concatenate([jnp.real(m)[obs], jnp.imag(m)[obs]])

        adj1 = jax.linear_transpose(synth1, jnp.zeros(self.ncol))
        self._synth = jax.vmap(synth1, in_axes=-1, out_axes=-1)
        self._adjoint = jax.vmap(lambda m: adj1(m)[0], in_axes=-1, out_axes=-1)

        if check:
            self._self_check(atol)

    def synth(self, a):
        """a: (ncol, B) -> maps (nrow, B); JAX-traceable, on device."""
        return self._synth(a)

    def adjoint(self, m):
        """m: (nrow, B) -> coefficients (ncol, B); exact transpose of synth."""
        return self._adjoint(m)

    def _self_check(self, atol: float):
        """Refuse to run if s2fft synthesis disagrees with ducc0 (the spin-2
        HEALPix bug); also asserts the linear-transpose adjoint identity."""
        import numpy as _np
        rng = _np.random.default_rng(0)
        a = rng.standard_normal((self.ncol, 1))
        ref = RealSHT(self.nside, self.index, self.spin, self.obs_pix)
        y_s2 = _np.asarray(self.synth(a))
        y_dc = ref.synth(a)
        rel = _np.abs(y_s2 - y_dc).max() / max(_np.abs(y_dc).max(), 1e-300)
        if not rel < atol:
            raise RuntimeError(
                f"s2fft spin-{self.spin} HEALPix synthesis disagrees with "
                f"ducc0 by {rel:.2e} (> {atol:.0e}). The installed s2fft has "
                "the HEALPix spin recursion-node bug; install the fixed "
                "branch or use backend='ducc'.")
        m = rng.standard_normal((self.nrow, 1))
        ip1 = float(y_s2[:, 0] @ m[:, 0])
        ip2 = float(a[:, 0] @ _np.asarray(self.adjoint(m))[:, 0])
        assert abs(ip1 - ip2) <= 1e-6 * (abs(ip1) + 1e-300), "adjoint identity"
