"""Utilities: real spherical-harmonic mode bookkeeping and C_ell handling.

SiMaster works internally in a *real* spherical-harmonic basis.  A real field
T(n) bandlimited to ``lmax`` is written

    T = sum_k a_k Y^R_k ,    k = (l, m),  m in [-l, l],  lmin <= l <= lmax,

where the Y^R are the orthonormal real spherical harmonics

    Y^R_{lm} = (1/sqrt2) ((-1)^m Y_{lm} + Y_{l,-m})        m > 0
    Y^R_{l0} = Y_{l0}
    Y^R_{lm} = (1/(i sqrt2)) ((-1)^|m| Y_{l|m|} - Y_{l,-|m|})   m < 0 .

For an isotropic field with spectrum C_l the real coefficients are i.i.d.
with variance C_l, which makes the covariance of the coefficient vector
block-diagonal: one (ncomp x ncomp) block per mode k, equal to C_l(k).

Relation to healpy's complex a_lm (m >= 0 storage):

    a^R_{l,0}  = Re a_{l0}
    a^R_{l,m}  = sqrt2 (-1)^m Re a_{lm}     (m > 0)
    a^R_{l,-m} = sqrt2 (-1)^m Im a_{lm}     (m > 0)
"""

from __future__ import annotations

import numpy as np
import healpy as hp


class RealAlmIndex:
    """Index map for the real spherical-harmonic coefficient vector.

    Modes are ordered l-major: for each l in [lmin, lmax], m runs over
    0, 1, -1, 2, -2, ..., l, -l.  This grouping keeps all modes of a given
    l contiguous, which makes per-l (band) selections simple slices.
    """

    def __init__(self, lmin: int, lmax: int):
        self.lmin, self.lmax = int(lmin), int(lmax)
        ls, ms = [], []
        for l in range(self.lmin, self.lmax + 1):
            ls.append(l); ms.append(0)
            for m in range(1, l + 1):
                ls.extend([l, l]); ms.extend([m, -m])
        self.l = np.array(ls, dtype=np.int32)
        self.m = np.array(ms, dtype=np.int32)
        self.nmodes = self.l.size

    def band_slice(self, l_lo: int, l_hi: int) -> slice:
        """Contiguous slice of modes with l_lo <= l <= l_hi (inclusive)."""
        i0 = np.searchsorted(self.l, l_lo, side="left")
        i1 = np.searchsorted(self.l, l_hi, side="right")
        return slice(int(i0), int(i1))

    # ---- conversions to/from healpy complex alm ----------------------------
    def real_to_healpy(self, a: np.ndarray, lmax_out: int | None = None) -> np.ndarray:
        """Real coefficient vector (..., nmodes) -> healpy complex alm."""
        lmax_out = self.lmax if lmax_out is None else lmax_out
        a = np.asarray(a)
        out = np.zeros(a.shape[:-1] + (hp.Alm.getsize(lmax_out),), dtype=np.complex128)
        l, m = self.l, self.m
        sgn = (-1.0) ** np.abs(m)
        pos = m >= 0
        idx_pos = hp.Alm.getidx(lmax_out, l[pos], m[pos])
        idx_neg = hp.Alm.getidx(lmax_out, l[~pos], -m[~pos])
        # m=0 modes: direct; m>0: real part; m<0: imaginary part
        m0 = m[pos] == 0
        re = np.zeros(out.shape, dtype=np.float64)
        im = np.zeros(out.shape, dtype=np.float64)
        # index sets are unique within re and within im separately
        re[..., idx_pos] = np.where(m0, a[..., pos], sgn[pos] * a[..., pos] / np.sqrt(2.0))
        im[..., idx_neg] = sgn[~pos] * a[..., ~pos] / np.sqrt(2.0)
        return re + 1j * im

    def healpy_to_real(self, alm: np.ndarray) -> np.ndarray:
        """Healpy complex alm -> real coefficient vector (..., nmodes)."""
        alm = np.asarray(alm)
        lmax_in = hp.Alm.getlmax(alm.shape[-1])
        l, m = self.l, self.m
        sgn = (-1.0) ** np.abs(m)
        idx = hp.Alm.getidx(lmax_in, l, np.abs(m))
        vals = alm[..., idx]
        out = np.where(
            m == 0, vals.real,
            np.where(m > 0, np.sqrt(2.0) * sgn * vals.real, np.sqrt(2.0) * sgn * vals.imag),
        )
        return out.astype(np.float64)


def cl_matrix(cl_dict, comp_names, lmax: int) -> np.ndarray:
    """Assemble the (ncomp, ncomp, lmax+1) fiducial spectrum matrix.

    Parameters
    ----------
    cl_dict : dict mapping (name_i, name_j) -> array C_l (length >= lmax+1).
        Missing pairs are taken to be zero.  Keys are symmetrized.
    comp_names : ordered list of component names, e.g. ['f0_T', 'f1_E', 'f1_B'].
    """
    nc = len(comp_names)
    out = np.zeros((nc, nc, lmax + 1))
    for (a, b), cl in cl_dict.items():
        if a not in comp_names or b not in comp_names:
            continue
        i, j = comp_names.index(a), comp_names.index(b)
        cl = np.asarray(cl, dtype=np.float64)
        if cl.size < lmax + 1:
            cl = np.pad(cl, (0, lmax + 1 - cl.size))
        out[i, j] = out[j, i] = cl[: lmax + 1]
    return out


def psd_floor(clmat: np.ndarray, floor_frac: float = 1e-12) -> np.ndarray:
    """Project each per-l (ncomp x ncomp) block onto the PSD cone.

    Eigenvalues are floored at ``floor_frac`` times the largest eigenvalue of
    the block (and at zero).  Used when iterating: estimated bandpowers can
    produce indefinite blocks which are not valid covariances.
    """
    clmat = np.array(clmat, copy=True)
    mats = np.moveaxis(clmat, -1, 0)  # (L, nc, nc)
    w, v = np.linalg.eigh(0.5 * (mats + np.swapaxes(mats, -1, -2)))
    wmax = np.clip(w.max(axis=-1, keepdims=True), 0.0, None)
    w = np.clip(w, floor_frac * wmax, None)
    fixed = np.einsum("lij,lj,lkj->lik", v, w, v)
    return np.moveaxis(fixed, 0, -1)


def matrix_sqrt_psd(mats: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root of stacked (..., n, n) matrices."""
    w, v = np.linalg.eigh(0.5 * (mats + np.swapaxes(mats, -1, -2)))
    w = np.clip(w, 0.0, None)
    return np.einsum("...ij,...j,...kj->...ik", v, np.sqrt(w), v)
