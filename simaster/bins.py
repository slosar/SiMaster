"""Bandpower binning, mirroring NaMaster's NmtBin (flat-in-C_l weights).

Bandpowers are defined as flat C_l within each band:  C_l = c_b for
l in [lo_b, hi_b].  The binning operator P averages, the interpolation
operator S = P^T (P P^T)^{-1} broadcasts a bandpower back to flat C_l.
"""

from __future__ import annotations

import numpy as np


class Bins:
    """A set of contiguous-or-not bandpowers, flat in ell.

    Use the constructors :meth:`from_edges` or :meth:`linear`.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=int)
        self.hi = np.asarray(hi, dtype=int)
        if np.any(self.hi < self.lo):
            raise ValueError("band hi < lo")
        if np.any(self.lo[1:] <= self.hi[:-1]):
            raise ValueError("bands must be sorted and non-overlapping")
        self.nbands = self.lo.size

    @classmethod
    def from_edges(cls, lo, hi):
        """Bands [lo_i, hi_i] inclusive on both ends (NaMaster: hi exclusive
        in from_edges; here inclusive -- see docs/migration.md)."""
        return cls(lo, hi)

    @classmethod
    def linear(cls, lmin: int, lmax: int, nlb: int):
        """Uniform bands of width nlb covering [lmin, lmax]."""
        lo = np.arange(lmin, lmax + 1, nlb)
        hi = np.minimum(lo + nlb - 1, lmax)
        return cls(lo, hi)

    @classmethod
    def from_nside_linear(cls, nside: int, nlb: int, lmin: int = 2):
        """NaMaster-like: uniform bands up to 3*nside - 1."""
        return cls.linear(lmin, 3 * nside - 1, nlb)

    def get_effective_ells(self):
        return 0.5 * (self.lo + self.hi)

    # ---- operators on C_l vectors ------------------------------------------
    def bin_cl(self, cl):
        """Flat average of C_l (last axis indexed by l, starting at l=0)."""
        cl = np.asarray(cl)
        out = np.empty(cl.shape[:-1] + (self.nbands,))
        for b in range(self.nbands):
            out[..., b] = cl[..., self.lo[b]:self.hi[b] + 1].mean(axis=-1)
        return out

    def unbin_cl(self, cb, lmax: int):
        """Broadcast bandpowers to flat C_l on [0, lmax] (zero outside bands)."""
        cb = np.asarray(cb)
        cl = np.zeros(cb.shape[:-1] + (lmax + 1,))
        for b in range(self.nbands):
            cl[..., self.lo[b]:min(self.hi[b], lmax) + 1] = cb[..., b:b + 1]
        return cl

    def extend_to_cover(self, lmin: int, lmax: int):
        """Return (bins, is_user_band) covering [lmin, lmax] with junk bands.

        Gaps below, between and above the user bands are filled with single
        junk bands so that the QML band basis spans every multipole in the
        covariance model (uncovered multipoles would alias into the
        estimates).  ``is_user_band`` flags the original bands.
        """
        lo, hi, user = [], [], []
        cursor = lmin
        for b in range(self.nbands):
            lb, hb = max(self.lo[b], lmin), min(self.hi[b], lmax)
            if hb < lb:
                continue
            if lb > cursor:
                lo.append(cursor); hi.append(lb - 1); user.append(False)
            lo.append(lb); hi.append(hb); user.append(True)
            cursor = hb + 1
        if cursor <= lmax:
            lo.append(cursor); hi.append(lmax); user.append(False)
        return Bins(lo, hi), np.array(user, dtype=bool)
