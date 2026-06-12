"""Exact (dense) QML reference implementation for small problems.

Builds the pixel covariance explicitly and evaluates the response (binned
Fisher) matrix and noise bias exactly.  Cost is O((ncomp*nobs)^3) in time and
O(n^2) in memory, so this is only for nside <= ~16 -- it exists to validate
the scalable Monte-Carlo machinery in :mod:`simaster.qml`, not for science.
No template deprojection here (use small alpha-mode comparisons instead).
"""

from __future__ import annotations

import numpy as np

from .utils import RealAlmIndex
from . import sht


class ExactQML:
    def __init__(self, fields, bins_ext, clmat, index: RealAlmIndex):
        self.fields, self.bins, self.index = fields, bins_ext, index
        K = index.nmodes
        # global G = W Y B, modes stacked like CovModel (comps major)
        Gs = []
        for f in fields:
            Y = sht.build_dense_Y(f.nside, index, f.spin, f.obs_pix)
            W = np.tile(f.mask, f.ncomp)
            if f.beam is not None:
                bl = np.concatenate([f.beam[index.l]] * f.ncomp)
                Y = Y * bl[None, :]
            Gs.append(W[:, None] * Y)
        nrow = sum(g.shape[0] for g in Gs)
        ncomp = sum(f.ncomp for f in fields)
        self.ncomp, self.K = ncomp, K
        G = np.zeros((nrow, ncomp * K))
        r0, c0 = 0, 0
        for g, f in zip(Gs, fields):
            G[r0:r0 + g.shape[0], c0 * K:(c0 + f.ncomp) * K] = g
            r0 += g.shape[0]; c0 += f.ncomp
        self.G = G
        self.noisevar = np.concatenate(
            [np.tile(1.0 / f.ivar, f.ncomp) for f in fields])
        # dense C
        clk = clmat[:, :, index.l]                    # (Nc, Nc, K)
        SG = np.zeros_like(G)
        for c in range(ncomp):
            for d in range(ncomp):
                SG[:, c * K:(c + 1) * K] += G[:, d * K:(d + 1) * K] * clk[d, c][None, :]
        self.C = SG @ G.T + np.diag(self.noisevar)
        self.Cinv = np.linalg.inv(self.C)
        V = self.Cinv @ G
        self.H = G.T @ V                              # (Nc*K, Nc*K)
        self.W2 = (V.T * self.noisevar[None, :]) @ V

        self.spec_pairs = [(i, j) for i in range(ncomp) for j in range(i, ncomp)]

    def _slice(self, c, b):
        sl = self.index.band_slice(self.bins.lo[b], self.bins.hi[b])
        return slice(c * self.K + sl.start, c * self.K + sl.stop)

    def response(self):
        """Exact R_AB = 1/2 Tr[C^-1 P_A C^-1 P_B], A = (spec, band).

        With H = G^T C^-1 G and P_(cd),b = G (sum_{k in b} e_ck e_dk^T + sym) G^T,

            R_AB = 1/2 sum_{(gd) in pairs(A)} sum_{(ef) in pairs(B)}
                       sum_{k in b, k' in b'} H_de[k,k'] H_gf[k,k'] ,

        pairs((c,d)) = {(c,d), (d,c)} for c != d and {(c,c)} for c == d.
        """
        nbb = self.bins.nbands
        ns = len(self.spec_pairs)
        R = np.zeros((ns * nbb, ns * nbb))

        def pairs(c, d):
            return [(c, d)] if c == d else [(c, d), (d, c)]

        for A, (c, d) in enumerate(self.spec_pairs):
            for B, (e, f) in enumerate(self.spec_pairs):
                for b in range(nbb):
                    for b2 in range(nbb):
                        t = 0.0
                        for (g, dd) in pairs(c, d):
                            for (ee, ff) in pairs(e, f):
                                h1 = self.H[self._slice(dd, b), self._slice(ee, b2)]
                                h2 = self.H[self._slice(g, b), self._slice(ff, b2)]
                                t += np.sum(h1 * h2)
                        R[A * nbb + b, B * nbb + b2] = 0.5 * t
        return R

    def noise_bias(self):
        """Exact n_A = 1/2 Tr[C^-1 N C^-1 P_A]."""
        nbb = self.bins.nbands
        out = np.zeros(len(self.spec_pairs) * nbb)
        for A, (c, d) in enumerate(self.spec_pairs):
            for b in range(nbb):
                blk = self.W2[self._slice(c, b), self._slice(d, b)]
                out[A * nbb + b] = np.trace(blk) * (0.5 if c == d else 1.0)
        return out

    def y_of(self, x):
        """Exact y_A(x) = 1/2 x^T C^-1 P_A C^-1 x."""
        a = self.G.T @ (self.Cinv @ x)                # (Nc*K,)
        nbb = self.bins.nbands
        out = np.zeros(len(self.spec_pairs) * nbb)
        for A, (c, d) in enumerate(self.spec_pairs):
            for b in range(nbb):
                ac = a[self._slice(c, b)]
                ad = a[self._slice(d, b)]
                out[A * nbb + b] = (0.5 if c == d else 1.0) * np.sum(ac * ad)
        return out
