"""Radical compression (Bond, Jaffe & Knox 2000): offset-lognormal likelihood.

The full field-level likelihood of bandpowers is non-Gaussian where the
number of modes is small (large scales): upward fluctuations carry larger
uncertainty than downward ones.  BJK showed that the change of variables

    Z_b = ln(c_b + x_b),     x_b = noise offset ("x-factor")

makes the curvature matrix approximately amplitude-independent, so the
likelihood is well approximated as Gaussian in Z (the *offset-lognormal*
form), with weight matrix

    M^(Z)_bb' = (c_hat_b + x_b) F_bb' (c_hat_b' + x_b').

A dataset is then "radically compressed" to {c_hat, x, F}.

In the QML framework the x-factors generalize beyond the ideal full-sky
case (x_l = N_l/B_l^2) through the exact identity n = R x: the noise bias
expressed in bandpower units,

    x = R^-1 n,

computed from the workspace's response matrix and noise bias (exact in the
'exact'/'subsampled' engines).  Cross-spectra (TE-like) can be negative and
keep a Gaussian form, as is standard; auto-spectra get the Z transform.
At small scales (many modes) the two forms agree -- the compression only
*matters* at low l, which is QML territory anyway.

**Hamimeche & Lewis (2008) transform (``transform='hl'``).**  The offset
lognormal replaces the total power ``c+x`` by ``ln(c+x)``; this Gaussianizes the
per-mode chi^2/Wishart likelihood only *approximately*.  HL use instead the exact
variance-stabilizing change of variables

    g(x) = sign(x-1) sqrt(2 (x - ln x - 1)),   x = (c_hat+x_f)/(c+x_f),

which makes ``-2lnL = nu (x - ln x - 1) = (nu/2) g(x)^2`` an *exact* Gaussian per
mode in the full-sky limit.  In the compressed form the only change is
``ln(ratio) -> g(ratio)`` in the residual (same x-factor offset, same Fisher
metric): ``g`` and ``ln`` agree to leading order in ``ratio-1`` and differ at
second order, with ``g`` the exact Wishart Gaussianizer.  (HL's full method also
handles the T/E/B *matrix* jointly; here each auto-spectrum band is still scalar.)
"""

from __future__ import annotations

import numpy as np


def g_vst(x):
    """Hamimeche & Lewis (2008) variance-stabilizing transform.

    ``g(x) = sign(x-1) sqrt(2 (x - ln x - 1))`` for ``x>0`` (``g(1)=0``): the
    exact Gaussianizing change of variables for the per-mode Wishart/chi^2
    likelihood, i.e. the offset-lognormal's ``ln x`` replaced by its exact
    counterpart.  Agrees with ``ln x`` to leading order (both ``~ x-1``) and
    differs at ``O((x-1)^2)``.  Stable near ``x=1`` via ``log1p``.
    """
    x = np.asarray(x, dtype=float)
    u = x - 1.0
    t = u - np.log1p(u)                      # = x - ln x - 1, accurate near x=1
    return np.sign(u) * np.sqrt(2.0 * np.clip(t, 0.0, None))


class CompressedLikelihood:
    """{c_hat, x, F} + spectrum metadata; callable offset-lognormal lnL.

    Parameters (all over the *user* bands, junk bands marginalized):
    c_hat : (nb,) best-estimate bandpowers
    x     : (nb,) x-factors (noise offsets in bandpower units)
    F     : (nb, nb) bandpower Fisher matrix (= inverse covariance)
    is_auto : (nb,) bool; True -> offset-lognormal coordinate, False
        (cross-spectra, or autos with c_hat + x <= 0) -> linear/Gaussian.
    transform : 'lognormal' (BJK offset-lognormal, default) or 'hl'
        (Hamimeche & Lewis exact variance-stabilizing g-transform). Same
        x-factor offset and Fisher metric; 'hl' replaces ln(ratio) by g(ratio).
    """

    def __init__(self, ells, spec_names, c_hat, x, F, is_auto,
                 transform="lognormal"):
        if transform not in ("lognormal", "hl"):
            raise ValueError("transform must be 'lognormal' or 'hl'")
        self.transform = transform
        self.ells = np.asarray(ells)
        self.spec_names = list(spec_names)
        self.c_hat = np.asarray(c_hat, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.is_auto = np.asarray(is_auto, dtype=bool)
        # autos with non-positive (c_hat+x) cannot be transformed
        self.use_log = self.is_auto & (self.c_hat + self.x > 0)
        self.d = np.where(self.use_log, self.c_hat + self.x, 1.0)  # fiducial total
        self.M = self.d[:, None] * self.F * self.d[None, :]
        self.u_hat = np.where(self.use_log, np.log(self.d), self.c_hat)

    def _u(self, c):
        c = np.asarray(c, dtype=float)
        bad = self.use_log & (c + self.x <= 0)
        u = np.where(self.use_log,
                     np.log(np.clip(c + self.x, 1e-300, None)), c)
        return u, bad.any(axis=-1) if bad.ndim else bad.any()

    def loglike(self, c):
        """Offset-lognormal (or HL) ln L (up to a constant) for theory bandpowers.

        ``c``: flat vector over (spec, band) in self.spec_names order, or a
        dict {spec name: (nbands,)}.  Returns -inf where an auto band has
        c + x <= 0.
        """
        if isinstance(c, dict):
            c = np.concatenate([np.asarray(c[s]) for s in self.spec_names])
        c = np.asarray(c, dtype=float)
        if self.transform == "hl":
            bad = self.use_log & (c + self.x <= 0)
            if np.any(bad):
                return -np.inf
            # ratio = (data total)/(theory total); X = d*g(ratio) [log], c_hat-c [lin]
            ratio = (self.c_hat + self.x) / np.where(self.use_log, c + self.x, 1.0)
            X = np.where(self.use_log, self.d * g_vst(ratio), self.c_hat - c)
            return float(-0.5 * X @ self.F @ X)
        u, bad = self._u(c)
        r = u - self.u_hat
        val = -0.5 * r @ self.M @ r
        return -np.inf if bad else float(val)

    def loglike_gaussian(self, c):
        """Plain Gaussian comparison: -1/2 (c-c_hat)^T F (c-c_hat)."""
        if isinstance(c, dict):
            c = np.concatenate([np.asarray(c[s]) for s in self.spec_names])
        r = np.asarray(c, dtype=float) - self.c_hat
        return float(-0.5 * r @ self.F @ r)

    def save(self, path):
        np.savez(path, ells=self.ells, spec_names=self.spec_names,
                 c_hat=self.c_hat, x=self.x, F=self.F, is_auto=self.is_auto,
                 transform=self.transform)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        transform = str(d["transform"]) if "transform" in d.files else "lognormal"
        return cls(d["ells"], [str(s) for s in d["spec_names"]],
                   d["c_hat"], d["x"], d["F"], d["is_auto"], transform=transform)


def compress(ws, result=None, data=None, transform="lognormal"):
    """Radically compress a QML estimate to {c_hat, x, F}.

    ws : QMLWorkspace with the response computed (exact engines preferred).
    result : a BandpowerResult from ws.estimate (single realization); if
        None it is computed from ``data`` (or the fields' own maps).
    transform : 'lognormal' (BJK, default) or 'hl' (Hamimeche-Lewis g-transform).
    """
    if result is None:
        result = ws.estimate(data)
    vec = result.vector()
    if vec.shape[0] != 1:
        raise ValueError("compress one realization at a time")
    c_hat = vec[0]

    # x-factors over all bands (junk included), then restrict to user bands
    x_full = ws.R_inv @ ws.n_hat
    nbb = ws.bins.nbands
    keep = np.concatenate([np.flatnonzero(ws.is_user_band) + s * nbb
                           for s in range(len(ws.spec_pairs))])
    x = x_full[keep]

    F = np.linalg.inv(result.cov)        # junk bands marginalized
    nb_user = int(ws.is_user_band.sum())
    is_auto = np.concatenate([np.full(nb_user, i == j)
                              for (i, j) in ws.spec_pairs])
    return CompressedLikelihood(result.ells, result.spec_names,
                                c_hat, x, F, is_auto, transform=transform)
