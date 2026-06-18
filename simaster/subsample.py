r"""Error budget for the column-*subsampled* response (Fisher) and noise bias.

The exact engine (:meth:`simaster.qml.QMLWorkspace.run_exact`) with
``sample_frac = f < 1`` estimates the binned response matrix ``R`` (the
bandpower Fisher matrix) *and* the noise bias ``n`` from a stratified,
``1/f``-renormalised random subset of the mode columns ``k'``.  Both estimates
are unbiased, but for a *given* draw they carry a finite-sample error -- which,
because it does not average away in a real analysis (you ran one seed), behaves
as a systematic.

This module quantifies that error from the per-mode contributions retained by
``run_exact(..., keep_samples=True)`` and turns it into covariances that can be
*added* to ``R^-1`` as a subsampling error budget, plus a per-band
suboptimality diagnostic ``sqrt(diag Cov_sub / diag R^-1)``.

Because ``R`` and ``n`` are built from the *same* column draw, their
subsampling errors are correlated, and the bandpower estimate
``c = R^-1 (y - n)`` feels both:

    delta c = -R^-1 (delta R) c - R^-1 delta n = -R^-1 [ delta(R c) + delta n ],

so the honest budget propagates the *joint* fluctuation ``delta(R c + n)``.
The noise bias also gets its own covariance (and hence a non-trivial
``n_hat_err``, which the subsampled engine otherwise leaves at zero).

Two independent estimators are provided:

* **analytic** -- ``R`` and ``n`` are *linear* in the per-mode contributions,
  so their subsampling covariance is the textbook stratified
  simple-random-sampling (SRS) *without-replacement* variance, with the
  finite-population correction ``(1 - n_l/N_l)`` built in (it vanishes at
  ``f = 1``).  Propagated to the bandpowers by the delta method.  Exact for
  ``R`` and ``n`` themselves, but *linearises* the matrix inverse.

* **bootstrap** -- resample the mode columns within each stratum (FPC-scaled
  multinomial), rebuild ``R*`` and ``n*`` with the *same* weights, and form
  ``c* = R*^-1 (y - n*)``; take the sample covariance over draws.  This carries
  the resampled ``R`` through the full nonlinear inverse and keeps the R/n
  correlation, so it captures the bias/skew the delta method misses (the
  regime where the analytic estimate is known to under-report).

Geometry that makes this cheap.  Because the *row* multipole of ``H`` is
summed exactly (one adjoint transform per solve) and only the *column*
multipole ``l'`` is subsampled, each sampled mode contributes to exactly one
*column band* of the pre-symmetrised ``R`` (a compact slab ``(ns, nbb, ns)``)
and to a single band of ``n`` (a length-``ns`` vector).  These per-mode
contributions are additive across modes and concatenable across modes computed
on different nodes, so a distributed run checkpoints its
:class:`SubsampleStore` per node and merges them before this post-processing --
see :meth:`SubsampleStore.merge`.
"""

from __future__ import annotations

import numpy as np


class SubsampleStore:
    """Per-mode contributions to the binned response and noise bias.

    Attributes
    ----------
    slab : (ns, nbb, ns, P) array
        ``slab[A, b1, B, p]`` is mode ``p``'s contribution to the
        *pre-symmetrised* binned response ``R[A, b1, B, b2(p)]`` at unit
        column weight, where ``b2(p)`` is the mode's column band.
    nslab : (ns, P) array
        ``nslab[A, p]`` is mode ``p``'s contribution (with the 0.5 auto-spectrum
        factor folded in) to the binned noise bias ``n[A, b2(p)]`` at unit
        column weight.
    strat : (P,) int array
        Stratum (column multipole index ``l' - lmin``) of each retained mode.
    band_of_l : (nl,) int array
        Band index of every multipole; ``col_band = band_of_l[strat]``.
    N_l, n_l : (nl,) int arrays
        Population and sampled mode counts per stratum.  ``scale_l = N_l/n_l``.
    ns, nbb, nl : ints
        Number of spectra, bands, and multipoles.
    R_hat, R_inv : (nb, nb) arrays
        The symmetrised response and its inverse (``nb = ns*nbb``).
    n_hat : (nb,) array
        The binned noise bias.
    R_det, n_det : arrays, optional
        Deterministic control-variate offsets.  The stochastic slabs then
        store residuals around these offsets; by default both are zero and
        the historical raw-subsampling behavior is recovered.
    is_user_band : (nbb,) bool array
        Which bands are reported (user) vs marginalised (junk).
    """

    def __init__(self, slab, nslab, strat, band_of_l, N_l, n_l, ns, nbb, nl,
                 R_hat, R_inv, n_hat, is_user_band, R_det=None, n_det=None):
        self.slab = np.ascontiguousarray(slab, dtype=np.float64)
        self.nslab = np.ascontiguousarray(nslab, dtype=np.float64)
        self.strat = np.asarray(strat, dtype=np.int64)
        self.band_of_l = np.asarray(band_of_l, dtype=np.int64)
        self.N_l = np.asarray(N_l, dtype=np.int64)
        self.n_l = np.asarray(n_l, dtype=np.int64)
        self.ns, self.nbb, self.nl = int(ns), int(nbb), int(nl)
        self.R_hat = None if R_hat is None else np.asarray(R_hat, float)
        self.R_inv = None if R_inv is None else np.asarray(R_inv, float)
        self.n_hat = None if n_hat is None else np.asarray(n_hat, float)
        self.is_user_band = np.asarray(is_user_band, dtype=bool)
        self.R_det = (np.zeros((self.ns * self.nbb, self.ns * self.nbb))
                      if R_det is None else np.asarray(R_det, float))
        self.n_det = (np.zeros(self.ns * self.nbb)
                      if n_det is None else np.asarray(n_det, float))

    # -- derived quantities ------------------------------------------------
    @property
    def P(self):
        """Number of retained modes."""
        return self.slab.shape[3]

    @property
    def nb(self):
        """Flattened band-vector length ``ns*nbb``."""
        return self.ns * self.nbb

    @property
    def col_band(self):
        """Column band ``b2`` of each retained mode, shape (P,)."""
        return self.band_of_l[self.strat]

    @property
    def scale_l(self):
        """Per-stratum renormalisation ``N_l/n_l`` (1 where unsampled)."""
        s = np.ones(self.nl)
        m = self.n_l > 0
        s[m] = self.N_l[m] / self.n_l[m]
        return s

    @property
    def w0(self):
        """Base (unbiased-estimator) weight of each mode, shape (P,)."""
        return self.scale_l[self.strat]

    def _onehot_band(self):
        """(P, nbb) one-hot mapping each mode to its column band."""
        G = np.zeros((self.P, self.nbb))
        G[np.arange(self.P), self.col_band] = 1.0
        return G

    def build_R(self, weights):
        """Symmetrised binned response from per-mode ``weights`` (P,).

        ``weights = w0`` reproduces ``R_hat``; bootstrap draws perturb it.
        """
        C = self.slab * np.asarray(weights)[None, None, None, :]
        Rpre = np.einsum("ijkp,pd->ijkd", C, self._onehot_band())
        R = self.R_det + Rpre.reshape(self.nb, self.nb)
        return 0.5 * (R + R.T)

    def build_n(self, weights):
        """Binned noise bias (nb,) from per-mode ``weights`` (P,).

        ``weights = w0`` reproduces ``n_hat``.
        """
        contrib = self.nslab * np.asarray(weights)[None, :]
        return self.n_det + (contrib @ self._onehot_band()).reshape(self.nb)

    def reconstruct_R(self):
        """Rebuild ``R_hat`` from the stored slabs (consistency check)."""
        return self.build_R(self.w0)

    def reconstruct_n(self):
        """Rebuild ``n_hat`` from the stored slabs (consistency check)."""
        return self.build_n(self.w0)

    def noise_cov_analytic(self):
        """Stratified SRS-without-replacement covariance of ``n_hat`` (nb, nb)."""
        return _stratified_cov(self, _per_mode_n(self))

    # -- distributed aggregation ------------------------------------------
    def merge(self, other):
        """Concatenate another store's modes (e.g. computed on another node).

        Slabs are concatenated along the mode axis and ``n_l`` summed;
        ``N_l`` (the fixed population) and the band geometry must agree.
        ``R_hat``/``R_inv``/``n_hat`` are recomputed from the merged slabs.
        """
        if (self.ns, self.nbb, self.nl) != (other.ns, other.nbb, other.nl):
            raise ValueError("stores have incompatible shapes")
        if not np.array_equal(self.N_l, other.N_l):
            raise ValueError("stores have different mode populations N_l")
        if (not np.allclose(self.R_det, other.R_det)
                or not np.allclose(self.n_det, other.n_det)):
            raise ValueError("stores have different deterministic controls")
        m = SubsampleStore(
            np.concatenate([self.slab, other.slab], axis=3),
            np.concatenate([self.nslab, other.nslab], axis=1),
            np.concatenate([self.strat, other.strat]),
            self.band_of_l, self.N_l, self.n_l + other.n_l,
            self.ns, self.nbb, self.nl, None, None, None, self.is_user_band,
            R_det=self.R_det, n_det=self.n_det)
        m.R_hat = m.reconstruct_R()
        m.R_inv = np.linalg.inv(m.R_hat)
        m.n_hat = m.reconstruct_n()
        return m

    def save(self, path):
        """Serialise to a ``.npz`` checkpoint (for distributed aggregation)."""
        np.savez(path, slab=self.slab, nslab=self.nslab, strat=self.strat,
                 band_of_l=self.band_of_l, N_l=self.N_l, n_l=self.n_l,
                 shape=np.array([self.ns, self.nbb, self.nl]),
                 R_hat=np.array([]) if self.R_hat is None else self.R_hat,
                 R_inv=np.array([]) if self.R_inv is None else self.R_inv,
                 n_hat=np.array([]) if self.n_hat is None else self.n_hat,
                 is_user_band=self.is_user_band,
                 R_det=self.R_det, n_det=self.n_det)

    @classmethod
    def load(cls, path):
        """Load a ``.npz`` checkpoint written by :meth:`save`."""
        d = np.load(path)
        ns, nbb, nl = (int(x) for x in d["shape"])
        rh, ri, nh = d["R_hat"], d["R_inv"], d["n_hat"]
        R_det = d["R_det"] if "R_det" in d else None
        n_det = d["n_det"] if "n_det" in d else None
        return cls(d["slab"], d["nslab"], d["strat"], d["band_of_l"],
                   d["N_l"], d["n_l"], ns, nbb, nl,
                   rh if rh.size else None, ri if ri.size else None,
                   nh if nh.size else None, d["is_user_band"],
                   R_det=R_det, n_det=n_det)


class SubsampleError:
    """Result of :func:`compute_subsample_error`.

    Covariances span the full band set (user + junk); use :meth:`restrict`
    or pass ``user_bands=True`` to the helpers to collapse onto the reported
    user bands the same way :meth:`simaster.qml.QMLWorkspace._package` does.

    Attributes
    ----------
    cov_analytic, cov_boot : (nb, nb) arrays
        Subsampling-induced covariance of the *bandpowers*, delta-method and
        bootstrap respectively.  Include the noise-bias term when
        ``include_noise_bias`` was set.  These are the terms to add to ``R^-1``.
    cov_n_analytic, cov_n_boot : (nb, nb) arrays
        Subsampling covariance of the *noise bias* ``n`` itself.
    cov_Rc_analytic : (nb, nb) array
        Analytic covariance of ``R_hat @ c`` alone (no noise term), for
        diagnostics / comparison with the response-only budget.
    c_base : (nb,) array
        Reference bandpowers ``R^-1 q`` (the point the error is evaluated at).
    c_boot_mean : (nb,) array
        Mean bootstrap estimate; ``c_boot_mean - c_base`` is the bootstrap
        estimate of the subsampling *bias* (the part the delta method drops).
    R_inv : (nb, nb) array
        Statistical bandpower covariance ``R^-1`` (the optimal-case error).
    include_noise_bias : bool
        Whether the bandpower covariances include the noise-bias fluctuation.
    n_boot : int
    """

    def __init__(self, cov_analytic, cov_boot, cov_Rc_analytic,
                 cov_n_analytic, cov_n_boot, c_base, c_boot_mean, R_inv,
                 is_user_band, ns, n_boot, include_noise_bias):
        self.cov_analytic = cov_analytic
        self.cov_boot = cov_boot
        self.cov_Rc_analytic = cov_Rc_analytic
        self.cov_n_analytic = cov_n_analytic
        self.cov_n_boot = cov_n_boot
        self.c_base = c_base
        self.c_boot_mean = c_boot_mean
        self.R_inv = R_inv
        self.is_user_band = np.asarray(is_user_band, bool)
        self.ns, self.n_boot = int(ns), int(n_boot)
        self.include_noise_bias = bool(include_noise_bias)

    @property
    def _keep(self):
        nbb = self.is_user_band.size
        u = np.flatnonzero(self.is_user_band)
        return np.concatenate([u + s * nbb for s in range(self.ns)])

    def restrict(self, M):
        """Slice a full-band matrix/vector onto the reported user bands."""
        M = np.asarray(M)
        k = self._keep
        return M[np.ix_(k, k)] if M.ndim == 2 else M[k]

    def cov(self, which="boot"):
        """Bandpower subsampling covariance, ``'boot'`` or ``'analytic'``."""
        return self.cov_boot if which == "boot" else self.cov_analytic

    def total_cov(self, which="boot", user_bands=True):
        """``R^-1 + Cov_sub`` -- statistical error inflated by subsampling."""
        tot = self.R_inv + self.cov(which)
        return self.restrict(tot) if user_bands else tot

    def suboptimality(self, which="boot", user_bands=True):
        """Per-band ``sqrt(diag Cov_sub / diag R^-1)``.

        The fractional inflation of each band's error bar caused by
        subsampling; ~0 means the draw cost you nothing, ~1 means the
        subsampling error rivals the statistical error.
        """
        cov = self.cov(which)
        r = np.sqrt(np.clip(np.diag(cov), 0, None)
                    / np.clip(np.diag(self.R_inv), 1e-300, None))
        return self.restrict(r) if user_bands else r

    def n_hat_err(self, which="analytic", user_bands=True):
        """Per-band standard error of the noise bias from subsampling."""
        cov = self.cov_n_boot if which == "boot" else self.cov_n_analytic
        e = np.sqrt(np.clip(np.diag(cov), 0, None))
        return self.restrict(e) if user_bands else e

    def bias(self, user_bands=True):
        """Bootstrap estimate of the subsampling bias ``E[c*] - c_base``."""
        b = self.c_boot_mean - self.c_base
        return self.restrict(b) if user_bands else b

    def __repr__(self):
        ra = self.suboptimality("analytic")
        rb = self.suboptimality("boot")
        return (f"SubsampleError(n_boot={self.n_boot}, "
                f"noise_bias={self.include_noise_bias}; user bands: "
                f"max suboptimality analytic={ra.max():.3f} "
                f"boot={rb.max():.3f}; "
                f"median analytic={np.median(ra):.3f} "
                f"boot={np.median(rb):.3f})")


# ---------------------------------------------------------------------------
# per-mode vectors and the stratified covariance estimator
# ---------------------------------------------------------------------------
def _stratified_cov(store, vecs):
    """Stratified SRS-without-replacement covariance of ``sum_p w0[p] vecs[:,p]``.

    ``Cov = sum_l N_l^2 (1 - n_l/N_l) S_l / n_l`` over strata, with ``S_l`` the
    (ddof=1) sample covariance of the per-mode vectors in stratum ``l``.  The
    ``(1 - n_l/N_l)`` finite-population correction makes the result vanish at
    ``f = 1``; strata with ``n_l < 2`` (e.g. the one-column-per-l floor) cannot
    yield a within-stratum variance and contribute zero.
    """
    cov = np.zeros((store.nb, store.nb))
    for li in range(store.nl):
        cols = np.flatnonzero(store.strat == li)
        n = cols.size
        if n < 2:
            continue
        N = float(store.N_l[li])
        v = vecs[:, cols]
        dev = v - v.mean(axis=1, keepdims=True)
        S = dev @ dev.T / (n - 1)
        cov += N * N * (1.0 - n / N) / n * S
    return cov


def _per_mode_Rc(store, c_full):
    r"""Per-mode contribution to the symmetrised vector ``R_hat @ c``.

    Returns ``m`` of shape ``(nb, P)`` with
    ``R_hat @ c = sum_p w0[p] * m[:, p]``, where each ``m[:, p]`` is
    ``0.5 (M_p + M_p^T) c`` for the unit-weight per-mode matrix ``M_p``.
    """
    ns, nbb, P = store.ns, store.nbb, store.P
    slab = store.slab
    cmat = np.asarray(c_full, float).reshape(ns, nbb)
    col_band = store.col_band
    # M_p c : rows (A,b1), columns live only in band b2(p)
    c2 = cmat[:, col_band]                          # (ns=B, P)
    m1 = np.einsum("ijkp,kp->ijp", slab, c2)        # (ns, nbb, P)
    # M_p^T c : supported only at (B, b2(p))
    term2 = np.einsum("ijkp,ij->kp", slab, cmat)    # (ns=B, P)
    m2 = np.zeros((ns, nbb, P))
    ki = np.arange(ns)[:, None]
    pi = np.arange(P)[None, :]
    m2[ki, col_band[None, :], pi] = term2
    return 0.5 * (m1.reshape(store.nb, P) + m2.reshape(store.nb, P))


def _per_mode_n(store):
    """Per-mode contribution to the noise bias ``n``, shape ``(nb, P)``.

    Each mode feeds a single band ``b2(p)`` across all spectra, so
    ``nm[A*nbb + b2(p), p] = nslab[A, p]`` and zero elsewhere.
    """
    ns, P = store.ns, store.P
    nm = np.zeros((store.nb, P))
    rows = np.arange(ns)[:, None] * store.nbb + store.col_band[None, :]
    pidx = np.broadcast_to(np.arange(P)[None, :], (ns, P))
    nm[rows, pidx] = store.nslab
    return nm


def analytic_Rc_cov(store, c_full):
    """Stratified covariance of ``R_hat @ c`` (response term only)."""
    return _stratified_cov(store, _per_mode_Rc(store, c_full))


def bootstrap_cov(store, q, c_full, n_boot=2000, seed=0,
                  include_noise_bias=True):
    """Bootstrap covariance of ``c* = R*^-1 (y - n*)`` over resampled draws.

    Within each stratum the ``n_l`` sampled modes are resampled by an
    FPC-scaled multinomial: weights ``w0 * (1 + sqrt(1 - n_l/N_l) (k_i - 1))``
    with ``k ~ Multinomial(n_l, 1/n_l)``.  ``R*`` and (when
    ``include_noise_bias``) ``n*`` are rebuilt with the *same* weights, so the
    R/n correlation is preserved and ``R*`` is carried through the full matrix
    inverse.  ``q = y - n_hat`` is the fixed data statistic; with the noise
    term, ``c* = R*^-1 (q + n_hat - n*)``.

    Returns ``(cov_c, c_mean, c_base, cov_n, n_mean)``.
    """
    rng = np.random.default_rng(seed)
    q = np.asarray(q, float).reshape(-1)
    c_base = np.linalg.solve(store.R_hat, q)
    w0 = store.w0
    n_hat = store.n_hat if store.n_hat is not None else store.reconstruct_n()
    # strata that can fluctuate (>=2 modes and not fully sampled)
    strata = []
    for li in range(store.nl):
        cols = np.flatnonzero(store.strat == li)
        n = cols.size
        if n < 2:
            continue
        N = float(store.N_l[li])
        fpc = np.sqrt(max(0.0, 1.0 - n / N))
        if fpc > 0:
            strata.append((cols, n, fpc))
    C = np.empty((n_boot, store.nb))
    Nn = np.empty((n_boot, store.nb))
    for b in range(n_boot):
        w = w0.copy()
        for cols, n, fpc in strata:
            counts = rng.multinomial(n, np.full(n, 1.0 / n))
            w[cols] = w0[cols] * (1.0 + fpc * (counts - 1))
        Rb = store.build_R(w)
        if include_noise_bias:
            nb_star = store.build_n(w)
            Nn[b] = nb_star
            C[b] = np.linalg.solve(Rb, q + n_hat - nb_star)
        else:
            Nn[b] = n_hat
            C[b] = np.linalg.solve(Rb, q)
    cov = np.cov(C, rowvar=False) if n_boot > 1 else np.zeros((store.nb,) * 2)
    cov_n = (np.cov(Nn, rowvar=False) if (n_boot > 1 and include_noise_bias)
             else np.zeros((store.nb,) * 2))
    return (np.atleast_2d(cov), C.mean(axis=0), c_base,
            np.atleast_2d(cov_n), Nn.mean(axis=0))


def compute_subsample_error(store, q, c_full, n_boot=2000, seed=0,
                            include_noise_bias=True):
    """Full subsampling error budget from a :class:`SubsampleStore`.

    Parameters
    ----------
    store : SubsampleStore
        Retained per-mode contributions (``R_hat``/``R_inv``/``n_hat`` set).
    q : (nb,) array
        Reference data statistic ``y - n`` at which the bandpower error is
        evaluated; ``c_base = R^-1 q``.
    c_full : (nb,) array
        Bandpowers used for the analytic delta-method propagation; normally
        ``R^-1 q`` (consistent with the bootstrap).
    n_boot : int
        Number of bootstrap resamples (0 to skip the bootstrap).
    include_noise_bias : bool
        Include the (correlated) noise-bias fluctuation in the *bandpower*
        covariances.  The noise-bias-only covariances and ``n_hat`` error are
        always computed.
    """
    if store.R_hat is None or store.R_inv is None:
        raise ValueError("store is missing R_hat/R_inv")
    Rinv = store.R_inv
    # response-only and noise-only per-mode vectors
    m_Rc = _per_mode_Rc(store, c_full)
    m_n = _per_mode_n(store)
    cov_Rc = _stratified_cov(store, m_Rc)
    cov_n_analytic = _stratified_cov(store, m_n)
    # joint delta(R c + n) when including the noise term
    g = m_Rc + m_n if include_noise_bias else m_Rc
    cov_analytic = Rinv @ _stratified_cov(store, g) @ Rinv
    if n_boot and n_boot > 0:
        cov_boot, c_mean, c_base, cov_n_boot, _ = bootstrap_cov(
            store, q, c_full, n_boot, seed, include_noise_bias)
    else:
        c_base = Rinv @ np.asarray(q).reshape(-1)
        cov_boot = np.zeros_like(cov_analytic)
        cov_n_boot = np.zeros_like(cov_n_analytic)
        c_mean = c_base.copy()
    return SubsampleError(cov_analytic, cov_boot, cov_Rc, cov_n_analytic,
                          cov_n_boot, c_base, c_mean, Rinv, store.is_user_band,
                          store.ns, n_boot, include_noise_bias)
