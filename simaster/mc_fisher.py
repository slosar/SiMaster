r"""MC Fisher with uncertainty -- the Monte-Carlo analog of :mod:`simaster.subsample`.

The MC engine (:meth:`simaster.qml.QMLWorkspace.run_mc`) estimates the bandpower
Fisher ``F`` as the sample covariance of the band-power quadratic statistics
``y`` over ``N`` fiducial draws (the OQE identity ``Cov[y] = F``).  A single draw
gives an unbiased but *noisy* ``F-hat`` whose error -- like the subsampling error
-- does not average away in a real analysis and so behaves as a systematic.

This module turns a set of per-seed estimates (each ``R_hat`` from ``nsims``
draws, plus its reference mean ``ybar``) into the combined ``F-hat`` **and** its
uncertainty:

* **element-wise sigma(F_ab)** -- the Wishart variance
  ``Var[F_ab] = (F_aa F_bb + F_ab^2) / (N_eff - 1)``, cross-checked by the
  seed-to-seed scatter (model-free, needs >=2 seeds);
* **the error-bar correction** -- a *frozen* MC ``F-hat`` (one realisation, as in
  any real run) makes the bandpower pulls scale as ``1/sqrt(h)`` with the Hartlap
  factor ``h = (N_eff - n_b - 2)/(N_eff - 1)`` (the inverse-Wishart /
  Dodelson-Schneider effect; verified to 0.5%).  So the honest 1-sigma bandpower
  error bar is ``sqrt(diag F-hat^-1)`` (i.e. the Hartlap-shrunk covariance
  ``h F-hat^-1`` *divided* by ``h``), which yields pull -> 1.
* **suboptimality** vs an exact ``F`` (if supplied), and a **held-out chi2**
  check using sims that did *not* enter ``F-hat`` (a fair, circularity-free bias
  test).

Per-seed estimates are concatenable (just append / :meth:`MCFisherStore.merge`),
so a distributed run (one multi-node job, one seed per rank) saves a store per
rank and merges them here -- mirroring :meth:`SubsampleStore.merge`.  Pure
numpy; no HPC/JAX dependency.
"""
from __future__ import annotations

import glob as _glob
import numpy as np


class MCFisherStore:
    """Per-seed MC Fisher estimates and their reference means.

    Parameters
    ----------
    R, ybar, nsims : lists (one entry per seed) of the (n_b, n_b) sample-cov
        Fisher ``R_hat``, the (n_b,) reference mean ``ybar_fid``, and the int
        ``nsims`` it was built from.
    n, F_l : optional per-seed noise-bias and l-resolved windows (carried through).
    """

    def __init__(self, R, ybar, nsims, n=None, F_l=None):
        self.R = [np.asarray(x, float) for x in R]
        self.Ybar = [np.asarray(x, float) for x in ybar]
        self.nsims = [int(x) for x in nsims]
        self.N = None if n is None else [np.asarray(x, float) for x in n]
        self.Fl = None if F_l is None else [np.asarray(x, float) for x in F_l]
        if not self.R:
            raise ValueError("MCFisherStore needs >=1 seed")
        self.nb = self.R[0].shape[0]

    @classmethod
    def from_files(cls, paths):
        """Load per-seed ``.npz`` (keys: R_hat, ybar_fid, nsims [, n_hat, F_l_hat])."""
        if isinstance(paths, str):
            paths = sorted(_glob.glob(paths))
        R, Y, NS, N, FL = [], [], [], [], []
        for p in sorted(paths):
            d = np.load(p)
            R.append(d["R_hat"]); Y.append(d["ybar_fid"]); NS.append(int(d["nsims"]))
            if "n_hat" in d.files: N.append(d["n_hat"])
            if "F_l_hat" in d.files: FL.append(d["F_l_hat"])
        return cls(R, Y, NS, N or None, FL or None)

    # ---- combination --------------------------------------------------------
    @property
    def K(self):
        return len(self.R)

    @property
    def n_eff(self):
        return int(sum(self.nsims))

    def _w(self):
        w = np.array(self.nsims, float)
        return w / w.sum()

    def merge(self, other):
        """Append another store's seeds (distributed-run reduction)."""
        self.R += other.R; self.Ybar += other.Ybar; self.nsims += other.nsims
        if self.N is not None and other.N is not None: self.N += other.N
        if self.Fl is not None and other.Fl is not None: self.Fl += other.Fl
        return self

    def fisher(self):
        """Combined F-hat = nsims-weighted mean of the per-seed estimates."""
        w = self._w()
        F = sum(wk * Rk for wk, Rk in zip(w, self.R))
        return 0.5 * (F + F.T)

    def ybar_mean(self):
        w = self._w()
        return sum(wk * yk for wk, yk in zip(w, self.Ybar))

    def hartlap(self):
        """Anderson-Hartlap factor for the combined estimate."""
        N, p = self.n_eff, self.nb
        return (N - p - 2.0) / (N - 1.0)

    # ---- uncertainty on F-hat ----------------------------------------------
    def fisher_sigma_wishart(self):
        """Analytic Wishart 1-sigma of each F-hat element, (n_b, n_b)."""
        F = self.fisher(); N = self.n_eff
        return np.sqrt((np.outer(np.diag(F), np.diag(F)) + F ** 2) / (N - 1.0))

    def fisher_sigma_seeds(self):
        """Model-free 1-sigma of the *combined* F-hat from seed scatter (K>=2)."""
        if self.K < 2:
            return None
        Rs = np.stack([0.5 * (R + R.T) for R in self.R])
        return Rs.std(0, ddof=1) / np.sqrt(self.K)

    # ---- bandpower covariance / error bars ----------------------------------
    def bandpower_cov(self, calibrated=True):
        """Bandpower covariance.

        ``calibrated=True`` returns ``F-hat^-1`` -- the honest error bar that
        accounts for the frozen-F-hat (inverse-Wishart) inflation, giving
        pull -> 1.  ``calibrated=False`` returns the Hartlap-shrunk ``h F-hat^-1``
        (unbiased *estimate of* the true ``F^-1`` mean, but under-covers a single
        realisation by ``1/sqrt(h)``).
        """
        Cinv = np.linalg.inv(self.fisher())
        return Cinv if calibrated else self.hartlap() * Cinv

    def predicted_pull(self):
        """Pull std a single frozen Hartlap-shrunk MC cov produces: 1/sqrt(h)."""
        return 1.0 / np.sqrt(self.hartlap())

    def suboptimality(self, F_exact):
        """Per-band sqrt(diag F-hat^-1 / diag F_exact^-1) (>=1; 1 = optimal)."""
        F_exact = 0.5 * (np.asarray(F_exact) + np.asarray(F_exact).T)
        d_hat = np.diag(np.linalg.inv(self.fisher()))
        d_ex = np.diag(np.linalg.inv(F_exact))
        return np.sqrt(np.clip(d_hat, 0, None) / d_ex)

    def dRnorm(self, F_exact):
        """||F_exact^-1 (F-hat - F_exact)||_2 = max|lambda-1|, gen-eig(F-hat, F_exact)."""
        from scipy import linalg as sla
        F_exact = 0.5 * (np.asarray(F_exact) + np.asarray(F_exact).T)
        lam = sla.eigh(self.fisher(), F_exact, eigvals_only=True)
        return float(np.max(np.abs(lam - 1.0)))

    def banded(self, band_of_index, N):
        """Banded Fisher estimate (see :class:`BandedFisher`) from this store's
        combined ``F-hat``, with the N->N+1 self-consistency uncertainty."""
        return BandedFisher(self.fisher(), band_of_index, N, n_eff=self.n_eff)

    def held_out_chi2(self, y_holdout, calibrated=True):
        """Chi2/dof and pull from independent band-power vectors that did NOT
        enter F-hat (circularity-free).  ``y_holdout`` is (n_b, M)."""
        y = np.asarray(y_holdout, float)
        if y.ndim == 1:
            y = y[:, None]
        F = self.fisher(); Cinv = np.linalg.inv(F)
        dev = y - self.ybar_mean()[:, None]
        c = Cinv @ dev                                   # bandpower deviations
        cov = self.bandpower_cov(calibrated=calibrated)
        chi2 = np.einsum("bm,bc,cm->m", c, F, c)         # c^T F c
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        pull = (c / sig[:, None]).ravel()
        return dict(chi2dof=float(chi2.mean() / self.nb),
                    pull_std=float(pull.std()), n_holdout=int(y.shape[1]))


def banded_index_map(n_spectra, n_bands):
    """ell-band ordinal of each bandpower index for the standard spec-major
    layout (bandpower index = spectrum*n_bands + ell_band), so that the
    ell-band offset of two indices is ``|map[i] - map[j]|``.  For a
    :class:`~simaster.qml.QMLWorkspace` ``w`` use
    ``banded_index_map(len(w.spec_pairs), w.bins.nbands)``.
    """
    return np.tile(np.arange(int(n_bands)), int(n_spectra))


def band_fisher(F, band_of_index, N):
    """Block-banded Fisher: zero entries whose ell-band offset exceeds ``N``.

    The mask couples a bandpower only to its ell-neighbours, so the true ``F``
    is banded in ell; the full cross-spectrum block at each ell offset is kept
    (offset 0 -- e.g. the TT/TE/EE correlation at the same ell-band -- is always
    retained).  ``band_of_index`` is from :func:`banded_index_map`.
    """
    bo = np.asarray(band_of_index)
    F = np.asarray(F, float)
    if bo.shape[0] != F.shape[0]:
        raise ValueError("band_of_index length must match F")
    off = np.abs(bo[:, None] - bo[None, :])
    return np.where(off <= N, F, 0.0)


class BandedFisher:
    """Banded Fisher estimate ``F_band(N)`` with a self-consistency uncertainty.

    Banding lets the MC Fisher be estimated from far fewer sims than ``n_b``
    (only the ~``(2N+1)`` ell-neighbour couplings per band carry signal; the
    rest of ``F-hat`` is pure MC noise -- see ``SiMasterTest/band_fisher.py``).
    The truncation/MC uncertainty is estimated *self-consistently* by also
    forming ``F_band(N+1)`` and reporting how much the result moves when the
    next ell-band is admitted -- if adding band ``N+1`` barely changes the
    Fisher (and the error bars), ``N`` has converged.

    Parameters
    ----------
    F_full : (n_b, n_b) full (sample-covariance) Fisher estimate.
    band_of_index : (n_b,) ell-band ordinals, from :func:`banded_index_map`.
    N : kept ell-band half-width (``|offset| <= N``).
    n_eff : optional effective sim count, carried for reference.
    """

    def __init__(self, F_full, band_of_index, N, n_eff=None):
        self.N = int(N)
        self.n_eff = None if n_eff is None else int(n_eff)
        self.band_of_index = np.asarray(band_of_index)
        F = np.asarray(F_full, float)
        self.F_full = 0.5 * (F + F.T)
        self.fisher = band_fisher(self.F_full, self.band_of_index, self.N)
        self.fisher_next = band_fisher(self.F_full, self.band_of_index, self.N + 1)
        self.is_pd = bool(np.all(np.linalg.eigvalsh(self.fisher) > 0))
        self.is_pd_next = bool(np.all(np.linalg.eigvalsh(self.fisher_next) > 0))

    # ---- science products ---------------------------------------------------
    @property
    def cov(self):
        """Bandpower covariance F_band(N)^-1 (the calibrated error bar)."""
        return np.linalg.inv(self.fisher)

    @property
    def errbar(self):
        """1-sigma bandpower error bars sqrt(diag F_band^-1)."""
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))

    # ---- self-consistency uncertainty (N -> N+1) ----------------------------
    def uncertainty(self):
        """Relative change from admitting the (N+1)-th ell-band.

        Returns a dict with the worst-direction Fisher change
        ``||F_N^-1 (F_{N+1} - F_N)||_2`` and the relative change in the
        bandpower error bars (median & max).  These bound the error made by
        truncating at ``N`` (they fold in both residual true coupling and the
        MC noise in band ``N+1``, so they are a conservative 1-sigma).
        """
        FN, FN1 = self.fisher, self.fisher_next
        Cinv = np.linalg.inv(FN)
        lam = np.linalg.eigvals(Cinv @ (FN1 - FN))
        worstdir = float(np.max(np.abs(lam)).real)
        s0 = np.sqrt(np.clip(np.diag(Cinv), 0, None))
        s1 = np.sqrt(np.clip(np.diag(np.linalg.inv(FN1)), 0, None))
        rel = np.abs(s1 / s0 - 1.0)
        return dict(N=self.N, worstdir_fisher=worstdir,
                    errbar_rel_median=float(np.median(rel)),
                    errbar_rel_max=float(np.max(rel)),
                    is_pd=self.is_pd, is_pd_next=self.is_pd_next)

    def converged(self, tol=0.01):
        """True if admitting band N+1 moves the error bars by < ``tol`` (median)."""
        return self.uncertainty()["errbar_rel_median"] < tol

    def summary(self, verbose=True):
        u = self.uncertainty()
        if verbose:
            ne = "" if self.n_eff is None else f" n_eff={self.n_eff}"
            print(f"[banded-fisher] N={self.N}{ne}  PD={u['is_pd']}")
            print(f"  uncertainty from admitting band N+1={self.N + 1}:")
            print(f"    worst-direction Fisher change ||F_N^-1 dF||_2 = "
                  f"{u['worstdir_fisher']:.4f}")
            print(f"    error-bar relative change: median={u['errbar_rel_median']:.4f} "
                  f"max={u['errbar_rel_max']:.4f}")
            print(f"    -> {'CONVERGED' if u['errbar_rel_median'] < 0.01 else 'not converged'} "
                  f"at 1% error-bar tol")
        return u


def banded_fisher(F_or_store, band_of_index, N, verbose=True):
    """Build a :class:`BandedFisher` from a full ``F`` array or an
    :class:`MCFisherStore`, print the N->N+1 self-consistency uncertainty, and
    return it."""
    if isinstance(F_or_store, MCFisherStore):
        F = F_or_store.fisher(); n_eff = F_or_store.n_eff
    else:
        F = F_or_store; n_eff = None
    bf = BandedFisher(F, band_of_index, N, n_eff=n_eff)
    bf.summary(verbose=verbose)
    return bf


def compute_mc_error(store, F_exact=None, verbose=True):
    """Assemble MC Fisher + uncertainty stats into a dict (and optionally print)."""
    F = store.fisher(); h = store.hartlap()
    sw = store.fisher_sigma_wishart(); ss = store.fisher_sigma_seeds()
    diag = np.diag(F)
    rel_w = float(np.median(sw[np.diag_indices(store.nb)] / np.abs(diag)))
    out = dict(n_eff=store.n_eff, K=store.K, nb=store.nb, hartlap=float(h),
               predicted_pull=float(store.predicted_pull()),
               median_rel_sigma_diag_wishart=rel_w)
    if ss is not None:
        out["seed_vs_wishart_ratio"] = float(
            np.median(ss[np.diag_indices(store.nb)]
                      / sw[np.diag_indices(store.nb)]))
    if F_exact is not None:
        out["dRnorm"] = store.dRnorm(F_exact)
        out["suboptimality_median"] = float(np.median(store.suboptimality(F_exact)))
    if verbose:
        print(f"[mc-fisher] n_eff={out['n_eff']} (K={out['K']} seeds) n_b={out['nb']} "
              f"hartlap={h:.3f}")
        print(f"  sigma(F_ab) Wishart: median rel on diag = {rel_w:.3f} "
              f"(~sqrt(2/n_eff)={np.sqrt(2/store.n_eff):.3f})")
        if ss is not None:
            print(f"  seed-empirical / Wishart sigma ratio (diag) = "
                  f"{out['seed_vs_wishart_ratio']:.2f}  (->1 validates Wishart)")
        print(f"  frozen-F-hat error-bar inflation 1/sqrt(h) = "
              f"{out['predicted_pull']:.3f}  -> use calibrated cov (F-hat^-1) for pull->1")
        if F_exact is not None:
            print(f"  vs exact F:  dRnorm={out['dRnorm']:.3f}  "
                  f"suboptimality(med)={out['suboptimality_median']:.3f}")
    return out
