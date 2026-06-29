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
