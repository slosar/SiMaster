"""Quadratic maximum-likelihood bandpower estimation.

Estimator (Tegmark 1997, generalized to multiple correlated fields and
arbitrary bands; see report/):

    y_A(d)  = 1/2 d^T M P_A M d,           P_A = dC/dc_A,
    c_hat   = R^-1 (y(d) - n),             n_A = 1/2 Tr[M N M P_A],
    R_AB    = 1/2 Tr[M P_A M P_B],         cov(c_hat) ~= R^-1 at fiducial.

M is the inverse-covariance filter: C^-1 (computed by preconditioned CG,
never by dense inversion) or its template-deprojected pseudo-inverse
(alpha -> infinity limit of C + alpha t t^T, via Sherman-Morrison-Woodbury).

The response matrix R and the noise bias n are evaluated by Monte Carlo:
for sims x ~ N(0, C_fid),  cov[y_A(x), y_B(x)] = R_AB exactly (this holds
for the deprojected filter too, since M C M = M), and the same sims provide
the bandpower window functions F_bl = cov[y_b, y_l] against the unbinned
per-multipole statistics at zero extra cost.  The estimator is unbiased for
any fiducial spectrum; a wrong fiducial only costs optimality.  Iterating
re-centers the fiducial on the estimate.
"""

from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp

from .utils import RealAlmIndex, cl_matrix, psd_floor
from .field import Field
from .bins import Bins
from .covariance import CovModel
from .cg import pcg, solve_C


class BandpowerResult:
    """Container for estimated bandpowers.

    Attributes
    ----------
    ells : effective multipoles of the user bands.
    cl : dict mapping spectrum name -> (nbands,) or (ndata, nbands).
    cov : covariance of the concatenated user-band vector.
    spec_names : spectrum names in the order used by ``cov``.
    """

    def __init__(self, ells, cl, cov, spec_names, windows=None, ls=None):
        self.ells, self.cl, self.cov = ells, cl, cov
        self.spec_names, self.windows, self.ls = spec_names, windows, ls
        self.deviation = False  # True: cl holds deviations from the fiducial

    def vector(self):
        """Concatenated (ndata, nspec*nbands) bandpower vector."""
        return np.concatenate([np.atleast_2d(self.cl[s]) for s in self.spec_names],
                              axis=-1)

    def chi2(self, theory):
        """chi^2 of each data vector against a theory bandpower vector.

        ``theory``: dict spec name -> (nbands,) (e.g. from
        QMLWorkspace.predict) or a flat vector.
        """
        if isinstance(theory, dict):
            theory = np.concatenate([np.asarray(theory[s]) for s in self.spec_names])
        r = self.vector() - theory[None, :]
        cinv = np.linalg.inv(self.cov)
        return np.einsum("bi,ij,bj->b", r, cinv, r)


class QMLWorkspace:
    """Precomputed QML machinery for a fixed set of fields, bins and fiducial.

    Parameters
    ----------
    fields : Field or list of Field (same nside).
    bins : Bins with the *requested* bands.  Junk bands are added
        automatically so the band basis covers every multipole in
        [lmin, lmax]; they are estimated and marginalized, not reported.
    cl_fid : fiducial signal spectra: dict {(name_i, name_j): C_l} with
        component names like 'f0_0', 'f1_E' (see Field.comp_names), or a
        full (ncomp, ncomp, lmax+1) array.
    lmax : covariance bandlimit; default 3*nside - 1.  The input maps are
        assumed bandlimited to lmax (generate validation sims accordingly;
        for real data choose lmax high enough that signal+noise above it is
        negligible -- aliasing from uncovered multipoles is *not* modeled).
    fisher_mode : 'exact', 'mc' or 'auto' (default).
        'exact' computes the response (binned Fisher) matrix, noise bias and
        bandpower windows deterministically by solving M G column-by-column
        (batched CG over all (comp, l, m) modes) -- no dense inversion, cost
        ~ n_modes CG solves, feasible up to nside ~ 64.  'mc' estimates them
        from fiducial simulations: unbiased, scalable to any nside, but the
        frozen MC noise in R adds ~ sigma * sqrt(SNR_tot^2 / n_sims) offsets
        to the estimates, where SNR_tot^2 is the *total* squared S/N over
        all bands -- size n_sims accordingly.  'auto' picks 'exact' when
        the mode count makes it cheaper than the requested n_sims_fisher.
    n_sims_fisher / n_sims_noise : Monte-Carlo sample sizes for 'mc' mode.
    template_alpha : None (default) -> exact deprojection (alpha -> inf,
        Woodbury); finite value -> add alpha_rel*tr(C)/||t||^2 * t t^T to C.
    deproject_low_ell : marginalize monopole+dipole of every spin-0 field.
    backend : 'dense' (GPU GEMM, nside <~ 64), 'ducc' (matrix-free, any
        nside), or 'auto'.
    """

    def __init__(self, fields, bins: Bins, cl_fid, lmax=None, lmin=2,
                 backend="auto", fisher_mode="auto", fisher_frac=0.25,
                 n_sims_fisher=2048, n_sims_noise=512, cg_tol=1e-5,
                 cg_maxiter=700, seed=1234, template_alpha=None,
                 deproject_low_ell=True, batch_size=256, cachedir=None,
                 verbose=True):
        if isinstance(fields, Field):
            fields = [fields]
        self.fields = fields
        nside = fields[0].nside
        for i, f in enumerate(fields):
            if f.nside != nside:
                raise ValueError("all fields must share nside")
            if f.name is None:
                f.name = f"f{i}"
        self.nside = nside
        self.lmin = int(lmin)
        self.lmax = int(lmax) if lmax is not None else 3 * nside - 1
        self.index = RealAlmIndex(self.lmin, self.lmax)
        self.verbose = verbose
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.batch_size = batch_size
        self.n_sims_fisher, self.n_sims_noise = n_sims_fisher, n_sims_noise
        self._key = jax.random.PRNGKey(seed)

        # components and spectra
        self.comp_names = sum([f.comp_names for f in fields], [])
        nc = len(self.comp_names)
        self.spec_pairs = [(i, j) for i in range(nc) for j in range(i, nc)]
        self.spec_names = [f"{self.comp_names[i]} x {self.comp_names[j]}"
                           for i, j in self.spec_pairs]

        # bands (user + junk to cover [lmin, lmax])
        self.user_bins = bins
        self.bins, self.is_user_band = bins.extend_to_cover(self.lmin, self.lmax)
        self.ls = np.arange(self.lmin, self.lmax + 1)
        band_of_l = np.zeros(self.ls.size, dtype=int)
        for b in range(self.bins.nbands):
            band_of_l[(self.ls >= self.bins.lo[b]) & (self.ls <= self.bins.hi[b])] = b
        self._band_of_l = jnp.asarray(band_of_l)
        self._l_of_mode = jnp.asarray(self.index.l - self.lmin)

        # fiducial spectra
        if isinstance(cl_fid, dict):
            clmat = cl_matrix(cl_fid, self.comp_names, self.lmax)
        else:
            clmat = np.asarray(cl_fid, dtype=np.float64)[:, :, : self.lmax + 1]
        clmat = clmat.copy()
        clmat[:, :, : self.lmin] = 0.0

        # automatic monopole/dipole deprojection templates for spin-0 fields
        extra_templates = []
        if deproject_low_ell:
            import healpy as hp
            for fi, f in enumerate(fields):
                if f.spin != 0:
                    continue
                vx, vy, vz = hp.pix2vec(nside, f.obs_pix)
                for tmpl in (np.ones_like(vx), vx, vy, vz):
                    extra_templates.append((fi, (f.mask * tmpl)[None, :]))

        if backend == "auto":
            backend = "dense" if nside <= 64 else "ducc"
        self.backend = backend
        nmodes_tot = len(self.comp_names) * self.index.nmodes
        if fisher_mode == "auto":
            fisher_mode = ("exact" if backend == "dense"
                           and nmodes_tot <= max(40000, 2 * n_sims_fisher)
                           else "mc")
        self.fisher_mode = fisher_mode
        self.fisher_frac = fisher_frac
        t0 = time.time()
        self.cov = CovModel(fields, clmat, self.index, backend=backend,
                            cachedir=cachedir)
        self._add_templates(extra_templates)
        if template_alpha is not None:
            self.cov.set_template_alpha(template_alpha)
        self._template_mode = "alpha" if template_alpha is not None else "woodbury"
        self._log(f"operators built ({backend}) in {time.time() - t0:.1f}s; "
                  f"nrow={self.cov.nrow}, nmodes={self.cov.ncomp * self.index.nmodes}, "
                  f"bands={self.bins.nbands} ({int(self.is_user_band.sum())} user)")

        self._mc_done = False
        self._V = None  # C^-1 T for Woodbury deprojection

    # ------------------------------------------------------------------ misc --
    def _log(self, msg):
        if self.verbose:
            print(f"[simaster] {msg}", flush=True)

    def _add_templates(self, extra):
        """Append automatically generated templates to the CovModel."""
        if not extra:
            return
        cols = [] if self.cov.Tmat is None else [np.asarray(self.cov.Tmat).T]
        for fi, t in extra:
            v = np.zeros(self.cov.nrow)
            v[self.cov.slices[fi]] = np.tile(t, (self.fields[fi].ncomp, 1)
                                             ).reshape(-1) if t.shape[0] == 1 \
                else t.reshape(-1)
            cols.append(v[None, :])
        T = np.concatenate(cols, axis=0).T
        self.cov.Tmat = jnp.asarray(T, dtype=self.cov.dtype)
        self.cov.n_templates = T.shape[1]

    def _next_key(self):
        self._key, k = jax.random.split(self._key)
        return k

    # ------------------------------------------------------------ the filter --
    def _solve(self, B):
        """C^-1 B by preconditioned CG (with preconditioner auto-repair)."""
        X, self.last_cg = solve_C(self.cov, B, tol=self.cg_tol,
                                  maxiter=self.cg_maxiter, log=self._log)
        return X

    def _prepare_deprojection(self):
        if self.cov.Tmat is None or self._template_mode == "alpha":
            self._V = None
            return
        V = self._solve(self.cov.Tmat)
        S = self.cov.Tmat.T @ V
        self._V = V
        self._S_inv = jnp.linalg.inv(S + 1e-30 * jnp.eye(S.shape[0]))

    def _filter(self, B):
        """Apply M = C^-1 (alpha mode) or the deprojecting pseudo-inverse."""
        Z = self._solve(B)
        if self._V is not None:
            Z = Z - self._V @ (self._S_inv @ (self.cov.Tmat.T @ Z))
        return Z

    # ------------------------------------------------------- quadratic stats --
    def _y_stats(self, Z):
        """Band and per-l quadratic statistics of filtered vectors.

        Returns (y_band (nspec*nbands, B), y_l (nspec*nl, B)).
        """
        A = self.cov.to_modes(Z)  # (Nc, K, B)
        nl = self.ls.size
        ybs, yls = [], []
        for (i, j) in self.spec_pairs:
            prod = A[i] * A[j]
            fac = 0.5 if i == j else 1.0
            yl = fac * jax.ops.segment_sum(prod, self._l_of_mode,
                                           num_segments=nl)
            yb = jax.ops.segment_sum(yl, self._band_of_l,
                                     num_segments=self.bins.nbands)
            yls.append(yl)
            ybs.append(yb)
        return jnp.concatenate(ybs, axis=0), jnp.concatenate(yls, axis=0)

    # ----------------------------------------------------------- exact engine --
    def run_exact(self, sample_frac=None, sample_seed=0):
        """Deterministic (or column-subsampled) response computation.

        Solves V = M G over (comp, l, m) mode columns with batched CG and
        accumulates the l-resolved response tensor

            T[a,b,c,d](l1,l2) = sum_{k in l1, k' in l2} H[ak,ck'] H[bk,dk'],
            H = G^T M G,

        from which the binned response R, per-multipole windows F_bl and
        noise bias n follow for any binning.  Cost: n_columns CG solves
        (batched); memory: O(Nc^4 nl^2), never the dense H.

        With ``sample_frac = f < 1`` only a random fraction of the k'
        columns is solved -- stratified per l' (>= 1 column each, sampled
        without replacement) and renormalized by N_l'/n_l', which keeps the
        estimator exactly unbiased.  Because the row index of H is always
        summed exactly (it comes from one adjoint SHT per solve), the
        sampling noise is *local in bands*: the induced offset on band A is
        ~ sigma_A * SNR_A * sqrt(rho (1-f)/n_A) with the band's own S/N,
        in contrast to the sims-MC engine whose Wishart noise couples all
        bands coherently (offset ~ sigma_A * sqrt(SNR_tot^2/N_sims)).
        The sampled R is symmetrized; for very small f check its
        conditioning.
        """
        self._prepare_deprojection()
        Nc, K = len(self.comp_names), self.index.nmodes
        nl = self.ls.size
        lmode = np.asarray(self.index.l) - self.lmin
        # column selection (all modes, or stratified subsample per l)
        if sample_frac is None or sample_frac >= 1.0:
            sel = np.arange(K)
            scale_l = np.ones(nl)
        else:
            rng = np.random.default_rng(sample_seed)
            sel = []
            scale_l = np.ones(nl)
            for li in range(nl):
                kl = np.flatnonzero(lmode == li)
                n = max(1, int(round(sample_frac * kl.size)))
                sel.append(rng.choice(kl, size=n, replace=False))
                scale_l[li] = kl.size / n
            sel = np.sort(np.concatenate(sel))
        nsel = sel.size
        # T and W2d accumulate on device; transferred to host once at the end
        # (per-l host syncs dominated the runtime otherwise)
        T_dev = jnp.zeros((Nc, Nc, Nc, Nc, nl, nl))
        W2d_dev = jnp.zeros((Nc, Nc, K))
        noisevar = self.cov.noisevar

        J = max(1, self.batch_size // Nc)
        t0 = time.time()
        for j0 in range(0, nsel, J):
            kcols = sel[j0:j0 + J]
            jc = kcols.size
            # one-hot mode batches: columns ordered (k major, comp minor)
            E = jnp.zeros((Nc, K, jc * Nc))
            kk = jnp.repeat(jnp.asarray(kcols), Nc)
            cc = jnp.tile(jnp.arange(Nc), jc)
            E = E.at[cc, kk, jnp.arange(jc * Nc)].set(1.0)
            Gc = self.cov.from_modes(E)
            V = self._filter(Gc)
            A = self.cov.to_modes(V)                      # (Nc, K, jc*Nc)
            H = A.reshape(Nc, K, jc, Nc).transpose(0, 1, 3, 2)  # (a,k,c',j)
            Vr = V.reshape(-1, jc, Nc)
            W2d_dev = W2d_dev.at[:, :, jnp.asarray(kcols)].add(
                jnp.einsum("pjc,p,pjd->cdj", Vr, noisevar, Vr))
            # accumulate T: loop over row-l blocks, all on device
            lj = jnp.asarray(lmode[kcols])
            for l1 in range(nl):
                sl = self.index.band_slice(l1 + self.lmin, l1 + self.lmin)
                Hl = H[:, sl.start:sl.stop]               # (a, kl, c', j)
                X = jnp.einsum("akcj,bkdj->abcdj", Hl, Hl)
                Xl = jax.ops.segment_sum(X.transpose(4, 0, 1, 2, 3),
                                         lj, num_segments=nl)
                T_dev = T_dev.at[:, :, :, :, l1, :].add(
                    Xl.transpose(1, 2, 3, 4, 0))
            if self.verbose and (j0 // J) % 8 == 0:
                self._log(f"exact response {j0 + jc}/{nsel} columns "
                          f"(cg {self.last_cg[0]} it)")
        # stratified-subsample renormalization (no-op when all columns run)
        T = np.asarray(T_dev) * scale_l[None, None, None, None, None, :]
        W2d = np.asarray(W2d_dev) * scale_l[lmode][None, None, :]

        # ---- assemble response / windows / noise bias -----------------------
        ns = len(self.spec_pairs)

        def pairs(c, d):
            return [(c, d)] if c == d else [(c, d), (d, c)]

        Rl = np.zeros((ns, nl, ns, nl))
        for A, (c, d) in enumerate(self.spec_pairs):
            for B, (e, f) in enumerate(self.spec_pairs):
                acc = 0.0
                for (g, dd) in pairs(c, d):
                    for (ee, ff) in pairs(e, f):
                        acc = acc + T[dd, g, ee, ff]
                Rl[A, :, B, :] = 0.5 * acc

        band_of_l = np.asarray(self._band_of_l)
        nbb = self.bins.nbands
        Fb = np.zeros((ns, nbb, ns, nl))
        for b in range(nbb):
            Fb[:, b] = Rl[:, band_of_l == b].sum(axis=1)
        self.F_l_hat = Fb.reshape(ns * nbb, ns * nl)
        Rb = np.zeros((ns, nbb, ns, nbb))
        for b in range(nbb):
            Rb[:, :, :, b] = Fb[:, :, :, band_of_l == b].sum(axis=3)
        R = Rb.reshape(ns * nbb, ns * nbb)
        # column subsampling breaks exact symmetry; symmetrizing averages
        # two semi-independent estimates (harmless no-op for the full run)
        self.R_hat = 0.5 * (R + R.T)

        n_l = np.zeros((ns, nl))
        for A, (c, d) in enumerate(self.spec_pairs):
            fac = 0.5 if c == d else 1.0
            n_l[A] = fac * np.bincount(lmode, weights=W2d[c, d], minlength=nl)
        nb_arr = np.zeros((ns, nbb))
        for b in range(nbb):
            nb_arr[:, b] = n_l[:, band_of_l == b].sum(axis=1)
        self.n_hat = nb_arr.reshape(-1)
        self.n_hat_err = np.zeros_like(self.n_hat)

        self.hartlap = 1.0
        self.R_inv = np.linalg.inv(self.R_hat)
        self.ybar_fid = None
        self._mc_done = True
        self._log(f"exact response done in {(time.time() - t0) / 60:.1f} min "
                  f"({nsel}/{K} columns, "
                  f"condition {np.linalg.cond(self.R_hat):.2e})")

    # ------------------------------------------------------------- MC engine --
    def run_mc(self, n_sims_fisher=None, n_sims_noise=None):
        """Estimate response matrix R, noise bias n, and window functions."""
        nf = n_sims_fisher or self.n_sims_fisher
        nn = n_sims_noise or self.n_sims_noise
        self._prepare_deprojection()
        nb = self.bins.nbands * len(self.spec_pairs)
        nl = self.ls.size * len(self.spec_pairs)

        t0 = time.time()
        s_y = np.zeros(nb); s_yy = np.zeros((nb, nb))
        s_l = np.zeros(nl); s_yl = np.zeros((nb, nl))
        done = 0
        while done < nf:
            B = min(self.batch_size, nf - done)
            x = self.cov.sample(self._next_key(), B)
            yb, yl = self._y_stats(self._filter(x))
            yb, yl = np.asarray(yb), np.asarray(yl)
            s_y += yb.sum(1); s_yy += yb @ yb.T
            s_l += yl.sum(1); s_yl += yb @ yl.T
            done += B
            self._log(f"fisher MC {done}/{nf} (cg iters {self.last_cg[0]}, "
                      f"res {self.last_cg[1]:.1e})")
        mu = s_y / nf; mul = s_l / nf
        self.R_hat = (s_yy - nf * np.outer(mu, mu)) / (nf - 1)
        self.F_l_hat = (s_yl - nf * np.outer(mu, mul)) / (nf - 1)
        self.ybar_fid = mu
        self.n_sims_fisher_used = nf

        s_n = np.zeros(nb); s_nn = np.zeros((nb, nb)); done = 0
        while done < nn:
            B = min(self.batch_size, nn - done)
            x = self.cov.sample_noise(self._next_key(), B)
            yb, _ = self._y_stats(self._filter(x))
            yb = np.asarray(yb)
            s_n += yb.sum(1); s_nn += yb @ yb.T
            done += B
        self.n_hat = s_n / nn
        self.n_hat_err = np.sqrt(np.diag((s_nn - nn * np.outer(self.n_hat, self.n_hat))
                                         / (nn - 1) / nn).clip(0))

        # Hartlap-corrected inverse response
        h = (nf - nb - 2) / (nf - 1)
        if h <= 0:
            raise ValueError(f"n_sims_fisher={nf} too small for {nb} bins")
        self.hartlap = h
        self.R_inv = h * np.linalg.inv(self.R_hat)
        self._mc_done = True
        self._log(f"MC done in {time.time() - t0:.1f}s "
                  f"(R condition {np.linalg.cond(self.R_hat):.2e}, hartlap {h:.3f})")

    # ------------------------------------------------------------- estimation --
    def run_mean_debias(self, n_sims=128):
        """Mean of y over independent fiducial sims, for the around-fiducial
        (sim-debiased) estimator

            c_hat = c_fid + R^-1 (y(d) - <y>_fid sims).

        The mean carries the *exact* response of the actual filter, so a
        stochastic R-hat error only multiplies (c_true - c_fid) instead of
        c_true -- essential when R comes from the 'subsampled' or 'mc'
        engines on signal-dominated data.  Mean error ~ sigma/sqrt(n_sims)
        per band, so n_sims ~ 1/eps^2 (~100) suffices.
        """
        nb = self.bins.nbands * len(self.spec_pairs)
        s = np.zeros(nb)
        done = 0
        while done < n_sims:
            B = min(self.batch_size, n_sims - done)
            x = self.cov.sample(self._next_key(), B)
            yb, _ = self._y_stats(self._filter(x))
            s += np.asarray(yb).sum(1)
            done += B
        self.ybar_debias = s / n_sims
        self.n_mean_sims = n_sims

    def fiducial_bandpowers(self):
        """Band means of the current fiducial spectra (exact if band-flat,
        e.g. after update_fiducial)."""
        out = np.zeros((len(self.spec_pairs), self.bins.nbands))
        for si, (i, j) in enumerate(self.spec_pairs):
            out[si] = self.bins.bin_cl(self.cov.clmat[i, j])
        return out.reshape(-1)

    def fiducial_yref(self):
        """Expected y at the fiducial: ybar_A = sum_l F_Al c^fid_l + n_A.

        Deterministic and exact in the 'exact'/'subsampled' engines (the
        l-resolved windows are available); in 'mc' mode falls back to the
        simulation mean.  This is the reference point for fitting flat
        band *deviations* away from a smooth (curved-in-l) fiducial.
        """
        if self.fisher_mode in ("exact", "subsampled"):
            clvec = np.concatenate([self.cov.clmat[i, j, self.ls]
                                    for (i, j) in self.spec_pairs])
            return self.F_l_hat @ clvec + self.n_hat
        if self.ybar_fid is None:
            raise RuntimeError("mc mode: run_mc() first")
        return self.ybar_fid

    def pack_data(self, maps_per_field):
        """Full-sky maps [[m] or [Q,U] per field] -> data matrix (nrow, B)."""
        import healpy as hp
        cols = []
        for f, maps in zip(self.fields, maps_per_field):
            npix = hp.nside2npix(f.nside)
            arr = np.asarray(maps, dtype=np.float64).reshape(-1, f.ncomp, npix)
            cols.append(arr[:, :, f.obs_pix].reshape(arr.shape[0], -1))
        return jnp.asarray(np.concatenate(cols, axis=1).T)

    def estimate(self, data=None, deviations=False):
        """Estimate bandpowers.

        data : None (use the fields' own maps), a (nrow, B) matrix from
            :meth:`pack_data`, or a list of per-field full-sky maps.
        deviations : if True, keep the full (smooth, curved-in-l) fiducial
            spectra inside the covariance model and fit only flat band
            *deviations* dc_b away from it:

                dc = R^-1 (y(d) - ybar_fid),   ybar_fid = F c^fid_l + n.

            E[dc] = R^-1 F (c_true - c_fid)_l: exactly zero at the
            fiducial and free of flat-band binning bias otherwise (the
            curvature lives in the fiducial, not in the band model).  The
            result's ``cl`` then contains deviations (result.deviation is
            True); add ``bins.bin_cl(fiducial)`` or use iterate() --
            which in this mode *adds* the flat deviations to the smooth
            fiducial -- for total bandpowers.
        Returns a BandpowerResult restricted to the user bands.
        """
        if not self._mc_done:
            if self.fisher_mode == "exact":
                self.run_exact()
            elif self.fisher_mode == "subsampled":
                self.run_exact(sample_frac=self.fisher_frac)
            else:
                self.run_mc()
        if data is None:
            data = jnp.asarray(np.concatenate(
                [f.data_vector() for f in self.fields]))[:, None]
        elif isinstance(data, (list, tuple)):
            data = self.pack_data(data)
        yd, _ = self._y_stats(self._filter(data))
        if deviations:
            c_full = self.R_inv @ (np.asarray(yd)
                                   - self.fiducial_yref()[:, None])
        elif getattr(self, "ybar_debias", None) is not None:
            # around-fiducial form: response error multiplies (c - c_fid)
            c_full = (self.fiducial_bandpowers()[:, None]
                      + self.R_inv @ (np.asarray(yd)
                                      - self.ybar_debias[:, None]))
        else:
            c_full = self.R_inv @ (np.asarray(yd) - self.n_hat[:, None])
        self._last_c_full = c_full
        self._last_deviation = deviations
        res = self._package(c_full, self.R_inv)
        res.deviation = deviations
        return res

    def _package(self, c_full, cov_full):
        nspec, nb_all = len(self.spec_pairs), self.bins.nbands
        keep = np.concatenate([np.flatnonzero(self.is_user_band) + s * nb_all
                               for s in range(nspec)])
        cl = {}
        for si, s in enumerate(self.spec_names):
            rows = np.flatnonzero(self.is_user_band) + si * nb_all
            cl[s] = c_full[rows].T.squeeze()
        cov = cov_full[np.ix_(keep, keep)]
        wind = self.window_functions()[keep] if self._mc_done else None
        return BandpowerResult(self.user_bins.get_effective_ells(), cl, cov,
                               self.spec_names, windows=wind, ls=self.ls)

    def window_functions(self):
        """W[(s,b), (s',l)] such that <c_hat> = W @ c_l(theory)."""
        return self.R_inv @ self.F_l_hat

    def predict(self, cl_theory):
        """Expected user-band estimates for theory spectra (window-convolved).

        cl_theory: dict {(name_i, name_j): C_l} or (ncomp, ncomp, lmax+1).
        Returns dict spec name -> (n_user_bands,).
        """
        if isinstance(cl_theory, dict):
            clmat = cl_matrix(cl_theory, self.comp_names, self.lmax)
        else:
            clmat = np.asarray(cl_theory)
        clvec = np.concatenate([clmat[i, j, self.ls] for (i, j) in self.spec_pairs])
        cb = self.window_functions() @ clvec
        nb_all = self.bins.nbands
        out = {}
        for si, s in enumerate(self.spec_names):
            rows = np.flatnonzero(self.is_user_band) + si * nb_all
            out[s] = cb[rows]
        return out

    # -------------------------------------------------------------- iteration --
    def update_fiducial(self, c_full):
        """Replace the fiducial spectra by flat bandpowers c (all bands)."""
        nb_all = self.bins.nbands
        clmat = np.zeros_like(self.cov.clmat)
        for si, (i, j) in enumerate(self.spec_pairs):
            cb = c_full[si * nb_all:(si + 1) * nb_all]
            cl = self.bins.unbin_cl(cb, self.lmax)
            clmat[i, j] = clmat[j, i] = cl
        clmat[:, :, : self.lmin] = 0.0
        clmat = psd_floor(clmat)
        self.cov.set_clmat(clmat)
        self._mc_done = False
        self.ybar_debias = None  # stale: tied to the previous fiducial

    def update_fiducial_deviations(self, dc_full):
        """Add flat band deviations to the (smooth) fiducial spectra,
        preserving their curvature within bands."""
        nb_all = self.bins.nbands
        clmat = np.array(self.cov.clmat, copy=True)
        for si, (i, j) in enumerate(self.spec_pairs):
            dcl = self.bins.unbin_cl(dc_full[si * nb_all:(si + 1) * nb_all],
                                     self.lmax)
            clmat[i, j] = clmat[j, i] = clmat[i, j] + dcl
        clmat[:, :, : self.lmin] = 0.0
        clmat = psd_floor(clmat)
        self.cov.set_clmat(clmat)
        self._mc_done = False
        self.ybar_debias = None

    def iterate(self, data=None, n_iter=2, deviations=False):
        """Newton-Raphson-style iteration: re-center fiducial on estimates.

        With ``deviations=True`` the fiducial keeps its smooth shape and
        the flat band deviations are *added* to it each iteration (the
        recommended mode with a curved fiducial); otherwise the fiducial
        is replaced by flat bandpowers.  Returns the list of
        BandpowerResult, one per iteration (the data batch mean drives the
        update when B > 1).
        """
        history = []
        for it in range(n_iter):
            res = self.estimate(data, deviations=deviations)
            history.append(res)
            if it < n_iter - 1:
                upd = self._last_c_full.mean(axis=1)
                if deviations:
                    self.update_fiducial_deviations(upd)
                else:
                    self.update_fiducial(upd)
                self._log(f"iteration {it + 1}: fiducial updated")
        return history
