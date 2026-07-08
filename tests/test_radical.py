"""Unit tests: BJK radical compression (offset-lognormal likelihood)."""

import numpy as np
import healpy as hp
import jax
import pytest

import simaster as sm
from simaster.exact import ExactQML

NSIDE, LMAX = 8, 20
NPIX = hp.nside2npix(NSIDE)


def setup(ivar0=4e4):
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, ivar0)
    l = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1e-2 / l[2:] ** 2
    b = sm.Bins.linear(2, LMAX, 5)
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX); cl_flat[:2] = 0
    f = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                        fisher_mode="exact", verbose=False,
                        deproject_low_ell=False, cg_tol=1e-9)
    return w, b, cl_flat, mask, ivar


def test_x_factor_identities():
    """c_hat + x = R^-1 y exactly (BJK total power D-hat), x > 0; and in
    the noise-dominated limit x approaches the white-noise N_l."""
    w, b, cl_flat, mask, ivar = setup()
    w.run_exact()
    d = w.cov.sample(jax.random.PRNGKey(2), 1)
    res = w.estimate(d)
    comp = sm.compress(w, result=res)
    yd, _ = w._y_stats(w._filter(d))
    total = w.R_inv @ np.asarray(yd)[:, 0]
    assert np.allclose(comp.c_hat + comp.x, total, rtol=1e-10)
    assert np.all(comp.x > 0)

    # noise-dominated: x ~ sigma^2 Omega_pix
    w2, b2, _, _, ivar2 = setup(ivar0=1e2)
    w2.run_exact()
    x2 = w2.R_inv @ w2.n_hat
    nl_white = (1.0 / ivar2[0]) * hp.nside2pixarea(NSIDE)
    assert np.allclose(x2, nl_white, rtol=0.3)


def test_g_vst():
    """HL variance-stabilizing transform: exact chi^2 Gaussianizer, monotone,
    stable near 1, and equal to ln(x) at leading order."""
    x = np.array([0.25, 0.6, 1.0, 1.7, 4.0])
    g = sm.g_vst(x)
    assert g[2] == 0.0 and np.all(np.diff(g) > 0)
    assert np.allclose(0.5 * g ** 2, x - np.log(x) - 1.0)      # -2lnL = nu * g^2/2
    assert np.all(np.isfinite(sm.g_vst(1 + np.array([0.0, 1e-13, -1e-13]))))
    u = 1e-3
    assert abs(sm.g_vst(1 + u) - np.log1p(u)) < u ** 2          # ln to leading order


def test_hl_compressed_likelihood():
    """HL CompressedLikelihood peaks at the estimate, reduces to the offset-
    lognormal near the fiducial, guards c+x<=0, and save/load round-trips."""
    import os
    import tempfile
    w, b, cl_flat, mask, ivar = setup()
    w.run_exact()
    d = w.cov.sample(jax.random.PRNGKey(3), 1)
    res = w.estimate(d)
    ln = sm.compress(w, result=res, transform="lognormal")
    hl = sm.compress(w, result=res, transform="hl")
    assert hl.transform == "hl"
    c0 = hl.c_hat
    assert np.isclose(hl.loglike(c0), 0.0, atol=1e-9)           # peak at the estimate
    assert hl.loglike(c0) > hl.loglike(1.1 * c0)               # peaked
    cs = c0 + 0.02 / np.sqrt(np.diag(hl.F))                     # 0.02 sigma deviation
    assert abs(hl.loglike(cs) / ln.loglike(cs) - 1.0) < 0.1    # HL -> lognormal near fid
    bad = c0.copy(); bad[np.flatnonzero(hl.use_log)[0]] = -9e9
    assert hl.loglike(bad) == -np.inf
    p = os.path.join(tempfile.mkdtemp(), "hl.npz"); hl.save(p)
    assert sm.CompressedLikelihood.load(p).transform == "hl"


def test_transform_residual_and_calibrated_Mf():
    """transform_residual limits + the M_f-calibrated (full-HL) likelihood is
    calibrated (mean chi^2 ~ dof) on chi^2-distributed fiducial sims, for both
    the lognormal and HL transforms."""
    rng = np.random.default_rng(0)
    nb = 6
    c_fid = np.array([300.0, 30.0, 12.0, 7.0, 5.0, 4.0])
    x = np.ones(nb)
    is_auto = np.ones(nb, bool)
    F = np.eye(nb)
    nu = np.array([3, 5, 9, 15, 25, 40])            # per-band effective d.o.f.

    def draw(n):                                     # skewed, positive: (c+x)*chi2_nu/nu - x
        return (c_fid + x) * (rng.chisquare(nu, size=(n, nb)) / nu) - x

    sim, test = draw(8000), draw(8000)
    # transform_residual: gaussian is exact; lognormal/HL -> (c_hat-c) near fiducial
    d = 1e-4 * c_fid
    assert np.allclose(sm.transform_residual(c_fid + d, c_fid, x, is_auto, "gaussian"), d)
    for tr in ("lognormal", "hl"):
        Xr = sm.transform_residual(c_fid + d, c_fid, x, is_auto, tr, c_fid=c_fid)
        assert np.allclose(Xr, d, rtol=1e-2)
    # calibrated likelihood: mean chi^2 ~ dof (the raw-Fisher form would inflate)
    for tr in ("lognormal", "hl"):
        cov_X, xbar = sm.build_Mf(sim, c_fid, x, is_auto, transform=tr)
        assert cov_X.shape == (nb, nb)
        c2 = np.array([-2 * sm.CompressedLikelihood(
            np.arange(nb), ["a"], ci, x, F, is_auto,
            transform=tr, cov_X=cov_X, xbar=xbar, c_fid=c_fid).loglike(c_fid)
            for ci in test])
        assert abs(c2.mean() - nb) < 0.2 * nb


def test_hl_matrix_transform():
    """Full HL matrix transform: reduces to the scalar form for one field, and
    for (T,E) jointly Gaussianizes the correlated TE cross-spectrum (which the
    per-spectrum scalar HL keeps Gaussian), giving a calibrated likelihood."""
    from scipy.stats import skew, wishart
    rng = np.random.default_rng(1)
    # n=1: matrix transform == scalar transform
    m = 5
    ch = np.abs(rng.normal(10, 3, m)); cc = np.abs(rng.normal(10, 2, m)); xs = np.ones(m)
    a = sm.transform_residual(ch, cc, xs, np.ones(m, bool), "hl", c_fid=cc)
    b = sm.transform_residual(ch, cc, xs, np.ones(m, bool), "hl", c_fid=cc, spec_pairs=[(0, 0)])
    assert np.allclose(a, b)

    # n=2 (T,E) Wishart data, few modes/band -> non-Gaussian, correlated TE
    nb = 3; sp = [(0, 0), (0, 1), (1, 1)]
    S = np.array([[[100.0, 30], [30, 40]], [[50, 15], [15, 25]], [[20, 5], [5, 12]]])
    xm = np.array([[[2.0, 0], [0, 3]]] * nb)
    Ctot = S + xm; nu = np.array([6, 12, 25])
    xf = np.concatenate([np.full(nb, 2.0), np.zeros(nb), np.full(nb, 3.0)])
    cfid = np.concatenate([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]])
    is_auto = np.array([True] * nb + [False] * nb + [True] * nb)

    def draw(n):
        out = np.zeros((n, 3 * nb))
        for bb in range(nb):
            W = wishart.rvs(df=nu[bb], scale=Ctot[bb] / nu[bb], size=n, random_state=rng)
            out[:, bb] = W[:, 0, 0] - xm[bb, 0, 0]
            out[:, nb + bb] = W[:, 0, 1]
            out[:, 2 * nb + bb] = W[:, 1, 1] - xm[bb, 1, 1]
        return out

    sim, test = draw(6000), draw(6000); dof = 3 * nb; F = np.eye(dof)
    covX, xbar = sm.build_Mf(sim, cfid, xf, is_auto, "hl", spec_pairs=sp)
    c2 = np.array([-2 * sm.CompressedLikelihood(
        np.arange(dof), ["a"], ci, xf, F, is_auto, transform="hl",
        cov_X=covX, xbar=xbar, c_fid=cfid, spec_pairs=sp).loglike(cfid) for ci in test])
    assert abs(c2.mean() - dof) < 0.15 * dof                    # calibrated
    # TE residual: matrix HL Gaussianizes it, scalar HL (spec_pairs=None) does not
    te = slice(nb, 2 * nb)
    Xm = np.array([sm.transform_residual(ci, cfid, xf, is_auto, "hl", c_fid=cfid, spec_pairs=sp)[te]
                   for ci in test[:2000]])
    Xs = np.array([sm.transform_residual(ci, cfid, xf, is_auto, "hl", c_fid=cfid)[te]
                   for ci in test[:2000]])
    assert np.nanmedian(np.abs(skew(Xm, 0))) < 0.5 * np.nanmedian(np.abs(skew(Xs, 0)))


@pytest.mark.slow
def test_offset_lognormal_beats_gaussian():
    """Offset-lognormal tracks the exact (dense) likelihood much better
    than a Gaussian; good to |d(-2lnL)| ~< 1 within ~1.5 sigma even
    for the lowest, most heterogeneous band.  The far
    low-C side of the exact likelihood is steeper than lognormal (BJK's
    known limitation; band nu-heterogeneity) -- documented, not asserted.
    """
    w, b, cl_flat, mask, ivar = setup()
    w.run_exact()
    d = np.asarray(w.cov.sample(jax.random.PRNGKey(11), 1))
    res = w.estimate(jax.numpy.asarray(d))
    comp = sm.compress(w, result=res)

    ex = ExactQML([w.fields[0]], w.bins, w.cov.clmat, w.index)
    G, nv = ex.G, ex.noisevar
    lmode = w.index.l

    def exact_lnL(cb_vec):
        clk = b.unbin_cl(cb_vec, LMAX)[lmode]
        C = (G * clk[None, :]) @ G.T + np.diag(nv)
        _, ld = np.linalg.slogdet(C)
        z = np.linalg.solve(C, d[:, 0])
        return -0.5 * (d[:, 0] @ z) - 0.5 * ld

    c0 = comp.c_hat.copy()
    sig = 1.0 / np.sqrt(comp.F[0, 0])
    grid = c0[0] + sig * np.linspace(-2.2, 3.5, 25)
    grid = grid[grid + comp.x[0] > 0]
    scan = [(exact_lnL(np.r_[v, c0[1:]]),
             comp.loglike(np.r_[v, c0[1:]]),
             comp.loglike_gaussian(np.r_[v, c0[1:]])) for v in grid]
    ex2, ln2, ga2 = (-2 * (np.array(a) - np.array(a).max())
                     for a in zip(*scan))
    core = ex2 < 2.5     # ~ +-1.5 sigma
    wide = ex2 < 9.0     # ~ +-3 sigma
    assert np.abs(ln2 - ex2)[core].max() < 1.0
    assert np.abs(ln2 - ex2)[core].max() < np.abs(ga2 - ex2)[core].max()
    assert np.abs(ln2 - ex2)[wide].max() < np.abs(ga2 - ex2)[wide].max()
