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
