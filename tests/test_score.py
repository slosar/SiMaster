"""Unit tests: field-level likelihood score and autodiff consistency."""

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp
import pytest

import simaster as sm
from simaster.score import score, quad_loglike

NSIDE, LMAX = 8, 20
NPIX = hp.nside2npix(NSIDE)


def make_ws():
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, 4e4)
    l = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1e-2 / l[2:] ** 2
    b = sm.Bins.linear(2, LMAX, 5)
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX); cl_flat[:2] = 0
    f = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    return sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                           fisher_mode="exact", verbose=False,
                           deproject_low_ell=False, cg_tol=1e-10)


def test_autodiff_grad_equals_y_statistic():
    """d(-1/2 d^T C^-1 d)/dc_A through CG implicit diff == y_A exactly."""
    w = make_ws()
    d = w.cov.sample(jax.random.PRNGKey(3), 1)
    yd, _ = w._y_stats(w._filter(d))
    cb = jnp.asarray(w.fiducial_bandpowers())
    g = jax.grad(quad_loglike, argnums=1)(w, cb, d)
    assert np.allclose(np.asarray(g), np.asarray(yd)[:, 0], rtol=1e-8)


@pytest.mark.slow
def test_score_identity():
    """Hutchinson score == y - n - R c_fid (exact engine reference)."""
    w = make_ws()
    w.run_exact()
    d = w.cov.sample(jax.random.PRNGKey(3), 1)
    s_hat, s_err = score(w, d, n_probes=512)
    yd, _ = w._y_stats(w._filter(d))
    s_exact = np.asarray(yd)[:, 0] - w.n_hat - w.R_hat @ w.fiducial_bandpowers()
    assert np.all(np.abs(s_hat - s_exact) < 5 * s_err + 1e-12)
