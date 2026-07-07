"""Dense-Cholesky filter (solver='cholesky'): equivalence with CG and with
the dense ExactQML reference, including the ill-conditioned (Planck-noise-like
kappa ~ 1e8) regime where CG cannot converge in any usable iteration count.
"""

import numpy as np
import healpy as hp
import jax
import pytest

import simaster as sm
from simaster.exact import ExactQML

NSIDE, LMAX = 8, 20
NPIX = hp.nside2npix(NSIDE)


def setup_T(ivar0=4e4, nlb=5):
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, ivar0)
    l = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1e-2 / l[2:] ** 2
    f = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    b = sm.Bins.linear(2, LMAX, nlb)
    return f, b, cl, mask, ivar


def test_cholesky_ill_conditioned_matches_dense_reference():
    """At signal/noise ~ 1e8 per mode (true-Planck-like conditioning, where
    the default CG stalls), the Cholesky filter reproduces the brute-force
    dense reference response, noise bias and y statistic."""
    # ivar = 1e9 -> N_l = 1.6e-11 vs C_l(2) = 2.5e-3: kappa ~ 1.5e8
    f, b, cl, mask, ivar = setup_T(ivar0=1e9)
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl}, lmax=LMAX, verbose=False,
                        deproject_low_ell=False, fisher_mode="exact",
                        solver="cholesky", batch_size=500)
    w.run_exact()
    ex = ExactQML([f], w.bins, w.cov.clmat, w.index)
    R = ex.response()
    dR = np.abs(w.R_hat - R) / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    assert dR.max() < 1e-6
    n = ex.noise_bias()
    assert np.abs(w.n_hat - n).max() < 1e-6 * np.abs(n).max()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(w.cov.nrow, 1))
    w._prepare_deprojection()
    yb, _ = w._y_stats(w._filter(jax.numpy.asarray(x)))
    assert np.allclose(np.asarray(yb)[:, 0], ex.y_of(x[:, 0]), rtol=1e-5)


@pytest.mark.slow
def test_cholesky_matches_cg_TEB():
    """T + spin-2, monopole/dipole deprojection on: solver='cholesky' and a
    tight-tolerance CG give the same response and the same estimates."""
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, 4e4)
    l = np.arange(LMAX + 1).astype(float)
    b = sm.Bins.linear(2, LMAX, 6)
    clTT = np.where(l >= 2, 1e-2 / np.maximum(l, 1) ** 2, 0)
    clEE = np.where(l >= 2, 4e-3 / np.maximum(l, 1) ** 2, 0)
    clBB = np.where(l >= 2, 1e-3 / np.maximum(l, 1) ** 2, 0)
    clTE = 0.4 * np.sqrt(clTT * clEE)
    cld = {('t_0', 't_0'): clTT, ('p_E', 'p_E'): clEE,
           ('p_B', 'p_B'): clBB, ('t_0', 'p_E'): clTE}

    def make():
        fT = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
        fP = sm.Field(mask, [np.zeros(NPIX)] * 2, spin=2, ivar=ivar, name="p")
        return [fT, fP]

    w_ch = sm.QMLWorkspace(make(), b, cld, lmax=LMAX, fisher_mode="exact",
                           solver="cholesky", batch_size=300, verbose=False)
    w_cg = sm.QMLWorkspace(make(), b, cld, lmax=LMAX, fisher_mode="exact",
                           cg_tol=1e-10, batch_size=300, verbose=False)
    w_ch.run_exact()
    w_cg.run_exact()
    scale = np.sqrt(np.outer(np.diag(w_cg.R_hat), np.diag(w_cg.R_hat)))
    assert (np.abs(w_ch.R_hat - w_cg.R_hat) / scale).max() < 1e-6
    assert np.allclose(w_ch.n_hat, w_cg.n_hat,
                       atol=1e-6 * np.abs(w_cg.n_hat).max())

    data = w_cg.cov.sample(jax.random.PRNGKey(3), 8)
    r_ch = w_ch.estimate(data)
    r_cg = w_cg.estimate(data)
    for s in r_cg.spec_names:
        sig = np.sqrt(np.diag(r_cg.cov)).max()
        assert np.abs(r_ch.cl[s] - r_cg.cl[s]).max() < 1e-5 * max(sig, 1e-30)


def test_cholesky_factor_invalidated_on_fiducial_update():
    """update_fiducial must drop the dense factor (it is tied to C(fiducial))
    and the next solve must rebuild it."""
    f, b, cl, mask, ivar = setup_T()
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl}, lmax=LMAX, verbose=False,
                        fisher_mode="exact", solver="cholesky", batch_size=500)
    res = w.estimate(w.cov.sample(jax.random.PRNGKey(5), 2))
    assert w._chol is not None
    w.update_fiducial(w._last_c_full.mean(axis=1))
    assert w._chol is None
    res2 = w.estimate(w.cov.sample(jax.random.PRNGKey(6), 2))
    assert w._chol is not None
    assert np.all(np.isfinite(res2.vector()))
