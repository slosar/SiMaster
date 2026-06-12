"""Unit tests: end-to-end QML estimation on small problems.

These compare the scalable Monte-Carlo machinery against the exact dense
reference and verify unbiasedness on simulated data.
"""

import numpy as np
import healpy as hp
import jax
import pytest

import simaster as sm
from simaster.exact import ExactQML

NSIDE, LMAX = 8, 20
NPIX = hp.nside2npix(NSIDE)


def setup_T(nlb=5, ivar0=4e4):
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, ivar0)
    l = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1e-2 / l[2:] ** 2
    f = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    b = sm.Bins.linear(2, LMAX, nlb)
    return f, b, cl, mask, ivar


@pytest.mark.slow
def test_mc_response_matches_exact():
    f, b, cl, mask, ivar = setup_T()
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl}, lmax=LMAX,
                        n_sims_fisher=8000, n_sims_noise=2000,
                        batch_size=1000, seed=3, verbose=False,
                        deproject_low_ell=False)
    w.run_mc()
    ex = ExactQML([f], w.bins, w.cov.clmat, w.index)
    R = ex.response()
    err = np.abs(w.R_hat - R) / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    assert err.max() < 6 * np.sqrt(2.0 / 8000)
    n = ex.noise_bias()
    assert np.allclose(w.n_hat, n, atol=6 * np.max(w.n_hat_err))


def test_y_statistic_matches_exact():
    f, b, cl, mask, ivar = setup_T()
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl}, lmax=LMAX, verbose=False,
                        deproject_low_ell=False, cg_tol=1e-10)
    ex = ExactQML([f], w.bins, w.cov.clmat, w.index)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(w.cov.nrow, 1))
    w._prepare_deprojection()
    yb, _ = w._y_stats(w._filter(jax.numpy.asarray(x)))
    y_exact = ex.y_of(x[:, 0])
    assert np.allclose(np.asarray(yb)[:, 0], y_exact, rtol=1e-6)


@pytest.mark.slow
def test_unbiased_flatband_T():
    """Data drawn from band-flat spectra: <c_hat> must equal the input
    bandpowers exactly (no window-function ambiguity)."""
    f, b, cl, mask, ivar = setup_T()
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX)
    cl_flat[:2] = 0
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                        fisher_mode="exact", batch_size=500, seed=4,
                        verbose=False)
    rng = np.random.default_rng(12)
    nreal = 300
    maps = np.array([hp.alm2map(hp.synalm(cl_flat, lmax=LMAX), NSIDE, lmax=LMAX)
                     * mask + rng.normal(0, 1.0 / np.sqrt(ivar)) for _ in range(nreal)])
    res = w.estimate(w.pack_data([maps]))
    est = res.cl['t_0 x t_0']
    target = b.bin_cl(cl_flat)
    pull = (est.mean(0) - target) / (est.std(0) / np.sqrt(nreal))
    assert np.abs(pull).max() < 4.0
    # chi2 distribution: mean ~ nbands
    chi2 = res.chi2(target)
    assert abs(chi2.mean() - b.nbands) < 5 * np.sqrt(2.0 * b.nbands / nreal)


@pytest.mark.slow
def test_unbiased_TEB():
    """T + spin-2 field, all six spectra, flat-band input."""
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, 4e4)
    l = np.arange(LMAX + 1).astype(float)
    b = sm.Bins.linear(2, LMAX, 6)
    def flat(c):
        out = b.unbin_cl(b.bin_cl(c), LMAX); out[:2] = 0; return out
    clTT = flat(np.where(l >= 2, 1e-2 / np.maximum(l, 1) ** 2, 0))
    clEE = flat(np.where(l >= 2, 4e-3 / np.maximum(l, 1) ** 2, 0))
    clBB = flat(np.where(l >= 2, 1e-3 / np.maximum(l, 1) ** 2, 0))
    clTE = flat(0.4 * np.sqrt(clTT * clEE))
    fT = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    fP = sm.Field(mask, [np.zeros(NPIX)] * 2, spin=2, ivar=ivar, name="p")
    cld = {('t_0', 't_0'): clTT, ('p_E', 'p_E'): clEE,
           ('p_B', 'p_B'): clBB, ('t_0', 'p_E'): clTE}
    w = sm.QMLWorkspace([fT, fP], b, cld, lmax=LMAX, fisher_mode="exact",
                        batch_size=300, seed=5, verbose=False)
    # data drawn with the internal sampler (distribution identical to
    # healpy synfast with these spectra; equivalence is tested separately)
    nreal = 400
    data = w.cov.sample(jax.random.PRNGKey(77), nreal)
    res = w.estimate(data)
    for spec, cl_in in [('t_0 x t_0', clTT), ('p_E x p_E', clEE),
                        ('p_B x p_B', clBB), ('t_0 x p_E', clTE),
                        ('t_0 x p_B', 0 * clTT), ('p_E x p_B', 0 * clTT)]:
        est = res.cl[spec]
        target = b.bin_cl(cl_in)
        pull = (est.mean(0) - target) / (est.std(0) / np.sqrt(nreal))
        assert np.abs(pull).max() < 4.5, f"biased {spec}: {pull}"


@pytest.mark.slow
def test_template_deprojection():
    """Strong contamination along a template is removed; woodbury == alpha."""
    f, b, cl, mask, ivar = setup_T()
    rng = np.random.default_rng(13)
    tmpl = hp.alm2map(hp.synalm(np.ones(LMAX + 1) * 1e-2, lmax=LMAX),
                      NSIDE, lmax=LMAX)
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX); cl_flat[:2] = 0
    nreal = 200
    sky = np.array([hp.alm2map(hp.synalm(cl_flat, lmax=LMAX), NSIDE, lmax=LMAX)
                    * mask + rng.normal(0, 1.0 / np.sqrt(ivar))
                    + 50.0 * rng.normal() * tmpl * mask for _ in range(nreal)])
    target = b.bin_cl(cl_flat)

    res = {}
    for mode, alpha in [("wood", None), ("alpha", 1e6)]:
        fc = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t",
                      templates=[tmpl * mask])
        w = sm.QMLWorkspace(fc, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                            fisher_mode="exact", batch_size=500, seed=6,
                            verbose=False, template_alpha=alpha)
        res[mode] = w.estimate(w.pack_data([sky])).cl['t_0 x t_0']
        pull = (res[mode].mean(0) - target) / (res[mode].std(0) / np.sqrt(nreal))
        assert np.abs(pull).max() < 4.0, f"{mode}: {pull}"
    # the two prescriptions agree realization by realization
    d = res["wood"] - res["alpha"]
    assert np.abs(d).max() < 0.15 * res["wood"].std(0).max()


def test_compat_shim():
    f, b, cl, mask, ivar = setup_T()
    rng = np.random.default_rng(14)
    m = hp.alm2map(hp.synalm(cl, lmax=LMAX), NSIDE, lmax=LMAX) * mask \
        + rng.normal(0, 1.0 / np.sqrt(ivar))
    f = sm.Field(mask, [m], ivar=ivar, name="t")
    out = sm.compute_full_master(f, f, b, cl_guess={('t_0', 't_0'): cl},
                                 lmax=LMAX, n_sims_fisher=512,
                                 n_sims_noise=128, batch_size=256,
                                 verbose=False)
    assert out.shape == (1, b.nbands)
    assert np.all(np.isfinite(out))


@pytest.mark.slow
def test_run_exact_matches_dense_reference():
    """Workspace exact engine == brute-force dense ExactQML."""
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, 4e4)
    l = np.arange(LMAX + 1).astype(float)
    b = sm.Bins.linear(2, LMAX, 6)
    clTT = np.where(l >= 2, 1e-2 / np.maximum(l, 1) ** 2, 0)
    clEE = np.where(l >= 2, 4e-3 / np.maximum(l, 1) ** 2, 0)
    clBB = np.where(l >= 2, 1e-3 / np.maximum(l, 1) ** 2, 0)
    clTE = 0.4 * np.sqrt(clTT * clEE)
    fT = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    fP = sm.Field(mask, [np.zeros(NPIX)] * 2, spin=2, ivar=ivar, name="p")
    cld = {('t_0', 't_0'): clTT, ('p_E', 'p_E'): clEE,
           ('p_B', 'p_B'): clBB, ('t_0', 'p_E'): clTE}
    w = sm.QMLWorkspace([fT, fP], b, cld, lmax=LMAX, fisher_mode="exact",
                        deproject_low_ell=False, cg_tol=1e-9, verbose=False,
                        batch_size=300)
    w.run_exact()
    ex = ExactQML([fT, fP], w.bins, w.cov.clmat, w.index)
    R = ex.response()
    dR = np.abs(w.R_hat - R) / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    assert dR.max() < 1e-8
    assert np.abs(w.n_hat - ex.noise_bias()).max() < 1e-8 * np.abs(ex.noise_bias()).max()


@pytest.mark.slow
def test_subsampled_response_unbiased():
    """Column-subsampled response (stratified, 1/f-renormalized) is an
    unbiased estimate of the exact one."""
    f, b, cl, mask, ivar = setup_T()
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX); cl_flat[:2] = 0
    w0 = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                         fisher_mode="exact", verbose=False,
                         deproject_low_ell=False)
    w0.run_exact()
    Rs, ns = [], []
    for s in range(6):
        w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                            fisher_mode="subsampled", verbose=False,
                            deproject_low_ell=False)
        w.run_exact(sample_frac=0.4, sample_seed=s)
        Rs.append(w.R_hat); ns.append(w.n_hat)
    Rm = np.mean(Rs, axis=0)
    scale = np.sqrt(np.outer(np.diag(w0.R_hat), np.diag(w0.R_hat)))
    # mean over 6 seeds approaches exact ~ sqrt(6) faster (Frobenius)
    fro_mean = np.linalg.norm((Rm - w0.R_hat) / scale)
    fro_one = np.median([np.linalg.norm((R - w0.R_hat) / scale) for R in Rs])
    assert fro_mean < 0.75 * fro_one
    assert (np.abs(Rm - w0.R_hat) / scale).max() < 0.05
    assert np.abs(np.mean(ns, axis=0) - w0.n_hat).max() < 0.05 * np.abs(w0.n_hat).max()
