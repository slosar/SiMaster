"""Subsampling error budget for the column-subsampled response matrix.

Validates that the analytic (delta-method) and bootstrap covariances built
from a single subsampled run reproduce the empirical scatter of the response
matrix over many independent column draws (the ground-truth subsampling error).
"""

import numpy as np
import pytest

import simaster as sm

NSIDE, LMAX = 8, 20
import healpy as hp
NPIX = hp.nside2npix(NSIDE)


def _setup():
    mask = np.zeros(NPIX); mask[: 2 * NPIX // 3] = 1.0
    ivar = np.full(NPIX, 4e4)
    l = np.arange(LMAX + 1)
    cl = np.zeros(LMAX + 1); cl[2:] = 1e-2 / l[2:] ** 2
    f = sm.Field(mask, [np.zeros(NPIX)], ivar=ivar, name="t")
    b = sm.Bins.linear(2, LMAX, 5)
    cl_flat = b.unbin_cl(b.bin_cl(cl), LMAX); cl_flat[:2] = 0
    w = sm.QMLWorkspace(f, b, {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                        fisher_mode="exact", verbose=False,
                        deproject_low_ell=False)
    return w, cl_flat


def test_store_reconstructs_R():
    """Per-mode slabs sum back to the symmetrised R_hat exactly."""
    w, _ = _setup()
    w.run_exact(sample_frac=0.4, sample_seed=0, keep_samples=True)
    store = w._subsample_store
    R = store.reconstruct_R()
    scale = np.sqrt(np.outer(np.diag(w.R_hat), np.diag(w.R_hat)))
    assert (np.abs(R - w.R_hat) / scale).max() < 1e-9


def test_store_reconstructs_n():
    """Per-mode noise slabs sum back to the noise bias n_hat exactly."""
    w, _ = _setup()
    w.run_exact(sample_frac=0.4, sample_seed=0, keep_samples=True)
    store = w._subsample_store
    n = store.reconstruct_n()
    assert np.abs(n - w.n_hat).max() < 1e-9 * np.abs(w.n_hat).max()


def test_n_hat_err_set_and_positive():
    """keep_samples fills the (otherwise zero) noise-bias standard error."""
    w, _ = _setup()
    w.run_exact(sample_frac=0.4, sample_seed=0, keep_samples=True)
    assert w.n_hat_err.shape == w.n_hat.shape
    assert np.all(w.n_hat_err >= 0)
    assert w.n_hat_err.max() > 0
    # full run -> exact noise bias -> zero error
    w2, _ = _setup()
    w2.run_exact(keep_samples=True)
    assert np.allclose(w2.n_hat_err, 0.0)


def test_full_run_zero_error():
    """f = 1 (all columns) has no subsampling error."""
    w, _ = _setup()
    w.run_exact(keep_samples=True)              # full run
    err = w.subsample_error(n_boot=64, seed=0)
    assert np.allclose(err.cov_analytic, 0.0)
    assert np.allclose(err.cov_boot, 0.0)


@pytest.mark.slow
def test_noise_bias_error_matches_empirical():
    """Analytic + bootstrap n_hat error reproduce the scatter over draws."""
    w, cl_flat = _setup()
    w.run_exact()                               # exact reference (f=1)

    f = 0.4
    ns_list = []
    for s in range(60):
        ws = sm.QMLWorkspace(w.fields[0], w.user_bins,
                             {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                             fisher_mode="exact", verbose=False,
                             deproject_low_ell=False)
        ws.run_exact(sample_frac=f, sample_seed=s)
        ns_list.append(ws.n_hat)
    cov_emp = np.cov(np.array(ns_list), rowvar=False)
    d_emp = np.sqrt(np.clip(np.diag(cov_emp), 0, None))

    ws = sm.QMLWorkspace(w.fields[0], w.user_bins,
                         {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                         fisher_mode="exact", verbose=False,
                         deproject_low_ell=False)
    ws.run_exact(sample_frac=f, sample_seed=123, keep_samples=True)
    err = ws.subsample_error(n_boot=1500, seed=1)

    # analytic n_hat_err is set on the workspace and matches the empirical one
    keep = err._keep
    d_ana = ws.n_hat_err[keep]
    d_boot = err.n_hat_err("boot")
    m = d_emp[keep] > 0.05 * d_emp[keep].max()
    assert np.all((d_ana[m] / d_emp[keep][m] > 0.5)
                  & (d_ana[m] / d_emp[keep][m] < 2.0)), d_ana[m] / d_emp[keep][m]
    assert np.all((d_boot[m] / d_emp[keep][m] > 0.5)
                  & (d_boot[m] / d_emp[keep][m] < 2.0))


@pytest.mark.slow
def test_analytic_matches_empirical_Rc():
    """Analytic Cov(R_hat c) reproduces the empirical scatter over draws."""
    from simaster.subsample import analytic_Rc_cov
    w, cl_flat = _setup()
    w.run_exact()                               # exact reference (f=1)
    c_fid = w.fiducial_bandpowers().reshape(-1)

    # ground truth: scatter of R_hat @ c over many independent column draws
    f = 0.4
    vs, analytics = [], []
    for s in range(60):
        ws = sm.QMLWorkspace(w.fields[0], w.user_bins,
                             {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                             fisher_mode="exact", verbose=False,
                             deproject_low_ell=False)
        ws.run_exact(sample_frac=f, sample_seed=s, keep_samples=True)
        vs.append(ws.R_hat @ c_fid)
        analytics.append(analytic_Rc_cov(ws._subsample_store, c_fid))
    cov_emp = np.cov(np.array(vs), rowvar=False)
    cov_ana = np.mean(analytics, axis=0)        # per-draw estimator -> average

    d_emp = np.sqrt(np.clip(np.diag(cov_emp), 0, None))
    d_ana = np.sqrt(np.clip(np.diag(cov_ana), 0, None))
    m = d_emp > 0.05 * d_emp.max()              # ignore near-null bands
    ratio = d_ana[m] / d_emp[m]
    assert np.all((ratio > 0.6) & (ratio < 1.6)), ratio


@pytest.mark.slow
def test_bootstrap_matches_analytic_and_empirical():
    """Bootstrap c-cov tracks the analytic budget and the empirical scatter."""
    w, cl_flat = _setup()
    w.run_exact()
    c_fid = w.fiducial_bandpowers().reshape(-1)
    q_ref = w.R_hat @ c_fid                      # fixed reference statistic

    # empirical subsampling cov of c_hat = R_seed^-1 q_ref over draws
    f = 0.4
    chats = []
    for s in range(60):
        ws = sm.QMLWorkspace(w.fields[0], w.user_bins,
                             {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                             fisher_mode="exact", verbose=False,
                             deproject_low_ell=False)
        ws.run_exact(sample_frac=f, sample_seed=s)
        chats.append(np.linalg.solve(ws.R_hat, q_ref))
    cov_emp = np.cov(np.array(chats), rowvar=False)

    # single-draw budget (analytic + bootstrap)
    ws = sm.QMLWorkspace(w.fields[0], w.user_bins,
                         {('t_0', 't_0'): cl_flat}, lmax=LMAX,
                         fisher_mode="exact", verbose=False,
                         deproject_low_ell=False)
    ws.run_exact(sample_frac=f, sample_seed=123, keep_samples=True)
    err = ws.subsample_error(ref="fiducial", n_boot=1500, seed=1)

    keep = err._keep
    de = np.sqrt(np.clip(np.diag(cov_emp), 0, None))
    da = np.sqrt(np.clip(np.diag(err.cov_analytic)[keep], 0, None))
    db = np.sqrt(np.clip(np.diag(err.cov_boot)[keep], 0, None))
    m = de > 0.05 * de.max()
    # both estimators reproduce the empirical per-band subsampling error
    assert np.all((da[m] / de[m] > 0.5) & (da[m] / de[m] < 2.0)), da[m] / de[m]
    assert np.all((db[m] / de[m] > 0.5) & (db[m] / de[m] < 2.0)), db[m] / de[m]
    # and analytic and bootstrap agree with each other in this moderate-f regime
    assert np.all((db[m] / da[m] > 0.6) & (db[m] / da[m] < 1.7)), db[m] / da[m]
