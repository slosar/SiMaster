"""Deflated / recycled CG: correctness (any W) and speed-up (harvested W).

The deflated solver returns the *exact* C^-1 B for any full-rank deflation
basis, so correctness is tested against plain CG with both a random and a
harvested basis; the iteration count is tested separately on a deliberately
ill-conditioned (masked, anisotropic-noise) problem.
"""

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp
import pytest

import simaster as sm
from simaster.utils import RealAlmIndex
from simaster.covariance import CovModel
from simaster.cg import solve_C, deflated_pcg, pcg
from simaster.deflation import (DeflationSpace, harvest_ritz, build_deflation,
                                dense_eig_deflation)


def make_cov(nside=16, lmax=31, aniso=4.0, seed=0):
    """A masked, strongly anisotropic-ivar spin-0 CovModel (slow CG)."""
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix)
    mask[: 2 * npix // 3] = 1.0
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    ivar = 1e4 * (1.0 + aniso * np.cos(3 * theta) ** 2)
    l = np.arange(lmax + 1)
    clTT = np.zeros(lmax + 1); clTT[2:] = 1e-2 / l[2:] ** 2
    f0 = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="t")
    idx = RealAlmIndex(2, lmax)
    clmat = sm.cl_matrix({("t_0", "t_0"): clTT}, ["t_0"], lmax)
    return CovModel([f0], clmat, idx, backend="dense")


def _settle(cov, seed=0):
    """Escalate the preconditioner to its SPD bound (as a real solve does)."""
    b = jnp.asarray(np.random.default_rng(seed).normal(size=(cov.nrow, 1)))
    solve_C(cov, b, tol=1e-8, maxiter=3000)


def test_deflated_matches_plain_random_W():
    """For an *arbitrary* full-rank W the deflated solve equals plain CG --
    correctness is independent of the deflation-space quality."""
    cov = make_cov()
    _settle(cov)
    n = cov.nrow
    rng = np.random.default_rng(2)
    B = jnp.asarray(rng.normal(size=(n, 3)))
    x0, _ = solve_C(cov, B, tol=1e-9, maxiter=5000)
    W, _ = np.linalg.qr(rng.normal(size=(n, 16)))     # random subspace
    defl = DeflationSpace(cov.apply_C, jnp.asarray(W))
    xd, (it, rel, indef) = deflated_pcg(cov.apply_C, cov.apply_precond, B, defl,
                                        tol=1e-9, maxiter=5000)
    assert not bool(indef)
    assert np.allclose(np.asarray(xd), np.asarray(x0), rtol=1e-6,
                       atol=1e-8 * np.abs(np.asarray(x0)).max())


def test_deflation_true_residual_small():
    """The reconstructed x = Qb + Pᵀx̃ solves C x = b (not just the deflated
    system): the *true* residual ‖C x − b‖ is at tolerance."""
    cov = make_cov()
    _settle(cov)
    n = cov.nrow
    B = jnp.asarray(np.random.default_rng(4).normal(size=(n, 2)))
    defl = build_deflation(cov.apply_C, cov.apply_precond, n, k=32, steps=80)
    xd, _ = solve_C(cov, B, tol=1e-9, maxiter=5000, deflation=defl)
    resid = np.asarray(cov.apply_C(xd)) - np.asarray(B)
    assert np.abs(resid).max() < 1e-6 * np.abs(np.asarray(B)).max()


def test_harvest_reduces_iterations():
    """Recycled deflation cuts CG iterations on an ill-conditioned problem,
    and the solution is unchanged."""
    cov = make_cov(aniso=4.0)
    _settle(cov)
    n = cov.nrow
    B = jnp.asarray(np.random.default_rng(5).normal(size=(n, 8)))
    x0, (it0, rel0) = solve_C(cov, B, tol=1e-8, maxiter=5000)
    defl = build_deflation(cov.apply_C, cov.apply_precond, n, k=48, steps=110)
    xd, (itd, reld) = solve_C(cov, B, tol=1e-8, maxiter=5000, deflation=defl)
    assert itd < 0.8 * it0                              # >= 1.25x fewer iters
    assert np.allclose(np.asarray(xd), np.asarray(x0), rtol=1e-6,
                       atol=1e-8 * np.abs(np.asarray(x0)).max())


def test_harvest_approaches_optimal():
    """The recycled basis recovers most of the speed-up of the *optimal*
    (dense largest-eigenvector) deflation space of the same size."""
    cov = make_cov(nside=16, lmax=31, aniso=4.0)
    _settle(cov)
    n = cov.nrow
    B = jnp.asarray(np.random.default_rng(6).normal(size=(n, 2)))
    _, (it0, _) = solve_C(cov, B, tol=1e-8, maxiter=5000)
    k = 48
    d_opt, ev = dense_eig_deflation(cov.apply_C, cov.apply_precond, n, k)
    _, (it_opt, _) = solve_C(cov, B, tol=1e-8, maxiter=5000, deflation=d_opt)
    d_h = build_deflation(cov.apply_C, cov.apply_precond, n, k, steps=120)
    _, (it_h, _) = solve_C(cov, B, tol=1e-8, maxiter=5000, deflation=d_h)
    assert ev[0] > 1.0                                 # slow modes are the large ones
    assert it_opt < it0                                # optimal deflation helps
    # harvested recovers most of the optimal iteration reduction
    assert (it0 - it_h) >= 0.7 * (it0 - it_opt)


def test_workspace_deflation_equivalence():
    """QMLWorkspace.estimate with deflation matches the non-deflated result
    (same R_hat / cl / cov) at fewer CG iterations."""
    nside, lmax = 16, 31
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(1)
    mask = np.zeros(npix); mask[: 2 * npix // 3] = 1.0
    theta, _ = hp.pix2ang(nside, np.arange(npix))
    ivar = 1e4 * (1 + 3 * np.cos(3 * theta) ** 2)
    l = np.arange(lmax + 1); clTT = np.zeros(lmax + 1); clTT[2:] = 1e-2 / l[2:] ** 2
    m = hp.synfast(clTT, nside) * mask + rng.normal(size=npix) / np.sqrt(ivar) * mask
    f0 = sm.Field(mask, [m], ivar=ivar, name="t")
    bins = sm.Bins.linear(2, lmax, nlb=6)

    def run(defl):
        w = sm.QMLWorkspace([f0], bins, {("t_0", "t_0"): clTT}, lmax=lmax,
                            backend="dense", fisher_mode="exact",
                            deproject_low_ell=False, deflation=defl,
                            cg_tol=1e-7, verbose=False)
        return w, w.estimate()

    w0, r0 = run(0)
    w1, r1 = run(48)
    assert w1._defl is not None and w1._defl.k == 48
    assert np.allclose(w0.R_hat, w1.R_hat, rtol=1e-5,
                       atol=1e-8 * np.abs(w0.R_hat).max())
    assert np.allclose(r0.cl["t_0 x t_0"], r1.cl["t_0 x t_0"], rtol=1e-5)
    assert np.allclose(r0.cov, r1.cov, rtol=1e-5)
    assert w1.last_cg[0] < w0.last_cg[0]               # fewer iterations


def test_deflation_invalidated_on_fiducial_update():
    """update_fiducial drops the stale deflation space (tied to C(fiducial))."""
    nside, lmax = 16, 31
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix); mask[: 2 * npix // 3] = 1.0
    l = np.arange(lmax + 1); clTT = np.zeros(lmax + 1); clTT[2:] = 1e-2 / l[2:] ** 2
    f0 = sm.Field(mask, [np.zeros(npix)], ivar=np.full(npix, 1e4), name="t")
    bins = sm.Bins.linear(2, lmax, nlb=6)
    w = sm.QMLWorkspace([f0], bins, {("t_0", "t_0"): clTT}, lmax=lmax,
                        backend="dense", fisher_mode="exact",
                        deproject_low_ell=False, deflation=16, verbose=False)
    w.build_deflation()
    assert w._defl is not None
    w.update_fiducial(w.fiducial_bandpowers())
    assert w._defl is None
