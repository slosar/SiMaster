"""Unit tests: real harmonic basis, dense synthesis matrices, conversions."""

import numpy as np
import healpy as hp
import pytest

from simaster.utils import RealAlmIndex, cl_matrix, psd_floor, matrix_sqrt_psd
from simaster import sht


def test_real_healpy_roundtrip():
    idx = RealAlmIndex(2, 15)
    rng = np.random.default_rng(1)
    a = rng.normal(size=idx.nmodes)
    alm = idx.real_to_healpy(a)
    a2 = idx.healpy_to_real(alm)
    assert np.allclose(a, a2, atol=1e-14)


def test_real_basis_iid_variance():
    """Real coefficients of an isotropic field are iid with variance C_l."""
    idx = RealAlmIndex(2, 10)
    cl = np.zeros(11); cl[2:] = 1.0 / np.arange(2, 11) ** 2
    rng = np.random.default_rng(2)
    samples = []
    for _ in range(400):
        alm = hp.synalm(cl, lmax=10)
        samples.append(idx.healpy_to_real(alm))
    var = np.var(samples, axis=0)
    assert np.allclose(var, cl[idx.l], rtol=0.35)


def test_dense_Y_orthonormality():
    """Y^T Y = I / Omega_pix for lmax <= 2 nside (exact HEALPix quadrature
    holds only approximately, but mode mixing must be tiny)."""
    nside, lmax = 8, 16
    idx = RealAlmIndex(2, lmax)
    obs = np.arange(hp.nside2npix(nside))  # full sky
    Y = sht.build_dense_Y(nside, idx, 0, obs)
    om = 4 * np.pi / hp.nside2npix(nside)
    G = Y.T @ Y * om
    # HEALPix quadrature is approximate: ~3% mode mixing at lmax = 2 nside,
    # eigenvalues within [0.85, 1.06] (and up to [0.28, 1.73] at 3 nside - 1,
    # which is why the CG preconditioner carries a 2x safety margin).
    err = np.abs(G - np.eye(idx.nmodes)).max()
    assert err < 5e-2
    ev = np.linalg.eigvalsh(G)
    assert ev.min() > 0.8 and ev.max() < 1.1


def test_dense_Y_spin2_matches_healpy():
    nside, lmax = 8, 16
    idx = RealAlmIndex(2, lmax)
    obs = np.arange(hp.nside2npix(nside))
    Y = sht.build_dense_Y(nside, idx, 2, obs)
    rng = np.random.default_rng(3)
    aE = rng.normal(size=idx.nmodes)
    aB = rng.normal(size=idx.nmodes)
    qu = (Y @ np.concatenate([aE, aB])).reshape(2, -1)
    almE = idx.real_to_healpy(aE)
    almB = idx.real_to_healpy(aB)
    Q, U = hp.alm2map_spin([almE, almB], nside, 2, lmax)
    assert np.allclose(qu[0], Q, atol=1e-12 * np.abs(Q).max() + 1e-13)
    assert np.allclose(qu[1], U, atol=1e-12 * np.abs(U).max() + 1e-13)


def test_realsht_adjoint_consistency():
    """Matrix-free RealSHT.synth/adjoint are exact transposes of each other."""
    nside, lmax = 8, 20
    idx = RealAlmIndex(2, lmax)
    obs = np.flatnonzero(np.arange(hp.nside2npix(nside)) % 3 != 0)
    for spin in (0, 2):
        op = sht.RealSHT(nside, idx, spin, obs)
        rng = np.random.default_rng(4)
        a = rng.normal(size=(op.ncol, 3))
        m = rng.normal(size=(op.nrow, 3))
        lhs = np.sum(m * op.synth(a))
        rhs = np.sum(a * op.adjoint(m))
        assert np.allclose(lhs, rhs, rtol=1e-12)


def test_cl_matrix_and_psd_floor():
    cl = {('a', 'a'): np.ones(10), ('a', 'b'): 2 * np.ones(10),
          ('b', 'b'): np.ones(10)}
    m = cl_matrix(cl, ['a', 'b'], 9)
    assert m.shape == (2, 2, 10)
    assert np.all(m[0, 1] == m[1, 0])
    fixed = psd_floor(m)
    w = np.linalg.eigvalsh(np.moveaxis(fixed, -1, 0))
    assert np.all(w >= -1e-12)
    s = matrix_sqrt_psd(np.moveaxis(fixed, -1, 0))
    assert np.allclose(np.einsum("lij,lkj->lik", s, s),
                       np.moveaxis(fixed, -1, 0), atol=1e-12)
