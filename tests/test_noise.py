"""Per-pixel block noise (PixelNoiseCov): operators vs dense reference."""

import numpy as np
import healpy as hp

import simaster as sm
from simaster.utils import RealAlmIndex
from simaster.covariance import CovModel
from simaster.exact import ExactQML
from simaster.noise import PixelNoiseCov


NSIDE, LMAX = 8, 16


def _random_iqu_cov(npix, seed=0, scale=1e-6):
    """A random symmetric positive-definite 3x3 per-pixel noise covariance."""
    rng = np.random.default_rng(seed)
    L = np.zeros((npix, 3, 3))
    L[:, 0, 0] = rng.uniform(4, 6, npix)
    L[:, 1, 1] = rng.uniform(4, 6, npix)
    L[:, 2, 2] = rng.uniform(4, 6, npix)
    L[:, 1, 0] = rng.uniform(-2, 2, npix)
    L[:, 2, 0] = rng.uniform(-2, 2, npix)
    L[:, 2, 1] = rng.uniform(-2, 2, npix)
    N = np.einsum("pij,pkj->pik", L, L) * scale       # PD, diag ~ 3e-5
    return N


def _setup(seed=0):
    npix = hp.nside2npix(NSIDE)
    mask = np.zeros(npix)
    mask[: 2 * npix // 3] = 1.0
    mask[npix // 4: npix // 3] *= 0.6                  # partial weights
    N = _random_iqu_cov(npix, seed=seed)
    cov_maps = [N[:, 0, 0], N[:, 0, 1], N[:, 0, 2],
                N[:, 1, 1], N[:, 1, 2], N[:, 2, 2]]
    f0, f2, noise = sm.iqu_from_cov(mask, None, cov_maps, names=("t", "p"))
    fields = [f0, f2]
    op = f0.obs_pix
    l = np.arange(LMAX + 1)
    clTT = np.zeros(LMAX + 1); clTT[2:] = 1e-2 / l[2:] ** 2
    clEE = np.zeros(LMAX + 1); clEE[2:] = 4e-3 / l[2:] ** 2
    clBB = np.zeros(LMAX + 1); clBB[2:] = 1e-3 / l[2:] ** 2
    clTE = np.zeros(LMAX + 1); clTE[2:] = 0.5 * np.sqrt(clTT[2:] * clEE[2:])
    cld = {('t_0', 't_0'): clTT, ('p_E', 'p_E'): clEE,
           ('p_B', 'p_B'): clBB, ('t_0', 'p_E'): clTE}
    idx = RealAlmIndex(2, LMAX)
    names = sum([f.comp_names for f in fields], [])
    clmat = sm.cl_matrix(cld, names, LMAX)
    return fields, clmat, idx, noise, N[op]


def test_block_diag_matches_input_blocks():
    """The exposed per-row diagonal equals the input II/QQ/UU at each pixel."""
    fields, _, _, noise, Nobs = _setup()
    nv = np.asarray(noise.noisevar)
    npix = fields[0].nobs
    assert np.allclose(nv[:npix], Nobs[:, 0, 0])           # I rows
    assert np.allclose(nv[npix:2 * npix], Nobs[:, 1, 1])   # Q rows
    assert np.allclose(nv[2 * npix:3 * npix], Nobs[:, 2, 2])  # U rows


def test_apply_inv_sqrt_against_dense():
    fields, _, _, noise, _ = _setup()
    Nd = noise.dense()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(noise.nrow, 4))
    assert np.allclose(np.asarray(noise.apply(x)), Nd @ x, rtol=1e-10,
                       atol=1e-12 * np.abs(Nd @ x).max())
    Ninv = np.linalg.inv(Nd)
    assert np.allclose(np.asarray(noise.apply_inv(x)), Ninv @ x, rtol=1e-9,
                       atol=1e-11 * np.abs(Ninv @ x).max())
    # sqrt_apply twice reconstructs N
    s = np.asarray(noise.sqrt_apply(noise.sqrt_apply(x)))
    assert np.allclose(s, Nd @ x, rtol=1e-9, atol=1e-11 * np.abs(Nd @ x).max())
    # diag(N^-1) exposed for the preconditioner
    assert np.allclose(np.asarray(noise.ivar_eff), np.diag(Ninv), rtol=1e-9)


def test_quad_cdj_against_dense():
    fields, _, _, noise, _ = _setup()
    Nd = noise.dense()
    Nc = sum(f.ncomp for f in fields)
    rng = np.random.default_rng(8)
    V = rng.normal(size=(noise.nrow, 3, Nc))
    got = np.asarray(noise.quad_cdj(V))
    want = np.einsum("pjc,pq,qjd->cdj", V, Nd, V)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-11 * np.abs(want).max())


def test_apply_C_matches_dense_with_block_noise():
    """Full matrix-free C with correlated I/Q/U noise == dense reference."""
    fields, clmat, idx, noise, _ = _setup()
    cov = CovModel(fields, clmat, idx, backend="dense", noise=noise)
    ex = ExactQML(fields, sm.Bins.linear(2, LMAX, LMAX - 1), clmat, idx,
                  noise=noise)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(cov.nrow, 2))
    a = np.asarray(cov.apply_C(x))
    b = ex.C @ x
    assert np.allclose(a, b, rtol=1e-9, atol=1e-11 * np.abs(b).max())


def test_preconditioner_solves_with_block_noise():
    """CG with the block-aware preconditioner converges on C x = b."""
    fields, clmat, idx, noise, _ = _setup()
    cov = CovModel(fields, clmat, idx, backend="dense", noise=noise)
    from simaster.cg import solve_C
    rng = np.random.default_rng(7)
    b = rng.normal(size=(cov.nrow, 3))
    x, (it, rel) = solve_C(cov, b, tol=1e-8, maxiter=600)
    resid = np.asarray(cov.apply_C(x)) - b
    assert np.abs(resid).max() < 1e-5 * np.abs(b).max()


def test_diagonal_special_case_unchanged():
    """No coupling groups -> apply reduces to the old noisevar*x path."""
    npix = hp.nside2npix(NSIDE)
    mask = np.zeros(npix); mask[: npix // 2] = 1.0
    ivar = np.full(npix, 4e4)
    f = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="t")
    noise = PixelNoiseCov([f])
    assert not noise.has_blocks
    rng = np.random.default_rng(1)
    x = rng.normal(size=(noise.nrow, 2))
    nv = np.asarray(noise.noisevar)
    assert np.allclose(np.asarray(noise.apply(x)), nv[:, None] * x)
    assert np.allclose(np.asarray(noise.apply_inv(x)), x / nv[:, None])
