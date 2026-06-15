"""Unit tests: covariance operator, backends, CG, exact-reference agreement."""

import numpy as np
import healpy as hp
import jax
import pytest

import simaster as sm
from simaster.utils import RealAlmIndex
from simaster.covariance import CovModel
from simaster.exact import ExactQML
from simaster.cg import pcg


NSIDE, LMAX = 8, 20


def make_fields(spin2=True, seed=0, aniso=False):
    npix = hp.nside2npix(NSIDE)
    rng = np.random.default_rng(seed)
    mask = np.zeros(npix)
    mask[: 2 * npix // 3] = 1.0
    mask[npix // 3: npix // 2] *= 0.7  # apodized-ish patch
    ivar = np.full(npix, 4e4)
    if aniso:
        theta, _ = hp.pix2ang(NSIDE, np.arange(npix))
        ivar = ivar * (1 + np.cos(3 * theta) ** 2)
    l = np.arange(LMAX + 1)
    clTT = np.zeros(LMAX + 1); clTT[2:] = 1e-2 / l[2:] ** 2
    clEE = np.zeros(LMAX + 1); clEE[2:] = 4e-3 / l[2:] ** 2
    clBB = np.zeros(LMAX + 1); clBB[2:] = 1e-3 / l[2:] ** 2
    clTE = np.zeros(LMAX + 1); clTE[2:] = 0.5 * np.sqrt(clTT[2:] * clEE[2:])
    f0 = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="t")
    fields = [f0]
    cld = {('t_0', 't_0'): clTT}
    if spin2:
        f2 = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=2 * ivar, name="p")
        fields.append(f2)
        cld.update({('p_E', 'p_E'): clEE, ('p_B', 'p_B'): clBB,
                    ('t_0', 'p_E'): clTE})
    return fields, cld


def build_cov(fields, cld, backend):
    idx = RealAlmIndex(2, LMAX)
    names = sum([f.comp_names for f in fields], [])
    clmat = sm.cl_matrix(cld, names, LMAX)
    return CovModel(fields, clmat, idx, backend=backend), clmat, idx


def test_backends_agree():
    fields, cld = make_fields()
    cov_d, clmat, idx = build_cov(fields, cld, "dense")
    cov_c, _, _ = build_cov(fields, cld, "ducc")
    rng = np.random.default_rng(5)
    x = rng.normal(size=(cov_d.nrow, 2))
    a = np.asarray(cov_d.apply_C(x))
    b = np.asarray(cov_c.apply_C(x))
    assert np.allclose(a, b, rtol=1e-10, atol=1e-12 * np.abs(a).max())


def _s2fft_cov(fields, cld):
    """Build an s2fft-backend CovModel, or skip if unavailable/unpatched.

    The 's2fft' backend self-checks against ducc0 at construction and raises
    if the installed s2fft has the HEALPix spin-2 recursion-node bug (any
    release <= 1.4.0), so stock CI installs skip cleanly."""
    pytest.importorskip("s2fft")
    try:
        return build_cov(fields, cld, "s2fft")[0]
    except RuntimeError as e:
        pytest.skip(f"s2fft backend unavailable: {e}")


def test_s2fft_backend_agrees():
    """Native-JAX s2fft Y/Yᵀ reproduce the dense covariance operator."""
    fields, cld = make_fields()
    cov_d, _, _ = build_cov(fields, cld, "dense")
    cov_s = _s2fft_cov(fields, cld)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(cov_d.nrow, 2))
    a = np.asarray(cov_d.apply_C(x))
    b = np.asarray(cov_s.apply_C(x))
    assert np.allclose(a, b, rtol=1e-9, atol=1e-11 * np.abs(a).max())


def test_s2fft_adjoint_is_exact_transpose():
    """⟨Y a, m⟩ = ⟨a, Yᵀ m⟩ for the s2fft synthesis (spin-2 field)."""
    fields, cld = make_fields()
    cov_s = _s2fft_cov(fields, cld)
    op = cov_s._sht[-1]                       # the spin-2 field
    rng = np.random.default_rng(11)
    a = rng.normal(size=(op.ncol, 1))
    m = rng.normal(size=(op.nrow, 1))
    lhs = float(np.asarray(op.synth(a))[:, 0] @ m[:, 0])
    rhs = float(a[:, 0] @ np.asarray(op.adjoint(m))[:, 0])
    assert abs(lhs - rhs) <= 1e-9 * (abs(lhs) + 1e-30)


def test_apply_C_matches_dense_matrix():
    fields, cld = make_fields()
    cov, clmat, idx = build_cov(fields, cld, "dense")
    ex = ExactQML(fields, sm.Bins.linear(2, LMAX, LMAX - 1), clmat, idx)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(cov.nrow, 2))
    a = np.asarray(cov.apply_C(x))
    b = ex.C @ x
    assert np.allclose(a, b, rtol=1e-9, atol=1e-11 * np.abs(b).max())


def test_cg_solves_C():
    fields, cld = make_fields(aniso=True)
    cov, clmat, idx = build_cov(fields, cld, "dense")
    rng = np.random.default_rng(7)
    b = rng.normal(size=(cov.nrow, 3))
    from simaster.cg import solve_C
    x, (it, rel) = solve_C(cov, b, tol=1e-8, maxiter=600)
    resid = np.asarray(cov.apply_C(x)) - b
    assert np.abs(resid).max() < 1e-5 * np.abs(b).max()


def test_precond_is_spd():
    fields, cld = make_fields(aniso=True)
    cov, _, _ = build_cov(fields, cld, "dense")
    rng = np.random.default_rng(8)
    x = rng.normal(size=(cov.nrow, 8))
    q = np.sum(np.asarray(x) * np.asarray(cov.apply_precond(x)), axis=0)
    assert np.all(q > 0)


def test_exact_hessian_matches_dense():
    """grad/Hessian/Fisher from exact_hessian reproduce the dense formulas
    H_AB = F_AB - dᵀ M C_A M C_B M d, g_A = ½dᵀMC_AMd - ½Tr(MC_A)."""
    fields, cld = make_fields()                       # spin0 + spin2
    idx = RealAlmIndex(2, LMAX)
    names = sum([f.comp_names for f in fields], [])
    clmat = sm.cl_matrix(cld, names, LMAX)
    bins = sm.Bins.linear(2, LMAX, nlb=6)
    ws = sm.QMLWorkspace(fields, bins, cld, lmax=LMAX, backend="dense",
                         fisher_mode="exact", deproject_low_ell=False,
                         verbose=False)
    ex = ExactQML(fields, ws.bins, clmat, idx)
    G, M = ex.G, ex.Cinv
    K = idx.nmodes
    rng = np.random.default_rng(3)
    d = rng.normal(size=(ex.C.shape[0],))

    # dense C_A for every (spectrum, band) parameter, workspace ordering
    nbb = ws.bins.nbands
    CA = []
    for (i, j) in ws.spec_pairs:
        for b in range(nbb):
            sl = idx.band_slice(ws.bins.lo[b], ws.bins.hi[b])
            Gi = G[:, i * K + sl.start: i * K + sl.stop]
            Gj = G[:, j * K + sl.start: j * K + sl.stop]
            CA.append(Gi @ Gi.T if i == j else Gi @ Gj.T + Gj @ Gi.T)
    z = M @ d
    Mz = [M @ (Ca @ z) for Ca in CA]                  # M C_A z
    nP = len(CA)
    F = 0.5 * np.array([[np.trace(M @ CA[A] @ M @ CA[B]) for B in range(nP)]
                        for A in range(nP)])
    Q = np.array([[(CA[A] @ z) @ Mz[B] for B in range(nP)] for A in range(nP)])
    g = np.array([0.5 * z @ CA[A] @ z - 0.5 * np.trace(M @ CA[A])
                  for A in range(nP)])
    H_dense = F - Q

    le = ws.exact_hessian(data=d[:, None])            # full band set
    assert le.grad.shape == (nP,) and le.hess.shape == (nP, nP)
    assert np.allclose(le.grad, g, rtol=1e-5, atol=1e-6 * np.abs(g).max())
    assert np.allclose(le.fisher, F, rtol=1e-5, atol=1e-6 * np.abs(F).max())
    assert np.allclose(le.hess, H_dense, rtol=1e-4,
                       atol=1e-5 * np.abs(H_dense).max())
    # fisher_estimate marginalizes the junk bands: invert the FULL F, then
    # restrict -- not the user-band submatrix.
    keep = le._keep
    c_full = le.c0 + np.linalg.solve(le.fisher, le.grad)
    cov_full = np.linalg.inv(le.fisher)
    c_user, cov_user = le.fisher_estimate(user_bands=True)
    assert np.allclose(c_user, c_full[keep], rtol=1e-9, atol=1e-12)
    assert np.allclose(cov_user, cov_full[np.ix_(keep, keep)], rtol=1e-9)
    # marginalizing differs from conditioning (submatrix inverse) when there
    # are junk bands -- guards against the naive slice.
    if keep.size < nP:
        cond = np.linalg.inv(le.fisher[np.ix_(keep, keep)])
        assert not np.allclose(cov_user, cond, rtol=1e-3)


@pytest.mark.slow
def test_sample_covariance_matches_C():
    """MC covariance of cov.sample converges to the dense C."""
    fields, cld = make_fields(spin2=False)
    cov, clmat, idx = build_cov(fields, cld, "dense")
    ex = ExactQML(fields, sm.Bins.linear(2, LMAX, LMAX - 1), clmat, idx)
    n = 6000
    x = np.asarray(cov.sample(jax.random.PRNGKey(0), n))
    Chat = x @ x.T / n
    scale = np.sqrt(np.outer(np.diag(ex.C), np.diag(ex.C)))
    err = np.abs(Chat - ex.C) / scale
    assert err.max() < 8 * np.sqrt(2.0 / n)
