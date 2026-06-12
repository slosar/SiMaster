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
