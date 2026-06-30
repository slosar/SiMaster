"""Banded Fisher estimator (simaster.mc_fisher.BandedFisher)."""
import numpy as np
import pytest

import simaster as sm


def _banded_truth(nspec, nbands, bandwidth, seed=0):
    """A PD Fisher that is exactly block-banded with the given ell-bandwidth."""
    rng = np.random.default_rng(seed)
    nb = nspec * nbands
    bidx = sm.banded_index_map(nspec, nbands)
    A = rng.standard_normal((nb, nb))
    F = A @ A.T + nb * np.eye(nb)
    off = np.abs(bidx[:, None] - bidx[None, :])
    F = np.where(off <= bandwidth, F, 0.0)
    return 0.5 * (F + F.T), bidx


def test_band_fisher_zeros_far_offdiagonal():
    F, bidx = _banded_truth(2, 5, bandwidth=4)
    Fb = sm.band_fisher(F, bidx, 1)
    # offset 0 and 1 kept, offset >=2 zeroed
    off = np.abs(bidx[:, None] - bidx[None, :])
    assert np.all(Fb[off > 1] == 0.0)
    assert np.allclose(Fb[off <= 1], F[off <= 1])


def test_index_map_layout():
    bidx = sm.banded_index_map(3, 4)
    assert list(bidx) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]


def test_banding_recovers_banded_truth_exactly():
    # if the truth is banded with bandwidth W, banding at N>=W is exact and the
    # N->N+1 self-consistency uncertainty collapses to ~0.
    F, bidx = _banded_truth(2, 8, bandwidth=3)
    bf = sm.BandedFisher(F, bidx, 3)
    assert bf.is_pd
    assert np.allclose(bf.fisher, F)              # nothing real was removed
    u = bf.uncertainty()
    assert u["worstdir_fisher"] < 1e-10
    assert u["errbar_rel_median"] < 1e-12
    assert bf.converged(tol=0.01)


def test_undertruncation_flagged_by_uncertainty():
    # truth bandwidth 4; truncating at N=1 must register a real change at N->2.
    F, bidx = _banded_truth(2, 8, bandwidth=4, seed=3)
    u1 = sm.BandedFisher(F, bidx, 1).uncertainty()
    u4 = sm.BandedFisher(F, bidx, 4).uncertainty()
    assert u1["worstdir_fisher"] > u4["worstdir_fisher"]   # converges with N
    assert u4["worstdir_fisher"] < 1e-10                   # exact at/above W


def test_store_banded_method_matches_function():
    F, bidx = _banded_truth(2, 6, bandwidth=2)
    store = sm.MCFisherStore([F], [np.zeros(F.shape[0])], [1000])
    bf = store.banded(bidx, 2)
    assert bf.n_eff == 1000
    assert np.allclose(bf.fisher, sm.band_fisher(store.fisher(), bidx, 2))


def test_cov_and_errbar_consistent():
    F, bidx = _banded_truth(2, 5, bandwidth=2)
    bf = sm.BandedFisher(F, bidx, 2)
    assert np.allclose(bf.cov, np.linalg.inv(bf.fisher))
    assert np.allclose(bf.errbar, np.sqrt(np.diag(bf.cov)))


def test_length_mismatch_raises():
    F, bidx = _banded_truth(2, 5, bandwidth=2)
    with pytest.raises(ValueError):
        sm.band_fisher(F, bidx[:-1], 1)
