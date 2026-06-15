"""Deflation / recycling for the conjugate-gradient inverse-covariance solve.

Every Fisher engine applies ``M = C^-1`` to *thousands* of right-hand sides
that all share the **same** covariance ``C``.  Preconditioned CG converges at
a rate set by the spread of the spectrum of ``P^-1 C`` -- a handful of
outlying eigenvalues (from masked large scales, deprojected templates,
strongly down-weighted pixels) that drag down *every* solve.  Deflation
removes that subspace from the Krylov iteration once and amortizes the cost
over all the RHS.

Which outliers?  SiMaster's preconditioner is a *guaranteed upper bound*:
once escalated to its SPD bound, ``P^-1 ⪰ C^-1``, equivalently ``C ⪰ P``, so
every eigenvalue of ``P^-1 C`` is ``>= 1`` with the bulk clustered at 1 and a
tail of *large* eigenvalues (the cut-sky / anisotropic-noise modes the
isotropic bound over-corrects).  The slow-converging directions are therefore
the **largest** eigenvalues of ``P^-1 C``, and those are what we deflate
(verified empirically: deflating the largest k cuts iterations ~2x, the
smallest k does nothing).

Given a deflation basis ``W`` (n x k, full column rank) spanning those slow
directions, define the (Galerkin) coarse operator and projectors

    E   = Wᵀ C W            (k x k, SPD),     Q  = W E^-1 Wᵀ,
Deflated CG solves the deflated system ``P C x̃ = P b`` (whose effective
condition number drops by deflating the extreme/outlying eigenvalues) and reconstructs
    x = Q b + Pᵀ x̃.

This is **exact** for any full-rank ``W``: ``C x = b`` holds the moment the
deflated residual ``P(b - C x̃)`` vanishes (proof: ``b - C x̃`` then lies in
``range(C W) = ker P``, so ``C x = C Q b + P C x̃ = (I-P)b + P b = b``).  So
the deflation space only changes the *speed*, never the *answer*.

The per-iteration overhead is one projector application, ``P v = v - (C W)
E^-1 (Wᵀ v)``: with ``C W`` precomputed once (k operator applies), it is only
small dense GEMMs and a k x k solve -- no extra SHTs.

We build ``W`` by **recycling** the Krylov information from a short
instrumented solve: PCG implicitly runs Lanczos on ``P^-1 C``, so the Ritz
vectors of its tridiagonal matrix approximate the eigenvectors of ``P^-1 C``,
and the *largest* Ritz vectors are exactly the slow directions to deflate
(:func:`harvest_ritz`).  Lanczos resolves the well-separated extreme
eigenvalues first, so a short run captures them accurately.  This reuses
solves we have to do anyway, the defining feature of "recycled"/eigCG methods
(Stathopoulos & Orginos 2010).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


class DeflationSpace:
    """Precomputed coarse operator for a deflation basis ``W``.

    Parameters
    ----------
    apply_A : callable (n, b) -> (n, b), the operator ``C``.
    W : (n, k) deflation basis (need only be full column rank; it is used
        as given -- orthonormalize beforehand for a well-conditioned ``E``).

    Holds ``W``, ``AW = C W`` (k operator applies, done once) and the
    Cholesky factor of ``E = Wᵀ C W``, and exposes the matrix-free coarse
    correction ``Q`` and the deflation projectors ``P`` and ``Pᵀ`` -- all
    cheap dense algebra, no further operator applies.
    """

    def __init__(self, apply_A, W):
        W = jnp.asarray(W)
        self.W = W
        self.k = int(W.shape[1])
        self.AW = apply_A(W)                       # C W  (n, k)
        E = W.T @ self.AW                          # Wᵀ C W
        self.E = 0.5 * (E + E.T)
        # Precompute the dense k x k inverse so the per-iteration coarse solve
        # is a small GEMM, not a LAPACK custom-call: inside a CG while_loop the
        # latter's fixed dispatch cost dominates (it can cost more than the SHT
        # itself).  E is small, SPD and well conditioned (W is orthonormal), so
        # an explicit inverse is accurate; jitter guards a rank-deficient W.
        eps = 1e-12 * jnp.trace(self.E) / max(self.k, 1)
        self.E_inv = jnp.linalg.inv(self.E + eps * jnp.eye(self.k))

    def _einv(self, X):
        return self.E_inv @ X

    def coarse(self, B):
        """Q B = W E^-1 Wᵀ B  (the coarse-grid solution component)."""
        return self.W @ self._einv(self.W.T @ B)

    def project(self, V):
        """P V = V - C W E^-1 Wᵀ V  = (I - C Q) V."""
        return V - self.AW @ self._einv(self.W.T @ V)

    def project_T(self, X):
        """Pᵀ X = X - W E^-1 (C W)ᵀ X = (I - Q C) X."""
        return X - self.W @ self._einv(self.AW.T @ X)


def harvest_ritz(apply_A, apply_M, probe, k, m, tol=0.0):
    """Approximate the ``k`` largest eigenvectors of ``P^-1 C`` by recycling.

    Runs ``m`` steps of preconditioned CG on a single ``probe`` RHS,
    instrumented to record the implicit Lanczos tridiagonal ``T`` (from the
    CG coefficients) and the preconditioned residuals.  The Ritz vectors of
    ``T`` approximate eigenvectors of ``M C = P^-1 C``; the ``k`` belonging to
    the largest Ritz values are the slow-converging directions to deflate
    (the SPD-bound preconditioner puts the bulk at 1 and the slow modes at the
    top -- see the module docstring).

    Parameters
    ----------
    apply_A, apply_M : the operator ``C`` and preconditioner ``M`` (n, b)->(n, b).
    probe : (n,) or (n, 1) starting vector (white noise works well).
    k : number of deflation vectors to return.
    m : number of Lanczos/CG steps (must exceed ``k``; ~3k-5k is plenty).
    tol : optional early stop on the relative preconditioned residual.

    Returns
    -------
    W : (n, k') orthonormal deflation basis, k' = min(k, steps taken).
    ritz : (k',) the associated (largest) Ritz values, descending.
    """
    r = jnp.asarray(probe, dtype=jnp.float64).reshape(-1, 1)
    z = apply_M(r)
    p = z
    rz = float(jnp.sum(r * z))
    d0 = rz if rz > 0 else 1.0
    Us, alphas, betas = [], [], []
    for _ in range(m):
        Us.append(z / jnp.sqrt(jnp.sum(r * z, axis=0)))     # Lanczos vector q_j
        Ap = apply_A(p)
        pap = float(jnp.sum(p * Ap))
        if not pap > 0:                                     # breakdown / done
            break
        alpha = rz / pap
        r = r - alpha * Ap
        z = apply_M(r)
        rz_new = float(jnp.sum(r * z))
        alphas.append(alpha)
        betas.append(rz_new / rz)
        p = z + (rz_new / rz) * p
        rz = rz_new
        if rz_new <= 0 or rz_new / d0 < tol ** 2:
            break

    mm = len(alphas)
    if mm == 0:
        raise RuntimeError("harvest_ritz: CG made no progress on the probe")
    a = np.asarray(alphas, dtype=np.float64)
    b = np.asarray(betas, dtype=np.float64)
    # tridiagonal of P^-1 C in the Lanczos basis (Golub & Van Loan / eigCG)
    diag = np.empty(mm)
    diag[0] = 1.0 / a[0]
    diag[1:] = 1.0 / a[1:] + b[:-1] / a[:-1]
    off = np.sqrt(np.clip(b[:-1], 0.0, None)) / a[:-1]
    T = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    evals, Y = np.linalg.eigh(T)                            # ascending
    kk = min(k, mm)
    U = jnp.concatenate(Us[:mm], axis=1)                    # (n, mm)
    W = U @ jnp.asarray(Y[:, -kk:])                         # largest-Ritz block
    W, _ = jnp.linalg.qr(W)                                 # orthonormalize
    return W, evals[-kk:][::-1]                             # descending


def build_deflation(apply_A, apply_M, n, k, steps=None, n_probes=1, seed=0,
                    tol=0.0):
    """Convenience: harvest a :class:`DeflationSpace` from random probes.

    Runs :func:`harvest_ritz` on ``n_probes`` independent white-noise probes,
    stacks the Ritz vectors, re-orthonormalizes, and wraps the result.  More
    probes broaden the captured subspace at linear cost.
    """
    steps = steps or max(2 * k + 10, 40)
    key = jax.random.PRNGKey(seed)
    cols = []
    for i in range(n_probes):
        key, sub = jax.random.split(key)
        probe = jax.random.normal(sub, (n, 1), dtype=jnp.float64)
        W, _ = harvest_ritz(apply_A, apply_M, probe, k, steps, tol=tol)
        cols.append(W)
    W = cols[0] if len(cols) == 1 else jnp.concatenate(cols, axis=1)
    if W.shape[1] > k:                                      # trim to k via QR
        W, _ = jnp.linalg.qr(W)
        W = W[:, :k]
    return DeflationSpace(apply_A, W)


def dense_eig_deflation(apply_A, apply_M, n, k):
    """Exact largest-eigenvector deflation basis (dense; small problems only).

    Forms ``M C`` densely and returns the eigenvectors of its ``k`` largest
    eigenvalues -- the *optimal* deflation space for the SPD-bound
    preconditioner, for validating that the recycled :func:`harvest_ritz`
    basis captures most of the speed-up.  Cost is O(n^2) memory / O(n^3) time,
    so this is a test/reference helper only.
    """
    eye = jnp.eye(n, dtype=jnp.float64)
    A = np.asarray(apply_A(eye))
    M = np.asarray(apply_M(eye))
    MA = M @ A                                              # ~ P^-1 C (nonsym)
    evals, evecs = np.linalg.eig(MA)
    order = np.argsort(evals.real)[::-1][:k]                # largest
    W = np.real(evecs[:, order])
    W, _ = np.linalg.qr(W)
    return DeflationSpace(apply_A, jnp.asarray(W)), evals.real[order]
