"""Batched preconditioned conjugate gradient in JAX.

Solves C X = B for many right-hand sides simultaneously; every iteration is
a single batched operator application (GEMMs on the dense backend), which is
what makes QML tractable on a GPU.  Convergence is monitored per column on
the preconditioned residual norm; iteration stops when every column is below
tolerance or at ``maxiter``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def pcg(apply_A, apply_M, B, tol=1e-6, maxiter=500, x0=None):
    """Solve A X = B with preconditioner M ~ A^-1, batched over columns.

    Returns (X, info), info = (niter, max relative residual, indefinite).
    ``indefinite`` is True if a negative preconditioned residual norm was
    encountered (M not SPD) -- the caller should rebuild the preconditioner
    with a safer setting and retry (see QMLWorkspace._solve).
    """
    X = jnp.zeros_like(B) if x0 is None else x0
    R = B - apply_A(X) if x0 is not None else B
    Z = apply_M(R)
    P = Z
    rz = jnp.sum(R * Z, axis=0)
    bnorm = jnp.sqrt(jnp.clip(jnp.sum(B * apply_M(B), axis=0), 0))
    bnorm = jnp.where(bnorm == 0, 1.0, bnorm)

    def bad(rz):
        return jnp.any(rz < -1e-12 * bnorm ** 2)

    def cond(state):
        X, R, Z, P, rz, it = state
        rel = jnp.sqrt(jnp.clip(rz, 0)) / bnorm
        ok = jnp.logical_not(bad(rz))
        return jnp.logical_and(jnp.logical_and(it < maxiter, jnp.max(rel) > tol), ok)

    def body(state):
        X, R, Z, P, rz, it = state
        AP = apply_A(P)
        alpha = rz / jnp.sum(P * AP, axis=0)
        X = X + alpha[None, :] * P
        R = R - alpha[None, :] * AP
        Z = apply_M(R)
        rz_new = jnp.sum(R * Z, axis=0)
        beta = rz_new / rz
        P = Z + beta[None, :] * P
        return X, R, Z, P, rz_new, it + 1

    state = (X, R, Z, P, rz, jnp.asarray(0))
    X, R, Z, P, rz, it = jax.lax.while_loop(cond, body, state)
    rel = jnp.max(jnp.sqrt(jnp.clip(rz, 0)) / bnorm)
    return X, (it, rel, bad(rz))


def solve_C(cov, B, tol=1e-6, maxiter=500, log=None):
    """Solve C X = B for a CovModel, escalating the preconditioner diagonal
    if it is found indefinite (possible while it uses the statistical mean
    rather than the guaranteed upper bound; escalation is capped, so this
    terminates).  Returns (X, (niter, rel_resid))."""
    for _ in range(8):
        X, (it, rel, indef) = pcg(cov.apply_C, cov.apply_precond, B,
                                  tol=tol, maxiter=maxiter)
        if not bool(indef):
            break
        scale = 2.0 * cov._precond_scale
        if log:
            log(f"preconditioner indefinite; escalating D scale to {scale}")
        cov._build_precond(scale)
    return X, (int(it), float(rel))
