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


def deflated_pcg(apply_A, apply_M, B, defl, tol=1e-6, maxiter=500):
    """Deflated preconditioned CG: ``pcg`` with the slow eigen-directions in
    ``defl`` (a :class:`~simaster.deflation.DeflationSpace`) projected out.

    Solves the deflated system ``P C x̃ = P B`` and returns the *exact*
    reconstruction ``Q B + Pᵀ x̃`` (see :mod:`simaster.deflation`), so the
    result equals plain CG's to tolerance for any full-rank deflation basis --
    only the iteration count differs.  The per-iteration cost is one operator
    apply (as in ``pcg``) plus the projector ``P`` (small dense GEMMs, no
    SHTs).  ``info`` matches ``pcg``: (niter, max rel. residual, indefinite) --
    and because ``C x − B = −R`` exactly here, that relative residual is the
    true ``‖C x − B‖_M/‖B‖_M`` of the original system, not the deflated one."""
    Qb = defl.coarse(B)                       # coarse-grid component, Q B
    R = defl.project(B)                       # r0 = P B  (x̃_0 = 0)
    Z = apply_M(R)
    P = Z
    rz = jnp.sum(R * Z, axis=0)
    # Normalize by the ORIGINAL RHS ‖B‖_M, not the deflated ‖P B‖_M: the
    # reconstruction x = Q b + Pᵀ x̃ gives  C x_k − B = −R_k  exactly (R is the
    # deflated residual), so ‖R_k‖_M/‖B‖_M is the *true* relative residual of
    # C x = B -- identical in meaning to ``pcg`` and to what ``solve_C``
    # reports.  (‖P B‖_M is not even a valid normalizer: P is C-orthogonal, not
    # M-orthogonal, so ‖P B‖_M can exceed ‖B‖_M.)
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
        AP = defl.project(apply_A(P))         # P C p (one operator apply)
        denom = jnp.sum(P * AP, axis=0)
        alpha = jnp.where(denom == 0, 0.0, rz / denom)
        X = X + alpha[None, :] * P
        R = R - alpha[None, :] * AP
        Z = apply_M(R)
        rz_new = jnp.sum(R * Z, axis=0)
        beta = jnp.where(rz == 0, 0.0, rz_new / rz)
        P = Z + beta[None, :] * P
        return X, R, Z, P, rz_new, it + 1

    state = (jnp.zeros_like(B), R, Z, P, rz, jnp.asarray(0))
    X, R, Z, P, rz, it = jax.lax.while_loop(cond, body, state)
    rel = jnp.max(jnp.sqrt(jnp.clip(rz, 0)) / bnorm)
    return Qb + defl.project_T(X), (it, rel, bad(rz))


def solve_C(cov, B, tol=1e-6, maxiter=500, log=None, deflation=None):
    """Solve C X = B for a CovModel, escalating the preconditioner diagonal
    if it is found indefinite (possible while it uses the statistical mean
    rather than the guaranteed upper bound; escalation is capped, so this
    terminates).  With ``deflation`` (a
    :class:`~simaster.deflation.DeflationSpace`) the slow eigen-directions are
    projected out -- same solution, fewer iterations.  Returns
    (X, (niter, rel_resid))."""
    if cov.backend == "almond":
        from .almond_device import solve_almond_device
        return solve_almond_device(cov, B, tol=tol, maxiter=maxiter,
                                   log=log, deflation=deflation)

    for _ in range(8):
        if deflation is None:
            X, (it, rel, indef) = pcg(cov.apply_C, cov.apply_precond, B,
                                      tol=tol, maxiter=maxiter)
        else:
            X, (it, rel, indef) = deflated_pcg(cov.apply_C, cov.apply_precond,
                                               B, deflation, tol=tol,
                                               maxiter=maxiter)
        if not bool(indef):
            break
        scale = 2.0 * cov._precond_scale
        if log:
            log(f"preconditioner indefinite; escalating D scale to {scale}")
        cov._build_precond(scale)
    return X, (int(it), float(rel))
