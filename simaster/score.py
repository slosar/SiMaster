"""Field-level log-likelihood: score and JAX-differentiable quadratic term.

    logL(c) = -1/2 d^T C(c)^-1 d - 1/2 ln det C(c) + const.

* The quadratic term is made differentiable through the CG solve with
  ``lax.custom_linear_solve`` (implicit differentiation: the backward pass
  is one more solve, no unrolling).  Its gradient w.r.t. bandpowers is
  identically the QML quadratic statistic y_A -- verified in the tests.
* The log-det never needs to be evaluated: its gradient
  -1/2 Tr[C^-1 P_A] is a single-solve Hutchinson trace (all bands from one
  batched probe solve), satisfying 1/2 Tr[C^-1 P_A] = n_A + (R c_fid)_A.
  Hence score_A = y_A(d) - n_A - (R c_fid)_A: QML is the Newton step.
* Second derivatives do NOT get cheaper with autodiff: the Hessian is the
  double-solve trace 1/2 Tr[C^-1 P_A C^-1 P_B] (one solve per band per
  probe via jax.hessian) -- use the subsampled-column response engine
  instead.  Hessian-vector products cost ~2 solves and enable
  truncated-Newton optimization or HMC over bandpowers.

Template deprojection is not folded into this module (filter is plain
C^-1); deproject upstream or include the alpha-term in C.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .cg import pcg


def score(ws, data, n_probes=128, key=None):
    """Exact-likelihood score dlogL/dc_A at the workspace fiducial.

    score_A = y_A(d) - 1/2 Tr[C^-1 P_A], the trace via Rademacher probes
    (one batched solve, all bands at once).  ``data``: (nrow, 1) vector.
    Returns (score, trace_estimate_err).
    """
    key = key if key is not None else jax.random.PRNGKey(0)
    ws._prepare_deprojection()
    yd, _ = ws._y_stats(ws._filter(data))
    yd = np.asarray(yd)[:, 0]
    # Hutchinson: Tr[M P_A] = E[(M v)^T P_A v], v Rademacher
    v = jax.random.rademacher(key, (ws.cov.nrow, n_probes),
                              dtype=ws.cov.dtype)
    Mv = ws._filter(v)
    A1 = ws.cov.to_modes(Mv)
    A2 = ws.cov.to_modes(v)
    nl = ws.ls.size
    tr = []
    for (i, j) in ws.spec_pairs:
        prod = A1[i] * A2[j] if i == j else A1[i] * A2[j] + A1[j] * A2[i]
        yl = jax.ops.segment_sum(prod, ws._l_of_mode, num_segments=nl)
        tr.append(jax.ops.segment_sum(yl, ws._band_of_l,
                                      num_segments=ws.bins.nbands))
    tr = np.asarray(jnp.concatenate(tr, axis=0))     # (nbands_tot, n_probes)
    return (yd - 0.5 * tr.mean(axis=1),
            0.5 * tr.std(axis=1) / np.sqrt(n_probes))


def quad_loglike(ws, cb, data):
    """-1/2 d^T C(cb)^-1 d, differentiable in the bandpower vector ``cb``.

    cb: (nspec * nbands,) flat bandpowers over the workspace's (extended)
    bands; the signal covariance is rebuilt from cb (flat in C_l), the
    noise and mask structure come from the workspace.  Differentiable with
    jax.grad / jax.jvp through the CG solve (implicit differentiation);
    grad(quad_loglike)(cb) equals the quadratic statistic y(cb-filter).
    Dense backend only (the ducc callback has no autodiff rule).
    """
    cov = ws.cov
    band_of_mode = jnp.asarray(np.asarray(ws._band_of_l)[
        np.asarray(ws.index.l) - ws.lmin])           # (K,)
    nbb = ws.bins.nbands
    Nc = len(ws.comp_names)

    def cl_k_of(cb):
        cbm = cb.reshape(len(ws.spec_pairs), nbb)
        clk = jnp.zeros((Nc, Nc, ws.index.nmodes), dtype=cov.dtype)
        for s, (i, j) in enumerate(ws.spec_pairs):
            vals = cbm[s][band_of_mode]
            clk = clk.at[i, j].set(vals)
            if i != j:
                clk = clk.at[j, i].set(vals)
        return clk

    def matvec(cb, x):
        A = cov.to_modes(x)
        A = jnp.einsum("cdk,dkb->ckb", cl_k_of(cb), A)
        return cov.from_modes(A) + cov.noisevar[:, None] * x

    def solve(mv, b):
        x, _ = pcg(mv, cov.apply_precond, b, tol=ws.cg_tol,
                   maxiter=ws.cg_maxiter)
        return x

    z = lax.custom_linear_solve(lambda x: matvec(cb, x), data, solve,
                                symmetric=True)
    return -0.5 * jnp.sum(data * z)
