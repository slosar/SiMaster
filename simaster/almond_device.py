"""Device-resident Almond covariance operators and batched PCG.

The legacy Almond backend entered every SHT through ``jax.pure_callback``.
That materialized NumPy arrays on the host twice per transform and therefore
twice per covariance/preconditioner application inside every CG iteration.

This module mirrors the small/static :class:`CovModel` state as CuPy views,
imports the solve RHS from JAX through DLPack once, and performs the complete
PCG loop with CuPy and Almond on one CUDA device. The solution is exported to
JAX through DLPack once at the end. No map or harmonic array crosses the host
boundary; the Python loop synchronizes only scalar convergence checks.
"""

from __future__ import annotations


class AlmondDeviceCov:
    """CuPy implementation of ``CovModel.apply_C/apply_precond``."""

    def __init__(self, cov):
        from almond.interop import as_cupy

        self.cov = cov
        self.cp = __import__("cupy")
        cp = self.cp
        self.as_cupy = as_cupy
        self.fields = cov.fields
        self.slices = cov.slices
        self.ncomp = cov.ncomp
        self.K = cov.K
        self.nrow = cov.nrow
        self.ops = cov._sht

        self.w = [as_cupy(x, dtype=cp.float64) for x in cov.w]
        self.beams = [None if x is None else as_cupy(x, dtype=cp.float64)
                      for x in cov.beams]
        noise = cov.noise
        self.noisevar = as_cupy(noise._noisevar, dtype=cp.float64)
        self.ivar_diag = as_cupy(noise._ivar_diag, dtype=cp.float64)
        self.groups = [{
            "rows": as_cupy(g.rows, dtype=cp.int64),
            "cov": as_cupy(g.cov, dtype=cp.float64),
            "inv": as_cupy(g.inv, dtype=cp.float64),
        } for g in noise.groups]
        self.refresh()

    def refresh(self):
        """Refresh arrays that can change during iteration or auto-repair."""
        cp = self.cp
        cov = self.cov
        self.cl_k = self.as_cupy(cov.cl_k, dtype=cp.float64)
        self.T_k = self.as_cupy(cov.T_k, dtype=cp.float64)
        self.Tmat = (None if cov.Tmat is None else
                     self.as_cupy(cov.Tmat, dtype=cp.float64))
        self.template_alpha = (
            None if cov.template_alpha is None else
            self.as_cupy(cov.template_alpha, dtype=cp.float64))

    def noise_apply(self, x):
        cp = self.cp
        out = self.noisevar[:, None] * x
        for g in self.groups:
            out[g["rows"]] = cp.einsum(
                "pij,pjb->pib", g["cov"], x[g["rows"]])
        return out

    def noise_apply_inv(self, x):
        cp = self.cp
        out = self.ivar_diag[:, None] * x
        for g in self.groups:
            out[g["rows"]] = cp.einsum(
                "pij,pjb->pib", g["inv"], x[g["rows"]])
        return out

    def to_modes(self, x):
        cp = self.cp
        outs = []
        for i, f in enumerate(self.fields):
            xf = (x[self.slices[i]].reshape(f.ncomp, f.nobs, -1)
                  * self.w[i][:, :, None]).reshape(f.ncomp * f.nobs, -1)
            a = self.ops[i].adjoint_device(xf).reshape(f.ncomp, self.K, -1)
            if self.beams[i] is not None:
                a *= self.beams[i][None, :, None]
            outs.append(a)
        return cp.concatenate(outs, axis=0)

    def from_modes(self, A):
        cp = self.cp
        outs = []
        c0 = 0
        for i, f in enumerate(self.fields):
            a = A[c0:c0 + f.ncomp]
            c0 += f.ncomp
            if self.beams[i] is not None:
                a = a * self.beams[i][None, :, None]
            m = self.ops[i].synth_device(
                a.reshape(f.ncomp * self.K, -1)).reshape(
                    f.ncomp, f.nobs, -1)
            outs.append((m * self.w[i][:, :, None]).reshape(
                f.ncomp * f.nobs, -1))
        return cp.concatenate(outs, axis=0)

    def apply_C(self, x):
        cp = self.cp
        A = self.to_modes(x)
        A = cp.einsum("cdk,dkb->ckb", self.cl_k, A)
        out = self.from_modes(A) + self.noise_apply(x)
        if self.template_alpha is not None:
            out += self.Tmat @ (self.template_alpha[:, None]
                                * (self.Tmat.T @ x))
        return out

    def apply_precond(self, y):
        cp = self.cp
        ny = self.noise_apply_inv(y)
        A = self.to_modes(ny)
        A = cp.einsum("kcd,dkb->ckb", self.T_k, A)
        return ny - self.noise_apply_inv(self.from_modes(A))


class _DeviceDeflation:
    def __init__(self, defl, as_cupy, cp):
        self.W = as_cupy(defl.W, dtype=cp.float64)
        self.AW = as_cupy(defl.AW, dtype=cp.float64)
        self.E_inv = as_cupy(defl.E_inv, dtype=cp.float64)

    def coarse(self, B):
        return self.W @ (self.E_inv @ (self.W.T @ B))

    def project(self, V):
        return V - self.AW @ (self.E_inv @ (self.W.T @ V))

    def project_T(self, X):
        return X - self.W @ (self.E_inv @ (self.AW.T @ X))


def _pcg_device(dc, B, tol, maxiter, deflation=None):
    """CuPy PCG matching :mod:`simaster.cg`, including deflation."""
    cp = dc.cp
    tiny = cp.finfo(cp.float64).tiny
    if deflation is None:
        X = cp.zeros_like(B)
        R = B.copy()
        Qb = None
    else:
        Qb = deflation.coarse(B)
        R = deflation.project(B)
        X = cp.zeros_like(B)
    Z = dc.apply_precond(R)
    P = Z.copy()
    rz = cp.sum(R * Z, axis=0)
    bnorm = cp.sqrt(cp.maximum(cp.sum(B * dc.apply_precond(B), axis=0), 0.0))
    bnorm = cp.where(bnorm == 0, 1.0, bnorm)

    indefinite = False
    rel = cp.sqrt(cp.maximum(rz, 0.0)) / bnorm
    relmax = float("inf")
    it = 0
    for it in range(int(maxiter)):
        rel = cp.sqrt(cp.maximum(rz, 0.0)) / bnorm
        status = cp.asnumpy(cp.stack((
            cp.asarray(cp.any(rz < -1e-12 * bnorm ** 2), dtype=cp.float64),
            cp.max(rel))))
        indefinite, relmax = bool(status[0]), float(status[1])
        if indefinite or relmax <= float(tol):
            break
        AP = dc.apply_C(P)
        if deflation is not None:
            AP = deflation.project(AP)
        denom = cp.sum(P * AP, axis=0)
        alpha = cp.where(denom != 0, rz / denom, 0.0)
        X += alpha[None, :] * P
        R -= alpha[None, :] * AP
        Z = dc.apply_precond(R)
        rz_new = cp.sum(R * Z, axis=0)
        beta = cp.where(cp.abs(rz) > tiny, rz_new / rz, 0.0)
        P = Z + beta[None, :] * P
        rz = rz_new
    else:
        it = int(maxiter)
        rel = cp.sqrt(cp.maximum(rz, 0.0)) / bnorm
        status = cp.asnumpy(cp.stack((
            cp.asarray(cp.any(rz < -1e-12 * bnorm ** 2), dtype=cp.float64),
            cp.max(rel))))
        indefinite, relmax = bool(status[0]), float(status[1])

    if deflation is not None:
        X = Qb + deflation.project_T(X)
    return X, (int(it), relmax, bool(indefinite))


def solve_almond_device(cov, B, *, tol=1e-6, maxiter=500, log=None,
                        deflation=None):
    """Solve ``C X=B`` entirely on the GPU and return a zero-copy JAX view."""
    from almond.interop import as_cupy, as_jax

    cp = __import__("cupy")
    dc = getattr(cov, "_almond_device_cov", None)
    if dc is None:
        dc = cov._almond_device_cov = AlmondDeviceCov(cov)
    dB = cp.ascontiguousarray(as_cupy(B, dtype=cp.float64))

    for _ in range(8):
        dc.refresh()
        ddefl = (None if deflation is None else
                 _DeviceDeflation(deflation, as_cupy, cp))
        X, (it, rel, indefinite) = _pcg_device(
            dc, dB, tol, maxiter, deflation=ddefl)
        if not indefinite:
            break
        scale = 2.0 * cov._precond_scale
        if log:
            log(f"preconditioner indefinite; escalating D scale to {scale}")
        cov._build_precond(scale)
    return as_jax(X), (it, rel)
