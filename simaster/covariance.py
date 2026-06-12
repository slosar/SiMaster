"""Matrix-free covariance model and operators.

The data vector x concatenates, over fields f, the observed-pixel maps of
each component (1 for spin 0; Q, U for spin 2).  Its covariance is

    C = U Chat U^T + N (+ sum_j alpha_j t_j t_j^T)

with U = W Y B (mask times real-basis synthesis times beam), Chat the
block-diagonal fiducial spectrum (one (ncomp x ncomp) block C_l per mode),
and N = diag(1/ivar) the always-diagonal pixel noise.  Everything is applied
matrix-free:

    C x = W Y B Cl B Y^T W x + x/ivar (+ template term),

where Y / Y^T are either dense GEMMs on the device ("dense" backend) or
exact ducc0 transforms through ``jax.pure_callback`` ("ducc" backend).

The preconditioner is the Woodbury inverse of the isotropic approximation
U Chat U^T + N with U^T N^-1 U replaced by its statistically averaged
diagonal D_c(l) = b_l^2 sum_p(w^2 ivar)/(4pi):

    P^-1 = N^-1 - N^-1 U T U^T N^-1,
    T_l  = (Chat_l^-1 + D_l)^-1 = Chat - Chat D^1/2 (1 + D^1/2 Chat D^1/2)^-1 D^1/2 Chat,

the second form remaining valid for singular Chat (e.g. zero fiducial BB).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .utils import RealAlmIndex, matrix_sqrt_psd
from . import sht


class CovModel:
    """Holds the fiducial covariance model and applies its operators.

    Parameters
    ----------
    fields : list of Field (same nside; masks may differ)
    clmat : (ncomp_tot, ncomp_tot, lmax+1) fiducial spectra incl. noise-free
        signal only (noise comes from ivar).
    index : RealAlmIndex shared by all fields.
    backend : 'dense' or 'ducc'.
    template_alpha : None for exact (Woodbury, alpha->inf) deprojection,
        or a finite number: covariance gains alpha_rel * tr(C)/(t^T t) * t t^T
        per template (the "correctly large prefactor" prescription).
    """

    def __init__(self, fields, clmat, index: RealAlmIndex, backend="dense",
                 cachedir=None, dtype=jnp.float64):
        self.fields = fields
        self.index = index
        self.backend = backend
        self.dtype = dtype
        self.nside = fields[0].nside
        self.K = index.nmodes
        self.lk = jnp.asarray(index.l)          # l of each mode
        self.ncomp = sum(f.ncomp for f in fields)

        # ---- layout ----------------------------------------------------------
        self.comp_of_field, self.slices, self.nrow = [], [], 0
        c0 = 0
        for f in fields:
            self.comp_of_field.append(list(range(c0, c0 + f.ncomp)))
            self.slices.append(slice(self.nrow, self.nrow + f.ncomp * f.nobs))
            self.nrow += f.ncomp * f.nobs
            c0 += f.ncomp

        # ---- per-field weights, noise ---------------------------------------
        self.w = [jnp.asarray(np.tile(f.mask, (f.ncomp, 1)), dtype=dtype)
                  for f in fields]                       # (ncomp, nobs)
        self.noisevar = jnp.concatenate(
            [jnp.asarray(np.tile(1.0 / f.ivar, f.ncomp), dtype=dtype)
             for f in fields])                            # (nrow,)
        self.ivar_vec = 1.0 / self.noisevar
        self.beams = [None if f.beam is None
                      else jnp.asarray(f.beam[index.l], dtype=dtype)
                      for f in fields]                    # b_l per mode

        # ---- SHT operators ----------------------------------------------------
        if backend == "dense":
            self.Y = [jnp.asarray(sht.build_dense_Y(
                self.nside, index, f.spin, f.obs_pix, cachedir=cachedir),
                dtype=dtype) for f in fields]
        elif backend == "ducc":
            self._sht = [sht.RealSHT(self.nside, index, f.spin, f.obs_pix)
                         for f in fields]
        else:
            raise ValueError(f"unknown backend {backend!r}")

        self.set_clmat(clmat)
        self._build_templates()

    # ------------------------------------------------------------------ Cl --
    def set_clmat(self, clmat):
        """(Re)set the fiducial spectra; cheap, used when iterating."""
        clmat = np.asarray(clmat, dtype=np.float64)
        if clmat.shape != (self.ncomp, self.ncomp, self.index.lmax + 1):
            raise ValueError(f"clmat must be (ncomp, ncomp, lmax+1) = "
                             f"({self.ncomp},{self.ncomp},{self.index.lmax + 1})")
        self.clmat = clmat
        self.cl_k = jnp.asarray(clmat[:, :, self.index.l], dtype=self.dtype)  # (Nc,Nc,K)
        chol = matrix_sqrt_psd(np.moveaxis(clmat, -1, 0))                     # (L+1,Nc,Nc)
        self.sqrt_cl_k = jnp.asarray(chol[self.index.l], dtype=self.dtype)    # (K,Nc,Nc)
        self._build_precond(getattr(self, "_precond_scale", 1.0))

    def _build_precond(self, d_scale: float = 1.0):
        """Per-l Woodbury factor T_l, robust to singular Chat_l.

        For the subtracted Woodbury form to be positive definite, D must
        upper-bound the spectrum of U^T N^-1 U.  The guaranteed bound is
        2 * max_p(w^2 ivar) * npix/(4pi) (the factor 2 covers HEALPix Gram
        eigenvalues, up to ~1.75 at l = 3 nside), but it can be far from
        the typical mode response when the weights are non-uniform, which
        costs CG iterations.  We therefore start from the statistical mean
        sum_p(w^2 ivar)/(4pi) and escalate by ``d_scale`` (doubling, capped
        at the guaranteed bound) only if CG detects indefiniteness --
        see QMLWorkspace._solve.
        """
        L = self.index.lmax
        npix_full = 12 * self.nside ** 2
        self._precond_scale = d_scale
        D = np.zeros((self.ncomp, L + 1))
        for f, comps in zip(self.fields, self.comp_of_field):
            w2i = (f.mask ** 2) * f.ivar
            d_mean = float(np.sum(w2i)) / (4.0 * np.pi)
            d_max = 2.0 * float(np.max(w2i)) * npix_full / (4.0 * np.pi)
            d = min(d_scale * d_mean, d_max)
            bl2 = np.ones(L + 1) if f.beam is None else f.beam[: L + 1] ** 2
            for c in comps:
                D[c] = d * bl2
        Dm = np.moveaxis(D, -1, 0)                       # (L+1, Nc)
        Ch = np.moveaxis(self.clmat, -1, 0)              # (L+1, Nc, Nc)
        sD = np.sqrt(Dm)
        inner = np.eye(self.ncomp) + sD[:, :, None] * Ch * sD[:, None, :]
        inner_inv = np.linalg.inv(inner)
        T = Ch - np.einsum("lij,lj,ljk,lk,lkm->lim", Ch, sD, inner_inv, sD, Ch)
        self.T_k = jnp.asarray(T[self.index.l], dtype=self.dtype)  # (K,Nc,Nc)

    # ------------------------------------------------------------- templates --
    def _build_templates(self):
        """Collect template vectors as a dense (nrow, Nt) matrix."""
        cols = []
        for f, sl in zip(self.fields, self.slices):
            for t in f.templates:
                v = np.zeros(self.nrow)
                v[sl] = t.reshape(-1)
                cols.append(v)
        self.Tmat = (jnp.asarray(np.stack(cols, axis=1), dtype=self.dtype)
                     if cols else None)
        self.n_templates = 0 if self.Tmat is None else self.Tmat.shape[1]
        self.template_alpha = None  # set by workspace if finite-alpha mode

    def set_template_alpha(self, alpha_rel):
        """Enable finite-alpha mode: alpha_j = alpha_rel * tr(C) / ||t_j||^2."""
        if self.Tmat is None:
            self.template_alpha = None
            return
        # tr(C) ~ sum_p w^2 sum_c sum_l (2l+1)/4pi Cl_cc b^2 + sum 1/ivar
        trC = float(jnp.sum(self.noisevar))
        l = np.arange(self.index.lmin, self.index.lmax + 1)
        for f, comps in zip(self.fields, self.comp_of_field):
            bl2 = np.ones_like(l, dtype=float) if f.beam is None else f.beam[l] ** 2
            for c in comps:
                sig = np.sum((2 * l + 1) / (4 * np.pi) * self.clmat[c, c, l] * bl2)
                trC += sig * float(np.sum(f.mask ** 2))
        tnorm = jnp.sum(self.Tmat ** 2, axis=0)
        self.template_alpha = jnp.asarray(alpha_rel * trC / tnorm, dtype=self.dtype)

    # ------------------------------------------------------------ SHT plumbing --
    def _Yt_field(self, i, xf):
        """(ncomp*nobs, B) -> (ncomp, K, B) for field i (Y^T W x done outside)."""
        f = self.fields[i]
        if self.backend == "dense":
            # (x^T Y)^T, not Y^T x: XLA would otherwise materialize a
            # transposed copy of Y (GBs) during autotuning
            a = (xf.T @ self.Y[i]).T
        else:
            shape = jax.ShapeDtypeStruct((f.ncomp * self.K, xf.shape[-1]),
                                         xf.dtype)
            a = jax.pure_callback(
                lambda m, op=self._sht[i]: op.adjoint(np.asarray(m, np.float64)),
                shape, xf, vmap_method="sequential")
        return a.reshape(f.ncomp, self.K, -1)

    def _Y_field(self, i, af):
        """(ncomp, K, B) -> (ncomp*nobs, B) for field i."""
        f = self.fields[i]
        af = af.reshape(f.ncomp * self.K, -1)
        if self.backend == "dense":
            return self.Y[i] @ af
        shape = jax.ShapeDtypeStruct((f.ncomp * f.nobs, af.shape[-1]), af.dtype)
        return jax.pure_callback(
            lambda a, op=self._sht[i]: op.synth(np.asarray(a, np.float64)),
            shape, af, vmap_method="sequential")

    def to_modes(self, x):
        """A = B Y^T W x, stacked over components: (Nc, K, B)."""
        outs = []
        for i, f in enumerate(self.fields):
            xf = (x[self.slices[i]].reshape(f.ncomp, f.nobs, -1)
                  * self.w[i][:, :, None]).reshape(f.ncomp * f.nobs, -1)
            a = self._Yt_field(i, xf)
            if self.beams[i] is not None:
                a = a * self.beams[i][None, :, None]
            outs.append(a)
        return jnp.concatenate(outs, axis=0)

    def from_modes(self, A):
        """x = W Y B A: (Nc, K, B) -> (nrow, B)."""
        outs = []
        c0 = 0
        for i, f in enumerate(self.fields):
            a = A[c0:c0 + f.ncomp]
            c0 += f.ncomp
            if self.beams[i] is not None:
                a = a * self.beams[i][None, :, None]
            m = self._Y_field(i, a).reshape(f.ncomp, f.nobs, -1)
            outs.append((m * self.w[i][:, :, None]).reshape(f.ncomp * f.nobs, -1))
        return jnp.concatenate(outs, axis=0)

    # ------------------------------------------------------------- operators --
    def apply_C(self, x):
        """C x for a batch x of shape (nrow, B)."""
        A = self.to_modes(x)
        A = jnp.einsum("cdk,dkb->ckb", self.cl_k, A)
        out = self.from_modes(A) + self.noisevar[:, None] * x
        if self.template_alpha is not None:
            out = out + self.Tmat @ (self.template_alpha[:, None]
                                     * (self.Tmat.T @ x))
        return out

    def apply_precond(self, y):
        """Approximate C^-1 y (see module docstring)."""
        ny = self.ivar_vec[:, None] * y
        A = self.to_modes(ny)
        A = jnp.einsum("kcd,dkb->ckb", self.T_k, A)
        v = self.from_modes(A)
        return ny - self.ivar_vec[:, None] * v

    def sample(self, key, nbatch: int):
        """Draw x ~ N(0, C) (fiducial signal + noise [+ alpha templates])."""
        k1, k2, k3 = jax.random.split(key, 3)
        xi = jax.random.normal(k1, (self.K, self.ncomp, nbatch), dtype=self.dtype)
        A = jnp.einsum("kcd,kdb->ckb", self.sqrt_cl_k, xi)
        x = self.from_modes(A)
        eta = jax.random.normal(k2, x.shape, dtype=self.dtype)
        x = x + jnp.sqrt(self.noisevar)[:, None] * eta
        if self.template_alpha is not None:
            zt = jax.random.normal(k3, (self.n_templates, nbatch), dtype=self.dtype)
            x = x + self.Tmat @ (jnp.sqrt(self.template_alpha)[:, None] * zt)
        return x

    def sample_noise(self, key, nbatch: int):
        eta = jax.random.normal(key, (self.nrow, nbatch), dtype=self.dtype)
        return jnp.sqrt(self.noisevar)[:, None] * eta
