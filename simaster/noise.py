"""Per-pixel block noise covariance.

The data vector concatenates, over fields, each field's observed-pixel
component maps (1 for spin 0, [Q, U] for spin 2), exactly as assembled by
:class:`simaster.covariance.CovModel`.  Its noise covariance ``N`` is
block-diagonal in *sky pixel*: most rows carry a scalar variance ``1/ivar``,
but a set of rows that share a sky pixel may be coupled by a small dense
block -- e.g. a per-pixel 3x3 I/Q/U noise covariance that crosses a spin-0
(I) field and a spin-2 (Q, U) field, as delivered by the Planck NPIPE
``wcov`` products (II, IQ, IU, QQ, QU, UU).

The scalar-diagonal model SiMaster used previously is the 1x1 special case:
with no coupling groups, :meth:`PixelNoiseCov.apply` and friends reduce to
``noisevar * x``.  Pixel-pixel correlations are *not* representable -- the
matrix-free design requires ``N`` to stay block-diagonal in pixel space.

``N`` enters the estimator only through a handful of operators, all of which
have a clean per-pixel block analogue:

    N x         (apply)        -- covariance.apply_C
    N^-1 x      (apply_inv)    -- covariance.apply_precond (Woodbury)
    N^(1/2) x   (sqrt_apply)   -- covariance.sample / sample_noise probes
    V^T N V     (quad_cdj)     -- qml noise-bias window accumulation
    dense N                    -- exact.ExactQML reference (tiny nside only)

``ivar_eff`` (the diagonal of ``N^-1``) feeds the preconditioner's per-l
sensitivity bound; for a diagonal block it is just ``ivar``.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def field_row_layout(fields):
    """Global row layout shared with :class:`CovModel`.

    Returns ``(slices, comp_of_field, nrow)`` where field ``i`` occupies
    ``slices[i]`` (a contiguous block ``ncomp*nobs`` long, component-major:
    all of component 0's pixels, then component 1's), and ``comp_of_field[i]``
    lists its *global* component indices.
    """
    slices, comp_of_field, nrow, c0 = [], [], 0, 0
    for f in fields:
        comp_of_field.append(list(range(c0, c0 + f.ncomp)))
        slices.append(slice(nrow, nrow + f.ncomp * f.nobs))
        nrow += f.ncomp * f.nobs
        c0 += f.ncomp
    return slices, comp_of_field, nrow


def _diag_noisevar(fields):
    """Baseline scalar noise variance per row, from each field's ivar."""
    return np.concatenate(
        [np.tile(1.0 / f.ivar, f.ncomp) for f in fields]).astype(np.float64)


class _Group:
    """A set of ``npix`` pixel-blocks of size ``sz``, sharing the operator.

    ``rows`` is ``(npix, sz)`` global row indices; ``cov`` is ``(npix, sz, sz)``
    symmetric PSD blocks.  Block inverse and symmetric square root are formed
    once via eigendecomposition (robust for the small, well-conditioned blocks
    here), and the strictly off-diagonal part is cached for the ``V^T N V``
    correction.
    """

    def __init__(self, rows, cov):
        rows = np.asarray(rows, dtype=np.int64)
        cov = np.asarray(cov, dtype=np.float64)
        npix, sz, sz2 = cov.shape
        if sz != sz2 or rows.shape != (npix, sz):
            raise ValueError("group rows/cov shape mismatch")
        cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))     # symmetrise
        w, Q = np.linalg.eigh(cov)
        if np.any(w <= 0):
            wmin = float(w.min())
            raise ValueError(f"non-positive-definite noise block (min eig "
                             f"{wmin:.3e}); check the pixel covariance maps")
        inv = np.einsum("pij,pj,pkj->pik", Q, 1.0 / w, Q)
        sqrt = np.einsum("pij,pj,pkj->pik", Q, np.sqrt(w), Q)
        diag = np.einsum("pii->pi", cov)                 # (npix, sz)
        offdiag = cov - np.einsum("pi,ij->pij", diag, np.eye(sz))

        self.rows = jnp.asarray(rows)
        self.cov = jnp.asarray(cov)
        self.inv = jnp.asarray(inv)
        self.sqrt = jnp.asarray(sqrt)
        self.offdiag = jnp.asarray(offdiag)
        self._rows_np = rows
        self._cov_np = cov
        self._inv_diag = np.einsum("pii->pi", inv)       # diag(N^-1) per row


class PixelNoiseCov:
    """Block-diagonal (per sky pixel) pixel-noise covariance operator.

    Parameters
    ----------
    fields : list of Field
        Same list, in the same order, passed to :class:`QMLWorkspace` /
        :class:`CovModel` -- fixes the global row layout.
    noisevar : (nrow,) array, optional
        Scalar per-row variance.  Default: each field's ``1/ivar``.  Entries
        for rows inside a coupling group are overwritten by that group's block
        diagonal so the exposed diagonal is always exact.
    groups : list of _Group, optional
        Coupling blocks (e.g. one per sky pixel for IQU).  Their row sets must
        be disjoint.
    """

    def __init__(self, fields, noisevar=None, groups=None):
        self.fields = list(fields)
        self.slices, self.comp_of_field, self.nrow = field_row_layout(fields)
        self.groups = list(groups) if groups else []

        nv = (_diag_noisevar(fields) if noisevar is None
              else np.asarray(noisevar, dtype=np.float64).copy())
        if nv.shape != (self.nrow,):
            raise ValueError(f"noisevar must be ({self.nrow},), got {nv.shape}")

        # block diagonal overrides the scalar baseline; off-diagonal-only ivar
        ivar_eff = 1.0 / nv
        seen = np.zeros(self.nrow, dtype=bool)
        for g in self.groups:
            r = g._rows_np.reshape(-1)
            if seen[r].any():
                raise ValueError("coupling groups share rows; blocks must be "
                                 "disjoint in pixel space")
            seen[r] = True
            nv[g._rows_np] = np.einsum("pii->pi", g._cov_np)
            ivar_eff[g._rows_np] = g._inv_diag

        self._noisevar = jnp.asarray(nv)
        self._ivar_diag = jnp.asarray(1.0 / nv)          # 1/diag(N) (scalar)
        self._sqrt_nv = jnp.asarray(np.sqrt(nv))
        self.ivar_eff = jnp.asarray(ivar_eff)            # diag(N^-1), per row

    # exposed diagonals (back-compat with the old scalar `noisevar`) ----------
    @property
    def noisevar(self):
        """Exact per-row diagonal of N (block diagonal where coupled)."""
        return self._noisevar

    @property
    def ivar_diag(self):
        """1/diag(N) -- the scalar reciprocal, *not* diag(N^-1)."""
        return self._ivar_diag

    @property
    def has_blocks(self):
        return bool(self.groups)

    # operators ---------------------------------------------------------------
    def apply(self, x):
        """N x for a batch x of shape (nrow, B)."""
        out = self._noisevar[:, None] * x
        for g in self.groups:
            xb = x[g.rows]                                # (npix, sz, B)
            cb = jnp.einsum("pij,pjb->pib", g.cov, xb)
            out = out.at[g.rows].set(cb)
        return out

    def apply_inv(self, x):
        """N^-1 x via per-pixel block solve (exact)."""
        out = self._ivar_diag[:, None] * x
        for g in self.groups:
            out = out.at[g.rows].set(
                jnp.einsum("pij,pjb->pib", g.inv, x[g.rows]))
        return out

    def sqrt_apply(self, x):
        """N^(1/2) x (symmetric square root, per pixel block)."""
        out = self._sqrt_nv[:, None] * x
        for g in self.groups:
            out = out.at[g.rows].set(
                jnp.einsum("pij,pjb->pib", g.sqrt, x[g.rows]))
        return out

    def quad_cdj(self, V):
        """V^T N V contracted as ('pjc,p,pjd->cdj'); V is (nrow, J, Nc).

        Returns (Nc, Nc, J).  The diagonal baseline plus, per group, the
        off-diagonal block correction sum_{p,s,s'} V[s] Noff[s,s'] V[s'].
        """
        out = jnp.einsum("pjc,p,pjd->cdj", V, self._noisevar, V)
        for g in self.groups:
            Vg = V[g.rows]                               # (npix, sz, J, Nc)
            out = out + jnp.einsum("psjc,pst,ptjd->cdj", Vg, g.offdiag, Vg)
        return out

    def dense(self):
        """Full dense (nrow, nrow) N (numpy); only for tiny-nside references."""
        M = np.diag(np.asarray(self._noisevar))
        for g in self.groups:
            rows = g._rows_np
            cov = g._cov_np
            for p in range(rows.shape[0]):
                idx = rows[p]
                M[np.ix_(idx, idx)] = cov[p]
        return M

    # constructors ------------------------------------------------------------
    @classmethod
    def iqu(cls, fields, *, II, IQ, IU, QQ, QU, UU, check=True):
        """Per-pixel 3x3 I/Q/U noise coupling across a spin-0 and spin-2 field.

        ``fields`` must contain exactly one spin-0 field (temperature, I) and
        one spin-2 field (polarization, Q/U) which share ``obs_pix``.  The six
        covariance maps are the upper triangle of the symmetric 3x3 pixel
        noise covariance, in K_CMB^2; they may be full-sky healpy RING arrays
        (restricted to the shared observed pixels here) or already restricted.
        """
        s0 = [f for f in fields if f.spin == 0]
        s2 = [f for f in fields if f.spin == 2]
        if len(s0) != 1 or len(s2) != 1:
            raise ValueError("PixelNoiseCov.iqu needs exactly one spin-0 (I) "
                             "and one spin-2 (Q,U) field")
        f0, f2 = s0[0], s2[0]
        if f0.obs_pix.shape != f2.obs_pix.shape or \
                not np.array_equal(f0.obs_pix, f2.obs_pix):
            raise ValueError("the I and Q/U fields must share obs_pix for a "
                             "per-pixel IQU noise coupling (use the same mask "
                             "and a consistent ivar cut; iqu_from_cov does this)")
        op = f0.obs_pix
        npix = op.size

        def restrict(m, label):
            m = np.asarray(m, dtype=np.float64).reshape(-1)
            if m.size == npix:
                return m
            if m.size > npix:
                return m[op]
            raise ValueError(f"{label} map too small ({m.size} < {npix})")

        II = restrict(II, "II"); QQ = restrict(QQ, "QQ"); UU = restrict(UU, "UU")
        IQ = restrict(IQ, "IQ"); IU = restrict(IU, "IU"); QU = restrict(QU, "QU")

        blk = np.empty((npix, 3, 3))
        blk[:, 0, 0] = II
        blk[:, 1, 1] = QQ
        blk[:, 2, 2] = UU
        blk[:, 0, 1] = blk[:, 1, 0] = IQ
        blk[:, 0, 2] = blk[:, 2, 0] = IU
        blk[:, 1, 2] = blk[:, 2, 1] = QU

        slices, _, _ = field_row_layout(fields)
        i0, i2 = fields.index(f0), fields.index(f2)
        s0sl, s2sl = slices[i0], slices[i2]
        Irow = s0sl.start + np.arange(npix)
        Qrow = s2sl.start + np.arange(npix)
        Urow = s2sl.start + npix + np.arange(npix)
        rows = np.stack([Irow, Qrow, Urow], axis=1)      # (npix, 3): (I,Q,U)

        noisevar = _diag_noisevar(fields)
        if check:
            # the field ivars only drive pixel selection; the block diagonal is
            # authoritative, but a gross mismatch usually means a units/order
            # bug, so flag it.
            want = np.stack([II, QQ, UU], axis=1).reshape(-1)
            have = noisevar[rows.reshape(-1)]
            bad = np.abs(want - have) > 0.25 * np.abs(want) + 1e-300
            if bad.mean() > 0.1:
                import warnings
                warnings.warn(
                    "PixelNoiseCov.iqu: field ivar disagrees with the block "
                    f"diagonal on {100*bad.mean():.0f}% of rows; the block "
                    "diagonal will be used. Check units/ordering if unexpected.")
        return cls(fields, noisevar=noisevar, groups=[_Group(rows, blk)])


def iqu_from_cov(mask, maps, cov_maps, *, beam=None, names=("T", "P"),
                 nest=False):
    """Build the I (spin-0) and Q/U (spin-2) fields *and* their coupled noise.

    A one-call helper for per-pixel correlated I/Q/U noise (e.g. the Planck
    NPIPE ``wcov`` 3x3 blocks), guaranteeing the two fields share a pixel set
    and that the noise block diagonal is consistent with each field's ivar.

    Parameters
    ----------
    mask : array
        Multiplicative mask (healpy RING, or NESTED with ``nest=True``).
    maps : [I, Q, U] or None
        Observed maps; ``None`` for a geometry-only workspace.
    cov_maps : [II, IQ, IU, QQ, QU, UU]
        Upper triangle of the symmetric 3x3 per-pixel noise covariance
        (K_CMB^2), in the same ordering/units as ``mask``.
    beam : array, optional
        Harmonic transfer function b_l applied to *both* fields.
    names : (str, str)
        Field names for the I and Q/U fields (default ("T", "P")).
    nest : bool
        If True, inputs are NESTED and are reordered to RING (the NPIPE map +
        wcov products are NESTED; the masks are RING -- reorder consistently).

    Returns
    -------
    (f0, f2, noise) : the spin-0 field, the spin-2 field, and a
        :class:`PixelNoiseCov` to pass as ``QMLWorkspace(noise_cov=...)``.
    """
    import healpy as hp
    from .field import Field

    II, IQ, IU, QQ, QU, UU = (np.asarray(c, dtype=np.float64) for c in cov_maps)
    mask = np.asarray(mask, dtype=np.float64)
    if nest:
        reord = lambda m: hp.reorder(m, n2r=True)
        mask = reord(mask)
        II, IQ, IU, QQ, QU, UU = map(reord, (II, IQ, IU, QQ, QU, UU))
        if maps is not None:
            maps = [reord(np.asarray(m, dtype=np.float64)) for m in maps]

    # a pixel is usable only where the mask is on and all three variances are
    # finite & positive; both fields must drop exactly the same pixels so the
    # 3x3 blocks line up row-for-row.
    good = (mask != 0) & (II > 0) & (QQ > 0) & (UU > 0)
    ivar_I = np.where(good, 1.0 / np.where(II > 0, II, 1.0), 0.0)
    ivar_P = np.where(good, 2.0 / np.where(QQ + UU > 0, QQ + UU, 1.0), 0.0)

    mI = None if maps is None else [maps[0]]
    mP = None if maps is None else [maps[1], maps[2]]
    f0 = Field(mask, mI, spin=0, ivar=ivar_I, beam=beam, name=names[0])
    f2 = Field(mask, mP, spin=2, ivar=ivar_P, beam=beam, name=names[1])
    noise = PixelNoiseCov.iqu([f0, f2], II=II, IQ=IQ, IU=IU,
                              QQ=QQ, QU=QU, UU=UU, check=False)
    return f0, f2, noise
