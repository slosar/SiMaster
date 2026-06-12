"""Field container, mirroring NaMaster's NmtField as closely as practical.

A field consists of
  * ``mask``  -- multiplicative response of the instrument/survey,
  * ``maps``  -- the *observed* maps (1 map for spin 0, [Q, U] for spin 2),
  * ``ivar``  -- inverse noise variance per pixel (same for Q and U).

The data model is  observed = mask * signal + noise,  noise ~ N(0, 1/ivar)
per pixel, independent between pixels (the noise covariance is diagonal in
pixel space by construction).  Pixels with mask == 0 or ivar == 0 are
discarded entirely.

Inputs may be RING-ordered healpy arrays or ``healsparse.HealSparseMap``
objects (converted internally; healsparse is NEST-ordered, we reorder).
"""

from __future__ import annotations

import numpy as np
import healpy as hp


def _to_ring_array(m, nside_expect=None):
    """Accept a healpy RING array or a HealSparseMap; return (array, nside)."""
    try:
        import healsparse
        if isinstance(m, healsparse.HealSparseMap):
            nside = m.nside_sparse
            arr = m.generate_healpix_map(nside=nside, reduction="mean")
            arr = np.where(arr == hp.UNSEEN, 0.0, arr)
            arr = hp.reorder(arr, n2r=True)
            return np.asarray(arr, dtype=np.float64), nside
    except ImportError:
        pass
    arr = np.asarray(m, dtype=np.float64)
    nside = hp.npix2nside(arr.size)
    if nside_expect is not None and nside != nside_expect:
        raise ValueError(f"nside mismatch: got {nside}, expected {nside_expect}")
    return arr, nside


class Field:
    """A spin-0 or spin-2 field on the (cut) sphere.

    Parameters
    ----------
    mask : array or HealSparseMap
        Multiplicative mask applied to the signal.
    maps : list of arrays/HealSparseMaps, or None
        Observed maps; ``[m]`` for spin 0, ``[Q, U]`` for spin 2.  May be
        None when the field only defines geometry (e.g. to build a
        workspace before data exist).
    spin : int, optional
        0 or 2; inferred from ``len(maps)`` if omitted.
    ivar : array or HealSparseMap, optional
        Inverse noise variance per pixel.  Default: infinite S/N is not
        allowed -- a value must be supplied for QML (noise defines C).
        Pixels with ivar == 0 are dropped (infinite noise).
    templates : list, optional
        Template maps to marginalize over (each shaped like ``maps``,
        i.e. [t] for spin 0 or [tQ, tU] for spin 2).  Their amplitude is
        marginalized by adding alpha * t t^T to the covariance (alpha -> inf
        limit by default; see QMLWorkspace(template_alpha=...)).
    beam : array, optional
        Harmonic transfer function b_l multiplying the signal.
    name : str, optional
        Used to label spectra; defaults to 'f{i}' assigned by the workspace.
    """

    def __init__(self, mask, maps, spin=None, ivar=None, templates=None,
                 beam=None, name=None):
        self.mask_full, self.nside = _to_ring_array(mask)
        if maps is not None and not isinstance(maps, (list, tuple)):
            maps = [maps]
        if spin is None:
            if maps is None:
                raise ValueError("spin must be given when maps is None")
            spin = 0 if len(maps) == 1 else 2
        self.spin = int(spin)
        self.ncomp = 1 if self.spin == 0 else 2
        if maps is not None and len(maps) != self.ncomp:
            raise ValueError(f"spin {self.spin} needs {self.ncomp} map(s)")

        if ivar is None:
            raise ValueError("QML estimation requires an ivar (inverse noise "
                             "variance) map; for nearly noiseless data supply "
                             "a large finite ivar.")
        self.ivar_full, _ = _to_ring_array(ivar, self.nside)

        self.obs_pix = np.flatnonzero((self.mask_full != 0) & (self.ivar_full > 0))
        self.nobs = self.obs_pix.size
        if self.nobs == 0:
            raise ValueError("no observed pixels (mask*ivar == 0 everywhere)")
        self.mask = self.mask_full[self.obs_pix]
        self.ivar = self.ivar_full[self.obs_pix]

        self.maps = None
        if maps is not None:
            arrs = [(_to_ring_array(m, self.nside)[0])[self.obs_pix] for m in maps]
            self.maps = np.stack(arrs)  # (ncomp, nobs)

        self.templates = []
        if templates:
            for t in templates:
                if not isinstance(t, (list, tuple)):
                    t = [t]
                if len(t) != self.ncomp:
                    raise ValueError("template shape must match field maps")
                tt = np.stack([(_to_ring_array(x, self.nside)[0])[self.obs_pix]
                               for x in t])
                self.templates.append(tt)

        self.beam = None if beam is None else np.asarray(beam, dtype=np.float64)
        self.name = name

    @property
    def comp_names(self):
        base = self.name or "f"
        return [f"{base}_0"] if self.spin == 0 else [f"{base}_E", f"{base}_B"]

    def data_vector(self):
        """Observed maps flattened to (ncomp*nobs,)."""
        if self.maps is None:
            raise ValueError("field has no maps")
        return self.maps.reshape(-1)
