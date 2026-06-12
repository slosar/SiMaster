"""NaMaster-style convenience interface.

For pymaster users:

    import pymaster as nmt                 import simaster as sm
    f0 = nmt.NmtField(mask, [m])           f0 = sm.Field(mask, [m], ivar=ivar)
    b  = nmt.NmtBin.from_nside_linear(...) b  = sm.Bins.from_nside_linear(...)
    cl = nmt.compute_full_master(f0,f0,b)  cl = sm.compute_full_master(f0, f0, b,
                                                    cl_guess=cl_fid)

Differences (QML is a different estimator):
  * ivar is mandatory -- QML needs a noise model (it also removes the noise
    bias automatically, so there is no cl_noise argument).
  * cl_guess (the fiducial spectra) is mandatory: dict or array, see
    QMLWorkspace.  Estimates are unbiased for any reasonable guess.
  * The return value matches NaMaster ordering:
    spin0 x spin0 -> [c00]; spin0 x spin2 -> [TE, TB];
    spin2 x spin2 -> [EE, EB, BE, BB].
"""

from __future__ import annotations

import numpy as np

from .field import Field
from .bins import Bins
from .qml import QMLWorkspace


def compute_full_master(f1: Field, f2: Field, b: Bins, cl_guess=None,
                        workspace=None, return_workspace=False, **kwargs):
    """QML analogue of pymaster.compute_full_master."""
    if cl_guess is None:
        raise ValueError("QML needs fiducial spectra: pass cl_guess "
                         "(dict {(comp_i, comp_j): C_l} or full matrix)")
    same = f1 is f2
    fields = [f1] if same else [f1, f2]
    if workspace is None:
        workspace = QMLWorkspace(fields, b, cl_guess, **kwargs)
    res = workspace.estimate()

    # NaMaster output ordering
    n1 = f1.comp_names
    n2 = n1 if same else f2.comp_names
    rows = []
    for c1 in n1:
        for c2 in n2:
            key = f"{c1} x {c2}" if f"{c1} x {c2}" in res.cl else f"{c2} x {c1}"
            rows.append(np.atleast_1d(res.cl[key]))
    out = np.array(rows)
    if return_workspace:
        return out, workspace, res
    return out
