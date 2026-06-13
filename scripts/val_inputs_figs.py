#!/usr/bin/env python
"""Input-visualization figures for the validation suite: for each of tests
1/2/3, a 3-panel mollview of (example observed data map, mask, ivar).

Cheap (synfast + mollview only, no QML/GPU), faithful to each test's setup.

  python scripts/val_inputs_figs.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from val_common import load_cmb_cls, load_mask, FIGDIR
from val2_ivar import make_ivar

NSIDE, LMAX = 32, 64
rng = np.random.default_rng(7)


def masked(m, mask, ivar=None):
    """Set unobserved pixels (mask==0 or ivar==0) to UNSEEN for display."""
    out = np.array(m, dtype=float)
    bad = (mask == 0)
    if ivar is not None:
        bad = bad | (ivar <= 0)
    out[bad] = hp.UNSEEN
    return out


def panel(datamap, mask, ivar, tag, title, data_label, data_unit,
          ivar_unit):
    obs = (mask != 0) & (ivar > 0)
    dlim = np.percentile(np.abs(datamap[obs]), 99)
    ivlim = ivar[obs]
    fig = plt.figure(figsize=(15, 4.2))
    hp.mollview(masked(datamap, mask, ivar), sub=(1, 3, 1),
                title=f"example data: {data_label}", unit=data_unit,
                cmap="RdBu_r", min=-dlim, max=dlim)
    # mask: observed (w=1) bright, unobserved dark, explicit range
    hp.mollview(mask, sub=(1, 3, 2), title="mask $w$ (white = observed)",
                cmap="gray", min=0, max=1)
    # uniform: 0-anchor (min==max would break the colorbar); non-uniform:
    # stretch to the observed range so even a few-percent modulation shows
    uniform = np.allclose(ivlim, ivlim.flat[0])
    vmin, vmax = (0.0, ivlim.max() * 1.05) if uniform \
        else (ivlim.min(), ivlim.max())
    hp.mollview(masked(ivar, mask), sub=(1, 3, 3),
                title="inverse noise variance"
                + (" (uniform)" if uniform else ""),
                unit=ivar_unit, cmap="viridis", min=vmin, max=vmax)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.savefig(os.path.join(FIGDIR, f"{tag}_inputs.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {tag}_inputs.png")


npix = hp.nside2npix(NSIDE)
mask = load_mask(NSIDE)
cls = load_cmb_cls(LMAX)

# ---- test 1: CMB T/Q/U, uniform noise ------------------------------------
sigma_T = 50.0
ivar1 = np.full(npix, 1.0 / sigma_T ** 2)
T, Q, U = hp.synfast([cls["TT"], cls["EE"], cls["BB"], cls["TE"]], NSIDE,
                     lmax=LMAX, pol=True, new=True)
d1 = mask * T + rng.normal(0, sigma_T, npix)
panel(d1, mask, ivar1, "val1",
      r"Test 1 inputs: CMB $T/Q/U$, NaMaster mask, uniform noise "
      r"($\sigma_T=50\,\mu$K)",
      r"observed $T$", r"$\mu$K", r"$\mu$K$^{-2}$")

# ---- test 2: same, anisotropic longitude-strip ivar ----------------------
ivar2 = make_ivar(NSIDE, mask) / sigma_T ** 2
d2 = mask * T + rng.normal(0, 1, npix) / np.sqrt(ivar2)
panel(d2, mask, ivar2, "val2",
      r"Test 2 inputs: as test 1 but ivar in longitude strips "
      r"($\times2$ rms variation)",
      r"observed $T$", r"$\mu$K", r"$\mu$K$^{-2}$")

# ---- test 3: LSS density + shear (pyccl), varying source-count ivar -------
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.81,
                      n_s=0.96)
z = np.linspace(0.01, 4.0, 600)
nz_l = np.exp(-0.5 * ((z - 0.75) / 0.05) ** 2)
nz_cnt = np.exp(-0.5 * ((z - 2.0) / 0.1) ** 2)
ells = np.arange(LMAX + 1)
tr_g = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz_l),
                              bias=(z, np.ones_like(z)))
tr_cnt = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz_cnt),
                                bias=(z, np.ones_like(z)))
cl_gg = ccl.angular_cl(cosmo, tr_g, tr_g, ells); cl_gg[:2] = 0.0
cl_cnt = ccl.angular_cl(cosmo, tr_cnt, tr_cnt, ells); cl_cnt[:2] = 0.0
pixarcmin2 = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0
ngal_mean = 15.0 * pixarcmin2
shape_noise = 0.3
# the shear ivar follows the non-uniform source count ngal = nbar (1+delta_z2)
np.random.seed(424242)                       # same map as val3_lss.py
delta_cnt = hp.synfast(cl_cnt, NSIDE, lmax=LMAX)
ngal_src = ngal_mean * np.clip(1.0 + delta_cnt, 1e-3, None)
ivar_s = ngal_src / shape_noise ** 2
g = hp.synfast(cl_gg, NSIDE, lmax=LMAX)
d3 = mask * g + rng.normal(0, 1, npix) / np.sqrt(np.full(npix, ngal_mean))
panel(d3, mask, ivar_s, "val3",
      r"Test 3 inputs: galaxy overdensity $\delta$ (lens); shear $ivar$ "
      r"$\propto$ source count $\bar n(1+\delta_{z=2})$ (non-uniform)",
      r"observed $\delta$", "", r"sources/pix $\,/\,\sigma_e^2$")

print("done")
