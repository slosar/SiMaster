#!/usr/bin/env python
"""Iteration demo: start from a deliberately wrong fiducial (x4 amplitude),
show that (i) the single-shot estimate is already unbiased, with inflated
errors, and (ii) one iteration re-centers the fiducial and recovers
near-optimal error bars.  nside=16, spin-0, exact response engine.
"""

import json
import os

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from val_common import load_cmb_cls, load_mask, FIGDIR, DATADIR, CACHEDIR
import simaster as sm

nside, lmax, nreal = 16, 47, 100
npix = hp.nside2npix(nside)
rng = np.random.default_rng(4)
mask = load_mask(nside)
cl = load_cmb_cls(lmax)["TT"]
sigma = 30.0
ivar = np.full(npix, 1 / sigma ** 2)
bins = sm.Bins.linear(2, lmax, 5)
bext, _ = bins.extend_to_cover(2, lmax)
cl_flat = bext.unbin_cl(bext.bin_cl(cl), lmax); cl_flat[:2] = 0

maps = np.array([hp.synfast(cl_flat, nside, lmax=lmax) * mask
                 + rng.normal(0, sigma, npix) for _ in range(nreal)])
target = bins.bin_cl(cl_flat)

out = {}
for label, fid_scale in [("true fiducial", 1.0), ("wrong fiducial (x4)", 4.0)]:
    f = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="T")
    w = sm.QMLWorkspace(f, bins, {("T_0", "T_0"): fid_scale * cl_flat},
                        lmax=lmax, fisher_mode="exact", verbose=False,
                        cachedir=CACHEDIR)
    data = w.pack_data([maps])
    hist = w.iterate(data, n_iter=2 if fid_scale != 1.0 else 1)
    for it, res in enumerate(hist):
        est = res.cl["T_0 x T_0"]
        name = f"{label}, iter {it}"
        out[name] = dict(
            bias_sigma=list((est.mean(0) - target) / est.std(0)),
            err_ratio=list(est.std(0)
                           / np.sqrt(np.diag(res.cov))[: bins.nbands]))
        print(name, "bias/sigma:", np.round(out[name]["bias_sigma"], 2),
              "emp/pred err:", np.round(out[name]["err_ratio"], 2))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ells = bins.get_effective_ells()
for name, r in out.items():
    axes[0].plot(ells, r["bias_sigma"], "o-", ms=3, label=name)
    axes[1].plot(ells, r["err_ratio"], "o-", ms=3, label=name)
axes[0].axhline(0, color="k", lw=.5); axes[0].set_ylabel(r"bias / $\sigma$")
axes[1].axhline(1, color="k", lw=.5)
axes[1].set_ylabel("empirical / predicted error")
for ax in axes:
    ax.set_xlabel(r"$\ell$"); ax.legend(fontsize=7)
fig.suptitle("Iteration from a wrong fiducial (nside=16, TT)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "val4_iteration.png"), dpi=150)
with open(os.path.join(DATADIR, "val4_summary.json"), "w") as fjs:
    json.dump(out, fjs, indent=2)
print("saved val4_iteration.png")
