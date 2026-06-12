#!/usr/bin/env python
"""Validation test 3: galaxy overdensity + weak lensing shear with pyccl.

Lenses (galaxy density, bias=1) at z=0.75, sources at z=1.5.  Theory C_l
from pyccl.  Realistic noise: 15 sources/arcmin^2, shape noise 0.3 per
component, per-pixel shear noise = 0.3/sqrt(ngal in pixel); galaxy shot
noise = 1/ngal per pixel (Poisson).  Shows unbiased recovery of density,
shear E/B and all cross-spectra over 100 realizations (band-flat input
variant for the chi^2, like val1).

Usage: python scripts/val3_lss.py [--quick]
"""

import argparse
import json
import os
import time

import numpy as np
import healpy as hp

from val_common import (load_mask, chi2_plot, spectra_plot, pulls_plot,
                        FIGDIR, DATADIR, CACHEDIR)
import simaster as sm

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=32)
P.add_argument("--lmax", type=int, default=None)
P.add_argument("--nreal", type=int, default=100)
P.add_argument("--nsims", type=int, default=16384)
P.add_argument("--nlb", type=int, default=8)
P.add_argument("--ngal-arcmin2", type=float, default=15.0)
P.add_argument("--shape-noise", type=float, default=0.3)
P.add_argument("--fisher", default="exact", choices=["exact", "mc"])
P.add_argument("--batch", type=int, default=128)
P.add_argument("--quick", action="store_true")
P.add_argument("--variants", default="flat,curved")
P.add_argument("--tag", default="val3")
args = P.parse_args()
if args.quick:
    args.nside, args.nsims, args.nreal = 16, 2048, 50

nside = args.nside
lmax = args.lmax or 3 * nside - 1
npix = hp.nside2npix(nside)
rng = np.random.default_rng(20260613)

# ---------------------------------------------------------------- theory --
import pyccl as ccl
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.81,
                      n_s=0.96)
z = np.linspace(0.01, 3.0, 400)
nz_lens = np.exp(-0.5 * ((z - 0.75) / 0.05) ** 2)
nz_src = np.exp(-0.5 * ((z - 1.50) / 0.05) ** 2)
tr_g = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz_lens),
                              bias=(z, np.ones_like(z)))
tr_k = ccl.WeakLensingTracer(cosmo, dndz=(z, nz_src))
ells = np.arange(lmax + 1)
cl_gg = ccl.angular_cl(cosmo, tr_g, tr_g, ells)
cl_gk = ccl.angular_cl(cosmo, tr_g, tr_k, ells)
cl_kk = ccl.angular_cl(cosmo, tr_k, tr_k, ells)
for c in (cl_gg, cl_gk, cl_kk):
    c[:2] = 0.0
cl_bb = np.zeros(lmax + 1)

# ----------------------------------------------------------------- noise --
mask = load_mask(nside)
pixarea_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600.0
ngal_pix = args.ngal_arcmin2 * pixarea_arcmin2
sigma_gamma = args.shape_noise / np.sqrt(ngal_pix)   # per Q/U component
ivar_g = np.full(npix, ngal_pix)                     # delta noise var = 1/ngal
ivar_s = np.full(npix, 1.0 / sigma_gamma ** 2)
print(f"[{args.tag}] nside={nside} lmax={lmax} fsky={mask.mean():.3f} "
      f"ngal/pix={ngal_pix:.0f} sigma_gamma={sigma_gamma:.2e}")

bins = sm.Bins.linear(2, lmax, args.nlb)
results = {}

for variant in args.variants.split(","):
    t0 = time.time()
    cls_v = {"gg": cl_gg.copy(), "ge": cl_gk.copy(), "ee": cl_kk.copy(),
             "bb": cl_bb.copy()}
    if variant == "flat":
        bext, _ = bins.extend_to_cover(2, lmax)
        for k in cls_v:
            cls_v[k] = bext.unbin_cl(bext.bin_cl(cls_v[k]), lmax)
            cls_v[k][:2] = 0.0

    fg = sm.Field(mask, [np.zeros(npix)], ivar=ivar_g, name="g")
    fs = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=ivar_s, name="s")
    cl_fid = {("g_0", "g_0"): cls_v["gg"], ("s_E", "s_E"): cls_v["ee"],
              ("s_B", "s_B"): cls_v["bb"], ("g_0", "s_E"): cls_v["ge"]}
    w = sm.QMLWorkspace([fg, fs], bins, cl_fid, lmax=lmax,
                        fisher_mode=args.fisher,
                        n_sims_fisher=args.nsims,
                        n_sims_noise=max(1024, args.nsims // 4),
                        batch_size=args.batch, seed=777, cachedir=CACHEDIR)
    w.run_exact() if args.fisher == 'exact' else w.run_mc()

    maps_g, maps_s = [], []
    for _ in range(args.nreal):
        g, q, u = hp.synfast([cls_v["gg"], cls_v["ee"], cls_v["bb"],
                              cls_v["ge"]], nside, lmax=lmax, pol=True,
                             new=True)
        maps_g.append(mask * g + rng.normal(0, 1, npix) / np.sqrt(ivar_g))
        maps_s.append(np.array([
            mask * q + rng.normal(0, sigma_gamma, npix),
            mask * u + rng.normal(0, sigma_gamma, npix)]))
    data = w.pack_data([np.array(maps_g), np.array(maps_s)])
    res = w.estimate(data)

    if variant == "flat":
        target = {s: w.user_bins.bin_cl(
            {"g_0 x g_0": cls_v["gg"], "s_E x s_E": cls_v["ee"],
             "s_B x s_B": cls_v["bb"], "g_0 x s_E": cls_v["ge"],
             "g_0 x s_B": np.zeros(lmax + 1),
             "s_E x s_B": np.zeros(lmax + 1)}[s]) for s in w.spec_names}
    else:
        target = w.predict(cl_fid)

    est_mean = {s: res.cl[s].mean(0) for s in w.spec_names}
    est_err = {s: res.cl[s].std(0) / np.sqrt(args.nreal) for s in w.spec_names}
    chi2 = res.chi2(target)
    dof = res.vector().shape[1]
    sig = np.sqrt(np.diag(res.cov)).reshape(len(w.spec_names), -1)
    pulls = np.concatenate([(res.cl[s] - target[s]) / sig[i]
                            for i, s in enumerate(w.spec_names)],
                           axis=1).ravel()

    tag = f"{args.tag}_{variant}" + ("" if args.fisher == "exact" else "_mc")
    pv = chi2_plot(chi2, dof, os.path.join(FIGDIR, f"{tag}_chi2.png"),
                   f"LSS density+shear nside={nside} ({variant} input)")
    labels = {"g_0 x g_0": r"$\delta\delta$", "g_0 x s_E": r"$\delta E$",
              "g_0 x s_B": r"$\delta B$", "s_E x s_E": r"$EE$",
              "s_E x s_B": r"$EB$", "s_B x s_B": r"$BB$"}
    spectra_plot(w, est_mean, est_err, target,
                 os.path.join(FIGDIR, f"{tag}_spectra.png"),
                 f"{args.tag} ({variant}): mean of {args.nreal} realizations",
                 labels=labels)
    pulls_plot(pulls, os.path.join(FIGDIR, f"{tag}_pulls.png"),
               f"{args.tag} ({variant}) per-band pulls")
    results[variant] = dict(chi2_mean=float(chi2.mean()), dof=int(dof),
                            ks_p=float(pv), pull_mean=float(pulls.mean()),
                            pull_std=float(pulls.std()),
                            minutes=(time.time() - t0) / 60.0)
    np.savez(os.path.join(DATADIR, f"{tag}.npz"), chi2=chi2, ells=res.ells,
             cov=res.cov,
             **{f"est_{s.replace(' ', '')}": res.cl[s] for s in w.spec_names},
             **{f"tgt_{s.replace(' ', '')}": target[s] for s in w.spec_names})
    print(f"[{tag}] chi2 {chi2.mean():.1f}/{dof}, KS p={pv:.3f}, "
          f"{results[variant]['minutes']:.1f} min")

with open(os.path.join(DATADIR, f"{args.tag}_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
