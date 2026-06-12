#!/usr/bin/env python
"""Validation test 1: CMB T/Q/U at nside=32, NaMaster mask, uniform noise.

100 realizations at fixed mask (fixed Fisher).  Two variants:

  (a) data drawn from the band-flattened input spectra; the QML estimates
      must be unbiased against the exact input bandpowers, and the chi^2 of
      the 6 x nbands bandpower vector must follow chi^2_dof -- this tests
      the estimator and its Fisher/covariance with no window ambiguity;
  (b) data drawn from the original (curved) spectra; estimates are compared
      to the window-convolved theory prediction.

Usage: python scripts/val1_cmb.py [--quick]
"""

import argparse
import json
import os
import time

import numpy as np
import healpy as hp

from val_common import (load_cmb_cls, load_mask, chi2_plot, spectra_plot,
                        pulls_plot, FIGDIR, DATADIR, CACHEDIR)
import simaster as sm

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=32)
P.add_argument("--lmax", type=int, default=None, help="default 3*nside-1")
P.add_argument("--nreal", type=int, default=100)
P.add_argument("--nsims", type=int, default=16384)
P.add_argument("--nlb", type=int, default=8)
P.add_argument("--sigma-T", type=float, default=30.0,
               help="white noise per pixel, uK")
P.add_argument("--fisher", default="exact", choices=["exact", "mc"])
P.add_argument("--batch", type=int, default=128)
P.add_argument("--quick", action="store_true", help="small fast config")
P.add_argument("--ivar-maker", default=None,
               help="python file providing make_ivar(nside, mask) "
                    "(used by val2)")
P.add_argument("--tag", default="val1")
args = P.parse_args()

if args.quick:
    args.nside, args.nsims, args.nreal = 16, 2048, 50

nside = args.nside
lmax = args.lmax or 3 * nside - 1
npix = hp.nside2npix(nside)
rng = np.random.default_rng(20260612)

cls_in = load_cmb_cls(lmax)
mask = load_mask(nside)
print(f"[{args.tag}] nside={nside} lmax={lmax} fsky={mask.mean():.3f} "
      f"nobs={int(mask.sum())}")

# noise model
if args.ivar_maker:
    ns = {}
    exec(open(args.ivar_maker).read(), ns)
    ivar_T = ns["make_ivar"](nside, mask) * 1.0 / args.sigma_T ** 2
else:
    ivar_T = np.full(npix, 1.0 / args.sigma_T ** 2)
ivar_P = ivar_T / 2.0  # Q,U noise = sqrt(2) x T, as usual

bins = sm.Bins.linear(2, lmax, args.nlb)
results = {}

for variant in ("flat", "curved"):
    t0 = time.time()
    cls_v = dict(cls_in)
    if variant == "flat":
        bext, _ = bins.extend_to_cover(2, lmax)
        for k in cls_v:
            cls_v[k] = bext.unbin_cl(bext.bin_cl(cls_v[k]), lmax)
            cls_v[k][:2] = 0.0

    fT = sm.Field(mask, [np.zeros(npix)], ivar=ivar_T, name="T")
    fP = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=ivar_P, name="P")
    cl_fid = {("T_0", "T_0"): cls_v["TT"], ("P_E", "P_E"): cls_v["EE"],
              ("P_B", "P_B"): cls_v["BB"], ("T_0", "P_E"): cls_v["TE"]}
    w = sm.QMLWorkspace([fT, fP], bins, cl_fid, lmax=lmax,
                        fisher_mode=args.fisher,
                        n_sims_fisher=args.nsims,
                        n_sims_noise=max(1024, args.nsims // 4),
                        batch_size=args.batch, seed=999, cachedir=CACHEDIR)
    w.run_exact() if args.fisher == 'exact' else w.run_mc()

    # 100 data realizations (fixed mask, fixed Fisher)
    maps_T, maps_P = [], []
    for _ in range(args.nreal):
        T, Q, U = hp.synfast([cls_v["TT"], cls_v["EE"], cls_v["BB"],
                              cls_v["TE"]], nside, lmax=lmax, pol=True,
                             new=True)
        nT = rng.normal(0, 1, npix) / np.sqrt(ivar_T)
        nQ = rng.normal(0, 1, npix) / np.sqrt(ivar_P)
        nU = rng.normal(0, 1, npix) / np.sqrt(ivar_P)
        maps_T.append(mask * T + nT)
        maps_P.append(np.array([mask * Q + nQ, mask * U + nU]))
    data = w.pack_data([np.array(maps_T), np.array(maps_P)])
    res = w.estimate(data)

    if variant == "flat":
        target = {s: w.user_bins.bin_cl(
            {"T_0 x T_0": cls_v["TT"], "P_E x P_E": cls_v["EE"],
             "P_B x P_B": cls_v["BB"], "T_0 x P_E": cls_v["TE"],
             "T_0 x P_B": np.zeros(lmax + 1),
             "P_E x P_B": np.zeros(lmax + 1)}[s]) for s in w.spec_names}
    else:
        target = w.predict({("T_0", "T_0"): cls_v["TT"],
                            ("P_E", "P_E"): cls_v["EE"],
                            ("P_B", "P_B"): cls_v["BB"],
                            ("T_0", "P_E"): cls_v["TE"]})

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
                   f"{args.tag} CMB TQU nside={nside} ({variant} input)")
    spectra_plot(w, est_mean, est_err, target,
                 os.path.join(FIGDIR, f"{tag}_spectra.png"),
                 f"{args.tag} ({variant}): mean of {args.nreal} realizations")
    pulls_plot(pulls, os.path.join(FIGDIR, f"{tag}_pulls.png"),
               f"{args.tag} ({variant}) per-band pulls")

    results[variant] = dict(
        chi2_mean=float(chi2.mean()), dof=int(dof), ks_p=float(pv),
        pull_mean=float(pulls.mean()), pull_std=float(pulls.std()),
        cg_iters=w.last_cg[0], minutes=(time.time() - t0) / 60.0)
    np.savez(os.path.join(DATADIR, f"{tag}.npz"),
             chi2=chi2, ells=res.ells, cov=res.cov,
             **{f"est_{s.replace(' ', '')}": res.cl[s] for s in w.spec_names},
             **{f"tgt_{s.replace(' ', '')}": target[s] for s in w.spec_names})
    print(f"[{tag}] chi2 mean {chi2.mean():.1f}/dof {dof}, KS p={pv:.3f}, "
          f"pulls {pulls.mean():.3f}+-{pulls.std():.3f}, "
          f"{results[variant]['minutes']:.1f} min")

with open(os.path.join(DATADIR, f"{args.tag}_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
