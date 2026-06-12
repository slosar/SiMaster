#!/usr/bin/env python
"""nside=1024 harness -- run on a machine with a large GPU (A100/H100 class).

Same physics as validation test 1 (CMB T/Q/U, NaMaster mask upgraded to
nside=1024, uniform noise) but at full resolution with the matrix-free
'ducc' backend and the Monte-Carlo response engine.  Designed to be
restartable and to checkpoint the expensive pieces.

Stages (run sequentially, each checkpointed under --workdir):
  1. fisher  -- MC response + noise bias (the expensive part)
  2. data    -- generate + estimate nreal realizations
  3. report  -- chi2 / plots, like val1

Read scripts/bench_feasibility.py output and the report's feasibility
section for the expected cost before launching.  Tune:
  --batch       RHS per CG solve batch (GPU RAM bound: ~5 vectors of
                3*nobs*batch float64 live at once)
  --nsims       MC sims; remember the frozen-R offset scales as
                sigma * sqrt(SNR_tot^2 / nsims) -- for signal-dominated CMB
                at lmax=2048 this wants O(1e5) sims; consider --nlb >= 16,
                noise-dominating sigma-T, or iterating instead.
"""

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import healpy as hp

from val_common import load_cmb_cls, NMT_TEST
import simaster as sm

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=1024)
P.add_argument("--lmax", type=int, default=None, help="default 2*nside")
P.add_argument("--nlb", type=int, default=16)
P.add_argument("--nreal", type=int, default=100)
P.add_argument("--nsims", type=int, default=100000)
P.add_argument("--batch", type=int, default=64)
P.add_argument("--sigma-T", type=float, default=30.0)
P.add_argument("--cg-tol", type=float, default=1e-5)
P.add_argument("--workdir", default="nside1024_work")
P.add_argument("--stage", default="all", choices=["fisher", "data", "all"])
args = P.parse_args()

nside = args.nside
lmax = args.lmax or 2 * nside
npix = hp.nside2npix(nside)
os.makedirs(args.workdir, exist_ok=True)

mask = hp.read_map(os.path.join(NMT_TEST, "mask.fits"))
mask = (hp.ud_grade(mask, nside) > 0.5).astype(np.float64)
ivar_T = np.full(npix, 1.0 / args.sigma_T ** 2)
cls_in = load_cmb_cls(min(lmax, 767))
cls = {k: np.pad(v, (0, max(0, lmax + 1 - v.size))) for k, v in cls_in.items()}

bins = sm.Bins.linear(2, lmax, args.nlb)
fT = sm.Field(mask, [np.zeros(npix)], ivar=ivar_T, name="T")
fP = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=ivar_T / 2, name="P")
cl_fid = {("T_0", "T_0"): cls["TT"], ("P_E", "P_E"): cls["EE"],
          ("P_B", "P_B"): cls["BB"], ("T_0", "P_E"): cls["TE"]}

w = sm.QMLWorkspace([fT, fP], bins, cl_fid, lmax=lmax, backend="ducc",
                    fisher_mode="mc", n_sims_fisher=args.nsims,
                    n_sims_noise=args.nsims // 8, batch_size=args.batch,
                    cg_tol=args.cg_tol, seed=2026)

fisher_ckpt = os.path.join(args.workdir, "fisher.npz")
if args.stage in ("fisher", "all"):
    if os.path.exists(fisher_ckpt):
        print("fisher checkpoint exists, skipping")
    else:
        t0 = time.time()
        w.run_mc()
        np.savez(fisher_ckpt, R_hat=w.R_hat, F_l_hat=w.F_l_hat,
                 n_hat=w.n_hat, hartlap=w.hartlap)
        print(f"fisher stage: {(time.time()-t0)/3600:.2f} h")

if args.stage in ("data", "all"):
    d = np.load(fisher_ckpt)
    w.R_hat, w.F_l_hat, w.n_hat = d["R_hat"], d["F_l_hat"], d["n_hat"]
    w.hartlap = float(d["hartlap"])
    w.R_inv = w.hartlap * np.linalg.inv(w.R_hat)
    w._mc_done = True
    rng = np.random.default_rng(7)
    est = []
    for i in range(args.nreal):
        T, Q, U = hp.synfast([cls["TT"], cls["EE"], cls["BB"], cls["TE"]],
                             nside, lmax=lmax, pol=True, new=True)
        T = mask * T + rng.normal(0, args.sigma_T, npix)
        Q = mask * Q + rng.normal(0, args.sigma_T * np.sqrt(2), npix)
        U = mask * U + rng.normal(0, args.sigma_T * np.sqrt(2), npix)
        res = w.estimate(w.pack_data([T[None], np.array([Q, U])[None]]))
        est.append(res.vector()[0])
        np.savez(os.path.join(args.workdir, "estimates.npz"),
                 est=np.array(est), cov=res.cov,
                 ells=res.ells, spec_names=res.spec_names)
        print(f"realization {i+1}/{args.nreal}")
print("done")
