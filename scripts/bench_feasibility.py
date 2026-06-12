#!/usr/bin/env python
"""Benchmark SiMaster cost drivers and project nside=1024 requirements.

Measures, per backend:
  * one covariance application C x (batched) -- the CG iteration unit
  * CG solve iteration counts at the validation settings
  * dense-backend G memory and build time vs nside
and prints a feasibility table for nside = 1024 (memory and wall-clock
estimates for the 'ducc' matrix-free backend, plus the GPU RAM needed by
the batched vectors and, hypothetically, by a dense backend).

Run on this machine:    python scripts/bench_feasibility.py
Run on a big-GPU node:  python scripts/bench_feasibility.py --nside 256 --full
"""

import argparse
import json
import os
import time

import numpy as np
import healpy as hp

from val_common import load_mask, DATADIR, CACHEDIR
import jax
import simaster as sm
from simaster.utils import RealAlmIndex
from simaster.covariance import CovModel
from simaster.cg import solve_C

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=32)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--lmax", type=int, default=None)
P.add_argument("--full", action="store_true",
               help="also run a full small estimation for end-to-end timing")
args = P.parse_args()

results = {"device": str(jax.devices()[0]),
           "nside": args.nside}


def timeit(fn, n=3):
    fn()  # warmup / compile
    t0 = time.time()
    for _ in range(n):
        out = fn()
        jax.block_until_ready(out) if hasattr(out, "block_until_ready") else None
    return (time.time() - t0) / n


for backend in ("dense", "ducc"):
    nside = args.nside
    lmax = args.lmax or 3 * nside - 1
    npix = hp.nside2npix(nside)
    mask = load_mask(nside)
    ivar = np.full(npix, 1.0 / 900.0)
    l = np.arange(lmax + 1).astype(float)
    cl = {"TT": np.where(l >= 2, 1000.0 / np.maximum(l, 2) ** 2, 0),
          "EE": np.where(l >= 2, 100.0 / np.maximum(l, 2) ** 2, 0),
          "BB": np.where(l >= 2, 10.0 / np.maximum(l, 2) ** 2, 0)}
    fT = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="T")
    fP = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=ivar / 2, name="P")
    idx = RealAlmIndex(2, lmax)
    clmat = sm.cl_matrix({("T_0", "T_0"): cl["TT"], ("P_E", "P_E"): cl["EE"],
                          ("P_B", "P_B"): cl["BB"]},
                         ["T_0", "P_E", "P_B"], lmax)
    t0 = time.time()
    cov = CovModel([fT, fP], clmat, idx, backend=backend,
                   cachedir=CACHEDIR if backend == "dense" else None)
    t_build = time.time() - t0
    x = jax.numpy.asarray(np.random.default_rng(0).normal(
        size=(cov.nrow, args.batch)))
    t_apply = timeit(lambda: cov.apply_C(x))
    t0 = time.time()
    _, (iters, rel) = solve_C(cov, x, tol=1e-5, maxiter=500)
    t_solve = time.time() - t0
    mem_G = 0
    if backend == "dense":
        mem_G = sum(y.size * 8 for y in cov.Y) / 1e9
    results[backend] = dict(
        build_s=t_build, apply_s=t_apply, solve_s=t_solve, cg_iters=int(iters),
        nrow=cov.nrow, nmodes=cov.ncomp * idx.nmodes, G_GB=mem_G,
        batch=args.batch)
    print(f"[{backend}] nside={nside}: build {t_build:.1f}s, "
          f"C-apply({args.batch}) {t_apply*1e3:.0f} ms, "
          f"solve {t_solve:.1f}s in {int(iters)} iters, G={mem_G:.2f} GB")

# ---------------------------------------------------------- 1024 projection --
nside_t = 1024
lmax_t = 2 * nside_t           # practical choice; 3*nside-1 noted in report
npix_t = 12 * nside_t ** 2
fsky = float(load_mask(32).mean())
nobs = fsky * npix_t
nrow = 3 * nobs
nmodes = 3 * ((lmax_t + 1) ** 2 - 4)
b = results["ducc"]
# ducc SHT cost scales ~ npix * lmax ~ nside^3; measure scaling directly:
scale = (nside_t / args.nside) ** 3
t_apply_1024 = b["apply_s"] * scale
batch = b["batch"]
proj = dict(
    nobs=nobs, nrow=nrow, nmodes=nmodes,
    vec_GB_per_batch=nrow * batch * 8 / 1e9,
    cg_state_GB=5 * nrow * batch * 8 / 1e9,   # X, R, Z, P, AP
    dense_G_GB=(nobs + 2 * nobs) * nmodes / 3 * 8 / 1e9 * 3,  # hypothetical
    t_apply_s_extrapolated=t_apply_1024,
    t_solve_s=t_apply_1024 * b["cg_iters"],
)
results["projection_1024"] = proj
print(json.dumps(results, indent=2))
with open(os.path.join(DATADIR, "bench.json"), "w") as f:
    json.dump(results, f, indent=2)
