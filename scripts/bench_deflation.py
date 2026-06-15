#!/usr/bin/env python
"""Benchmark deflated / recycled CG against plain preconditioned CG.

On a masked, anisotropic-noise problem (the regime where the isotropic
preconditioner is loosest and CG is slowest), this compares plain CG with
deflated CG over a batch of right-hand sides, reporting:

  * the iteration count (the SHT-bound cost unit),
  * the wall-time per solve (including the per-iteration projector overhead),
  * the one-off harvest/build cost and how many solves amortize it,
  * the solution agreement (deflation must not change the answer).

Deflation reuses one C across all RHS, so the build cost is paid once and the
saving compounds over the thousands of solves in a Fisher run.  The relative
saving is largest for the matrix-free (ducc / s2fft) backends, where each CG
iteration is an SHT and the dense projector overhead is negligible.

    python scripts/bench_deflation.py                 # nside 16, dense
    python scripts/bench_deflation.py --nside 32 --k 64 --batch 64
"""

import argparse
import time

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp

import simaster as sm
from simaster.utils import RealAlmIndex
from simaster.covariance import CovModel
from simaster.cg import solve_C
from simaster.deflation import build_deflation, dense_eig_deflation

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=16)
P.add_argument("--lmax", type=int, default=None)
P.add_argument("--aniso", type=float, default=4.0, help="ivar = 1+aniso*cos^2")
P.add_argument("--k", type=int, default=48, help="deflation subspace size")
P.add_argument("--steps", type=int, default=None, help="harvest Lanczos steps")
P.add_argument("--batch", type=int, default=64, help="RHS per solve")
P.add_argument("--backend", default="dense")
P.add_argument("--tol", type=float, default=1e-8)
P.add_argument("--optimal", action="store_true",
               help="also time the dense largest-eigenvector (optimal) basis")
args = P.parse_args()

nside = args.nside
lmax = args.lmax or 3 * nside - 1
npix = hp.nside2npix(nside)
mask = np.zeros(npix); mask[: 2 * npix // 3] = 1.0
theta, _ = hp.pix2ang(nside, np.arange(npix))
ivar = 1e4 * (1.0 + args.aniso * np.cos(3 * theta) ** 2)
l = np.arange(lmax + 1); clTT = np.zeros(lmax + 1); clTT[2:] = 1e-2 / l[2:] ** 2
f0 = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="t")
idx = RealAlmIndex(2, lmax)
clmat = sm.cl_matrix({("t_0", "t_0"): clTT}, ["t_0"], lmax)
cov = CovModel([f0], clmat, idx, backend=args.backend)
n = cov.nrow
print(f"device {jax.devices()[0]}  backend {args.backend}  nside {nside} "
      f"lmax {lmax}  nrow {n}  batch {args.batch}")


def timed_solve(B, defl=None):
    X, (it, rel) = solve_C(cov, B, tol=args.tol, maxiter=5000, deflation=defl)
    X.block_until_ready()
    t0 = time.time()
    X, (it, rel) = solve_C(cov, B, tol=args.tol, maxiter=5000, deflation=defl)
    X.block_until_ready()
    return X, it, time.time() - t0


B = jnp.asarray(np.random.default_rng(0).normal(size=(n, args.batch)))

# settle the preconditioner to its SPD bound (as a real run does), then time
x0, it0, t0 = timed_solve(B)
print(f"\nplain CG          : {it0:5d} iters   {t0:7.3f} s")

steps = args.steps or max(2 * args.k + 10, 40)
tb = time.time()
defl = build_deflation(cov.apply_C, cov.apply_precond, n, args.k, steps=steps)
jax.block_until_ready(defl.AW)
tb = time.time() - tb
xd, itd, td = timed_solve(B, defl)
err = float(np.abs(np.asarray(xd - x0)).max() / np.abs(np.asarray(x0)).max())
print(f"deflated (k={defl.k:3d})   : {itd:5d} iters   {td:7.3f} s   "
      f"[harvest {tb:.2f} s, {steps} steps]")

if args.optimal:
    d_opt, ev = dense_eig_deflation(cov.apply_C, cov.apply_precond, n, args.k)
    xo, ito, to = timed_solve(B, d_opt)
    print(f"deflated optimal  : {ito:5d} iters   {to:7.3f} s   "
          f"[top eigenvalues {np.round(ev[:3], 0)}]")

print(f"\niteration speed-up : {it0/itd:.2f}x")
print(f"wall-time speed-up : {t0/td:.2f}x  (per solve, excl. one-off harvest)")
saved = t0 - td
if saved > 0:
    print(f"harvest amortizes after ~{tb/saved:.1f} solves "
          f"(a Fisher run does thousands)")
print(f"solution agreement : max rel. diff {err:.1e}  (deflation is exact)")
