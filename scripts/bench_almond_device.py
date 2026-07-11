#!/usr/bin/env python3
"""Benchmark legacy Almond callbacks against SiMaster 0.2 device PCG.

Run on a GPU node with the NG checkouts first on ``PYTHONPATH``. The script
prints one JSON record suitable for inclusion in the release notes.
"""

from __future__ import annotations

import argparse
import json
import time

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np

import almond
import simaster as sm
from almond.interop import as_cupy
from simaster.almond_device import AlmondDeviceCov, solve_almond_device
from simaster.cg import pcg
from simaster.covariance import CovModel
from simaster.utils import RealAlmIndex


def timed(fn, nrep):
    vals = []
    result = None
    for _ in range(nrep):
        cp.cuda.get_current_stream().synchronize()
        t0 = time.perf_counter()
        result = fn()
        target = result[0] if isinstance(result, tuple) else result
        if isinstance(target, cp.ndarray):
            cp.cuda.get_current_stream().synchronize()
        else:
            jax.block_until_ready(target)
        cp.cuda.get_current_stream().synchronize()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals)), result


def build_problem(nside, lmax, batch):
    npix = 12 * nside ** 2
    rng = np.random.default_rng(20260711)
    # A deterministic 70%-sky cut with mildly anisotropic noise exercises the
    # same gather/scatter and preconditioner paths as production.
    mask = (rng.random(npix) < 0.7).astype(np.float64)
    ivar = mask * (2.0e3 * (1.0 + 0.3 * rng.random(npix)))
    z = np.zeros(npix)
    f0 = sm.Field(mask, [z], ivar=ivar, name="t")
    f2 = sm.Field(mask, [z, z], spin=2, ivar=1.5 * ivar, name="p")
    ell = np.arange(lmax + 1)
    tt = np.zeros_like(ell, dtype=float)
    ee = np.zeros_like(ell, dtype=float)
    bb = np.zeros_like(ell, dtype=float)
    # Keep the inverse-covariance problem in the 5--20 iteration regime used
    # by the documented large-nside production filter. This benchmark targets
    # callback overhead, not the separate extreme-S/N conditioning problem.
    tt[2:] = 1e-5 / (ell[2:] + 1) ** 1.5
    ee[2:] = 0.4 * tt[2:]
    bb[2:] = 0.1 * tt[2:]
    names = f0.comp_names + f2.comp_names
    cl = sm.cl_matrix({("t_0", "t_0"): tt,
                       ("p_E", "p_E"): ee,
                       ("p_B", "p_B"): bb}, names, lmax)
    cov = CovModel([f0, f2], cl, RealAlmIndex(2, lmax), backend="almond")
    rhs = jnp.asarray(rng.standard_normal((cov.nrow, batch)))
    return cov, rhs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--lmax", type=int)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    lmax = args.lmax if args.lmax is not None else 3 * args.nside - 1
    cov, rhs = build_problem(args.nside, lmax, args.batch)
    dc = AlmondDeviceCov(cov)

    # Warm every lazy Almond buffer, XLA callback shape and cuFFT plan.
    legacy_apply = cov.apply_C(rhs)
    jax.block_until_ready(legacy_apply)
    direct_apply = dc.apply_C(as_cupy(rhs))
    cp.cuda.get_current_stream().synchronize()

    t_callback_apply, legacy_apply = timed(lambda: cov.apply_C(rhs), args.reps)
    t_device_apply, direct_apply = timed(
        lambda: dc.apply_C(as_cupy(rhs)), args.reps)
    direct_apply_h = cp.asnumpy(direct_apply)
    legacy_apply_h = np.asarray(legacy_apply)

    # Legacy reference: original JAX PCG around pure_callback. New path:
    # CuPy PCG with DLPack only at solve boundaries.
    legacy_solve = lambda: pcg(cov.apply_C, cov.apply_precond, rhs,
                               tol=args.tol, maxiter=args.maxiter)
    direct_solve = lambda: solve_almond_device(
        cov, rhs, tol=args.tol, maxiter=args.maxiter)
    # Warm both solver implementations and allow the direct path to apply the
    # same preconditioner auto-repair used in production before timing either.
    direct_solve()[0].block_until_ready()
    legacy_solve()[0].block_until_ready()
    t_callback_cg, old = timed(legacy_solve, args.reps)
    t_device_cg, new = timed(
        direct_solve, args.reps)
    x_old, old_info = old
    x_new, new_info = new
    old_it, old_rel = int(old_info[0]), float(old_info[1])
    new_it, new_rel = int(new_info[0]), float(new_info[1])
    x_old_h, x_new_h = np.asarray(x_old), np.asarray(x_new)

    record = {
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "jax": jax.__version__,
        "almond": almond.__version__,
        "simaster": sm.__version__,
        "nside": args.nside,
        "lmax": lmax,
        "batch": args.batch,
        "tol": args.tol,
        "apply_callback_s": t_callback_apply,
        "apply_device_s": t_device_apply,
        "apply_speedup": t_callback_apply / t_device_apply,
        "apply_relative_difference": float(
            np.linalg.norm(direct_apply_h - legacy_apply_h)
            / np.linalg.norm(legacy_apply_h)),
        "cg_callback_s": t_callback_cg,
        "cg_device_s": t_device_cg,
        "cg_speedup": t_callback_cg / t_device_cg,
        "cg_callback_iters": old_it,
        "cg_device_iters": new_it,
        "cg_callback_rel": old_rel,
        "cg_device_rel": new_rel,
        "solution_relative_difference": float(
            np.linalg.norm(x_new_h - x_old_h) / np.linalg.norm(x_old_h)),
    }
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
