"""Portable per-rank worker for the auto MC-Fisher harness (:mod:`simaster.fisher_auto`).

One rank performs one unit of work -- a pilot-``k`` timing, or one MC seed -- and
writes a file to a shared ``outdir``.  Any :class:`~simaster.fisher_auto.Scheduler`
just runs ``python -m simaster.fisher_worker ... --rank R`` for R in 0..n_tasks-1;
nothing here is HPC-specific.

Problem contract -- an importable module exposing::

    build_workspace(nside, *, fisher_mode, deflation, cg_tol,
                    seed=1234, n_sims_fisher=512, n_sims_noise=256) -> QMLWorkspace

so each rank rebuilds the *same* problem independently.
"""
import argparse, importlib, json, os, time
import numpy as np


def task_pilot(prob, a):
    """Time deflation harvest + one 96-batch filter solve for this rank's k."""
    import jax.numpy as jnp
    grid = [int(x) for x in a.k_grid.split(",")]
    k = grid[a.rank % len(grid)]
    ws = prob.build_workspace(a.nside, fisher_mode="subsampled", deflation=k,
                              cg_tol=a.cg_tol)
    t0 = time.time(); ws._ensure_deflation(); harv = time.time() - t0
    ws._V = None                                   # skip deproj for raw filter timing
    rhs = jnp.asarray(np.random.default_rng(0).normal(size=(ws.cov.nrow, 96)))
    t0 = time.time(); z = ws._filter(rhs); z.block_until_ready(); dt = time.time() - t0
    per = dt / 96.0
    r = dict(k=int(k), harvest_s=harv, per_solve_s=per, cg_iters=int(ws.last_cg[0]),
             total_h=(harv + a.nsims * per) / 3600.0)
    json.dump(r, open(os.path.join(a.outdir, f"pilot_k{k}.json"), "w"), indent=2)
    print(f"[worker pilot] k={k} iters={r['cg_iters']} harvest={harv:.0f}s "
          f"per_solve={per:.2f}s total({a.nsims})={r['total_h']:.2f}h", flush=True)


def task_mc(prob, a):
    """One MC Fisher seed (rank = seed): run_mc, save R_hat + reference mean."""
    ws = prob.build_workspace(a.nside, fisher_mode="mc", deflation=a.k, cg_tol=a.cg_tol,
                              seed=1000 + a.rank, n_sims_fisher=a.nsims,
                              n_sims_noise=max(256, a.nsims // 2))
    ws.run_mc(n_sims_fisher=a.nsims)
    np.savez(os.path.join(a.outdir, f"mc_s{a.rank}.npz"),
             R_hat=np.asarray(ws.R_hat), ybar_fid=np.asarray(ws.ybar_fid),
             n_hat=np.asarray(ws.n_hat), F_l_hat=np.asarray(ws.F_l_hat),
             nsims=a.nsims, seed=a.rank, hartlap=float(getattr(ws, "hartlap", 1.0)))
    print(f"[worker mc] seed={a.rank} nsims={a.nsims} k={a.k} "
          f"hartlap={getattr(ws, 'hartlap', 1.0):.3f}", flush=True)


TASKS = {"pilot": task_pilot, "mc": task_mc}


def main(argv=None):
    P = argparse.ArgumentParser()
    P.add_argument("--task", required=True, choices=list(TASKS))
    P.add_argument("--problem", required=True, help="importable module with build_workspace()")
    P.add_argument("--nside", type=int, required=True)
    P.add_argument("--rank", type=int, default=0)
    P.add_argument("--k", type=int, default=0, help="deflation k (mc task)")
    P.add_argument("--k-grid", default="200,400,800,1600", help="pilot task k grid")
    P.add_argument("--nsims", type=int, default=512)
    P.add_argument("--cg-tol", type=float, default=1e-2)
    P.add_argument("--outdir", required=True)
    a = P.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    prob = importlib.import_module(a.problem)
    TASKS[a.task](prob, a)


if __name__ == "__main__":
    main()
