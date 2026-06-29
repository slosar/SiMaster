"""Semi-automatic MC-Fisher harness (portable; no HPC specifics).

Drives the three-step workflow over a pluggable :class:`Scheduler`:

  i)   *pilot*   -- time a deflation-k grid, pick the k minimising MC wall;
  ii)  *mc*      -- run ``n_seeds`` independent MC Fisher seeds at the optimal k,
                    one per rank (the Scheduler fans these across nodes as a
                    single allocation);
  iii) *combine* -- merge the seeds into F-hat + uncertainty
                    (:func:`simaster.mc_fisher.compute_mc_error`) and, with a
                    small batch of *held-out* sims that did not enter F-hat,
                    a circularity-free chi2/pull check.

A site provides a :class:`Scheduler` (e.g. ``simaster.nersc.SlurmScheduler``);
the only requirement is "run ``python -m simaster.fisher_worker <argv> --rank R``
for R in 0..n-1, in parallel, and block until done".  :class:`LocalScheduler`
(in-process, sequential) makes the whole thing run anywhere for testing/small jobs.
"""
from __future__ import annotations

import abc, glob, importlib, json, os
import numpy as np

from .mc_fisher import MCFisherStore, compute_mc_error


class Scheduler(abc.ABC):
    """Run a worker across ``n_tasks`` ranks in parallel and block until done."""
    @abc.abstractmethod
    def map(self, n_tasks: int, worker_argv: list): ...


class LocalScheduler(Scheduler):
    """Run ranks sequentially in-process. Portable; for testing / small problems."""
    def map(self, n_tasks, worker_argv):
        from . import fisher_worker
        for r in range(n_tasks):
            fisher_worker.main(worker_argv + ["--rank", str(r)])


def _argv(task, problem, nside, outdir, cg_tol, nsims, k=0, k_grid=None):
    a = ["--task", task, "--problem", problem, "--nside", str(nside),
         "--outdir", outdir, "--cg-tol", repr(cg_tol), "--nsims", str(nsims),
         "--k", str(k)]
    if k_grid is not None:
        a += ["--k-grid", ",".join(str(int(x)) for x in k_grid)]
    return a


def optimal_k_from_pilot(outdir, nsims):
    """Pick k minimising total MC wall (harvest + nsims*per_solve) from pilot JSONs."""
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(outdir, "pilot_k*.json"))]
    if not rows:
        raise RuntimeError("no pilot results found")
    for r in rows:
        r["total_h"] = (r["harvest_s"] + nsims * r["per_solve_s"]) / 3600.0
    best = min(rows, key=lambda r: r["total_h"])
    return int(best["k"]), sorted(rows, key=lambda r: r["k"])


def _draw_holdout_y(problem, nside, n, k, cg_tol):
    """Band-power vectors from independent sims (disjoint seed) -- not in F-hat."""
    prob = importlib.import_module(problem)
    ws = prob.build_workspace(nside, fisher_mode="mc", deflation=k, cg_tol=cg_tol,
                              seed=900_000)
    ws._ensure_deflation(); ws._prepare_deprojection()
    x = ws.cov.sample(ws._next_key(), n)
    yb, _ = ws._y_stats(ws._filter(x))
    return np.asarray(yb)


def run_auto(scheduler, problem, nside, outdir, *, nsims=512, n_seeds=6, k="auto",
             k_grid=(200, 400, 800, 1600), n_holdout=100, cg_tol=1e-2,
             F_exact_path=None, log=print):
    """Run pilot -> MC seeds -> combine. Returns (MCFisherStore, report dict)."""
    os.makedirs(outdir, exist_ok=True)

    # i) pilot for optimal k
    if k == "auto":
        log(f"[auto] pilot: k grid {tuple(k_grid)} ({len(k_grid)} ranks)")
        scheduler.map(len(k_grid),
                      _argv("pilot", problem, nside, outdir, cg_tol, nsims, k_grid=k_grid))
        k, pilot = optimal_k_from_pilot(outdir, nsims)
        log(f"[auto] optimal k={k}  (pilot: " +
            ", ".join(f"k{r['k']}={r['total_h']:.2f}h" for r in pilot) + ")")

    # ii) MC seeds at optimal k (one per rank)
    log(f"[auto] MC: {n_seeds} seeds x nsims={nsims} at k={k}")
    scheduler.map(n_seeds, _argv("mc", problem, nside, outdir, cg_tol, nsims, k=k))

    # iii) combine + uncertainty
    store = MCFisherStore.from_files(os.path.join(outdir, "mc_s*.npz"))
    F_exact = None
    if F_exact_path and os.path.exists(F_exact_path):
        d = np.load(F_exact_path); F_exact = d["R_hat"]
    log(f"[auto] combine: {store.K} seeds, n_eff={store.n_eff}")
    report = compute_mc_error(store, F_exact=F_exact)
    report["optimal_k"] = int(k)

    # held-out (circularity-free) chi2/pull
    if n_holdout:
        y = _draw_holdout_y(problem, nside, n_holdout, k, cg_tol)
        report["heldout_calibrated"] = store.held_out_chi2(y, calibrated=True)
        report["heldout_hartlap"] = store.held_out_chi2(y, calibrated=False)
        log(f"[auto] held-out chi2 ({n_holdout} indep sims): "
            f"calibrated pull={report['heldout_calibrated']['pull_std']:.3f} "
            f"(F-hat^-1), vs Hartlap-shrunk pull="
            f"{report['heldout_hartlap']['pull_std']:.3f}")

    json.dump(report, open(os.path.join(outdir, "mc_fisher_report.json"), "w"), indent=2)
    log(f"[auto] wrote {os.path.join(outdir, 'mc_fisher_report.json')}")
    return store, report
