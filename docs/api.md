# API reference (essentials)

## `simaster.Field(mask, maps, spin=None, ivar=None, templates=None, beam=None, name=None)`

A spin-0 (`maps=[m]`) or spin-2 (`maps=[Q, U]`) field. `mask`, `maps`,
`ivar`, `templates` accept healpy RING arrays or
`healsparse.HealSparseMap`. Pixels with `mask==0` or `ivar==0` are dropped.
`maps` may be `None` to define geometry only (pass data later through
`QMLWorkspace.estimate`). `name` prefixes component labels
(`{name}_0` or `{name}_E`, `{name}_B`).

## `simaster.Bins`

- `Bins.from_edges(lo, hi)` — inclusive band edges.
- `Bins.linear(lmin, lmax, nlb)`, `Bins.from_nside_linear(nside, nlb)`.
- `get_effective_ells()`, `bin_cl(cl)`, `unbin_cl(cb, lmax)`.

## `simaster.QMLWorkspace(fields, bins, cl_fid, **opts)`

Precomputes everything tied to (fields, fiducial, bins). Important options:

| option | default | meaning |
|---|---|---|
| `lmax` | `3*nside-1` | covariance bandlimit (aliasing caveat in method.md) |
| `lmin` | 2 | lowest multipole in the band basis |
| `backend` | auto | `'dense'` (GPU GEMM, nside≲64), `'ducc'` (matrix-free CPU), or `'s2fft'` (native-JAX on-device; opt-in, needs fixed s2fft — see method.md) |
| `fisher_mode` | auto | `'exact'`, `'subsampled'`, or `'mc'` (see method.md) |
| `fisher_frac` | 0.25 | fraction of mode columns solved in `'subsampled'` mode |
| `fisher_control_variate` | None | experimental: `'pseudo_cl'` uses a deterministic pseudo-Cl/MASTER-style local-diagonal `Cinv` response as a control variate for the exact/subsampled Fisher engine |
| `n_sims_fisher`, `n_sims_noise` | 2048/512 | `'mc'`-mode sample sizes |
| `template_alpha` | None | None = exact Woodbury deprojection; finite = add `alpha*tr(C)/||t||^2 t t^T` |
| `deproject_low_ell` | True | marginalize monopole+dipole of spin-0 fields |
| `cg_tol`, `cg_maxiter` | 1e-5, 700 | inverse-covariance solver |
| `deflation` | 0 | `k>0`: deflated/recycled CG — project the `k` slowest eigen-directions of `P⁻¹C` (harvested once) out of every solve; same result, ~1.5–2× fewer iters (method.md) |
| `deflation_steps`, `deflation_probes` | None, 1 | harvest Lanczos steps (default `~max(2k+10,40)`) and number of random probes |
| `batch_size` | 256 | RHS per GPU batch |
| `cachedir` | None | disk cache for dense synthesis matrices |
| `seed` | 1234 | MC reproducibility |
| `verbose` | True | progress logging |

Methods:

- `estimate(data=None, deviations=False)` → `BandpowerResult`. `data`: None
  (fields' maps), a list of per-field full-sky map arrays (batched: shape
  `(nreal, ncomp, npix)`), or a packed `(nrow, B)` matrix from `pack_data`.
  With `deviations=True` the smooth fiducial is kept in the covariance and
  only flat band *deviations* are fitted (see method.md); `result.deviation`
  flags this and `result.cl` then holds deviations.
- `run_exact(sample_frac=None, sample_seed=0, keep_samples=False)` /
  `run_mc(n_sims_fisher=None, n_sims_noise=None)` — force the response
  computation (an engine is run automatically on first `estimate`).
  `sample_frac=f` gives the subsampled engine directly. `keep_samples=True`
  retains the per-mode contributions (in `_subsample_store`) for the
  subsampling error budget and fills the otherwise-zero `n_hat_err`. With
  `fisher_control_variate='pseudo_cl'`, the stored per-mode contributions are
  residuals around the deterministic pseudo-Cl control, and the same
  `subsample_error()` machinery applies.
- `subsample_error(ref="fiducial", data=None, n_boot=2000, seed=0,
  include_noise_bias=True)` → `SubsampleError` — error budget for the
  column-subsampling inaccuracy of the Fisher matrix `R` and noise bias `n`,
  from a prior `run_exact(sample_frac=f, keep_samples=True)`. Returns the
  analytic (stratified, delta-method) and column-`bootstrap` covariances of
  the bandpowers (to add to `R⁻¹`), the per-band suboptimality
  `√(diag Cov_sub / diag R⁻¹)`, and the noise-bias covariance / error.
  Evaluated at the fiducial (default), supplied `data`, or an arbitrary `ref`
  bandpower vector. The `SubsampleStore` is checkpointable
  (`save`/`load`/`merge`) for multi-node aggregation.
- `build_deflation(k=None, steps=None, n_probes=None, seed=0)` — (re)build
  the deflated/recycled-CG subspace from the current `C` (harvested by
  recycling a short instrumented solve); called automatically before the
  response loop when `deflation>0`, exposed for manual rebuilds (e.g. after
  `update_fiducial`). Same solutions, fewer CG iterations (method.md).
- `run_mean_debias(n_sims=128)` — compute the fiducial-sim mean for the
  around-fiducial (sim-debiased) estimator `ĉ = c_fid + R⁻¹(y − ⟨y⟩)`,
  needed to deflate stochastic-`R̂` error for the subsampled/MC engines
  (method.md).
- `exact_hessian(data=None)` → `LikelihoodExpansion`. Exact gradient and
  Hessian of the Gaussian log-likelihood in the bandpower basis at the
  current fiducial: `∂lnL/∂c = y − ⟨y⟩` and `∂²lnL/∂c² = F − Q` with
  `Q_AB = dᵀ C⁻¹ C_A C⁻¹ C_B C⁻¹ d` (so `E[∂²lnL] = −F`). Costs `1 + nparam`
  CG solves; use as a pure 2nd-order likelihood expansion about a good
  fiducial (see method.md). Use `fisher_mode='exact'` for an exact `F`.
- `predict(cl_theory)` — window-convolved expectation of the estimates.
- `window_functions()` — `W[(s,b),(s',l)]` with `<ĉ> = W cl`.
- `iterate(data=None, n_iter=2, deviations=False)` — Newton–Raphson
  re-centering of the fiducial (adds flat deviations to the smooth fiducial
  when `deviations=True`); returns a list of `BandpowerResult`.
- `update_fiducial(c_full)`, `update_fiducial_deviations(dc_full)`,
  `fiducial_bandpowers()`, `pack_data(maps_per_field)` — lower-level helpers.

## `simaster.BandpowerResult`

`ells`, `cl` (dict spectrum name → `(nreal, nbands)`), `cov`, `spec_names`,
`windows`, `ls`, `deviation`; methods `vector()`, `chi2(theory)`.

## `simaster.LikelihoodExpansion`

Returned by `QMLWorkspace.exact_hessian`. Holds the full-band `c0`, `grad`,
`hess` (= `F − Q`), and `fisher` (= `F`), plus `ells`/`user_ells`/
`is_user_band`. `newton_estimate(user_bands=True, floor=0)` →
`(c_hat, cov)` from the Newton step `c0 + (−hess)⁻¹ grad` with covariance
`(−hess)⁻¹`; `fisher_estimate(user_bands=True)` uses `F` instead (always SPD).
With `user_bands=True` the junk bands are marginalized (full inverse, then
restrict).

## `simaster.DeflationSpace` / `build_deflation` / `harvest_ritz` (advanced)

Deflated/recycled CG primitives (also reachable through
`QMLWorkspace(deflation=k)`). `harvest_ritz(apply_A, apply_M, probe, k, m)`
recycles a short instrumented PCG run into the `k` largest-Ritz vectors of
`P⁻¹C` (the slow directions); `build_deflation(apply_A, apply_M, n, k, ...)`
wraps that into a `DeflationSpace`, which precomputes the coarse operator
`E=WᵀCW` and exposes the projectors used by `simaster.cg.deflated_pcg` /
`solve_C(..., deflation=...)`. The solve is exact for any full-rank basis —
only the iteration count depends on `W` (see method.md).

## `simaster.compute_full_master(f1, f2, bins, cl_guess=..., **opts)`

NaMaster-style one-call interface; returns spectra in NaMaster row ordering.

## `simaster.compress(workspace, result=None, data=None)` → `CompressedLikelihood`

Bond–Jaffe–Knox radical compression: reduces an estimate to `{c_hat, x, F}`
with x-factors `x = R⁻¹n` and an offset-lognormal likelihood
(`Z_b = ln(c_b + x_b)` for autos; crosses stay Gaussian). The result has
`loglike(c_theory)`, `loglike_gaussian(c_theory)`, `save(path)` and
`CompressedLikelihood.load(path)`. See method.md and the report.

## `simaster.score` (advanced)

Field-level likelihood tools: `score(workspace, data, n_probes=128)` returns
the exact-likelihood score `y − ½Tr[C⁻¹P_A]` (Hutchinson trace), and
`quad_loglike(workspace, cb, data)` is the `−½dᵀC⁻¹d` term made
differentiable through the CG solve (`jax.grad` reproduces the QML
statistic). See method.md.
