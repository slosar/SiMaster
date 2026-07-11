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

## `simaster.PixelNoiseCov` / `simaster.iqu_from_cov` — correlated pixel noise

Per-pixel **block-diagonal** noise covariance `N`. The default estimator noise
is a scalar `1/ivar` per row; `PixelNoiseCov` generalizes this to small dense
per-pixel blocks — e.g. a 3×3 I/Q/U block that couples a spin-0 (I) field and a
spin-2 (Q, U) field, as delivered by the Planck NPIPE `wcov` products
(II, IQ, IU, QQ, QU, UU). `N` stays block-diagonal *in sky pixel* (matrix-free);
pixel–pixel correlations are not representable. It enters the estimator only
through `apply` (`N x`), `apply_inv` (`N⁻¹ x`), `sqrt_apply` (`N^½ x`),
`quad_cdj` (`Vᵀ N V` noise-bias windows) and `dense` (tiny-nside reference);
`ivar_eff = diag(N⁻¹)` feeds the preconditioner bound.

- `PixelNoiseCov(fields, noisevar=None, groups=None)` — low-level constructor;
  `groups` are disjoint per-pixel coupling blocks.
- `PixelNoiseCov.iqu(fields, *, II, IQ, IU, QQ, QU, UU)` — build the 3×3 I/Q/U
  coupling from exactly one spin-0 and one spin-2 field sharing `obs_pix`.
- `iqu_from_cov(mask, maps, cov_maps, *, beam=None, names=("T","P"), nest=False)`
  → `(f0, f2, noise)` — one-call helper that builds the I and Q/U `Field`s *and*
  their coupled `PixelNoiseCov`, guaranteeing a shared pixel set and an
  ivar-consistent block diagonal. `cov_maps` is `[II, IQ, IU, QQ, QU, UU]` in
  K_CMB²; pass `nest=True` for NESTED inputs (NPIPE maps/wcov are NESTED).
  Feed the returned `noise` as `QMLWorkspace(..., noise_cov=noise)`.

## `simaster.QMLWorkspace(fields, bins, cl_fid, **opts)`

Precomputes everything tied to (fields, fiducial, bins). Important options:

| option | default | meaning |
|---|---|---|
| `lmax` | `3*nside-1` | covariance bandlimit (aliasing caveat in method.md) |
| `lmin` | 2 | lowest multipole in the band basis |
| `backend` | auto | `'dense'` (GPU GEMM, nside≲64), `'ducc'` (matrix-free CPU), `'s2fft'` (native-JAX on-device; opt-in, needs fixed s2fft — see method.md), or `'almond'` (CUDA/CuPy; device-resident covariance, preconditioner, and PCG with DLPack at solve boundaries) |
| `noise_cov` | None | `PixelNoiseCov` for per-pixel *block* noise (e.g. correlated I/Q/U); default is diagonal `1/ivar` per field. Its field list/order and `nrow` must match the workspace |
| `fisher_mode` | auto | `'exact'`, `'subsampled'`, or `'mc'` (see method.md) |
| `fisher_frac` | 0.25 | fraction of mode columns solved in `'subsampled'` mode |
| `fisher_control_variate` | None | experimental: `'pseudo_cl'` uses a deterministic pseudo-Cl/MASTER-style local-diagonal `Cinv` response as a control variate for the exact/subsampled Fisher engine |
| `n_sims_fisher`, `n_sims_noise` | 2048/512 | `'mc'`-mode sample sizes |
| `template_alpha` | None | None = exact Woodbury deprojection; finite = add `alpha*tr(C)/||t||^2 t t^T` |
| `deproject_low_ell` | True | marginalize monopole+dipole of spin-0 fields |
| `cg_tol`, `cg_maxiter` | 1e-5, 700 | inverse-covariance solver |
| `solver` | `'cg'` | `'cholesky'`: assemble dense C once + LAPACK factorization; conditioning-independent exact filter, O(nrow²) memory (CPU-node tool, nside ≤ 64; see method.md) |
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
  `run_mc(n_sims_fisher=None, n_sims_noise=None, third_moment=True)` — force the
  response computation (an engine is run automatically on first `estimate`).
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
- **Bandpower third moment (offset-lognormal / x-factor diagnostic).**
  `run_mc(third_moment=True)` (default) also accumulates, from the *same* Fisher
  sims (~free), the tensor `c3_hat[A,B] = ⟨δc_A² δc_B⟩` — its diagonal is the
  per-band third central moment (`skew_hat`, per-band skewness), its off-diagonal
  the `xxy` cross term. `param_third_moment(w)` / `param_skewness(w)` propagate it
  into a *parameter* direction `θ = Σ_A w_A c_A` (using `xxx`+`xxy`, dropping the
  sub-dominant fully-off-diagonal `xyz`): a coherent amplitude can be strongly
  non-Gaussian even when every individual band looks Gaussian, which is when the
  offset-lognormal (`compress`, analytic `x = R⁻¹n`) correction matters.
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

## `simaster.mc_fisher` — MC Fisher with uncertainty

`MCFisherStore(R, ybar, nsims, n=None, F_l=None)` holds per-seed MC Fisher
estimates (sample-covariance `R̂`, reference mean `ȳ`, `nsims`); build from saved
files with `MCFisherStore.from_files(paths_or_glob)` and combine nodes with
`.merge(other)`. Methods: `fisher()` (combined `F̂`), `hartlap()`,
`fisher_sigma_wishart()` / `fisher_sigma_seeds()` (element-wise σ(F̂); analytic +
model-free), `bandpower_cov(calibrated=True)` (`F̂⁻¹`, the pull→1 error bar — the
frozen-`F̂` inflation is `1/√h`), `suboptimality(F_exact)`, `dRnorm(F_exact)`,
and `held_out_chi2(y_holdout, calibrated=True)` (circularity-free χ²/pull on sims
not in `F̂`). `compute_mc_error(store, F_exact=None)` returns/prints the summary.

**Banded Fisher.** The mask couples a bandpower only to its ℓ-neighbours, so the
true `F` is banded in ℓ (worst-direction error from keeping ±N ℓ-bands halves per
band; error bars are unaffected — see `SiMasterTest/band_fisher.py`). This lets
`F` be estimated from far fewer than `n_b` sims (the rest of `F̂` is MC noise).
`banded_index_map(n_spectra, n_bands)` gives the ℓ-band ordinal of each bandpower
(for a workspace: `banded_index_map(len(w.spec_pairs), w.bins.nbands)`);
`band_fisher(F, band_of_index, N)` zeros couplings beyond `N` ℓ-bands (keeping the
full cross-spectrum block at each offset). `store.banded(band_of_index, N)` (or
`BandedFisher(F, band_of_index, N)`) returns the banded estimate with `.fisher`,
`.cov` (`=F_band⁻¹`), `.errbar`, and a **self-consistency uncertainty**
`.uncertainty()` / `.summary()`: it also forms `F_band(N+1)` and reports how much
admitting the next ℓ-band moves the Fisher (`‖F_N⁻¹ΔF‖₂`) and the error bars
(median/max) — converged when that change is below tolerance.

## `simaster.fisher_auto` / `simaster.run_auto` — semi-automatic harness

`run_auto(scheduler, problem, nside, outdir, *, nsims=512, n_seeds=6, k='auto',
k_grid=(200,400,800,1600), n_holdout=100, cg_tol=1e-2, F_exact_path=None)` runs
pilot → optimal-`k` → parallel MC → combine → held-out χ² and returns
`(MCFisherStore, report)`. `Scheduler` is the pluggable ABC (`map(n_tasks,
worker_argv)`); `LocalScheduler` runs ranks in-process (portable). A *problem* is
an importable module exposing `build_workspace(nside, *, fisher_mode, deflation,
cg_tol, seed, n_sims_fisher, n_sims_noise) -> QMLWorkspace`. The per-rank unit is
`python -m simaster.fisher_worker`.

## `simaster.nersc` — HPC adapter (SLURM/NERSC)

`SlurmScheduler(cpus_per_task=256, omp_threads=128)` implements `Scheduler.map`
by fanning ranks out with `srun` inside the current allocation, so the whole
workflow is one multi-node SLURM job. `python -m simaster.nersc.run_auto` is the
in-allocation driver and `simaster/nersc/run_auto.sh` the reference sbatch.

## `simaster.compute_full_master(f1, f2, bins, cl_guess=..., **opts)`

NaMaster-style one-call interface; returns spectra in NaMaster row ordering.

## `simaster.compress(workspace, result=None, data=None, transform="lognormal")` → `CompressedLikelihood`

Bond–Jaffe–Knox radical compression: reduces an estimate to `{c_hat, x, F}`
with x-factors `x = R⁻¹n` and an offset-lognormal likelihood
(`Z_b = ln(c_b + x_b)` for autos; crosses stay Gaussian). The result has
`loglike(c_theory)`, `loglike_gaussian(c_theory)`, `save(path)` and
`CompressedLikelihood.load(path)`. See method.md and the report.

`transform='hl'` uses the Hamimeche & Lewis (2008) exact variance-stabilizing
transform `g(x)=sign(x-1)√(2(x−ln x−1))` (exposed as `simaster.g_vst`) in place
of `ln`. `g` is the exact Gaussianizer of the per-mode Wishart/χ² likelihood
(vs `ln`'s approximation) and reduces the residual per-band skewness more.

**Calibrated (full HL) form.** Its aggressive tail means a *calibrated*
likelihood needs HL's transformed-space covariance `M_f=cov(X)` (from fiducial
sims), not the raw bandpower Fisher. `run_mc(store_bandpowers=True)` keeps the
per-sim bandpowers (`mc_bandpowers`); `M_f, x̄ = build_Mf(mc_bandpowers[:,keep],
c_fid, x, is_auto, transform)`; then `CompressedLikelihood(..., cov_X=M_f,
xbar=x̄, c_fid=…)` evaluates `−2lnL=(X−x̄)ᵀM_f⁻¹(X−x̄)`. `transform_residual` is
the public building block. Passing `spec_pairs=` (the component-index pair of
each spectrum) selects the **full HL matrix transform** — the per-band n×n
field-covariance eigen-transform that handles T/E/B cross-spectra jointly
(reduces exactly to the scalar form for one field). See
`experiments/hl_lognormal_likelihood/`.

## `simaster.score` (advanced)

Field-level likelihood tools: `score(workspace, data, n_probes=128)` returns
the exact-likelihood score `y − ½Tr[C⁻¹P_A]` (Hutchinson trace), and
`quad_loglike(workspace, cb, data)` is the `−½dᵀC⁻¹d` term made
differentiable through the CG solve (`jax.grad` reproduces the QML
statistic). See method.md.
