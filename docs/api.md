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
| `backend` | auto | `'dense'` (GPU GEMM, nside≲64) or `'ducc'` (matrix-free) |
| `fisher_mode` | auto | `'exact'`, `'subsampled'`, or `'mc'` (see method.md) |
| `fisher_frac` | 0.25 | fraction of mode columns solved in `'subsampled'` mode |
| `n_sims_fisher`, `n_sims_noise` | 2048/512 | `'mc'`-mode sample sizes |
| `template_alpha` | None | None = exact Woodbury deprojection; finite = add `alpha*tr(C)/||t||^2 t t^T` |
| `deproject_low_ell` | True | marginalize monopole+dipole of spin-0 fields |
| `cg_tol`, `cg_maxiter` | 1e-5, 700 | inverse-covariance solver |
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
- `run_exact(sample_frac=None, sample_seed=0)` / `run_mc(n_sims_fisher=None,
  n_sims_noise=None)` — force the response computation (an engine is run
  automatically on first `estimate`). `sample_frac=f` gives the subsampled
  engine directly.
- `run_mean_debias(n_sims=128)` — compute the fiducial-sim mean for the
  around-fiducial (sim-debiased) estimator `ĉ = c_fid + R⁻¹(y − ⟨y⟩)`,
  needed to deflate stochastic-`R̂` error for the subsampled/MC engines
  (method.md).
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
