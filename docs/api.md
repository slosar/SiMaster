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
| `backend` | auto | `'dense'` (GPU GEMM) or `'ducc'` (matrix-free) |
| `fisher_mode` | auto | `'exact'` or `'mc'` (see method.md) |
| `n_sims_fisher`, `n_sims_noise` | 2048/512 | MC-mode sample sizes |
| `template_alpha` | None | None = exact Woodbury deprojection; finite = add `alpha*tr(C)/||t||^2 t t^T` |
| `deproject_low_ell` | True | marginalize monopole+dipole of spin-0 fields |
| `cg_tol`, `cg_maxiter` | 1e-5, 700 | inverse-covariance solver |
| `batch_size` | 256 | RHS per GPU batch |
| `seed` | 1234 | MC reproducibility |

Methods:

- `estimate(data=None)` → `BandpowerResult`. `data`: None (fields' maps), a
  list of per-field full-sky map arrays (batched: shape
  `(nreal, ncomp, npix)`), or a packed `(nrow, B)` matrix from `pack_data`.
- `run_exact()` / `run_mc()` — force the response computation.
- `predict(cl_theory)` — window-convolved expectation of the estimates.
- `window_functions()` — `W[(s,b),(s',l)]` with `<ĉ> = W cl`.
- `update_fiducial(c_full)`, `iterate(data, n_iter)`.

## `simaster.BandpowerResult`

`ells`, `cl` (dict spectrum name → `(nreal, nbands)`), `cov`, `spec_names`,
`vector()`, `chi2(theory)`.

## `simaster.compute_full_master(f1, f2, bins, cl_guess=..., **opts)`

NaMaster-style one-call interface; returns spectra in NaMaster row ordering.
