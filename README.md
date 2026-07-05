# SiMaster

[![tests](https://github.com/slosar/SiMaster/actions/workflows/ci.yml/badge.svg)](https://github.com/slosar/SiMaster/actions/workflows/ci.yml)

**S**pherical-harmonic **i**nverse-covariance (QML) power spectrum estimation —
a GPU-accelerated, optimal alternative to pseudo-`C_l` codes, with a
[NaMaster](https://github.com/LSSTDESC/NaMaster)-like interface.

SiMaster implements the quadratic maximum-likelihood (QML / optimal quadratic)
bandpower estimator of Tegmark (1997), generalized to multiple correlated
spin-0 and spin-2 fields on cut skies. Unlike classic QML codes it never
builds or inverts a dense pixel covariance: the inverse-covariance filter is
applied with preconditioned conjugate gradients, the Fisher (response) matrix
and noise bias are evaluated by batched GPU Monte Carlo, and the bandpower
window functions come from the same simulations for free.

```python
import numpy as np
import simaster as sm

# observed = mask * signal + noise,  noise ~ N(0, 1/ivar) per pixel
f0 = sm.Field(mask, [delta_map], ivar=ivar_g)                # spin 0
f2 = sm.Field(mask, [gamma1, gamma2], spin=2, ivar=ivar_s)   # spin 2
b  = sm.Bins.linear(2, 95, nlb=8)

cl_fid = {("f0_0", "f0_0"): cl_gg, ("f1_E", "f1_E"): cl_EE,
          ("f0_0", "f1_E"): cl_gE, ("f1_B", "f1_B"): cl_BB}

w = sm.QMLWorkspace([f0, f2], b, cl_fid)
result = w.estimate()
print(result.ells, result.cl["f0_0 x f1_E"])   # bandpowers
cov = result.cov                                # bandpower covariance
```

## Features

- spin-0 and spin-2 fields, auto- and cross-spectra (including EB/TB)
- HEALPix and [healsparse](https://healsparse.readthedocs.io) inputs
- per-pixel inverse-variance noise (`ivar`); pixels with `mask==0` or
  `ivar==0` are dropped exactly
- template marginalization by adding `alpha * t t^T` to the covariance
  (finite `alpha` or the exact `alpha -> infinity` Woodbury limit; monopole
  and dipole of spin-0 fields are deprojected automatically)
- flat-in-`C_l` bandpowers on arbitrary (non-)uniform `l` bins, with junk
  bands covering the rest of the modelled range to prevent aliasing of
  unbinned power
- a smooth (curved-in-`l`) fiducial can be kept inside the covariance while
  only flat band *deviations* are fitted (`estimate(deviations=True)`)
- three response/Fisher engines: `exact` (deterministic, batched-CG mode
  probing), `subsampled` (unbiased column subsampling — several × cheaper
  than Monte Carlo at fixed accuracy), and `mc` (scales to nside 1024, with an
  uncertainty budget — see below)
- MC Fisher with an uncertainty budget (`simaster.mc_fisher`): the combined
  `F̂` plus element-wise Wishart σ(F̂) (cross-checked by the seed-to-seed
  scatter) and a closed-form `1/√Hartlap` error-bar calibration so bandpower
  pulls → 1, with a held-out-sim χ² bias check. A semi-automatic harness
  (`simaster.run_auto`) does pilot → optimal deflation-`k` → parallel MC →
  combine over a pluggable `Scheduler` — `LocalScheduler` runs anywhere;
  `simaster.nersc.SlurmScheduler` runs the whole workflow as one multi-node
  SLURM job
- measurable error budget for the `subsampled` engine
  (`run_exact(sample_frac=f, keep_samples=True)` then
  `QMLWorkspace.subsample_error()`): analytic stratified covariance + column
  bootstrap of the subsampling error on the Fisher matrix *and* noise bias,
  as a covariance to add to `R⁻¹` plus a per-band suboptimality diagnostic;
  the per-mode store is checkpointable/mergeable for multi-node runs
- single estimation around a fiducial spectrum, or Newton–Raphson-style
  iteration (`QMLWorkspace.iterate`); around-fiducial sim-debiased estimator
- Bond–Jaffe–Knox radical compression to `{c_hat, x, F}` with an
  offset-lognormal likelihood (`simaster.compress`), and field-level
  likelihood scores / autodiff (`simaster.score`)
- exact gradient *and* Hessian of the Gaussian log-likelihood in the
  bandpower basis (`QMLWorkspace.exact_hessian`) — a per-realization
  second-order likelihood expansion about a fiducial, `1 + nparam` CG solves
- deflated / recycled CG (`deflation=k`): the `k` slowest-converging
  eigen-directions of `P⁻¹C` are recycled from a short instrumented solve and
  projected out of every subsequent inverse-covariance solve, cutting CG
  iterations ~1.5–2× with no change to the result — a pure win shared by all
  Fisher engines
- three exact SHT backends: `dense` (precomputed real-SH synthesis matrices;
  everything is GPU GEMM — best for nside <= 64), `ducc` (matrix-free CPU
  transforms, scales to nside 1024+), and `s2fft` (native-JAX matrix-free
  transforms that stay on the accelerator and are differentiable — opt-in,
  needs an s2fft build with exact HEALPix spin-2 synthesis; see
  `docs/method.md`)
- pure Python; double precision end to end

## Install

```bash
conda create -n simaster -c conda-forge python=3.11 healpy ducc0 numpy scipy
conda activate simaster
pip install -e ".[gpu]"        # or plain `pip install -e .` for CPU-only JAX
```

## Documentation

- `docs/method.md` — the estimator, filters, MC response matrix, MC Fisher
  uncertainty + the automatic harness, caveats
- `docs/api.md` — API reference
- `docs/migration.md` — switching from NaMaster
- `notebooks/` — worked examples, including a NaMaster comparison
- `report/` — LaTeX validation report with QA plots

## Validation

`scripts/val1_cmb.py` (CMB T/Q/U, NaMaster test mask, uniform noise),
`scripts/val1_cmb.py --ivar-maker scripts/val2_ivar.py --tag val2`
(anisotropic noise in longitude strips), `scripts/val3_lss.py` (galaxy
density + weak lensing with pyccl theory spectra). Each runs 100
realizations at nside=32 at fixed mask and checks the chi^2 distribution of
the recovered bandpowers against the input. `scripts/bench_feasibility.py`
measures the cost drivers and projects nside=1024 requirements.

## License

SiMaster is free software, licensed under the GNU General Public License
version 3 or (at your option) any later version (GPL-3.0-or-later). See the
[LICENSE](LICENSE) file for the full text.

Copyright (C) 2026 Anze Slosar and contributors.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
