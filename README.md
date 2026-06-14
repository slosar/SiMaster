# SiMaster

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
  than Monte Carlo at fixed accuracy), and `mc` (scales to nside 1024)
- single estimation around a fiducial spectrum, or Newton–Raphson-style
  iteration (`QMLWorkspace.iterate`); around-fiducial sim-debiased estimator
- Bond–Jaffe–Knox radical compression to `{c_hat, x, F}` with an
  offset-lognormal likelihood (`simaster.compress`), and field-level
  likelihood scores / autodiff (`simaster.score`)
- two exact SHT backends: `dense` (precomputed real-SH synthesis matrices;
  everything is GPU GEMM — best for nside <= 64) and `ducc` (matrix-free,
  scales to nside 1024+)
- pure Python; double precision end to end

## Install

```bash
conda create -n simaster -c conda-forge python=3.11 healpy ducc0 numpy scipy
conda activate simaster
pip install -e ".[gpu]"        # or plain `pip install -e .` for CPU-only JAX
```

## Documentation

- `docs/method.md` — the estimator, filters, MC response matrix, caveats
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
