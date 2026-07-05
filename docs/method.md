# The SiMaster method

## Data model

Each input field provides three maps (HEALPix RING or healsparse):

- `signal` — the observed map,
- `mask` — multiplicative response `w`,
- `ivar` — per-pixel inverse noise variance.

The model is `observed = w * s + n`, with `n ~ N(0, 1/ivar)` independent per
pixel, so the noise covariance `N` is diagonal in pixel space by
construction. Pixels with `w == 0` *or* `ivar == 0` are removed from the
data vector entirely (infinite noise = zero weight). A spin-0 field
contributes one map, a spin-2 field contributes `[Q, U]` and its spectra are
E and B.

`N` need not be strictly diagonal: passing a `PixelNoiseCov`
(`QMLWorkspace(..., noise_cov=...)`) lets it carry small **per-sky-pixel
blocks** — e.g. the 3×3 I/Q/U covariance that couples a spin-0 (I) field and a
spin-2 (Q, U) field, as in the Planck NPIPE `wcov` products. `N` stays
block-diagonal in pixel space (matrix-free); `apply`/`apply_inv`/`sqrt`/`Vᵀ N V`
all have exact per-block analogues, and the preconditioner uses the true
`diag(N⁻¹)`. Pixel–pixel correlations remain out of scope. Build both fields
and the coupled noise in one call with `simaster.iqu_from_cov` (see api.md).

The concatenated data vector `x` over all field components has covariance

```
C = U Ĉ Uᵀ + N (+ Σ_j α_j t_j t_jᵀ),     U = W Y B,
```

where `Y` is the synthesis matrix of *real* orthonormal spherical harmonics
(real basis: the coefficient covariance of an isotropic field is strictly
block-diagonal, one `(ncomp × ncomp)` spectrum block `C_l` per `(l, m)`
mode, and all algebra stays real), `W` the mask, `B` optional beams, and
`t_j` are templates to marginalize.

## Estimator

SiMaster implements the Tegmark (1997) QML estimator generalized to multiple
correlated fields and bandpowers flat in `C_l`:

```
y_A(x)  = 1/2 xᵀ M P_A M x,        P_A = ∂C/∂c_A
ĉ       = R⁻¹ (y(x) − n),          n_A = 1/2 Tr[M N M P_A]
R_AB    = 1/2 Tr[M P_A M P_B],     cov(ĉ) ≈ R⁻¹ at the fiducial
```

`A = (spectrum pair, band)`. The filter `M` is `C⁻¹` or its
template-deprojected pseudo-inverse (below). The estimator is **unbiased
for any fiducial spectrum** (only optimality degrades), which justifies
single-shot estimation around a fiducial; `QMLWorkspace.iterate` re-centers
the fiducial on the estimate (a Newton–Raphson step on the field-level
likelihood).

Computing `y` is cheap: one filtered vector `z = M x`, one adjoint SHT
`a = BᵀYᵀWᵀz`, and per-spectrum products `Σ_k a_c a_d` accumulated per `l`
and per band.

## No dense matrices: CG with a guaranteed-SPD preconditioner

`M x` is computed by preconditioned conjugate gradients; the operator `C·x`
costs two SHTs (GEMMs on the dense backend) plus diagonal work, batched over
many right-hand sides. The preconditioner is the Woodbury inverse of the
isotropic approximation:

```
P⁻¹ = N⁻¹ − N⁻¹ U T Uᵀ N⁻¹,    T_l = (Ĉ_l⁻¹ + D_l)⁻¹
```

evaluated in a form valid for singular `Ĉ_l` (zero fiducial BB is fine).
`D` starts at the statistical mean `Σ_p w² ivar / 4π` of the mode response
`UᵀN⁻¹U` and, if CG ever detects an indefinite direction (possible on cut
skies because the true mode response is not diagonal), is escalated toward
the guaranteed upper bound `2 max_p(w² ivar) · npix/4π`, beyond which
`P⁻¹ ⪰ C⁻¹ ≻ 0` provably. Uniform-weight problems converge in ~10
iterations; strongly varying `w² ivar` costs more (the bound is then loose) —
this is the main performance caveat for highly anisotropic noise.

## Deflated / recycled CG (`deflation`)

Every Fisher engine applies `M = C⁻¹` to **thousands** of right-hand sides
that all share the *same* `C`. The CG convergence rate is set by the spread
of the spectrum of `P⁻¹C`, and — because the preconditioner is a guaranteed
upper bound (`P⁻¹ ⪰ C⁻¹`, so `C ⪰ P` and every eigenvalue of `P⁻¹C` is `≥ 1`)
— the bulk sits at 1 with a tail of *large* eigenvalues: a handful of slow
cut-sky / anisotropic-noise directions that drag down every solve. Deflation
removes that subspace from the Krylov iteration once and amortizes the cost
over all the RHS.

Given a basis `W` (`nrow × k`) spanning the slow directions, with the
Galerkin coarse operator `E = Wᵀ C W` and projectors `P = I − C W E⁻¹ Wᵀ`,
`Pᵀ = I − W E⁻¹ (CW)ᵀ`, deflated CG solves the deflated system `P C x̃ = P b`
and reconstructs

```
x = W E⁻¹ Wᵀ b  +  Pᵀ x̃.
```

This is **exact for any full-rank `W`** — `C x = b` the instant the deflated
residual `P(b − C x̃)` vanishes (then `b − C x̃ ∈ range(CW) = ker P`, so
`C x = (I−P)b + P b = b`). The deflation space changes only the *speed*, never
the *answer*; tests check the recovered solution against plain CG to CG
tolerance. The per-iteration overhead is one projector `P v = v − (CW) E⁻¹
(Wᵀ v)`: with `CW` precomputed once (k operator applies) and `E⁻¹` stored
dense, it is small GEMMs and **no extra SHTs** (a dense `k×k` inverse, not a
LAPACK `cho_solve`, because inside the CG `while_loop` the custom-call
dispatch can cost more than the SHT itself).

`W` is built by **recycling** the Krylov information of a short instrumented
solve: PCG implicitly runs Lanczos on `P⁻¹C`, so the Ritz vectors of its
tridiagonal approximate the eigenvectors of `P⁻¹C`, and the *largest* Ritz
vectors are the slow directions (Lanczos resolves well-separated extreme
eigenvalues first, so a short `~2k`-step run captures them). This reuses
solves we do anyway — the defining feature of eigCG/recycled methods.

Set `deflation=k` (off by default). The harvested space matches the optimal
(dense largest-eigenvector) deflation of the same `k` and, on a masked,
anisotropic-noise problem at nside 16, cuts CG by **~1.9×** (e.g. 484→259
iterations, 1.8× wall-time on the dense backend; the relative gain is larger
for the matrix-free `ducc`/`s2fft` backends, where each iteration is an SHT
and the dense projector is negligible). The one-off harvest amortizes after
~1 solve, and memory is `O(nrow·k)`. `build_deflation()` exposes a manual
(re)build, e.g. after `update_fiducial` (which invalidates the stale space,
since `W` is tied to `C(fiducial)`). Natural next step: thick-restart to cap
the harvest memory, and per-iteration recycling across consecutive RHS.

## SHT backends (`dense` / `ducc` / `s2fft` / `almond`)

Every `C·x` needs a synthesis `Y` (real-basis coefficients → observed-pixel
map) and its **exact transpose** `Yᵀ`. The three backends differ only in how
that adjoint pair is realized; all agree to ~1e-13 and are interchangeable:

- **`dense`** — the real-basis synthesis matrix `Y`, restricted to observed
  pixels, is precomputed once (cached on disk) and stored on the device, so
  every SHT is a single GEMM. Memory is `N_obs · (lmax+1)²`, so this is the
  fastest backend for `nside ≤ 64` (≤ 256 on large GPUs). Default for
  `nside ≤ 64`.
- **`ducc`** — matrix-free transforms through ducc0 (the engine behind
  healpy), exact in double precision and multithreaded on the CPU. Memory is
  `O(N_obs)`, so it scales to `nside = 1024+`. The transform runs on the CPU
  via `jax.pure_callback`: correct and scalable, but each SHT crosses the
  host↔device boundary and is opaque to XLA fusion and autodiff. Default for
  `nside > 64`.
- **`s2fft`** — native-JAX matrix-free transforms ([s2fft](https://github.com/astro-informatics/s2fft)).
  The synthesis is `s2fft.inverse` evaluated inside XLA, so the whole
  `C·x` chain stays **on the accelerator** (no host round-trip) and is
  differentiable. We build `Y` by scattering the real coefficients into
  s2fft's 2-D `flm` layout — for spin 0 the conjugate-symmetric scalar
  coefficients, for spin 2 the spin-(+2) coefficients `−(E + iB)` with the
  E/B reality symmetry filling the negative-`m` half — and obtain `Yᵀ`
  exactly as `jax.linear_transpose` of that differentiable synthesis (**not**
  the quadrature-based `s2fft.forward`, which is only an approximate inverse
  on HEALPix). This is the route to a GPU SHT at `nside ≳ 256` and to an
  end-to-end differentiable likelihood; on a small fp64 GPU (or CPU-bound
  hardware) ducc0 is still competitive, since the win is on-device locality
  and fusion rather than raw FLOPs (see the report's feasibility section).

  **Caveat — requires a fixed s2fft.** Released s2fft (≤ 1.4.0) has a
  Wigner-d recursion-node bug: HEALPix rings sit at `cos θ` values that land
  exactly on Wigner-d nodes for spin ≠ 0, where an intermediate
  renormalization produces a silently-dropped NaN, so HEALPix **spin-2**
  synthesis carries ~5–13 % pointwise error (spin-0 is unaffected, and the
  shipped HEALPix tests only covered spin 0). This is fatal for QML, which
  relies on exact adjoint pairs. It is fixed on the upstream branch
  `fix/healpix-spin-recursion-node` (guard the renormalization at exact
  nodes). `S2fftSHT` **self-checks against ducc0 at construction** and raises
  a `RuntimeError` if the installed s2fft fails the spin-2 check, so an
  unpatched install cannot silently produce wrong spectra. The backend is
  therefore **opt-in** (`backend='s2fft'`), not selected by `'auto'`. Note
  that s2fft's HEALPix `forward`→`inverse` round trip still shows the usual
  few-percent HEALPix quadrature error — that is expected and irrelevant
  here, because QML uses `Yᵀ` (the exact transpose), never the quadrature
  `forward`.
- **`almond`** — matrix-free transforms through [Almond](../../almond), an
  in-house CUDA/CuPy SHT library implementing ducc0's exact algorithm on the
  GPU (spin-0 and spin-2, synthesis and exact adjoint `Yᵀ`, float64, healpy
  conventions, validated against ducc0 to ~1e-12). It rides the same
  `jax.pure_callback` path as `ducc` (numpy-in/numpy-out `AlmondRealSHT`), so
  it is a drop-in alternative that keeps the SHT on the GPU. On a dedicated
  A100 it beats 64-thread ducc0 by ~6× (spin-0) / ~2.5–3× (spin-2). Opt-in
  (`backend='almond'`), not selected by `'auto'`; requires the `almond`
  package installed and a GPU. **Note:** with JAX and CuPy sharing one device,
  set `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (and `PREALLOCATE=false`) or the
  two allocators fragment the GPU and OOM at `nside ≳ 128`.

## Response (Fisher), noise bias, windows

Two engines:

- **exact** (`fisher_mode='exact'`): solve `V = M G` for all
  `(comp, l, m)` columns with batched CG and accumulate the `l`-resolved
  response tensor `T[a,b,c,d](l,l')` chunk by chunk (the dense `H = GᵀMG`
  is never stored). Exact `R`, exact noise bias `n = diag blocks of VᵀNV`,
  exact bandpower windows `F_bl` — for any binning. Cost ≈ `n_modes` CG
  solves; the default up to nside 64.

- **subsampled** (`fisher_mode='subsampled'`, `fisher_frac=f`): the exact
  engine run on a random fraction f of the mode *columns*, stratified per
  `l'` (>= 1 column each, without replacement) and renormalized by
  `N_l'/n_l'` — exactly unbiased, and since the row index of H is always
  summed exactly, the sampling error is *local in bands*:
  `offset(c_A)/sigma_A ~ SNR_A sqrt(rho (1-f)/n_A)` with the band's own
  S/N and a mask-geometry factor rho (~0.1 for the NaMaster test mask).
  Measured head-to-head at matched solve counts, its frozen-response
  offsets are 6-9x smaller than the sims-MC engine's (whose Wishart noise
  couples all bands: `offset ~ sqrt(SNR_tot^2/N_sims)`), i.e. ~40-80x
  cheaper at equal accuracy. The sampled R is symmetrized; check
  conditioning for very small f. This is the recommended scalable mode,
  combined with iteration when bands are strongly signal-dominated.
- **pseudo-Cl control variate** (`fisher_control_variate='pseudo_cl'` with
  the exact/subsampled engine): replace the sampled object by
  `R0 + sample(R_exact - R0)`, where `R0` is a deterministic MASTER-style
  coupling built with local diagonal signal+noise inverse pixel weights.  For
  spin-2 fields this diagonal uses the isotropic zero-lag Q/U signal variance
  implied by the E/B trace; the approximation is the local diagonal filter
  itself.  The control is computed in SiMaster's discrete HEALPix
  normalization using the same SHT operators, so a full-column run is
  algebraically identical to the ordinary exact engine; with
  `keep_samples=True`, the retained slabs are residuals and
  `subsample_error()` measures the residual sampling error.  This is
  experimental: a poor local-diagonal control can increase variance, so
  production runs should compare the reported suboptimality with and without
  the option.
- **mc** (`fisher_mode='mc'`): for sims `x ~ N(0, C̃)`,
  `cov[y_A, y_B] = R_AB` exactly (this also holds for the deprojected
  filter because `M C̃ M = M`), the same sims give windows
  `F_bl = cov[y_b, y_l]`, and noise-only probes give `n`.  The response
  sims must be Gaussian (the covariance identity uses Gaussian fourth
  moments; Rademacher probes would bias it through their fourth cumulant).
  The noise-bias trace `n_A = 1/2 Tr[M N M P_A]` is single-`M`, so it *is*
  plain Hutchinson estimation: there we use `N^(1/2) d` with Rademacher
  `d` (exact, since N is diagonal), which is unbiased with strictly lower
  variance than Gaussian noise draws — measured ~1.1-1.3x here; the gain
  is bounded because `P_A` is a band projector whose Frobenius mass is
  off-diagonal in pixel space, which no probe distribution suppresses. Scales to any
  nside. **Caveat:** the frozen MC noise of `R̂` adds a fixed offset to all
  estimates with rms `≈ σ_A · sqrt(SNR²_tot / n_sims)` where `SNR²_tot` is
  the *total* squared signal-to-noise summed over all bands. For
  signal-dominated data, size `n_sims` accordingly (or use exact mode, or
  iterate so that the fiducial absorbs the offset). The inverse-Wishart
  Hartlap factor is applied; residual non-Gaussian corrections to it are
  `O(n_bins/n_sims)`.

## MC Fisher uncertainty and the automatic harness

The MC engine returns a *noisy* `R̂`; like the subsampling error it does not
average away in one analysis, so `simaster.mc_fisher` quantifies it — the MC
analog of `simaster.subsample`. From a set of per-seed estimates (each `R̂` from
`nsims` draws plus its reference mean `ȳ`), `MCFisherStore` gives the combined
`F̂` and:

* **element-wise σ(F̂_ab)** — the Wishart variance
  `Var[F̂_ab] = (F_aa F_bb + F_ab²)/(N_eff − 1)`, cross-checked by the
  seed-to-seed scatter (model-free, needs ≥2 seeds). Verified: the median
  relative σ on the diagonal equals `√(2/N_eff)`.
* **error-bar calibration.** A *frozen* `F̂` (one realization, as in any real
  run) makes the bandpower *pulls* scale as `1/√h` with the Anderson–Hartlap
  factor `h = (N_eff − n_b − 2)/(N_eff − 1)` — the inverse-Wishart /
  Dodelson–Schneider effect, verified to 0.5% over `N_eff = 512…3072`. So the
  honest 1σ bandpower error bar is `√diag(F̂⁻¹)` (the Hartlap-shrunk covariance
  divided by `h`); `MCFisherStore.bandpower_cov(calibrated=True)` returns it and
  `pull → 1`. A single short MC under-covers: e.g. nside=128 `nsims=512`
  (`h≈0.43`) gives pull 1.52 while χ²/dof≈1 (misleading); averaging 6 seeds
  (`N_eff=3072`, `h≈0.91`) → pull 1.05 — seed-averaging is mandatory at high
  nside.
* **suboptimality / dRnorm** vs an exact `F` if supplied, and a **held-out χ²**
  (`held_out_chi2`) on independent sims that did *not* enter `F̂` — a
  circularity-free bias check.

Per-seed estimates are concatenable (`MCFisherStore.merge`), so a distributed
run saves one store per rank and merges them. See `compute_mc_error`.

### Automatic harness (`simaster.run_auto`, `simaster.fisher_auto`)

`run_auto(scheduler, problem, nside, outdir, k='auto', …)` runs the whole
workflow over a pluggable `Scheduler`:

  i.   **pilot** — time a deflation-`k` grid and pick the `k` minimising MC wall
       (harvest + `nsims`·per-solve); for ducc at nside=128 this is a shallow
       basin around `k≈800` (CG iters keep falling with `k` but per-solve
       flattens and harvest grows).
  ii.  **MC** — `n_seeds` independent seeds at the optimal `k`, one per rank.
  iii. **combine** — merge into `F̂` + uncertainty (`compute_mc_error`) and a
       held-out-χ² check.

The `Scheduler` ABC's only contract is "run
`python -m simaster.fisher_worker … --rank R` for `R` in `0..n−1`, in parallel,
and block". `LocalScheduler` runs ranks in-process (portable, any machine). HPC
backends live outside the portable core: `simaster.nersc.SlurmScheduler` fans
ranks out with `srun` *inside one allocation*, so the entire pilot→MC→combine
runs as a **single** SLURM job (never a job array) — see
`simaster/nersc/run_auto.sh`. A *problem* is any importable module exposing
`build_workspace(nside, *, fisher_mode, deflation, cg_tol, …) -> QMLWorkspace`.

## Bands, junk bands, aliasing

Bandpowers are flat in `C_l` over user bands; SiMaster automatically adds
"junk" bands so the band basis covers every multipole in `[lmin, lmax]`.
All bands are estimated jointly and the junk bands are marginalized (full
`R⁻¹`), not reported. **The covariance model is bandlimited at `lmax`**
(default `3 nside − 1`): power in the data above `lmax` is not modeled and
will alias — choose `lmax`/resolution so that signal+noise above it is
negligible. Validation simulations are generated bandlimited, making the
model exact.

Monopole and dipole of every spin-0 field are deprojected by default
(`deproject_low_ell`), since `l < 2` is outside the band basis.

## Template marginalization

Two equivalent prescriptions (validated against each other):

- `template_alpha = α` (finite): adds `α_rel · tr(C)/||t||² · t tᵀ` to `C`,
  the classic "large prefactor" recipe. The same term is included when
  drawing MC sims, keeping `R` consistent. Very large `α` degrades CG
  conditioning; `α ≲ 1e6` is safe in double precision.
- `template_alpha = None` (default): the exact `α → ∞` limit via
  Sherman–Morrison–Woodbury,
  `M = C⁻¹ − C⁻¹T (TᵀC⁻¹T)⁻¹ TᵀC⁻¹`, implemented as one extra batched
  solve for `C⁻¹T`. Numerically stable, exactly nulls the template
  directions.

## Numeric caveats (summary)

1. Everything is float64; QML quadratic forms difference large numbers and
   float32 is not supported.
2. CG tolerance (default 1e-5) enters the filter; data and sims use the
   same filter so the leading effect cancels in `R⁻¹y`, but do not relax it
   beyond ~1e-4.
3. MC-mode frozen-`R̂` offsets: see above.
4. The bandlimit/aliasing caveat: see above. HEALPix quadrature
   non-orthonormality (up to ±70% Gram-eigenvalue spread at
   `l = 3 nside − 1`) is *not* an error source for the estimator — `Y` is
   part of the model — but it does inflate the variance of the highest
   bands and is why the top bands near `3 nside` should be treated as junk.
5. Window functions: with a curved input spectrum, compare estimates to
   `QMLWorkspace.predict(cl_theory)` (window-convolved), not to naively
   binned theory.

## The around-fiducial (sim-debiased) estimator

With a stochastic response matrix (subsampled or MC engines), the plain
estimator `ĉ = R̂⁻¹(ŷ − n̂)` carries a frozen bias `R⁻¹δR̂·c_true` for *any*
fiducial. The around-fiducial form

```
ĉ = c_fid + R̂⁻¹( ŷ(d) − ⟨ŷ⟩_fid-sims )      (QMLWorkspace.run_mean_debias)
```

uses the mean of ŷ over ~1/ε² (~100) independent fiducial sims, which has
expectation `R_true·c_fid + n_true` with the *exact* response of the actual
filter — so the R̂ error multiplies `(c_true − c_fid)` instead of `c_true`
(verified: a 20%-wrong fiducial deflates subsampled offsets by exactly the
predicted ×5). Combined with one `iterate()` re-centering this is what makes
stochastic response engines viable for signal-dominated data; see the report
for the Planck-scale budget. The fiducial bandpowers used in this form are
band means of the fiducial spectra — exact when the fiducial is band-flat
(which `update_fiducial` guarantees).

## Field-level likelihood, score, and autodiff

`simaster.score` exposes the exact-likelihood view. The quadratic term
`-1/2 d^T C^-1 d` is differentiable through the CG solve with
`lax.custom_linear_solve` (implicit differentiation: the backward pass is
one extra solve), and `jax.grad` of it w.r.t. the bandpowers reproduces the
QML statistic `y_A` to machine precision (tested). The log-det term never
needs evaluating: its gradient `-1/2 Tr[C^-1 P_A]` is a single-solve
Hutchinson trace satisfying `1/2 Tr[C^-1 P_A] = n_A + (R c_fid)_A`, so the
full score is `y - n - R c_fid` — QML is the Newton step on this likelihood.
Second derivatives do *not* get cheaper via autodiff (the Hessian is the
double-solve trace; one solve per band per probe), so the subsampled-column
engine remains the Fisher method of choice; Hessian-*vector* products cost
~2 solves and enable truncated-Newton optimization or HMC over bandpowers,
and gradients w.r.t. any upstream parametrization (cosmological parameters,
calibration, beams) chain through `clmat` for free.

## Exact Hessian: second-order likelihood expansion (`exact_hessian`)

`QMLWorkspace.exact_hessian(data)` returns the **exact** gradient and Hessian
of the Gaussian log-likelihood in the bandpower basis, evaluated at the
current fiducial `c0` for one (or a batch of) data realization(s). For
`-2 lnL = dᵀ C⁻¹ d + ln det C` with `C(c)` *linear* in the bandpowers (so
`C_{,ij} = 0`), writing `M = C⁻¹`:

```
∂_A lnL   = ½ dᵀ M C_A M d − ½ Tr(M C_A)          = y_A − ⟨y_A⟩
∂²_AB lnL = F_AB − Q_AB
   F_AB = ½ Tr(M C_A M C_B)        (the response/Fisher matrix R)
   Q_AB = dᵀ M C_A M C_B M d = (C_A z)ᵀ M (C_B z),     z = M d.
```

Because `E[d dᵀ] = C`, `E[Q] = 2F` and hence `E[hess] = −F`, recovering the
Fisher matrix in the mean. The point is that `exact_hessian` keeps the *data*
term `Q` rather than its expectation, so `−hess = Q − F` is the genuine
per-realization curvature. The gradient is exactly the QML score
`y − n − R c_fid` of the previous section.

The data term is matrix-free and cheap. With `z = M d` (one CG solve) and
`v_A = C_A z = U E_A Uᵀ z = from_modes(E_A · to_modes(z))` (a band/spectrum
projection in harmonic space, no new SHTs), `Q = Vᵀ M V` needs one further
batched CG solve per band parameter — `1 + n_param` solves total, far fewer
than the per-mode response. `C_A` here is exactly the operator behind the
`y`-statistic (the symmetric `U_i E_b U_jᵀ + U_j E_b U_iᵀ` for a cross-band),
so `F`, `y` and `Q` are mutually consistent and the result is validated
against the dense `F − dᵀMC_AMC_BMd` to CG tolerance.

This is meant for treating a good fiducial's first and second derivatives as
the whole inference — a pure 2nd-order (Gaussian) expansion
`lnL(c) ≈ lnL(c0) + grad·(c−c0) + ½(c−c0)ᵀ hess (c−c0)`. The implied MLE is
the single Newton step `c0 + (−hess)⁻¹ grad` with parameter covariance
`(−hess)⁻¹`; `LikelihoodExpansion.newton_estimate()` does this with the exact
curvature, `fisher_estimate()` with `F` (always SPD). `hess`, `grad` and `F`
span the full band set (user **and** junk bands, both real parameters); the
estimate helpers therefore **marginalize** over the junk bands — invert the
full matrix, then restrict — rather than conditioning on them (which slicing
the Hessian would do). `−hess` is SPD only in expectation, so for a noisy
realization `newton_estimate(floor=…)` clips its eigenvalues; with a good
fiducial (`Q ≈ 2F`) it is typically already SPD.

## Radical compression (offset-lognormal likelihood, Bond–Jaffe–Knox 2000)

`simaster.compress(workspace, result)` reduces a QML estimate to the BJK
triplet `{c_hat, x, F}` (`CompressedLikelihood`, with `save`/`load` and a
callable `loglike`). The x-factors generalize the ideal `x_l = N_l/B_l^2`
through the exact identity `c_hat + x = R^-1 y` (total signal+noise power,
BJK's D-hat): `x = R^-1 n`, available exactly from the workspace response
and noise bias. The likelihood is Gaussian in `Z_b = ln(c_b + x_b)` for
auto-spectra with weight `M^(Z) = (c_hat+x) F (c_hat+x)`; cross-spectra
(which can be negative) stay Gaussian, as is standard. Verified against
the exact dense likelihood at nside=8: |delta(-2lnL)| <~ 1 within +-1.5
sigma even for the lowest Delta_l=5 band, always better than a plain
Gaussian out to 3 sigma; the far low-C tail of the exact likelihood is
steeper than lognormal (anti-conservative for lower limits — BJK's known
limitation, driven by nu-heterogeneity within wide low-l bands; use
narrower low-l bands if the deep tail matters). At high l the form
converges to Gaussian, so compression only changes anything at large
scales — QML territory.

## Curved fiducial + flat deviations (`estimate(deviations=True)`)

Instead of modelling the spectrum as flat within bands, keep the full
smooth (curved-in-l) fiducial C_l^fid inside the covariance and fit only
flat band deviations away from it:

```
dc = R^-1 ( y(d) - ybar_fid ),    ybar_fid = F_bl C^fid_l + n,
```

with ybar_fid deterministic and exact in the exact/subsampled engines (the
l-resolved windows F_bl are available; mc mode uses the sim mean). Then
E[dc] = R^-1 F (c_true - c_fid)_l — identically zero at the fiducial and
free of flat-band binning bias otherwise, since all spectral curvature
lives in the fiducial. `iterate(deviations=True)` adds the fitted flat
deviations to the smooth fiducial each step (preserving its shape) instead
of replacing it by flat bandpowers. This is the recommended mode whenever
a good smooth fiducial exists (CMB, LCDM-like LSS): the bandpower windows
then only matter at second order in the residual.
