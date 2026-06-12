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
