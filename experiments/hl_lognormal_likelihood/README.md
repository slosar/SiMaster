# Bandpower likelihood approximations: Gaussian vs offset-lognormal vs Hamimeche–Lewis

A controlled, reproducible test comparing three approximate bandpower likelihoods
— **Gaussian**, the **Bond–Jaffe–Knox (2000) offset-lognormal**, and
**Hamimeche & Lewis (2008)** — each in a **raw** (bandpower-Fisher metric) and a
**M_f-calibrated** (transformed-space covariance from fiducial sims) form. Written
as a self-contained artifact for the SiMaster paper.

## The question

At low ℓ / few modes per band the QML bandpower likelihood is non-Gaussian
(positively skewed: upward fluctuations carry more weight). Three things matter
for cosmological inference:

1. **Per-band Gaussianization** — which change of variables makes each band's
   marginal most Gaussian?
2. **Calibration** — does the *joint* likelihood give χ²-to-truth ≈ dof?
3. **Parameter bias** — does a coherent amplitude come out unbiased with correct
   error bars? (A goodness-of-fit test is blind to a bias that lives in one
   parameter direction; the parameter test is the one that matters.)

## The five likelihoods

Per auto band, with x-factor offset `x = R⁻¹n` and `ratio = (ĉ+x)/(c+x)`, define
the transformed residual (`simaster.transform_residual`):

| transform | `X_b` |
|---|---|
| gaussian | `ĉ − c` |
| lognormal (BJK) | `(c_fid+x)·ln(ratio)` |
| hl | `(c_fid+x)·g(ratio)`, `g(x)=sign(x−1)√(2(x−ln x−1))` (`simaster.g_vst`) |

`g` is the **exact** Gaussianizing transform of the per-mode Wishart/χ²
(`−2lnL = ν·g²/2`); `ln` is BJK's leading-order approximation. The likelihood is

    −2 ln L = (X − x̄)ᵀ M⁻¹ (X − x̄)

with two metric choices:

- **raw** — `M = F⁻¹` (bandpower Fisher), `x̄ = 0`. This is what `compress()`
  returns with no sims.
- **M_f** — `M = cov(X)`, `x̄ = ⟨X⟩` estimated over fiducial sims
  (`simaster.build_Mf`, Hartlap-corrected). This is HL's actual prescription:
  a Gaussian in the *transformed* space with its own covariance. Needs
  `QMLWorkspace.run_mc(store_bandpowers=True)`.

So `gaussian`, `lognormal_raw`, `lognormal_Mf`, `hl_raw`, `hl_Mf`.

## Setup (fully deterministic)

- Single spin-0 (temperature) field, **nside=16** (lmax=32), cut sky
  (`fsky ≈ 0.75` galactic strip), white noise `σ = 8 µK/pix`.
- **Narrow low-ℓ bins** (single-ℓ at low ℓ) → few modes/band → strong
  non-Gaussianity, where the transforms differ most.
- Red band-flat fiducial (`C_ℓ ∝ 1/ℓ(ℓ+1)`), exact dense **Cholesky** filter,
  fixed seeds ⇒ reproducible.
- MC Fisher with `n_mc` fiducial sims (response `R`, noise bias `n`, x-factors,
  and the per-sim bandpowers for `M_f`); `n_test` **independent** realizations
  for the metrics.

## What is measured

- **(A)** per-band residual skewness of `ĉ`, `ln(ĉ+x)`, `g((ĉ+x)/(c_fid+x))`.
- **(B)** χ²-to-truth distribution of the five likelihoods (mean, KS-p vs χ²_dof).
- **(C)** amplitude recovery: fit `A` (theory `= A·c_fid`) under each likelihood,
  report `pull = (Â−1)/σ_A` (bias = mean pull; correct spread = std ≈ 1) and the
  68% interval coverage.

## Results

`nside=16`, `n_mc=8000`, `n_test=4000`, `nb` = 25 bands (`dof=25`). Figures in
`results/`; full numbers in `results/summary.json`.

**Per-band Gaussianization** — mean|skew|: raw **0.565** → lognormal **0.351** →
**HL 0.062**. HL Gaussianizes ~6× better (nearly perfectly).

| model | χ²/dof | KS p | ampl. bias `⟨Â⟩−1` | pull mean | pull std | cov68 |
|---|---|---|---|---|---|---|
| gaussian       | 1.056 | 0.000 | −0.028 | −0.565 | 0.97 | 0.621 |
| lognormal_raw  | 1.108 | 0.000 | +0.063 | **+1.199** | 1.13 | 0.405 |
| lognormal_Mf   | 1.001 | 0.000 | −0.001 | −0.058 | 0.99 | 0.687 |
| hl_raw         | 1.227 | 0.000 | +0.094 | **+1.744** | 1.25 | 0.267 |
| **hl_Mf**      | **0.998** | **0.134** | +0.018 | +0.305 | 0.96 | 0.662 |

### Takeaways

1. **HL's `g` Gaussianizes each band best** — residual skew 0.565→0.351→**0.062**.
2. **Calibration requires `M_f`, and only HL fully calibrates.** The *raw*
   offset-lognormal and HL over-weight (χ²/dof>1; `g`'s aggressive tail makes
   `hl_raw` worst) because the raw Fisher is not the covariance of the
   transformed variable. With `M_f`, both get χ²/dof≈1 — but **only `hl_Mf`
   passes the KS test (p=0.13)**: `lognormal_Mf` has the right mean yet fails KS
   (p=0.000) because `ln`'s residual per-band skew (0.35) still shapes the joint
   χ². HL's near-perfect Gaussianization (0.06) is what passes it.
3. **The raw compressed likelihoods are amplitude-biased** (pull +1.2 to +1.7σ,
   coverage 0.27–0.41) — *worse than Gaussian* — a caution against using
   `compress()`'s default (no-sims) form for parameter inference at low ℓ. `M_f`
   removes the bias and restores ~0.68 coverage.
4. **Amplitude vs shape.** A fixed-covariance Gaussian recovers the amplitude
   only mildly biased (−0.57σ) yet fails the χ² shape (KS 0.000) — matching HL's
   remark that some Gaussian approximations give reliable constraints without
   capturing the per-ℓ shape. Both `M_f` forms recover the amplitude well
   (`lognormal_Mf` coverage 0.69, `hl_Mf` 0.66); the HL advantage is in the full
   joint distribution / higher moments, which is where the KS test separates them.

**One-line summary for the paper:** the exact HL `g`-transform with the
fiducial-sim covariance `M_f` is the only one of the five that is simultaneously
(near-)perfectly Gaussianizing per band, calibrated in the joint χ² (KS-passing),
and unbiased in the amplitude with correct coverage.

## Reproduce

```bash
JAX_PLATFORMS=cpu python experiment.py            # defaults: nside 16, n_mc 8000, n_test 4000
# faster smoke:
JAX_PLATFORMS=cpu python experiment.py --n-mc 800 --n-test 400
```
Outputs `results/{summary.json, skewness.png, chi2.png, amplitude.png, run.log}`.

## Library pieces used

- `simaster.g_vst`, `simaster.transform_residual`, `simaster.build_Mf`
- `simaster.CompressedLikelihood(..., transform='lognormal'|'hl', cov_X=, xbar=, c_fid=)`
- `QMLWorkspace.run_mc(store_bandpowers=True)`

## References

- G. Hamimeche & A. Lewis, *Phys. Rev. D* **77**, 103013 (2008), arXiv:0801.0554.
- J. R. Bond, A. H. Jaffe & L. Knox, *Astrophys. J.* **533**, 19 (2000).
