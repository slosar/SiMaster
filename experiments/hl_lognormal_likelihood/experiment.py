#!/usr/bin/env python
r"""Controlled comparison of bandpower likelihood approximations
======================================================================

Gaussian  vs  BJK offset-lognormal  vs  Hamimeche & Lewis (2008), each in the
"raw" (bandpower-Fisher metric) and "M_f-calibrated" (transformed-space
covariance from fiducial sims) forms.

Setup: a single spin-0 (temperature) field on a cut sky at nside=16 with
single-ell / narrow bins at low ell, where the QML bandpower likelihood is
strongly non-Gaussian (few modes per band).  Everything is built from a fixed
seed and the exact dense Cholesky filter, so the run is deterministic.

Transforms (per auto band, offset x = R^-1 n, ratio = (c_hat+x)/(c+x)):
    gaussian  :  X = c_hat - c
    lognormal :  X = (c_fid+x) ln(ratio)            [BJK]
    hl        :  X = (c_fid+x) g(ratio),  g(x)=sign(x-1)sqrt(2(x-ln x-1))   [HL]
Likelihood:  -2 ln L = (X - xbar)^T M^{-1} (X - xbar), with
    raw  :  M = bandpower Fisher^{-1}, xbar = 0   (compress() default)
    M_f  :  M = cov(X), xbar = <X>  over fiducial sims   (build_Mf; full HL)

We report, over many independent realizations:
  (A) per-band residual skewness under each transform (lower = better Gaussianized)
  (B) chi^2-to-truth calibration of the 5 likelihoods (mean, KS p vs chi^2_dof)
  (C) amplitude-parameter recovery: pull = (A_hat-1)/sigma_A -> bias + 68% coverage,
      the paper-relevant test (a coherent amplitude is biased by a Gaussian
      likelihood even when each band looks Gaussian; does HL fix it?).

Run:  JAX_PLATFORMS=cpu python experiment.py [--n-mc 8000 --n-test 4000 --nside 16]
Outputs: results/{summary.json, skewness.png, chi2.png, amplitude.png}.
"""
import argparse
import json
import os

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize

import simaster as sm
from simaster.radical import CompressedLikelihood

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=16)
P.add_argument("--n-mc", type=int, default=8000, help="fiducial sims for M_f + Fisher")
P.add_argument("--n-test", type=int, default=4000, help="independent test realizations")
P.add_argument("--sigma", type=float, default=8.0, help="white noise per pixel (uK)")
args = P.parse_args()
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ setup ----
nside, lmax = args.nside, 2 * args.nside
npix = hp.nside2npix(nside)
ell = np.arange(lmax + 1)
cl = np.zeros(lmax + 1); cl[2:] = 1000.0 / (ell[2:] * (ell[2:] + 1.0))   # ~flat D_l
mask = np.ones(npix); mask[hp.query_strip(nside, 0, np.deg2rad(30))] = 0.0  # fsky~0.75
ivar = np.where(mask > 0, 1.0 / args.sigma ** 2, 0.0)

# narrow low-ell bins: single-ell at low ell, pairs higher -> few modes/band
lo, hi, l = [], [], 2
while l <= lmax:
    wdt = 1 if l < 20 else 2
    lo.append(l); hi.append(min(l + wdt - 1, lmax)); l = hi[-1] + 1
bins = sm.Bins.from_edges(np.array(lo), np.array(hi))
cl_flat = bins.unbin_cl(bins.bin_cl(cl), lmax); cl_flat[:2] = 0.0        # band-flat fiducial

fld = sm.Field(mask, [np.zeros(npix)], ivar=ivar, name="T")
w = sm.QMLWorkspace(fld, bins, {("T_0", "T_0"): cl_flat}, lmax=lmax, backend="dense",
                    solver="cholesky", fisher_mode="mc", deproject_low_ell=False,
                    seed=1234, verbose=False)
print(f"nside={nside} lmax={lmax} fsky={mask.mean():.2f} bands={bins.nbands} "
      f"sigma={args.sigma}uK  n_mc={args.n_mc} n_test={args.n_test}", flush=True)

# ---- fiducial MC: response, x-factors, and the per-sim bandpowers for M_f ----
w.run_mc(n_sims_fisher=args.n_mc, n_sims_noise=args.n_mc // 2, store_bandpowers=True)
keep = np.flatnonzero(w.is_user_band)
nb = keep.size
cov_bp = np.asarray(w.R_inv)[np.ix_(keep, keep)]
F = np.linalg.inv(cov_bp)
x = (np.asarray(w.R_inv) @ np.asarray(w.n_hat))[keep]
c_fid = w.fiducial_bandpowers()[keep]
is_auto = np.ones(nb, bool)
sim_bp = np.asarray(w.mc_bandpowers)[:, keep]
ells = w.user_bins.get_effective_ells()
sig_b = np.sqrt(np.diag(cov_bp))

covX_ln, xbar_ln = sm.build_Mf(sim_bp, c_fid, x, is_auto, transform="lognormal")
covX_hl, xbar_hl = sm.build_Mf(sim_bp, c_fid, x, is_auto, transform="hl")

# ---- independent test realizations ----
rng = np.random.default_rng(20260708)
maps = []
for _ in range(args.n_test):
    m = hp.synfast(cl, nside, lmax=lmax, verbose=False)
    maps.append(mask * m + np.where(mask > 0, rng.normal(0, args.sigma, npix), 0.0))
test_bp = np.asarray(w.estimate([np.asarray(maps)]).vector())     # (n_test, nb)

# ================================================================= (A) skew ==
def skews(sample):
    raw = stats.skew(sample, axis=0)
    ln, hl = [], []
    for b in range(nb):
        col = sample[:, b]
        if np.all(col + x[b] > 0) and c_fid[b] + x[b] > 0:
            ln.append(stats.skew(np.log(col + x[b])))
            hl.append(stats.skew(sm.g_vst((col + x[b]) / (c_fid[b] + x[b]))))
        else:
            ln.append(np.nan); hl.append(np.nan)
    return raw, np.array(ln), np.array(hl)

sk_raw, sk_ln, sk_hl = skews(test_bp)

# ============================================= (B) chi^2 to truth (5 models) ==
dell = np.arange(nb)


def make(c_hat):
    base = dict(c_hat=c_hat, x=x, F=F, is_auto=is_auto)
    return {
        "gaussian": CompressedLikelihood(dell, ["T"], transform="lognormal", **base),
        "lognormal_raw": CompressedLikelihood(dell, ["T"], transform="lognormal", **base),
        "lognormal_Mf": CompressedLikelihood(dell, ["T"], transform="lognormal",
                                             cov_X=covX_ln, xbar=xbar_ln, c_fid=c_fid, **base),
        "hl_raw": CompressedLikelihood(dell, ["T"], transform="hl", **base),
        "hl_Mf": CompressedLikelihood(dell, ["T"], transform="hl",
                                      cov_X=covX_hl, xbar=xbar_hl, c_fid=c_fid, **base),
    }


def nll(like, name, c):     # -2 ln L
    return -2.0 * (like.loglike_gaussian(c) if name == "gaussian" else like.loglike(c))


names = ["gaussian", "lognormal_raw", "lognormal_Mf", "hl_raw", "hl_Mf"]
chi2 = {n: np.full(args.n_test, np.nan) for n in names}
Ahat = {n: np.full(args.n_test, np.nan) for n in names}
Aerr = {n: np.full(args.n_test, np.nan) for n in names}
for i in range(args.n_test):
    likes = make(test_bp[i])
    for n in names:
        lk = likes[n]
        chi2[n][i] = nll(lk, n, c_fid)
        # amplitude fit: theory = A * c_fid ; maximize loglike over A
        f = lambda A: nll(lk, n, A * c_fid)
        r = optimize.minimize_scalar(f, bounds=(0.3, 3.0), method="bounded")
        A0 = r.x; Ahat[n][i] = A0
        # 1-sigma from local curvature of -2lnL (= 1 at +-1 sigma)
        h = 1e-3
        curv = (f(A0 + h) - 2 * f(A0) + f(A0 - h)) / h ** 2
        Aerr[n][i] = np.sqrt(2.0 / curv) if curv > 0 else np.nan

dof = nb


def chi2_stat(c2):
    v = c2[np.isfinite(c2)]
    return float(v.mean()), float(stats.kstest(v, lambda t: stats.chi2.cdf(t, dof)).pvalue), int((~np.isfinite(c2)).sum())


def amp_stat(n):
    a, e = Ahat[n], Aerr[n]
    ok = np.isfinite(a) & np.isfinite(e) & (e > 0)
    pull = (a[ok] - 1.0) / e[ok]
    cov68 = float(np.mean(np.abs(pull) < 1.0))
    return float(np.mean(a[ok]) - 1.0), float(np.mean(pull)), float(np.std(pull)), cov68, int(ok.sum())


summary = {"config": vars(args), "nb": nb, "fsky": float(mask.mean()),
           "ells": ells.tolist(),
           "skew": {"raw": sk_raw.tolist(), "lognormal": sk_ln.tolist(), "hl": sk_hl.tolist()},
           "mean_abs_skew": {"raw": float(np.nanmean(np.abs(sk_raw))),
                             "lognormal": float(np.nanmean(np.abs(sk_ln))),
                             "hl": float(np.nanmean(np.abs(sk_hl)))},
           "chi2": {}, "amplitude": {}}
print(f"\nnb={nb} dof={dof}")
print(f"mean|skew|: raw {summary['mean_abs_skew']['raw']:.3f}  "
      f"lognormal {summary['mean_abs_skew']['lognormal']:.3f}  "
      f"HL {summary['mean_abs_skew']['hl']:.3f}")
print(f"\n{'model':16s} {'chi2/dof':>8s} {'KS p':>6s} {'A bias':>8s} "
      f"{'pull mean':>9s} {'pull std':>8s} {'cov68':>6s}")
for n in names:
    m, p, ex = chi2_stat(chi2[n])
    bias, pm, ps, cov, nok = amp_stat(n)
    summary["chi2"][n] = dict(mean_chi2=m, dof=dof, ks_p=p, n_excluded=ex)
    summary["amplitude"][n] = dict(bias=bias, pull_mean=pm, pull_std=ps, cov68=cov, n=nok)
    print(f"{n:16s} {m/dof:8.3f} {p:6.3f} {bias:+8.4f} {pm:+9.3f} {ps:8.3f} {cov:6.3f}")

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

# ================================================================= figures ===
# (A) per-band skewness vs ell
fig, a = plt.subplots(figsize=(6.4, 4.0))
a.axhline(0, color="k", lw=0.6)
a.plot(ells, sk_raw, "o-", ms=4, label="raw")
a.plot(ells, sk_ln, "s-", ms=4, label="lognormal ln(c+x)")
a.plot(ells, sk_hl, "^-", ms=4, label="HL g-transform")
a.set_xlabel(r"band $\ell$"); a.set_ylabel("residual skewness")
a.legend(); a.set_title("Per-band Gaussianization (lower |skew| is better)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "skewness.png"), dpi=150); plt.close(fig)

# (B) chi^2 distributions
fig, ax = plt.subplots(1, 5, figsize=(16, 3.4), sharex=True, sharey=True)
xhi = dof + 6 * np.sqrt(2 * dof); xg = np.linspace(0.1, xhi, 300); hb = np.linspace(0, xhi, 30)
for a, n in zip(ax, names):
    v = chi2[n][np.isfinite(chi2[n])]
    m, p, ex = chi2_stat(chi2[n])
    a.hist(np.clip(v, None, xhi), hb, density=True, alpha=0.6)
    a.plot(xg, stats.chi2.pdf(xg, dof), "k-", lw=1.4)
    a.set_title(f"{n}\nchi2/dof {m/dof:.2f}, KS {p:.3f}", fontsize=9)
    a.set_xlabel(r"$\chi^2$")
ax[0].set_ylabel("density")
fig.suptitle(f"Bandpower $\\chi^2$ to truth (dof={dof})", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "chi2.png"), dpi=150); plt.close(fig)

# (C) amplitude bias + coverage
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
bias = [summary["amplitude"][n]["bias"] for n in names]
pmean = [summary["amplitude"][n]["pull_mean"] for n in names]
cov = [summary["amplitude"][n]["cov68"] for n in names]
xp = np.arange(len(names))
ax[0].bar(xp, pmean, color=["C3", "C0", "C0", "C1", "C1"])
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xticks(xp); ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[0].set_ylabel(r"amplitude pull mean $\langle(\hat A-1)/\sigma_A\rangle$")
ax[0].set_title("Parameter bias (0 = unbiased)")
ax[1].bar(xp, cov, color=["C3", "C0", "C0", "C1", "C1"])
ax[1].axhline(0.68, color="k", ls="--", lw=0.8, label="0.68 target")
ax[1].set_xticks(xp); ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[1].set_ylabel("68% interval coverage"); ax[1].legend(); ax[1].set_ylim(0, 1)
ax[1].set_title("Interval coverage")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "amplitude.png"), dpi=150); plt.close(fig)
print("\nwrote results/{summary.json, skewness.png, chi2.png, amplitude.png}")
