#!/usr/bin/env python
"""Extra val2 QA: (1) chi^2 with vs without x-factors (offset-lognormal vs
Gaussian), (2) Fisher/response matrix computed three ways (exact,
column-subsampled, Monte-Carlo).

Uses the val2 physical setup: CMB T/Q/U, NaMaster mask, anisotropic
longitude-strip ivar (factor-2 rms variation).  Builds the workspace once,
snapshots the response from each engine, and reuses the exact engine for
the data estimates + x-factors.

  python scripts/val2_fisher_xfactor.py [--nside 32] [--nreal 100]
                                        [--nsims-mc 2048] [--frac 0.25]
"""

import argparse
import json
import os

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from val_common import (load_cmb_cls, load_mask, FIGDIR, DATADIR, CACHEDIR)
from val2_ivar import make_ivar
import simaster as sm

P = argparse.ArgumentParser()
P.add_argument("--nside", type=int, default=32)
P.add_argument("--lmax", type=int, default=None)
P.add_argument("--nlb", type=int, default=8)
P.add_argument("--nreal", type=int, default=100)
P.add_argument("--nsims-mc", type=int, default=2048)
P.add_argument("--frac", type=float, default=0.25)
P.add_argument("--sigma-T", type=float, default=50.0)
P.add_argument("--batch", type=int, default=128)
P.add_argument("--tag", default="val2x")
args = P.parse_args()

nside = args.nside
lmax = args.lmax or 2 * nside
npix = hp.nside2npix(nside)
rng = np.random.default_rng(20260613)

cls = load_cmb_cls(lmax)
mask = load_mask(nside)
ivar_T = make_ivar(nside, mask) / args.sigma_T ** 2     # anisotropic strips
ivar_P = ivar_T / 2.0
print(f"[{args.tag}] nside={nside} lmax={lmax} fsky={mask.mean():.3f}")

# band-flat fiducial (matches val2 band-flat variant: clean chi^2)
bins = sm.Bins.linear(2, lmax, args.nlb)
bext, _ = bins.extend_to_cover(2, lmax)
clf = {k: bext.unbin_cl(bext.bin_cl(cls[k]), lmax) for k in cls}
for k in clf:
    clf[k][:2] = 0.0

fT = sm.Field(mask, [np.zeros(npix)], ivar=ivar_T, name="T")
fP = sm.Field(mask, [np.zeros(npix)] * 2, spin=2, ivar=ivar_P, name="P")
cl_fid = {("T_0", "T_0"): clf["TT"], ("P_E", "P_E"): clf["EE"],
          ("P_B", "P_B"): clf["BB"], ("T_0", "P_E"): clf["TE"]}
w = sm.QMLWorkspace([fT, fP], bins, cl_fid, lmax=lmax, fisher_mode="exact",
                    batch_size=args.batch, seed=321, cachedir=CACHEDIR)

nbb = w.bins.nbands
ns = len(w.spec_pairs)
keep = np.concatenate([np.flatnonzero(w.is_user_band) + s * nbb
                       for s in range(ns)])


def snapshot():
    return dict(R=w.R_hat.copy(), Rinv=w.R_inv.copy(), n=w.n_hat.copy())


# ---- response three ways (snapshot after each; they overwrite) -----------
print("exact response ...")
w.run_exact()
S_exact = snapshot()
F_l = w.F_l_hat.copy()

print(f"subsampled response (f={args.frac}) ...")
w.run_exact(sample_frac=args.frac, sample_seed=1)
S_sub = snapshot()

print(f"MC response ({args.nsims_mc} sims) ...")
w.fisher_mode = "mc"
w.run_mc(n_sims_fisher=args.nsims_mc, n_sims_noise=max(512, args.nsims_mc // 4))
S_mc = snapshot()

# restore exact for the estimator
w.R_hat, w.R_inv, w.n_hat = S_exact["R"], S_exact["Rinv"], S_exact["n"]
w.F_l_hat = F_l
w.fisher_mode = "exact"
w._mc_done = True


def cov_user(S):
    return S["Rinv"][np.ix_(keep, keep)]


def sigma_user(S):
    return np.sqrt(np.diag(cov_user(S)))


# =====================================================================
#  Figure 1: Fisher matrix three ways
# =====================================================================
sig_e, sig_s, sig_m = sigma_user(S_exact), sigma_user(S_sub), sigma_user(S_mc)
Ce, Cs, Cm = cov_user(S_exact), cov_user(S_sub), cov_user(S_mc)
ells = w.user_bins.get_effective_ells()
nub = int(w.is_user_band.sum())

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (a) per-band TT error sigma_b for the three methods
tt = slice(0, nub)   # first spectrum (T_0 x T_0)
ax[0].plot(ells, sig_e[tt], "k-o", ms=4, label="exact", zorder=3)
ax[0].plot(ells, sig_s[tt], "C0s--", ms=4, label=f"subsampled f={args.frac}")
ax[0].plot(ells, sig_m[tt], "C3^:", ms=4, label=f"MC {args.nsims_mc} sims")
ax[0].set_xlabel(r"$\ell$"); ax[0].set_ylabel(r"$\sigma_b$ (TT band error)")
ax[0].set_yscale("log"); ax[0].legend(fontsize=8)
ax[0].set_title("(a) bandpower errors, TT")

# (b) scatter of all cov elements vs exact
iu = np.triu_indices(Ce.shape[0])
scale = np.sqrt(np.outer(np.diag(Ce), np.diag(Ce)))[iu]
xe = (Ce[iu]) / scale
ax[1].plot(xe, Cs[iu] / scale, "C0.", ms=3, alpha=0.5,
           label="subsampled")
ax[1].plot(xe, Cm[iu] / scale, "C3.", ms=3, alpha=0.5, label="MC")
lim = [xe.min() * 1.1, xe.max() * 1.1]
ax[1].plot(lim, lim, "k-", lw=0.8)
ax[1].set_xlabel("exact $R^{-1}_{bb'}$ (corr-normalized)")
ax[1].set_ylabel("approx $R^{-1}_{bb'}$")
ax[1].legend(fontsize=8); ax[1].set_title("(b) all covariance elements")

# (c) histogram of fractional residuals vs exact
res_s = ((Cs - Ce) / scale.reshape(Ce.shape) if False else
         (Cs[iu] - Ce[iu]) / scale)
res_m = (Cm[iu] - Ce[iu]) / scale
bins_h = np.linspace(-0.3, 0.3, 41)
ax[2].hist(res_s, bins_h, alpha=0.6, density=True,
           label=f"subsampled (rms {res_s.std():.3f})")
ax[2].hist(res_m, bins_h, alpha=0.6, density=True,
           label=f"MC (rms {res_m.std():.3f})")
ax[2].set_xlabel(r"$(R^{-1}_{\rm approx}-R^{-1}_{\rm exact})$ / corr")
ax[2].set_ylabel("density"); ax[2].legend(fontsize=8)
ax[2].set_title("(c) residuals vs exact")

fig.suptitle(f"{args.tag}: response matrix three ways "
             f"(CMB T/Q/U, anisotropic noise, nside={nside})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, f"{args.tag}_fisher_3way.png"), dpi=150)
plt.close(fig)
print("wrote fisher_3way figure")

# =====================================================================
#  Figure 2: chi^2 with vs without x-factors
# =====================================================================
# data estimates from the exact engine
maps_T, maps_P = [], []
for _ in range(args.nreal):
    T, Q, U = hp.synfast([clf["TT"], clf["EE"], clf["BB"], clf["TE"]],
                         nside, lmax=lmax, pol=True, new=True)
    maps_T.append(mask * T + rng.normal(0, 1, npix) / np.sqrt(ivar_T))
    maps_P.append(np.array([mask * Q + rng.normal(0, 1, npix) / np.sqrt(ivar_P),
                            mask * U + rng.normal(0, 1, npix) / np.sqrt(ivar_P)]))
data = w.pack_data([np.array(maps_T), np.array(maps_P)])
res = w.estimate(data)

# x-factors (BJK): x = R^-1 n, restricted to user bands; autos get Z=ln(c+x)
x_full = S_exact["Rinv"] @ S_exact["n"]
x = x_full[keep]
is_auto = np.concatenate([np.full(nub, i == j) for (i, j) in w.spec_pairs])
c_tgt = np.concatenate([w.user_bins.bin_cl(
    {"T_0 x T_0": clf["TT"], "P_E x P_E": clf["EE"], "P_B x P_B": clf["BB"],
     "T_0 x P_E": clf["TE"], "T_0 x P_B": np.zeros(lmax + 1),
     "P_E x P_B": np.zeros(lmax + 1)}[s]) for s in w.spec_names])
F = np.linalg.inv(Ce)

# build estimate matrix (nreal, nuser_total) in spec_names order
est = np.concatenate([np.atleast_2d(res.cl[s]) for s in w.spec_names], axis=1)

# metric for the Z-space chi^2, evaluated at the truth (BJK: fix at peak/true)
use_log = is_auto & (c_tgt + x > 0)
D = np.where(use_log, c_tgt + x, 1.0)
M_Z = D[:, None] * F * D[None, :]
Z_tgt = np.where(use_log, np.log(np.where(use_log, c_tgt + x, 1.0)), c_tgt)

chi2_G, chi2_Z, n_bad = [], [], 0
for i in range(args.nreal):
    ci = est[i]
    rG = ci - c_tgt
    chi2_G.append(rG @ F @ rG)
    arg = ci + x
    bad = use_log & (arg <= 0)
    n_bad += int(bad.any())
    Zi = np.where(use_log, np.log(np.clip(arg, 1e-300, None)), ci)
    rZ = Zi - Z_tgt
    chi2_Z.append(rZ @ M_Z @ rZ)
chi2_G, chi2_Z = np.array(chi2_G), np.array(chi2_Z)
dof = est.shape[1]

ks_G = stats.kstest(chi2_G, lambda v: stats.chi2.cdf(v, dof)).pvalue
ks_Z = stats.kstest(chi2_Z, lambda v: stats.chi2.cdf(v, dof)).pvalue

x_hi = dof + 6 * np.sqrt(2 * dof)
over_G = int((chi2_G > x_hi).sum())
over_Z = int((chi2_Z > x_hi).sum())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
xg = np.linspace(max(0.1, dof - 5 * np.sqrt(2 * dof)), x_hi, 300)
hbins = np.linspace(0, x_hi, 26)
for a, (c2, ksp, over, lab) in zip(
        ax, [(chi2_G, ks_G, over_G, "Gaussian $\\chi^2$ (no x-factor)"),
             (chi2_Z, ks_Z, over_Z,
              "offset-lognormal $\\chi^2$ (x-factors)")]):
    a.hist(np.clip(c2, None, x_hi), bins=hbins, density=True, alpha=0.6,
           label=f"{args.nreal} realizations")
    a.plot(xg, stats.chi2.pdf(xg, dof), "k-", lw=1.5,
           label=fr"$\chi^2_{{{dof}}}$")
    a.set_xlabel(r"$\chi^2$"); a.set_ylabel("density")
    extra = f"\n({over} beyond axis)" if over else ""
    a.set_title(f"{lab}\nmean {c2.mean():.1f} (dof {dof}), "
                f"KS p = {ksp:.3f}{extra}", fontsize=9)
    a.legend(fontsize=8)
fig.suptitle(f"{args.tag}: bandpower $\\chi^2$ vs truth, with / without "
             f"x-factors (CMB T/Q/U, anisotropic noise, nside={nside})",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, f"{args.tag}_xfactor_chi2.png"), dpi=150)
plt.close(fig)

# mechanism: per-band skewness reduction by the Z = ln(c+x) transform,
# for every auto-spectrum band (cross-spectra excluded).  Points below the
# 1:1 line = the x-factor symmetrizes that band; the transform helps most
# in the signal-dominated bands and can over-correct where x ~ c.
auto_idx = np.flatnonzero(is_auto)
sk_raw = np.array([stats.skew(est[:, b]) for b in auto_idx])
sk_z = np.array([stats.skew(np.log(np.clip(est[:, b] + x[b], 1e-300, None)))
                 for b in auto_idx])
snr = c_tgt[auto_idx] / np.sqrt(np.diag(Ce)[auto_idx])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
sc = ax[0].scatter(np.abs(sk_raw), np.abs(sk_z), c=np.log10(snr + 1e-9),
                   cmap="viridis", s=40)
mx = max(np.abs(sk_raw).max(), np.abs(sk_z).max()) * 1.1
ax[0].plot([0, mx], [0, mx], "k-", lw=0.8)
ax[0].set_xlabel(r"$|\mathrm{skew}(\hat c_b)|$ (raw)")
ax[0].set_ylabel(r"$|\mathrm{skew}(\ln(\hat c_b+x_b))|$")
ax[0].set_title("per auto-band skewness\n(below line: x-factor helps)",
                fontsize=9)
plt.colorbar(sc, ax=ax[0], label=r"$\log_{10}$ band S/N")
# the single most signal-dominated auto band, raw vs Z histograms
b0 = int(auto_idx[np.argmax(snr)])
ax[1].hist((est[:, b0] - c_tgt[b0]) / np.sqrt(Ce[b0, b0]), bins=18,
           density=True, alpha=0.6, label=f"raw, skew {stats.skew(est[:,b0]):+.2f}")
z = np.log(est[:, b0] + x[b0]); zt = np.log(c_tgt[b0] + x[b0])
ax[1].hist((z - zt) / z.std(), bins=18, density=True, alpha=0.6,
           label=f"$\\ln(c+x)$, skew {stats.skew(z):+.2f}")
xx = np.linspace(-4, 4, 200)
ax[1].plot(xx, stats.norm.pdf(xx), "k-", lw=1)
ax[1].set_xlabel("standardized residual")
ax[1].set_title(f"most signal-dominated auto band\n(S/N {snr.max():.0f})",
                fontsize=9)
ax[1].legend(fontsize=8)
fig.suptitle(f"{args.tag}: x-factor Gaussianization by band", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, f"{args.tag}_xfactor_mechanism.png"), dpi=150)
plt.close(fig)

summary = dict(
    nside=nside, lmax=lmax, dof=int(dof), nreal=args.nreal,
    chi2_gauss_mean=float(chi2_G.mean()), ks_gauss=float(ks_G),
    chi2_lognorm_mean=float(chi2_Z.mean()), ks_lognorm=float(ks_Z),
    n_realizations_with_neg_arg=int(n_bad),
    fisher_resid_rms_sub=float(res_s.std()),
    fisher_resid_rms_mc=float(res_m.std()),
    sub_frac=args.frac, nsims_mc=args.nsims_mc)
json.dump(summary, open(os.path.join(DATADIR, f"{args.tag}_summary.json"),
                        "w"), indent=2)
np.savez(os.path.join(DATADIR, f"{args.tag}.npz"),
         R_exact=S_exact["R"], R_sub=S_sub["R"], R_mc=S_mc["R"],
         n=S_exact["n"], x=x, c_tgt=c_tgt, est=est,
         chi2_G=chi2_G, chi2_Z=chi2_Z, ells=ells)
print(json.dumps(summary, indent=2))
