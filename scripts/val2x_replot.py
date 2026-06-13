#!/usr/bin/env python
"""Reprocess the saved val2x run: regenerate the x-factor chi^2 and
mechanism figures with proper handling of realizations that fall outside
the offset-lognormal's domain (some band driven to c_hat + x <= 0 by
mask-induced band coupling -- a real limitation, reported, not hidden).
Reads report/data/val2x.npz; no QML re-run.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from val_common import FIGDIR, DATADIR

tag = "val2x"
d = np.load(os.path.join(DATADIR, f"{tag}.npz"))
est, x, c_tgt = d["est"], d["x"], d["c_tgt"]
chi2_G, chi2_Z = d["chi2_G"], d["chi2_Z"]
nreal, ntot = est.shape
nspec = 6
nub = ntot // nspec
dof = ntot
# spec order: TT, TE, TB, EE, EB, BB  -> autos are blocks 0, 3, 5
auto_spec = [0, 3, 5]
is_auto = np.zeros(ntot, bool)
for s in auto_spec:
    is_auto[s * nub:(s + 1) * nub] = True

# realizations valid for the offset-lognormal: every auto band has c+x>0
arg = est + x[None, :]
valid = ~((arg[:, is_auto] <= 0).any(axis=1))
n_invalid = int((~valid).sum())
sig_emp = est.std(0)                          # empirical band error
snr = c_tgt / sig_emp

chi2_Zv = chi2_Z[valid]
ks_G = stats.kstest(chi2_G, lambda v: stats.chi2.cdf(v, dof)).pvalue
ks_Z = stats.kstest(chi2_Zv, lambda v: stats.chi2.cdf(v, dof)).pvalue
nside = int(d["ells"].size and 32)

# ------------------------------------------------------------- chi2 figure
x_hi = dof + 6 * np.sqrt(2 * dof)
over_G = int((chi2_G > x_hi).sum())
over_Z = int((chi2_Zv > x_hi).sum())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
xg = np.linspace(max(0.1, dof - 5 * np.sqrt(2 * dof)), x_hi, 300)
hb = np.linspace(0, x_hi, 26)
panels = [(chi2_G, ks_G, over_G, 0,
           "Gaussian $\\chi^2$ (no x-factor)"),
          (chi2_Zv, ks_Z, over_Z, n_invalid,
           "offset-lognormal $\\chi^2$ (x-factors)")]
for a, (c2, ksp, over, ninv, lab) in zip(ax, panels):
    a.hist(np.clip(c2, None, x_hi), bins=hb, density=True, alpha=0.6,
           label=f"{len(c2)} realizations")
    a.plot(xg, stats.chi2.pdf(xg, dof), "k-", lw=1.5,
           label=fr"$\chi^2_{{{dof}}}$")
    a.set_xlabel(r"$\chi^2$"); a.set_ylabel("density")
    notes = []
    if over:
        notes.append(f"{over} beyond axis")
    if ninv:
        notes.append(f"{ninv} excluded ($c+x\\leq 0$)")
    extra = "\n(" + "; ".join(notes) + ")" if notes else ""
    a.set_title(f"{lab}\nmean {c2.mean():.1f} (dof {dof}), "
                f"KS p = {ksp:.3f}{extra}", fontsize=9)
    a.legend(fontsize=8)
fig.suptitle(f"{tag}: bandpower $\\chi^2$ vs truth, with / without x-factors "
             f"(CMB T/Q/U, anisotropic noise, nside={nside})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, f"{tag}_xfactor_chi2.png"), dpi=150)
plt.close(fig)

# -------------------------------------------------------- mechanism figure
auto_idx = np.flatnonzero(is_auto)
ev = est[valid]
sk_raw = np.array([stats.skew(ev[:, b]) for b in auto_idx])
sk_z = np.array([stats.skew(np.log(np.clip(ev[:, b] + x[b], 1e-300, None)))
                 for b in auto_idx])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
sc = ax[0].scatter(np.abs(sk_raw), np.abs(sk_z), c=np.log10(snr[auto_idx]),
                   cmap="viridis", s=40)
mx = max(np.abs(sk_raw).max(), np.abs(sk_z).max()) * 1.1
ax[0].plot([0, mx], [0, mx], "k-", lw=0.8)
ax[0].set_xlabel(r"$|\mathrm{skew}(\hat c_b)|$ (raw)")
ax[0].set_ylabel(r"$|\mathrm{skew}(\ln(\hat c_b+x_b))|$")
ax[0].set_title("per auto-band skewness\n(below line: x-factor helps)",
                fontsize=9)
plt.colorbar(sc, ax=ax[0], label=r"$\log_{10}$ band S/N")
b0 = int(auto_idx[np.argmax(snr[auto_idx])])
ax[1].hist((ev[:, b0] - c_tgt[b0]) / sig_emp[b0], bins=18, density=True,
           alpha=0.6, label=f"raw, skew {stats.skew(ev[:,b0]):+.2f}")
z = np.log(ev[:, b0] + x[b0]); zt = np.log(c_tgt[b0] + x[b0])
ax[1].hist((z - zt) / z.std(), bins=18, density=True, alpha=0.6,
           label=f"$\\ln(c+x)$, skew {stats.skew(z):+.2f}")
xx = np.linspace(-4, 4, 200)
ax[1].plot(xx, stats.norm.pdf(xx), "k-", lw=1)
ax[1].set_xlabel("standardized residual"); ax[1].legend(fontsize=8)
ax[1].set_title(f"most signal-dominated auto band "
                f"(S/N {snr[auto_idx].max():.0f})", fontsize=9)
fig.suptitle(f"{tag}: x-factor Gaussianization by band", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, f"{tag}_xfactor_mechanism.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------- summary json
summ = json.load(open(os.path.join(DATADIR, f"{tag}_summary.json")))
summ.update(chi2_lognorm_mean=float(chi2_Zv.mean()), ks_lognorm=float(ks_Z),
            ks_gauss=float(ks_G), chi2_gauss_mean=float(chi2_G.mean()),
            n_realizations_with_neg_arg=n_invalid,
            n_realizations_valid=int(valid.sum()))
json.dump(summ, open(os.path.join(DATADIR, f"{tag}_summary.json"), "w"),
          indent=2)
print(json.dumps(summ, indent=2))
