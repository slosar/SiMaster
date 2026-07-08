#!/usr/bin/env python
r"""Full Hamimeche-Lewis MATRIX transform: joint T/E handling of cross-spectra
======================================================================

The scalar transforms treat each spectrum independently and keep the cross
(TE/TB/EB) Gaussian.  For correlated, few-mode bands the TE bandpower is itself
non-Gaussian, so that is wrong.  HL's full method assembles the per-band
``n_comp x n_comp`` field power matrix, whitens by the theory, applies ``g`` to
its eigenvalues, and re-sandwiches with the fiducial -- transforming all spectra
jointly.  ``simaster.transform_residual(..., spec_pairs=...)`` does this;
``spec_pairs`` gives each spectrum's component-index pair (e.g. TT=(0,0),
TE=(0,1), EE=(1,1)).

This is a self-contained, deterministic (T,E) demonstration on synthetic Wishart
bandpowers (so the "true" per-band distribution is known exactly): correlated
(corr ~ 0.85) and few modes per band.  We compare, in the M_f-calibrated form:
gaussian, lognormal, HL-scalar (TE Gaussian), HL-matrix (TE joint).

Run:  python experiment_matrix.py
Outputs: results/{matrix_summary.json, matrix.png}.
"""
import json
import os

import numpy as np
from scipy.stats import wishart, kstest, chi2, skew
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simaster as sm
from simaster.radical import CompressedLikelihood

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
N = 8000

# --- (T,E) setup: strongly correlated TE, few modes/band (non-Gaussian) -------
nb = 5
spec_pairs = [(0, 0), (0, 1), (1, 1)]                  # TT, TE, EE
labels = ["TT", "TE", "EE"]
# signal matrices S_b (corr ~ 0.85), decreasing with band; diagonal noise
base = np.array([[100.0, 55.0], [55.0, 40.0]])
S = np.array([base * f for f in (1.0, 0.6, 0.35, 0.2, 0.12)])           # (nb,2,2)
xm = np.array([[[3.0, 0.0], [0.0, 4.0]]] * nb)                          # noise (diag)
Ctot = S + xm
nu = np.array([4, 6, 9, 14, 22])                                       # modes/band
xf = np.concatenate([np.full(nb, 3.0), np.zeros(nb), np.full(nb, 4.0)])
c_fid = np.concatenate([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]])
is_auto = np.array([True] * nb + [False] * nb + [True] * nb)
dof = 3 * nb


def draw(n):
    out = np.zeros((n, 3 * nb))
    for b in range(nb):
        W = wishart.rvs(df=nu[b], scale=Ctot[b] / nu[b], size=n, random_state=rng)
        out[:, b] = W[:, 0, 0] - xm[b, 0, 0]
        out[:, nb + b] = W[:, 0, 1]
        out[:, 2 * nb + b] = W[:, 1, 1] - xm[b, 1, 1]
    return out


sim, test = draw(N), draw(N)
F = np.eye(dof)
print(f"(T,E) nb={nb} dof={dof} corr(TE)~{base[0,1]/np.sqrt(base[0,0]*base[1,1]):.2f} "
      f"nu={nu.tolist()} N={N}", flush=True)

# --- likelihoods (all M_f-calibrated) -----------------------------------------
models = {
    "gaussian": ("gaussian", None),
    "lognormal": ("lognormal", None),
    "hl_scalar": ("hl", None),
    "hl_matrix": ("hl", spec_pairs),
}
chi2v, cov_of = {}, {}
for name, (tr, sp) in models.items():
    covX, xbar = sm.build_Mf(sim, c_fid, xf, is_auto, transform=tr, spec_pairs=sp)
    cov_of[name] = (covX, xbar, tr, sp)
    c2 = np.empty(len(test))
    for i, ci in enumerate(test):
        if tr == "gaussian":                            # Gaussian: X = c - c_fid
            r = (ci - c_fid) - xbar
            c2[i] = r @ np.linalg.solve(covX, r)
        else:
            cl = CompressedLikelihood(np.arange(dof), ["a"], ci, xf, F, is_auto,
                                      transform=tr, cov_X=covX, xbar=xbar,
                                      c_fid=c_fid, spec_pairs=sp)
            c2[i] = -2 * cl.loglike(c_fid)
    chi2v[name] = c2

# --- per-spectrum residual skewness: scalar HL vs matrix HL --------------------
def resid(sp):
    return np.array([sm.transform_residual(ci, c_fid, xf, is_auto, "hl",
                                           c_fid=c_fid, spec_pairs=sp) for ci in test[:3000]])

Xs, Xm = resid(None), resid(spec_pairs)
sk_raw = [np.nanmedian(np.abs(skew(test[:, s * nb:(s + 1) * nb], 0))) for s in range(3)]
sk_sc = [np.nanmedian(np.abs(skew(Xs[:, s * nb:(s + 1) * nb], 0))) for s in range(3)]
sk_mx = [np.nanmedian(np.abs(skew(Xm[:, s * nb:(s + 1) * nb], 0))) for s in range(3)]

summary = {"nb": nb, "dof": dof, "nu": nu.tolist(),
           "corr_TE": float(base[0, 1] / np.sqrt(base[0, 0] * base[1, 1])),
           "skew_by_spec": {"raw": dict(zip(labels, sk_raw)),
                            "hl_scalar": dict(zip(labels, sk_sc)),
                            "hl_matrix": dict(zip(labels, sk_mx))},
           "chi2": {}}
print(f"\n{'model':12s} {'chi2/dof':>8s} {'KS p':>7s}")
for name, c2 in chi2v.items():
    p = kstest(c2, lambda t: chi2.cdf(t, dof)).pvalue
    summary["chi2"][name] = dict(mean_chi2=float(c2.mean()), ks_p=float(p))
    print(f"{name:12s} {c2.mean()/dof:8.3f} {p:7.3f}")
print("\nmedian |residual skew| by spectrum (raw -> HL-scalar -> HL-matrix):")
for s, lab in enumerate(labels):
    print(f"  {lab}: {sk_raw[s]:.3f} -> {sk_sc[s]:.3f} -> {sk_mx[s]:.3f}")
json.dump(summary, open(os.path.join(OUT, "matrix_summary.json"), "w"), indent=2)

# --- figure -------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
xp = np.arange(3); w = 0.27
ax[0].bar(xp - w, sk_raw, w, label="raw")
ax[0].bar(xp, sk_sc, w, label="HL scalar (TE kept Gaussian)")
ax[0].bar(xp + w, sk_mx, w, label="HL matrix (TE joint)")
ax[0].set_xticks(xp); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("median |residual skewness|")
ax[0].set_title("Per-spectrum Gaussianization"); ax[0].legend(fontsize=8)
names = list(chi2v); ks = [summary["chi2"][n]["ks_p"] for n in names]
ax[1].bar(np.arange(len(names)), ks, color=["C3", "C0", "C1", "C2"])
ax[1].axhline(0.05, color="k", ls="--", lw=0.8, label="0.05")
ax[1].set_xticks(np.arange(len(names))); ax[1].set_xticklabels(names, rotation=25, ha="right", fontsize=8)
ax[1].set_ylabel(r"$\chi^2$ KS p (calibrated)"); ax[1].set_title("Joint calibration"); ax[1].legend(fontsize=8)
fig.suptitle(f"HL matrix vs scalar transform, correlated (T,E), dof={dof}", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "matrix.png"), dpi=150); plt.close(fig)
print("\nwrote results/{matrix_summary.json, matrix.png}")
