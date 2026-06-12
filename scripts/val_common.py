"""Shared helpers for the validation suite."""

import os
import sys

# must be set before jax initializes: the desktop occupies part of the GPU,
# and XLA's GEMM autotuner profiles kernels with an extra operand copy
# (>1 GB for the dense synthesis matrices) -- disable it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                           + " --xla_gpu_autotune_level=0").strip()

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESOURCES = os.path.join(os.path.dirname(REPO), "resources")
NMT_TEST = os.path.join(RESOURCES, "code", "NaMaster", "test")
FIGDIR = os.path.join(REPO, "report", "figs")
DATADIR = os.path.join(REPO, "report", "data")
CACHEDIR = os.path.join(REPO, ".simaster_cache")
for d in (FIGDIR, DATADIR, CACHEDIR):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, REPO)


def load_cmb_cls(lmax):
    """NaMaster test-suite CMB spectra (l, TT, EE, BB, TE), in uK^2."""
    l, tt, ee, bb, te = np.loadtxt(os.path.join(NMT_TEST, "cls.txt"),
                                   unpack=True)
    out = {}
    for name, c in [("TT", tt), ("EE", ee), ("BB", bb), ("TE", te)]:
        c = c[: lmax + 1].copy()
        c[:2] = 0.0
        out[name] = c
    return out


def load_mask(nside):
    """NaMaster test-suite mask, degraded to nside and binarized."""
    m = hp.read_map(os.path.join(NMT_TEST, "mask.fits"))
    m = hp.ud_grade(m, nside)
    return (m > 0.5).astype(np.float64)


def chi2_plot(chi2_vals, dof, fname, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.linspace(max(0.1, dof - 5 * np.sqrt(2 * dof)),
                    dof + 6 * np.sqrt(2 * dof), 300)
    ax.hist(chi2_vals, bins=24, density=True, alpha=0.6,
            label=f"{len(chi2_vals)} realizations")
    ax.plot(x, stats.chi2.pdf(x, dof), 'k-', lw=1.5,
            label=fr"$\chi^2_{{{dof}}}$")
    ks = stats.kstest(chi2_vals, lambda v: stats.chi2.cdf(v, dof))
    ax.set_xlabel(r"$\chi^2$"); ax.set_ylabel("density")
    ax.set_title(f"{title}\nmean {np.mean(chi2_vals):.1f} (dof {dof}), "
                 f"KS p = {ks.pvalue:.3f}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
    return ks.pvalue


def spectra_plot(ws, res_mean, res_err, targets, fname, title,
                 labels=None, ell_scale=True):
    """Grid of bandpower panels: mean estimate vs target."""
    specs = ws.spec_names
    n = len(specs)
    ncol = min(3, n); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow),
                             squeeze=False)
    ells = ws.user_bins.get_effective_ells()
    f = ells * (ells + 1) / (2 * np.pi) if ell_scale else np.ones_like(ells)
    for i, s in enumerate(specs):
        ax = axes[i // ncol][i % ncol]
        ax.errorbar(ells, f * res_mean[s], yerr=f * res_err[s], fmt='o',
                    ms=3, label='QML mean $\\pm$ err of mean')
        ax.plot(ells, f * targets[s], 'k-', lw=1, label='target')
        ax.set_title((labels or {}).get(s, s), fontsize=9)
        ax.set_xlabel(r"$\ell$")
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def pulls_plot(pulls_flat, fname, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.linspace(-5, 5, 200)
    ax.hist(pulls_flat, bins=30, density=True, alpha=0.6)
    ax.plot(x, stats.norm.pdf(x), 'k-')
    ax.set_xlabel("pull"); ax.set_title(
        f"{title}\nmean {np.mean(pulls_flat):.3f}, std {np.std(pulls_flat):.3f}",
        fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)
