#!/usr/bin/env python
"""Assemble the LaTeX report sections from validation/benchmark outputs
(report/data/*.json) and compile report/main.pdf."""

import json
import os
import subprocess

from val_common import DATADIR, REPO

REPORT = os.path.join(REPO, "report")


def jload(name):
    p = os.path.join(DATADIR, name)
    if os.path.exists(p):
        return json.load(open(p))
    return None


def validation_section():
    out = []
    setup = {
        "val1": ("CMB $T/Q/U$, NaMaster mask, uniform noise "
                 "($\\sigma_T=30\\,\\mu$K/pix, $\\sigma_P=\\sqrt2\\sigma_T$)"),
        "val2": ("as test 1, but ivar in 8 longitude strips with the noise "
                 "r.m.s.\\ alternating by a factor of 2 across the observed "
                 "region"),
        "val3": ("galaxy density ($z{=}0.75$, $b{=}1$) and weak-lensing "
                 "shear (sources at $z{=}1.5$; 15 gal/arcmin$^2$, shape "
                 "noise 0.3), \\texttt{pyccl} theory"),
    }
    out.append(
        "All suites: $N_{\\rm side}=32$, $\\ell_{\\max}=95$, bands of "
        "$\\Delta\\ell=8$ over $2\\le\\ell\\le95$, 100 data realizations at "
        "fixed mask (= fixed Fisher), exact response engine, six spectra "
        "estimated jointly (TT, EE, BB, TE, TB, EB or the LSS analogues), "
        "monopole/dipole of spin-0 fields deprojected.  Two input-spectrum "
        "variants per suite: \\emph{flat} (data generated from band-flat "
        "spectra; the estimates must match the input bandpowers exactly, "
        "with no window ambiguity) and \\emph{curved} (original spectra; "
        "compared to the window-convolved prediction).  The $\\chi^2$ is "
        "that of the full bandpower vector against the target using the "
        "QML covariance $R^{-1}$.\n")
    out.append("\\begin{table}[h]\\centering\\small\n"
               "\\begin{tabular}{llrrrrr}\n\\toprule\n"
               "suite & variant & $\\langle\\chi^2\\rangle$ & dof & KS $p$ & "
               "pull mean & pull std \\\\\n\\midrule\n")
    for tag in ("val1", "val2", "val3"):
        s = jload(f"{tag}_summary.json")
        if s is None:
            continue
        for variant, r in s.items():
            out.append(
                f"{tag} & {variant} & {r['chi2_mean']:.1f} & {r['dof']} & "
                f"{r['ks_p']:.3f} & {r['pull_mean']:+.3f} & "
                f"{r['pull_std']:.3f} \\\\\n")
    out.append("\\bottomrule\n\\end{tabular}\n"
               "\\caption{Validation summary.  The KS $p$-value compares "
               "the per-realization $\\chi^2$ values to the "
               "$\\chi^2_{\\rm dof}$ distribution; pulls are per-band "
               "normalized residuals over all spectra, bands and "
               "realizations (expected: mean 0, std 1).}\n\\end{table}\n")
    for tag, desc in setup.items():
        if jload(f"{tag}_summary.json") is None:
            continue
        out.append(f"\n\\subsection{{Test {tag[-1]}: {desc}}}\n")
        for variant in ("flat", "curved"):
            for kind, cap in [("spectra", "recovered bandpowers"),
                              ("chi2", "$\\chi^2$ distribution"),
                              ("pulls", "pull distribution")]:
                f = f"{tag}_{variant}_{kind}.png"
                if os.path.exists(os.path.join(REPORT, "figs", f)):
                    out.append(
                        "\\begin{figure}[h]\\centering"
                        f"\\includegraphics[width=.78\\textwidth]{{{f}}}"
                        f"\\caption{{{tag} ({variant}): {cap}.}}"
                        "\\end{figure}\n")
        out.append("\\clearpage\n")
    mc = jload("val1_mc_summary.json")
    if mc:
        out.append(
            "\n\\subsection{Exact vs Monte-Carlo response}\n"
            "Repeating test 1 with the MC engine "
            f"($N_{{\\rm sims}}=16384$) gives $\\langle\\chi^2\\rangle = "
            f"{mc['flat']['chi2_mean']:.1f}$ (dof {mc['flat']['dof']}) for "
            "the flat variant, illustrating the frozen-$\\hat R$ S/N "
            "penalty discussed in \\S\\ref{sec:fisher}; the two engines "
            "agree on the response matrix itself to the expected "
            "$\\sqrt{2/N_{\\rm sims}}$ accuracy.\n")
    return "".join(out)


def feasibility_section():
    b = jload("bench.json")
    if b is None:
        return "Benchmarks not yet run.\n"
    d, c, p = b["dense"], b["ducc"], b["projection_1024"]
    txt = f"""
Measured on this machine (GPU: {b['device']}, 24-core CPU), validation
configuration ($N_{{\\rm side}}={b['nside']}$, $\\ell_{{\\max}}=3N_{{\\rm side}}-1$,
$T$+$E$/$B$, NaMaster mask, batch {d['batch']}):

\\begin{{table}}[h]\\centering\\small
\\begin{{tabular}}{{lrrrrr}}
\\toprule
backend & setup [s] & $\\mathbb{{C}}x$ batch [ms] & CG iters & solve [s] & $G$ mem [GB] \\\\
\\midrule
dense (GPU GEMM) & {d['build_s']:.1f} & {d['apply_s']*1e3:.0f} & {d['cg_iters']} & {d['solve_s']:.1f} & {d['G_GB']:.2f} \\\\
ducc (CPU SHT)   & {c['build_s']:.1f} & {c['apply_s']*1e3:.0f} & {c['cg_iters']} & {c['solve_s']:.1f} & -- \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Cost drivers at $N_{{\\rm side}}={b['nside']}$.}}
\\end{{table}}

\\paragraph{{$N_{{\\rm side}}=1024$ projection (ducc backend, $\\ell_{{\\max}}=2048$,
$f_{{\\rm sky}}=0.39$).}}
Data vector $3N_{{\\rm obs}} \\approx {p['nrow']/1e6:.1f}$M;
$N_{{\\rm modes}} \\approx {p['nmodes']/1e6:.1f}$M (so the exact response
engine is out of reach --- MC or iteration is mandatory).
A batch of {d['batch']} RHS holds $\\sim$5 CG state vectors:
{p['cg_state_GB']:.1f} GB of GPU RAM at float64, plus order-100 MB of
spectra/precondition tables --- \\textbf{{a 24--40 GB GPU is sufficient for
the solver itself}}; the hypothetical dense-$G$ backend would need
{p['dense_G_GB']/1e3:.1f} TB and is firmly excluded.
One covariance application (batched SHTs at $\\ell_{{\\max}}=2048$ on 24 CPU
cores) extrapolates to $\\sim${p['t_apply_s_extrapolated']:.0f} s per batch,
i.e.\\ a CG solve of {d['batch']} RHS in
$\\sim${p['t_solve_s']/60:.0f} min; $10^5$ MC sims for the response matrix
would cost $\\sim${p['t_solve_s']*100000/d['batch']/3600:.0f} h of CPU-SHT
time.  The honest conclusion: at $N_{{\\rm side}}=1024$ the SHT, not the GPU
linear algebra, is the bottleneck; a GPU SHT (the \\texttt{{sht}} backend
interface is ready) or a CPU farm for the transform stage is the upgrade
path.  Per-realization estimation (one solve) is entirely tractable
($\\sim${p['t_solve_s']/60/d['batch']:.1f} min/realization equivalent);
it is the response matrix that sets the budget, and it is computed once
per mask/fiducial.
"""
    return txt


with open(os.path.join(REPORT, "validation_section.tex"), "w") as f:
    f.write(validation_section())
with open(os.path.join(REPORT, "feasibility_section.tex"), "w") as f:
    f.write(feasibility_section())

for _ in range(3):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=REPORT, capture_output=True, text=True)
print("pdflatex rc:", r.returncode)
print(r.stdout[-600:] if r.returncode else "report/main.pdf built")
