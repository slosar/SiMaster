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
                 "($\\sigma_T=50\\,\\mu$K/pix, $\\sigma_P=\\sqrt2\\sigma_T$)"),
        "val2": ("as test 1, but ivar in 8 longitude strips with the noise "
                 "r.m.s.\\ alternating by a factor of 2 across the observed "
                 "region"),
        "val3": ("galaxy density ($z{=}0.75$, $b{=}1$) and weak-lensing "
                 "shear (sources at $z{=}1.5$; 15 gal/arcmin$^2$, shape "
                 "noise 0.3), \\texttt{pyccl} theory"),
    }
    out.append(
        "All suites: $N_{\\rm side}=32$, $\\ell_{\\max}=64=2N_{\\rm side}$ "
        "(chosen for the 6\\,GB GPU shared with the desktop session; pass "
        "\\texttt{--lmax 95} on a bigger card), $\\sigma_T=50\\,\\mu$K "
        "(S/N per mode crosses unity at $\\ell\\approx49$), bands of "
        "$\\Delta\\ell=8$ over $2\\le\\ell\\le64$, 100 data realizations at "
        "fixed mask (= fixed Fisher), exact response engine, six spectra "
        "estimated jointly (TT, EE, BB, TE, TB, EB or the LSS analogues), "
        "monopole/dipole of spin-0 fields deprojected.  Two input-spectrum "
        "variants per suite (both full curved-sky; the names refer to the "
        "input spectrum shape): \\emph{band-flat} (data generated from "
        "band-flat spectra, which lie exactly in the estimator's model "
        "space; the estimates must match the input bandpowers exactly, "
        "with no window ambiguity) and \\emph{smooth} (original "
        "curved-in-$\\ell$ spectra; "
        "compared to the window-convolved prediction).  The $\\chi^2$ is "
        "that of the full bandpower vector against the target using the "
        "QML covariance $R^{-1}$.  Test 3 replaces the smooth variant by "
        "the \\emph{deviation} mode: the smooth \\texttt{pyccl} fiducial "
        "is kept inside the covariance and flat band deviations are "
        "fitted; the target is exactly zero, with no window or binning "
        "ambiguity.\n")
    out.append("\\begin{table}[h]\\centering\\small\n"
               "\\begin{tabular}{llrrrrr}\n\\toprule\n"
               "suite & variant & $\\langle\\chi^2\\rangle$ & dof & KS $p$ & "
               "pull mean & pull std \\\\\n\\midrule\n")
    for tag in ("val1", "val2", "val3"):
        s = jload(f"{tag}_summary.json")
        if s is None:
            continue
        vname = {"flat": "band-flat", "curved": "smooth",
                 "dev": "deviations (smooth fid.)"}
        for variant, r in s.items():
            out.append(
                f"{tag} & {vname.get(variant, variant)} & "
                f"{r['chi2_mean']:.1f} & {r['dof']} & "
                f"{r['ks_p']:.3f} & {r['pull_mean']:+.3f} & "
                f"{r['pull_std']:.3f} \\\\\n")
    out.append("\\bottomrule\n\\end{tabular}\n"
               "\\caption{Validation summary.  The KS $p$-value compares "
               "the per-realization $\\chi^2$ values to the "
               "$\\chi^2_{\\rm dof}$ distribution; pulls are per-band "
               "normalized residuals over all spectra, bands and "
               "realizations (expected: mean 0, std 1).}\n\\end{table}\n"
               "Of the six suite/variant combinations, five have "
               "$\\chi^2/\\mathrm{dof}$ and KS $p$-values fully consistent "
               "with the expected distribution; the val1 band-flat entry "
               "($p=0.009$) is a low but structureless outlier (the pull "
               "histogram is unit-width and unbiased), attributable to the "
               "particular 100-realization draw --- the same band-flat "
               "configuration passes cleanly in val2 ($p=0.085$) and val3 "
               "($p=0.77$), and at $N_{\\rm side}=16$ ($p=0.83$).\n")
    for tag, desc in setup.items():
        if jload(f"{tag}_summary.json") is None:
            continue
        out.append(f"\n\\subsection{{Test {tag[-1]}: {desc}}}\n")
        for variant in ("flat", "curved", "dev"):
            for kind, cap in [("spectra", "recovered bandpowers"),
                              ("chi2", "$\\chi^2$ distribution"),
                              ("pulls", "pull distribution")]:
                f = f"{tag}_{variant}_{kind}.png"
                if os.path.exists(os.path.join(REPORT, "figs", f)):
                    out.append(
                        "\\begin{figure}[h]\\centering"
                        f"\\includegraphics[width=.78\\textwidth]{{{f}}}"
                        f"\\caption{{{tag} "
                        f"({ {'flat': 'band-flat', 'curved': 'smooth', 'dev': 'deviations from smooth fiducial'}[variant] }"
                        f" input): {cap}.}}"
                        "\\end{figure}\n")
        out.append("\\clearpage\n")
    out.append(
        "\n\\subsection{Column-subsampled response (stochastic exact engine)}\n"
        "A third response engine solves only a random fraction $f$ of the "
        "mode columns of $MG$, stratified per $\\ell'$ and renormalized by "
        "$N_{\\ell'}/n_{\\ell'}$ (exactly unbiased; sampling without "
        "replacement makes it continuously exact as $f\\to1$).  Because the "
        "row index of $H$ is summed exactly for every solved column, the "
        "sampling error stays \\emph{local in bands}: "
        "$\\delta c_A/\\sigma_A \\sim \\mathrm{SNR}_A\\sqrt{\\rho(1-f)/n_A}$ "
        "with the band's own S/N, versus the coherent "
        "$\\sqrt{\\mathrm{SNR}^2_{\\rm tot}/N_{\\rm sims}}$ of the sims-MC "
        "engine.  Measured head-to-head ($N_{\\rm side}=8$, signal-dominated, "
        "$\\sim$110 solves each): frozen-response offsets of "
        "0.13--0.18$\\sigma$ (subsampled, $f=0.25$) versus "
        "0.7--1.6$\\sigma$ (sims-MC) --- a 6--9$\\times$ accuracy gain at "
        "equal cost, i.e.\\ $\\mathcal{O}(50\\times)$ fewer solves at equal "
        "accuracy.  This is the recommended engine at scales where the full "
        "exact run is unaffordable, combined with iteration for strongly "
        "signal-dominated bands.\n")
    out.append(
        "\n\\subsection{Radical compression (offset-lognormal likelihood)}\n"
        "Following Bond, Jaffe \\& Knox (2000), "
        "\\texttt{simaster.compress} reduces an estimate to the triplet "
        "$\\{\\hat c_b, x_b, F_{bb'}\\}$ with the likelihood Gaussian in "
        "$Z_b=\\ln(c_b+x_b)$ (auto-spectra; crosses stay Gaussian).  The "
        "x-factors generalize $x_\\ell=\\mathcal N_\\ell/B_\\ell^2$ via "
        "the exact identity $\\hat c + x = R^{-1}\\hat y$ (total power), "
        "i.e.\\ $x=R^{-1}n$ from the workspace response and noise bias.  "
        "Against the exact dense likelihood at $N_{\\rm side}=8$ the "
        "offset-lognormal is good to $|\\Delta(-2\\ln L)|\\lesssim1$ "
        "within $\\pm1.5\\sigma$ even in the lowest $\\Delta\\ell=5$ "
        "band and always beats the Gaussian out to $3\\sigma$; the far "
        "low-$C$ tail of the exact likelihood is steeper (use narrower "
        "low-$\\ell$ bands if deep lower tails matter).\n")
    out.append(
        "\n\\subsection{Optimality vs pseudo-$C_\\ell$}\n"
        "On 100 common realizations (spin-0 CMB-like field, $N_{\\rm side}=16$, "
        "same mask and bins; \\texttt{notebooks/simaster\\_vs\\_namaster.ipynb}) "
        "the empirical QML error bars are $0.38\\times$, $0.52\\times$ and "
        "$0.78\\times$ the NaMaster pseudo-$C_\\ell$ ones in the three lowest "
        "bands, converging to unity at $\\ell\\gtrsim25$, and the predicted "
        "QML covariance matches the empirical scatter at the few-percent "
        "level.\n")
    mc = jload("val1_mc_summary.json")
    if mc:
        out.append(
            "\n\\subsection{Exact vs Monte-Carlo response}\n"
            "Repeating test 1 with the MC engine "
            f"($N_{{\\rm sims}}=8192$) gives $\\langle\\chi^2\\rangle = "
            f"{mc['flat']['chi2_mean']:.1f}$ (dof {mc['flat']['dof']}) for "
            "the flat variant, illustrating the frozen-$\\hat R$ S/N "
            "penalty discussed in \\S\\ref{sec:fisher}; the two engines "
            "agree on the response matrix itself to the expected "
            "$\\sqrt{2/N_{\\rm sims}}$ accuracy.\n")

    vx = jload("val2x_summary.json")
    if vx:
        out.append(
            "\n\\subsection{Fisher three ways and x-factors (val2 setup)}\n"
            "Using the val2 configuration (CMB $T/Q/U$, anisotropic "
            f"longitude-strip noise, $N_{{\\rm side}}={vx['nside']}$, "
            f"$\\ell_{{\\max}}={vx['lmax']}$), the response matrix computed "
            "by the three engines agrees closely: relative to the exact "
            "$R^{-1}$, the corr-normalized residual r.m.s.\\ is "
            f"{vx['fisher_resid_rms_sub']:.3f} for column-subsampling "
            f"($f={vx['sub_frac']}$) and {vx['fisher_resid_rms_mc']:.3f} for "
            f"Monte Carlo ($N_{{\\rm sims}}={vx['nsims_mc']}$) "
            f"({vx['fisher_resid_rms_mc'] / vx['fisher_resid_rms_sub']:.1f}"
            "$\\times$ noisier at comparable cost), confirming "
            "\\S\\ref{sec:fishercost} at the validation scale "
            "(Fig.~\\ref{fig:val2x_fisher}).\n")
        out.append(
            "\\begin{figure}[h]\\centering"
            "\\includegraphics[width=\\textwidth]{val2x_fisher_3way.png}"
            "\\caption{Response (Fisher) matrix three ways at the val2 "
            "setup: (a) per-band TT errors overlay; (b) all $R^{-1}$ "
            "elements vs the exact engine; (c) residual distribution --- "
            "column-subsampling is several times tighter than Monte Carlo "
            "at comparable cost.}\\label{fig:val2x_fisher}\\end{figure}\n")
        out.append(
            "Expressing the bandpower likelihood with BJK x-factors "
            "($Z_b=\\ln(\\hat c_b+x_b)$, autos only) improves the agreement "
            "of the goodness-of-fit statistic with $\\chi^2_{\\rm dof}$: the "
            f"KS $p$-value rises from {vx['ks_gauss']:.3f} (plain Gaussian "
            f"$\\chi^2$) to {vx['ks_lognorm']:.3f} (offset-lognormal), the "
            "Gaussian's heavy upper tail (from upward auto-power "
            "fluctuations) being removed by the log transform "
            "(Fig.~\\ref{fig:val2x_chi2}).  The transform symmetrizes the "
            "signal-dominated bands and can mildly over-correct "
            "noise-dominated ones; with strong mask-induced band coupling a "
            "few realizations have $\\hat c_b+x_b$ driven near zero in some "
            "band (the offset-lognormal assumes weakly-coupled bands), "
            "producing the rare upper-$\\chi^2$ outliers noted on the "
            "figure.\n")
        out.append(
            "\\begin{figure}[h]\\centering"
            "\\includegraphics[width=.92\\textwidth]{val2x_xfactor_chi2.png}\\\\"
            "\\includegraphics[width=.92\\textwidth]"
            "{val2x_xfactor_mechanism.png}"
            "\\caption{Top: bandpower $\\chi^2$ against the true spectrum "
            "with and without x-factors.  Bottom: per-auto-band skewness "
            "before/after the $\\ln(c+x)$ transform (left; points below the "
            "line are symmetrized) and the standardized residual of the "
            "most signal-dominated band (right).}"
            "\\label{fig:val2x_chi2}\\end{figure}\n")
    return "".join(out)


def feasibility_section():
    b = jload("bench.json")
    if b is None:
        return "Benchmarks not yet run.\n"
    d, c, p = b["dense"], b["ducc"], b["projection_1024"]
    shtb = jload("bench_sht.json")
    note = ("naive $N_{\\rm side}^3$ extrapolation from 32 (pessimistic, "
            "overhead-dominated)")
    if shtb and "1024" in shtb:
        # replace the cubic extrapolation with measured nside=1024 SHTs:
        # one C-apply per RHS = (spin0 + spin2) x (synth + adjoint)
        s0, s2 = shtb["1024"]["spin0_B8"], shtb["1024"]["spin2_B8"]
        per_rhs = (s0["synth_s"] + s0["adjoint_s"]
                   + s2["synth_s"] + s2["adjoint_s"]) / 8.0
        p = dict(p)
        p["t_apply_s_extrapolated"] = per_rhs * d["batch"]
        p["t_solve_s"] = p["t_apply_s_extrapolated"] * c["cg_iters"]
        note = ("\\emph{measured} full-sky ducc transforms at "
                "$N_{\\rm side}=1024$, $\\ell_{\\max}=2048$ "
                f"({per_rhs:.1f}\\,s per RHS per $\\mathbb{{C}}$-apply on "
                "this CPU)")
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
cores) costs $\\sim${p['t_apply_s_extrapolated']:.0f}\\,s per batch of
{d['batch']} ({note}), i.e.\\ a CG solve of {d['batch']} RHS in
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


def fisher_methods_section():
    """Fisher engines compared + cost to not degrade a Planck-like analysis."""
    # per-solve cost: 2 SHT pairs (spin0 + spin2) per C-apply, ~100 CG iters
    shtb = jload("bench_sht.json")
    ncores = os.cpu_count() or 24
    if shtb and "1024" in shtb:
        s0, s2 = shtb["1024"]["spin0_B8"], shtb["1024"]["spin2_B8"]
        apply_core_s = (s0["synth_s"] + s0["adjoint_s"]
                        + s2["synth_s"] + s2["adjoint_s"]) / 8.0 * ncores
        src = "measured on this machine's 24-core CPU (ducc0)"
    else:
        apply_core_s = 26.0
        src = "estimated from typical ducc0 throughput (to be replaced by "\
              "the measured value once \\texttt{bench\\_sht\\_scaling.py} has run)"
    iters = 100
    solve_core_h = apply_core_s * iters / 3600.0

    # Planck-like budget numbers
    snr2_tot = 1.0e6
    eps = 0.1
    rho, fsky, dl = 0.1, 0.6, 30
    nbands = 2000 // dl
    ncols_cov = 250 * nbands * 3
    n_exact = 3 * (2001 ** 2)
    n_mc_naive = snr2_tot / eps ** 2
    n_mc_fid = nbands * 6 / eps ** 2          # SNR_eff^2 ~ n_bands(x6 spectra)
    f_noiter = rho * fsky / (2 * eps ** 2)
    f_noiter = f_noiter / (1 + f_noiter)

    def ch(n):  # core-hours -> pretty
        v = n * solve_core_h
        return f"{v:,.0f}".replace(",", r"\,")

    return f"""
The response (binned Fisher) matrix $R_{{AB}}=\\frac12\\Tr[MP_AMP_B]$ is the
only piece of the QML pipeline whose cost grows faster than one solve per
data vector, so the engine choice decides feasibility at high resolution.
\\simaster\\ implements three engines (\\S\\ref{{sec:fisher}}); their error
budgets differ qualitatively:

\\begin{{itemize}}
\\item \\textbf{{Exact}}: all $N_{{\\rm modes}}$ columns of $MG$ solved;
  deterministic. Cost $N_{{\\rm modes}}$ CG solves.
\\item \\textbf{{Column-subsampled exact}} (fraction $f$, stratified per
  $\\ell'$, $N_{{\\ell'}}/n_{{\\ell'}}$-renormalized; exactly unbiased): the
  row index of $H$ is summed exactly, so the response error is local in
  bands, $\\delta c_A/\\sigma_A \\simeq \\mathrm{{SNR}}_A
  \\sqrt{{\\rho(1-f)/n_A}}$, with $\\rho\\simeq0.1$ measured for the
  NaMaster mask.
\\item \\textbf{{Sims-MC}} ($\\mathrm{{cov}}[\\hat y]$ over $N_{{\\rm sims}}$
  fiducial sims): Wishart noise couples all bands,
  $\\delta c_A/\\sigma_A \\simeq \\sqrt{{\\mathrm{{SNR}}^2_{{\\rm tot}}/
  N_{{\\rm sims}}}}$ with $\\mathrm{{SNR}}^2_{{\\rm tot}}=c^{{\\sf T}}Rc$.
\\end{{itemize}}

\\paragraph{{Requirement.}} We demand that the Fisher-induced systematic not
dominate the error budget: offset $\\le \\epsilon\\,\\sigma_A$ per band with
$\\epsilon=0.1$ (error-bar inflation $<0.5\\%$); all counts below scale as
$1/\\epsilon^2$.

\\paragraph{{Planck-like configuration.}} $N_{{\\rm side}}=1024$,
$\\ell_{{\\max}}=2000$, $T/E/B$, $f_{{\\rm sky}}\\simeq{fsky}$,
TT signal-dominated to $\\ell\\sim1500$, hence
$\\mathrm{{SNR}}^2_{{\\rm tot}}\\approx10^6$ and
$N_{{\\rm modes}}=3(\\ell_{{\\max}}+1)^2\\approx1.2\\times10^7$.
One $\\C$-apply costs {apply_core_s:.0f} core-s ({src}); with
$\\sim$entire-mission anisotropic noise we budget {iters} CG iterations,
i.e.\\ {solve_core_h:.2f} core-h per solve.

\\begin{{table}}[h]\\centering\\small
\\begin{{tabular}}{{lrr}}
\\toprule
engine & solves needed & CPU cost [core-h] \\\\
\\midrule
exact (all columns) & $1.2\\times10^7$ & {ch(n_exact)} \\\\
sims-MC, no iteration & $\\mathrm{{SNR}}^2_{{\\rm tot}}/\\epsilon^2
  = 10^8$ & {ch(n_mc_naive)} \\\\
subsampled, no iteration & $f\\ge{f_noiter:.2f}$ of all columns &
  {ch(f_noiter * n_exact)} \\\\
\\textbf{{subsampled + iteration}} & $\\sim250$ cols/band
  $\\to {ncols_cov:,.0f}$\\hspace{{-2mm}} & \\textbf{{{ch(ncols_cov)}}} \\\\
sims-MC + iteration & $\\sim{n_mc_fid:,.0f}$ & {ch(n_mc_fid)} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Fisher cost for a Planck-like analysis at
$\\epsilon=0.1$.  ``+ iteration'' means the \\emph{{around-fiducial}}
estimator $\\hat c = c_{{\\rm fid}} + \\hat R^{{-1}}(\\hat y(d) -
\\langle\\hat y\\rangle_{{\\rm fid\\ sims}})$ with one Newton--Raphson
re-centering of the fiducial: the sim mean carries the \\emph{{exact}}
response of the filter, so the $\\hat R$ error multiplies
$(c-c_{{\\rm fid}})$ instead of $c$ (verified numerically: a 20\\%-wrong
fiducial deflates the subsampled offsets by the predicted factor 5;
the plain $\\hat R^{{-1}}(\\hat y-\\hat n)$ form does \\emph{{not}}
benefit --- its bias is $R^{{-1}}\\delta\\hat R\\,c$ for any fiducial).
The $1/\\epsilon^2\\approx100$ extra sims for
$\\langle\\hat y\\rangle$ are negligible.  The budget is then set by the
\\emph{{covariance}} accuracy of $R^{{-1}}$ ($\\sim$2\\%:
$n_A\\ge\\rho/2(0.01)^2\\approx250$ columns per band, $\\Delta\\ell={dl}$,
3 components).}}
\\end{{table}}

Without iteration, neither stochastic engine is affordable for
signal-dominated data: the subsampling fraction obeys
$f/(1-f)\\ge\\rho f_{{\\rm sky}}(S/(S+N))^2/2\\epsilon^2$ --- note the band
width cancels --- giving $f\\simeq{f_noiter:.2f}$, while sims-MC needs
$10^8$ solves.  With a good fiducial, the around-fiducial estimator and
one re-centering make the offset multiply $(c-c_{{\\rm fid}})$ instead of
$c$, and both engines drop to
$\\mathcal{{O}}(5\\times10^4)$ solves: about {ch(ncols_cov)} core-h
($\\approx${ncols_cov * solve_core_h / 24 / 256:.0f} days on a 256-core
farm), or $\\sim$1 week on a single A100-class GPU once a GPU SHT backend
exists ($\\sim$0.15\\,s per $\\C$-apply), with $\\sim$38\\,GB of GPU RAM at
batch 64.  The noise bias never matters: Rademacher probes have
per-probe scatter equal to the per-realization $\\sigma_A$, so
$1/\\epsilon^2=100$ probes suffice.  We therefore consider the Planck
configuration feasible with the subsampled engine + iteration on a
mid-size CPU farm today, and on a single node with a GPU SHT.
"""


with open(os.path.join(REPORT, "validation_section.tex"), "w") as f:
    f.write(validation_section())
with open(os.path.join(REPORT, "feasibility_section.tex"), "w") as f:
    f.write(feasibility_section())
with open(os.path.join(REPORT, "fisher_methods_section.tex"), "w") as f:
    f.write(fisher_methods_section())

for _ in range(3):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=REPORT, capture_output=True, text=True)
print("pdflatex rc:", r.returncode)
print(r.stdout[-600:] if r.returncode else "report/main.pdf built")
