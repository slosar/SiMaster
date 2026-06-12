# Migrating from NaMaster

SiMaster deliberately mirrors the pymaster workflow:

| pymaster | simaster | notes |
|---|---|---|
| `nmt.NmtField(mask, [m])` | `sm.Field(mask, [m], ivar=ivar)` | `ivar` is **mandatory**: QML needs a noise model. |
| `nmt.NmtField(mask, [q, u])` | `sm.Field(mask, [q, u], spin=2, ivar=ivar)` | spectra are E/B, as in NaMaster. |
| `templates=[[t]]` | `templates=[t]` | same deprojection idea; see below. |
| `beam=bl` | `beam=bl` | |
| `nmt.NmtBin.from_nside_linear(nside, nlb)` | `sm.Bins.from_nside_linear(nside, nlb)` | |
| `nmt.NmtBin.from_edges(lo, hi)` | `sm.Bins.from_edges(lo, hi)` | **hi is inclusive** in simaster. |
| `nmt.NmtWorkspace` + `compute_coupling_matrix` | `sm.QMLWorkspace(fields, bins, cl_fid)` | the QML analogue precomputes the filter + response. |
| `nmt.compute_full_master(f1, f2, b)` | `sm.compute_full_master(f1, f2, b, cl_guess=...)` | same output ordering ([c00] / [TE,TB] / [EE,EB,BE,BB]). |
| `cl_noise` argument | — | the noise bias is computed internally from `ivar` and subtracted. |
| `w.decouple_cell(...)` | done internally | `estimate()` returns decoupled bandpowers directly. |
| bandpower windows `w.get_bandpower_windows()` | `QMLWorkspace.window_functions()` / `predict(cl_th)` | |

Key conceptual differences:

1. **A fiducial spectrum is required** (`cl_fid` / `cl_guess`). QML filters
   the data by the inverse fiducial covariance. Estimates are unbiased for
   any reasonable guess; error bars are optimal when the guess matches the
   truth (use `iterate()` if you care).
2. **Errors come for free.** `result.cov` is the bandpower covariance
   (`R⁻¹`); no Gaussian-covariance workspace needed.
3. **Mode coupling is removed exactly**, not via the pseudo-Cl coupling
   matrix; on large scales the error bars will be noticeably smaller than
   NaMaster's for the same data.
4. **Cost.** QML is for large scales. nside ≤ 64 runs in minutes on a small
   GPU; nside = 1024 is a cluster-GPU exercise (see the report). For high-l
   work, keep using NaMaster — the two share binning conventions so hybrid
   analyses are straightforward.
