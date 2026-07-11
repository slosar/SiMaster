# Changelog

## 0.2.0

- Gave SiMaster an explicit new release version, 0.2.0, in both package
  metadata and `simaster.__version__`.
- Replaced per-SHT `jax.pure_callback` traffic inside Almond-backed solves
  with a CuPy-native covariance, Woodbury preconditioner, and batched PCG.
  The RHS enters CuPy once through DLPack and the solution returns to JAX once;
  all CG iterations remain on the GPU.
- The device covariance supports mixed spin-0/spin-2 fields, masks and beams,
  scalar or block-diagonal pixel noise, finite-alpha templates, and an existing
  deflation space.
- Added a GPU regression comparing the device operator and solver against the
  established callback path.
- Added `scripts/bench_almond_device.py` to report callback/device covariance
  and full-CG timings, iteration counts, numerical differences, and speedups.
- Added the optional dependency group `simaster[almond]` requiring Almond 0.5.

NERSC A100 result at nside 128/lmax 383/batch 4: a covariance apply improves
from 59.46 ms to 4.183 ms (14.22x), and a converged 20-iteration PCG improves
from 1.200 s to 0.1867 s (6.43x). The operators differ by 4.9e-18 and the
solutions by 2.1e-16 relative. The CPU suite (48 tests), device operator/PCG,
and a complete exact-Fisher Almond QML smoke test pass.
