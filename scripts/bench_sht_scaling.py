#!/usr/bin/env python
"""Measure ducc0 SHT cost directly at large nside (CPU-only, minutes).

The covariance application is 2 SHT round-trips per field (adjoint
synthesis + synthesis); this measures them at nside = 256..1024 so the
nside=1024 feasibility projection rests on data, not on cubic
extrapolation from nside=32 (which is overhead-dominated and pessimistic).
"""

import json
import os
import time

import numpy as np

from val_common import DATADIR
from simaster.utils import RealAlmIndex
from simaster import sht

out = {}
for nside in (256, 512, 1024):
    lmax = 2 * nside
    idx = RealAlmIndex(2, lmax)
    npix = 12 * nside ** 2
    obs = np.arange(npix)  # full sky (upper bound; cut sky only cheaper)
    res = {}
    for spin in (0, 2):
        op = sht.RealSHT(nside, idx, spin, obs)
        for B in (1, 8):
            a = np.random.default_rng(0).normal(size=(op.ncol, B))
            t0 = time.time()
            m = op.synth(a)
            t_syn = time.time() - t0
            t0 = time.time()
            op.adjoint(m)
            t_adj = time.time() - t0
            res[f"spin{spin}_B{B}"] = dict(synth_s=t_syn, adjoint_s=t_adj)
            print(f"nside={nside} spin={spin} B={B}: "
                  f"synth {t_syn:.2f}s adjoint {t_adj:.2f}s")
    out[str(nside)] = res

with open(os.path.join(DATADIR, "bench_sht.json"), "w") as f:
    json.dump(out, f, indent=2)
print("saved bench_sht.json")
