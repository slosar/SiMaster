"""make_ivar for validation test 2: longitude strips, noise varying by a
factor of 2 (in rms) across the observed region.  Passed to val1_cmb.py via
--ivar-maker; returns the *relative* ivar pattern (multiplies 1/sigma_T^2).

    python scripts/val1_cmb.py --ivar-maker scripts/val2_ivar.py --tag val2
"""

import numpy as np
import healpy as hp


def make_ivar(nside, mask):
    npix = hp.nside2npix(nside)
    _, phi = hp.pix2ang(nside, np.arange(npix))
    # 8 longitude strips; rms alternates smoothly between sigma and 2*sigma
    # -> ivar between 1 and 1/4
    strip = np.floor(phi / (2 * np.pi) * 8).astype(int) % 2
    rms_factor = np.where(strip == 0, 1.0, 2.0)
    return 1.0 / rms_factor ** 2
