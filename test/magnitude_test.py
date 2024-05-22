# -*- coding: utf-8 -*-
import numpy as np

import pymiles.filter as flib
from pymiles.magnitudes import sun_magnitude


def test_solar_mags():
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    sun_mags = sun_magnitude(filters=filts, zeropoint="AB")
    ref = {
        "SLOAN_SDSS.g": 5.140807167815929,
        "SLOAN_SDSS.i": 4.536431419781877,
        "SLOAN_SDSS.r": 4.648245475828894,
        "SLOAN_SDSS.u": 6.390409355059589,
        "SLOAN_SDSS.z": 4.521180796814598,
    }
    for k in ref.keys():
        np.testing.assert_allclose(
            ref[k], sun_mags[k], rtol=1e-5, err_msg=k, verbose=True
        )
