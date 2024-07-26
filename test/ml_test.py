# -*- coding: utf-8 -*-
import numpy as np

import pymiles.filter as flib


def test_ml(miles_single):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = miles_single.mass_to_light(filters=filts, mass_in=["star+remn", "total"])
    ref = {
        "SLOAN_SDSS.g": np.array([1.98712619]),
        "SLOAN_SDSS.i": np.array([np.nan]),
        "SLOAN_SDSS.r": np.array([1.67578224]),
        "SLOAN_SDSS.u": np.array([np.nan]),
        "SLOAN_SDSS.z": np.array([np.nan]),
    }
    assert np.allclose(
        miles_single.meta["Mass_star_remn"], np.array([0.622357175092374])
    )
    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outmls["star+remn"][k], rtol=1e-5, err_msg=k)
