# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u

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
    assert np.isclose(miles_single.Mass_star_remn, 0.622357175092374)
    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outmls["star+remn"][k], rtol=1e-5, err_msg=k)


def test_ml_mass_invariant(miles_ssp):
    s0 = miles_ssp.interpolate(age=5.7, met=-0.45, imf_slope=1.3)
    s1 = miles_ssp.interpolate(age=5.7, met=-0.45, imf_slope=1.3, mass=1e10 * u.Msun)

    fnames = flib.search("SLOAN_SDSS.g")
    filts = flib.get(fnames)
    ml0 = s0.mass_to_light(filters=filts, mass_in="total")
    ml1 = s1.mass_to_light(filters=filts, mass_in="total")

    assert np.isclose(ml0["SLOAN_SDSS.g"], ml1["SLOAN_SDSS.g"])
