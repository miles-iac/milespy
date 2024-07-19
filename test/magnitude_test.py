# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np

import pymiles.filter as flib
from pymiles.magnitudes import sun_magnitude
from pymiles.spectra import spectra


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
        np.testing.assert_allclose(ref[k], sun_mags[k], rtol=1e-5, err_msg=k)


def test_cube_mags():
    # Create a mock datacube
    wave = u.Quantity(np.linspace(3000.0, 7000.0, 100), unit=u.AA)
    flux = u.Quantity(np.random.random((30, 30, 100)), unit=u.L_sun / u.AA)

    cube = spectra(flux=flux, spectral_axis=wave)

    fnames = flib.search("sdss.r")
    filts = flib.get(fnames)
    mags = cube.magnitudes(filters=filts)

    assert mags["SLOAN_SDSS.r"].shape == cube.dim
