# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np

import pymiles.filter as flib
from pymiles.magnitudes import sun_magnitude
from pymiles.spectra import Spectra


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

    cube = Spectra(flux=flux, spectral_axis=wave)

    fnames = flib.search("sdss.r")
    filts = flib.get(fnames)
    mags = cube.magnitudes(filters=filts)

    assert mags["SLOAN_SDSS.r"].shape == cube.dim


def test_mags(miles_single):
    # Compute mags
    fnames = flib.search("sdss")
    filts = flib.get(fnames)
    outmags = miles_single.magnitudes(filters=filts, zeropoint="AB")
    assert np.allclose(outmags["OAJ_JPAS.gSDSS"], np.array([6.36344262]))
    assert np.allclose(outmags["OAJ_JPAS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPAS.rSDSS"], np.array([5.70916609]))
    assert np.allclose(outmags["OAJ_JPLUS.gSDSS"], np.array([6.34960724]))
    assert np.allclose(outmags["OAJ_JPLUS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPLUS.rSDSS"], np.array([5.7056477]))
    assert np.allclose(outmags["OAJ_JPLUS.zSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.g"], np.array([6.40127153]))
    assert np.allclose(outmags["SLOAN_SDSS.i"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.r"], np.array([5.72369018]))
    assert np.allclose(outmags["SLOAN_SDSS.u"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.z"], np.array([np.nan]), equal_nan=True)
