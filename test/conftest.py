# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np
import pytest

from pymiles import SSPLibrary
from pymiles.spectra import Spectra


@pytest.fixture
def miles_ssp():
    return SSPLibrary(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
    )


@pytest.fixture
def emiles_ssp():
    return SSPLibrary(
        source="EMILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
    )


@pytest.fixture
def miles_single(miles_ssp):
    return miles_ssp.interpolate(age=5.7, met=-0.45, imf_slope=1.3)


@pytest.fixture
def emiles_single(emiles_ssp):
    return emiles_ssp.interpolate(age=1.0, met=-0.45, imf_slope=1.3)


@pytest.fixture
def miles_cube(miles_single):
    wave = miles_single.spectral_axis
    flux = u.Quantity(np.random.random((30, 30, 4300)), unit=u.L_sun / u.AA)
    flux[...] = miles_single.flux

    return Spectra(flux=flux, spectral_axis=wave)
