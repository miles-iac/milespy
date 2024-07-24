# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np
import pytest

import pymiles.ls_indices as lslib
from pymiles.spectra import spectra
from pymiles.ssp_models import ssp_models as ssp


@pytest.fixture
def miles_ssp():
    return ssp(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
    )


@pytest.fixture
def miles_single(miles_ssp):
    return miles_ssp.interpolate(age=5.7, met=-0.45, imf_slope=1.3)


def test_cube_ls(miles_single):
    # Create a mock datacube
    wave = miles_single.spectral_axis
    flux = u.Quantity(np.random.random((30, 30, 4300)), unit=u.L_sun / u.AA)
    flux[...] = miles_single.flux

    cube = spectra(flux=flux, spectral_axis=wave)

    names = lslib.search("Fe.*")
    indeces = lslib.get(names)
    outls = cube.line_strength(indeces)

    assert outls["Fe4033"].shape == cube.dim
    np.testing.assert_allclose(outls["Fe4033"], 0.45146105, rtol=1e-5)
