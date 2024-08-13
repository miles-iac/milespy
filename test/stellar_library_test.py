# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pymiles import StellarLibrary


@pytest.fixture
def lib():
    return StellarLibrary(source="MILES_STARS", version="9.1")


def test_get_starname(lib):
    assert lib.get_starname(id=100) == ["HD017382"]


def test_get_starname_multiple(lib):
    tmp = lib.get_starname(id=[100, 101])
    assert tmp[0] == "HD017382"
    assert tmp[1] == "HD017548"


def test_search_by_id(lib):
    tmp = lib.search_by_id(id=100)
    assert tmp.starname == "HD017382"


def test_search_by_id_multiple(lib):
    tmp = lib.search_by_id(id=[100, 101])
    assert tmp.starname[0] == "HD017382"
    assert tmp.starname[1] == "HD017548"
    assert tmp.nspec == 2


@pytest.mark.mpl_image_compare
def test_search_by_id_img(lib):
    tmp = lib.search_by_id(id=100)
    fig, ax = plt.subplots()
    ax.plot(tmp.spectral_axis, tmp.flux[0])
    ax.set_title(tmp.starname)
    return fig


def test_stars_in_range(lib):
    tmp = lib.in_range(
        teff_lims=[4500.0, 5000.0], logg_lims=[2.0, 2.5], FeH_lims=[0.0, 0.2]
    )
    assert tmp.data.shape == (14, 4367)
    assert tmp.teff.min() >= 4500.0
    assert tmp.teff.max() <= 5000.0
    assert tmp.logg.min() >= 2.0
    assert tmp.logg.max() <= 2.5
    assert tmp.FeH.min() >= 0.0
    assert tmp.FeH.max() <= 0.2


def test_search_closest(lib):
    # Search by params (Gets the closest spectra to those params)
    tmp = lib.closest(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    assert tmp.id == 743
    assert tmp.teff == 5041.0
    assert tmp.logg == 3.04
    assert tmp.FeH == -0.04
    assert np.isnan(tmp.MgFe)  # is this ok?


@pytest.mark.mpl_image_compare
def test_search_closest_img(lib):
    fig, ax = plt.subplots()
    tmp = lib.closest(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    ax.plot(tmp.spectral_axis, tmp.flux)
    ax.set_title(tmp.starname)
    return fig


@pytest.mark.mpl_image_compare
def test_interpolated_spectrum(lib):
    # Get spectrum by params (gets interpolated spectrum for those params)
    tmp1 = lib.closest(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    tmp2 = lib.interpolate(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    fig, ax = plt.subplots()
    ax.plot(tmp1.spectral_axis, tmp1.flux, label="Closest star")
    ax.plot(tmp2.spectral_axis, tmp2.flux, label="Interpolated star")
    ax.legend()
    return fig


def test_interp_array_wrong_shape(lib):
    with pytest.raises(ValueError):
        lib.interpolate(teff=[10.0, 12.0], logg=[-0.3, 0.0], FeH=[0.0])
