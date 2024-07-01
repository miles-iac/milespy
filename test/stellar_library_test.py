# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pymiles.stellar_library import stellar_library as stl


@pytest.fixture
def lib():
    return stl(source="MILES_STARS", version="9.1")


def test_get_starname(lib):
    assert lib.get_starname(id=100) == ["HD017382"]


def test_get_starname_multiple(lib):
    tmp = lib.get_starname(id=[100, 101])
    assert tmp[0] == "HD017382"
    assert tmp[1] == "HD017548"


def test_search_by_id(lib):
    tmp = lib.search_by_id(id=100)
    assert tmp.starname == ["HD017382"]
    assert tmp.wave_init == 3500.0
    assert tmp.wave_last == 7429.4


def test_search_by_id_multiple(lib):
    tmp = lib.search_by_id(id=[100, 101])
    assert tmp.starname[0] == "HD017382"
    assert tmp.starname[1] == "HD017548"
    assert tmp.nspec == 2


@pytest.mark.mpl_image_compare
def test_search_by_id_img(lib):
    tmp = lib.search_by_id(id=100)
    fig, ax = plt.subplots()
    ax.plot(tmp.wave, tmp.spec)
    ax.set_title(tmp.starname[0])
    return fig


def test_stars_in_range(lib):
    tmp = lib.get_stars_in_range(
        teff_lims=[4500.0, 5000.0], logg_lims=[2.0, 2.5], FeH_lims=[0.0, 0.2]
    )
    assert tmp.spec.shape == (4367, 14)
    assert tmp.teff.min() >= 4500.0
    assert tmp.teff.max() <= 5000.0
    assert tmp.logg.min() >= 2.0
    assert tmp.logg.max() <= 2.5
    assert tmp.FeH.min() >= 0.0
    assert tmp.FeH.max() <= 0.2


def test_search_closest(lib):
    # Search by params (Gets the closest spectra to those params)
    tmp = lib.search_by_params(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    assert tmp.id == 743
    assert tmp.teff == 5041.0
    assert tmp.logg == 3.04
    assert tmp.FeH == -0.04
    assert np.isnan(tmp.MgFe)  # is this ok?


@pytest.mark.mpl_image_compare
def test_search_closest_img(lib):
    fig, ax = plt.subplots()
    tmp = lib.search_by_params(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    ax.plot(tmp.wave, tmp.spec)
    ax.set_title(tmp.starname)
    return fig


@pytest.mark.mpl_image_compare
def test_interpolated_spectrum(lib):
    # Get spectrum by params (gets interpolated spectrum for those params)
    # Pablo had some prints inside this functions?..
    tmp1 = lib.search_by_params(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    tmp2 = lib.get_spectrum_by_params_delaunay(teff=5000.0, logg=3.0, FeH=0.0, MgFe=0.0)
    fig, ax = plt.subplots()
    ax.plot(tmp1.wave, tmp1.spec, label="Closest star")
    ax.plot(tmp2.wave, tmp2.spec, label="Interpolated star")
    ax.legend()
    return fig
