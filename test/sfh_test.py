# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest

import pymiles.filter as flib
from pymiles import SFH


@pytest.fixture
def sfh():
    # Initializing instance
    sfh = SFH(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="T",
    )

    # Let's play with the methods
    # First we select the age range we want
    sfh.set_time_range(start=13.5, end=0.01)

    # Then we define the SFR
    sfh.tau_sfr(start=11, tau=1.5, met=0.1)

    # Chemical and IMF evolution can also be included
    sfh.met_evol_sigmoid(start=-2.1, end=0.2, tc=10, gamma=2.0)
    sfh.alp_evol_sigmoid(start=0.4, end=0.0, tc=10)
    sfh.imf_evol_linear(start=0.5, end=2.3, t_start=11.5, t_end=9.0)

    return sfh


@pytest.mark.mpl_image_compare
def test_sfh_img(sfh):
    fig, ax = plt.subplots()
    ax.plot(sfh.time, sfh.imf_evol, label="IMF slope")
    ax.plot(sfh.time, sfh.met_evol, label="[M/H]")
    ax.plot(sfh.time, sfh.sfr * 10, label="SFR (scaled)")
    ax.set_xlabel("Look-back time")
    ax.legend()
    return fig


@pytest.fixture
def pred(sfh):
    # And finally some predictions
    return sfh.generate_spectra()


def test_predictions(pred):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = pred.mass_to_light(filters=filts, mass_in="star+remn")

    np.testing.assert_almost_equal(pred.meta["age"], np.array([9.747229444806024]))
    np.testing.assert_almost_equal(pred.meta["met"], np.array([-0.9777525480919624]))
    np.testing.assert_almost_equal(outmls["SLOAN_SDSS.g"], np.array([3.01242965]))


@pytest.mark.mpl_image_compare
def test_spectra_img(pred):
    # And finally the spectra
    fig, ax = plt.subplots()
    ax.plot(pred.spectral_axis, pred.flux)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.legend()
    return fig
