# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
from astropy import units as u
from packaging.version import Version

import pymiles.filter as flib
from pymiles import SFH
from pymiles import SSPLibrary


@pytest.fixture
def miles_ssp_basti():
    return SSPLibrary(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="T",
    )


@pytest.fixture
def sfh(miles_ssp_basti):
    times = np.unique(miles_ssp_basti.models.age) << u.Gyr
    sfh = SFH(times[times <= 13.5 * u.Gyr])

    # Then we define the SFR
    sfh.tau_sfr(start=11 * u.Gyr, tau=1.5 * u.Gyr)

    # Chemical and IMF evolution can also be included
    sfh.sigmoid_met(start=-2.1, end=0.2, tc=10 * u.Gyr, gamma=2.0 / u.Gyr)
    sfh.sigmoid_alpha(start=0.4, end=0.0, tc=10 * u.Gyr)
    sfh.linear_imf(start=0.5, end=2.3, t_start=11.5 * u.Gyr, t_end=9.0 * u.Gyr)

    return sfh


@pytest.mark.mpl_image_compare
@pytest.mark.skipif(
    Version(scipy.__version__) < Version("1.11"),
    reason="scipy.integrate.simpson results change",
)
def test_sfh_img(sfh):
    fig, ax = plt.subplots()
    ax.plot(sfh.time, sfh.imf, label="IMF slope")
    ax.plot(sfh.time, sfh.met, label="[M/H]")
    ax.plot(sfh.time, sfh.sfr.to(u.Msun / u.Gyr) * 10, label="SFR (scaled)")
    ax.set_xlabel("Look-back time")
    ax.legend()
    ax.set_xlim(0, 14)
    ax.set_ylim(-3, 6)
    return fig


@pytest.fixture
def pred(sfh, miles_ssp_basti):
    # And finally some predictions
    return miles_ssp_basti.from_sfh(sfh)


def test_predictions(pred):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = pred.mass_to_light(filters=filts, mass_in="star+remn")

    np.testing.assert_almost_equal(pred.age, 9.747229444806024)
    np.testing.assert_almost_equal(pred.met, -0.9777525480919624)
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
