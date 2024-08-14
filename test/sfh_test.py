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
def test_sfh_img(pred, sfh):
    fig, axs = plt.subplots(2, 2, sharex=True)

    # SFR
    axs[0, 0].plot(sfh.time, sfh.sfr.to(u.Msun / u.Gyr), label="SFR")
    axs[0, 0].axvline(
        np.average(sfh.time, weights=sfh.time_weights).value, c="k", ls="--"
    )
    axs[0, 0].axvline(pred.age, c="r", ls=":")
    axs[0, 0].set_title("SFR")

    # IMF slope
    axs[1, 0].plot(sfh.time, sfh.imf, label="IMF slope")
    axs[1, 0].axhline(np.average(sfh.imf, weights=sfh.sfr).value, c="k", ls="--")
    axs[1, 0].axhline(pred.imf_slope, c="r", ls=":")
    axs[1, 0].set_title("IMF slope")

    # Metallicity
    axs[0, 1].plot(sfh.time, sfh.met, label="[M/H]")
    axs[0, 1].axhline(np.average(sfh.met, weights=sfh.sfr).value, c="k", ls="--")
    axs[0, 1].axhline(pred.met, c="r", ls=":")
    axs[0, 1].set_title("Metallicity")

    # Alpha
    axs[1, 1].plot(sfh.time, sfh.alpha, label="alpha/Fe")
    axs[1, 1].axhline(np.average(sfh.alpha, weights=sfh.sfr).value, c="k", ls="--")
    axs[1, 1].axhline(pred.alpha, c="r", ls=":")
    axs[1, 1].set_title("[alpha/Fe]")

    axs[1, 0].set_xlabel("Gyr")
    axs[1, 1].set_xlabel("Gyr")

    return fig


@pytest.fixture
def pred(sfh, miles_ssp_basti):
    # And finally some predictions
    return miles_ssp_basti.from_sfh(sfh)


def test_predictions(pred, sfh):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = pred.mass_to_light(filters=filts, mass_in="star+remn")

    expected_imf = np.average(sfh.imf, weights=sfh.sfr).value
    expected_age = np.average(sfh.time, weights=sfh.sfr).value
    expected_alpha = np.average(sfh.alpha, weights=sfh.sfr).value
    expected_met = np.average(sfh.met, weights=sfh.sfr).value

    assert np.isclose(pred.age, expected_age, rtol=0.05)
    assert np.isclose(pred.met, expected_met, rtol=0.05)
    assert np.isclose(pred.alpha, expected_alpha, rtol=0.05)
    assert np.isclose(pred.imf_slope, expected_imf, rtol=0.05)
    assert np.isclose(outmls["SLOAN_SDSS.g"], np.array([2.95471517]))


@pytest.mark.mpl_image_compare
def test_spectra_img(pred):
    # And finally the spectra
    fig, ax = plt.subplots()
    ax.plot(pred.spectral_axis, pred.flux)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    return fig
