# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from milespy import SSPLibrary


@pytest.mark.mpl_image_compare
def test_ssp_interp_alpha():
    # This test reproduces Fig 10 of Vazdekis et al. 2015
    miles_ssp = SSPLibrary(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="T",
    )
    fig, ax = plt.subplots()

    # 12 Gyr
    enhanced = miles_ssp.interpolate(
        age=12.0 * u.Gyr, met=0.0 * u.dex, imf_slope=1.3, alpha=0.4 * u.dex
    )
    base = miles_ssp.interpolate(age=12.0 * u.Gyr, met=0.0 * u.dex, imf_slope=1.3)
    ax.plot(base.spectral_axis, enhanced.flux / base.flux, label="Age = 12 Gyr", c="k")

    # 2 Gyr
    enhanced = miles_ssp.interpolate(
        age=2.0 * u.Gyr, met=0.0 * u.dex, imf_slope=1.3, alpha=0.4 * u.dex
    )
    base = miles_ssp.interpolate(age=2.0 * u.Gyr, met=0.0 * u.dex, imf_slope=1.3)
    ax.plot(base.spectral_axis, enhanced.flux / base.flux, label="Age = 2 Gyr", c="g")

    ax.axhline(1.0, c="k")
    ax.legend(loc=0)
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0.9, 2.0)
    return fig


@pytest.mark.mpl_image_compare
def test_ssp_interp_closest_img(miles_ssp):
    close = miles_ssp.closest(
        age=[2.57, 3.8, 11.3] << u.Gyr,
        met=[0.193, 0.042, -0.412] << u.dex,
        imf_slope=[1.124, 1.104, 1.206] << u.dimensionless_unscaled,
    )
    interp = miles_ssp.interpolate(
        age=[2.57, 3.8, 11.3] << u.Gyr,
        met=[0.193, 0.042, -0.412] << u.dex,
        imf_slope=[1.124, 1.104, 1.206] << u.dimensionless_unscaled,
    )

    fig, ax = plt.subplots()
    c = ["r", "g", "b"]
    for i in range(3):
        ax.plot(interp[i].spectral_axis, interp[i].flux, ls="-", c=c[i])
        ax.plot(close[i].spectral_axis, close[i].flux, ls="-", c=c[i], alpha=0.5)

    return fig


@pytest.mark.mpl_image_compare
def test_ssp_interp_img(miles_ssp):
    miles_1 = miles_ssp.interpolate(age=5.7 * u.Gyr, met=-0.45 * u.dex, imf_slope=1.3)
    # Also get the closest ones, which should be the base for the interpolation
    miles_vertices = miles_ssp.interpolate(
        age=5.7 * u.Gyr, met=-0.45 * u.dex, imf_slope=1.3, closest=True, simplex=True
    )
    fig, ax = plt.subplots()
    for i in range(miles_vertices.nspec):
        ax.plot(
            miles_vertices.spectral_axis,
            miles_vertices.flux[i, :],
            alpha=0.3,
            label=f"Vertex (age={miles_vertices.age[i]}, "
            f"met={miles_vertices.met[i]})",
        )
    ax.plot(
        miles_1.spectral_axis,
        miles_1.flux,
        alpha=0.5,
        c="k",
        label=f"Interpolated (age={miles_1.age}, " f"met={miles_1.met})",
    )
    ax.legend(loc=0)
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0, 6e-5)
    return fig


def test_ssp_trim(miles_ssp):
    miles_ssp.trim(4000 * u.AA, 5000 * u.AA)
    assert miles_ssp.models.spectral_axis.min() > 4000 * u.AA
    assert miles_ssp.models.spectral_axis.max() < 5000 * u.AA


def test_ssp_out_of_range(miles_ssp):
    with pytest.raises(ValueError):
        miles_ssp.in_range(age_lims=[25.0, 30.0] << u.Gyr, met_lims=[4, 5] << u.dex)


def test_ssp_in_range(miles_ssp):
    miles_1 = miles_ssp.in_range(
        age_lims=[15.0, 20.0] << u.Gyr, met_lims=[-0.1, 0.5] << u.dex
    )
    assert miles_1.nspec == 56
    assert miles_1.age.min() > 15.0 * u.Gyr
    assert miles_1.age.max() < 20.0 * u.Gyr
    assert miles_1.met.max() < 0.5 * u.dex
    assert miles_1.met.min() > -0.1 * u.dex


def test_assert_alpha_in_fix(miles_ssp):
    with pytest.raises(ValueError):
        miles_ssp.in_list(
            age=[0.2512, 0.0708, 1.4125] << u.Gyr,
            met=[0.22, 0.0, -1.71] << u.dex,
            imf_slope=[1.3, 1.3, 1.3] << u.dimensionless_unscaled,
            alpha=[0.0, 0.4, 0.0] << u.dex,
        )


def test_ssp_in_list(miles_ssp):
    miles_1 = miles_ssp.in_list(
        age=[0.2512, 0.0708, 1.4125] << u.Gyr,
        met=[0.22, 0.0, -1.71] << u.dex,
        imf_slope=[1.3, 1.3, 1.3] << u.dimensionless_unscaled,
        mass=[0.5, 2.0, 10.0] << u.Msun,
    )
    assert miles_1.age.shape == (3,)
    assert np.array_equal(miles_1.age, np.array([0.2512, 0.0708, 1.4125]) << u.Gyr)
    assert np.array_equal(miles_1.met, np.array([0.22, 0.0, -1.71]) << u.dex)
    assert np.array_equal(miles_1.imf_slope, np.array([1.3, 1.3, 1.3]))
    assert np.array_equal(miles_1.mass, np.array([0.5, 2.0, 10.0]) << u.Msun)


def test_ssp_not_in_list(miles_ssp):
    with pytest.raises(ValueError):
        _ = miles_ssp.in_list(
            age=[1e6, 1e6, 1e6] << u.Gyr,
            met=[1e6, 1e6, 1e6] << u.dex,
            imf_slope=[1e6, 1e6, 1e6] << u.dimensionless_unscaled,
        )


def test_ssp_interp(miles_single):
    assert miles_single.age == 5.7 * u.Gyr
    assert miles_single.met == -0.45 * u.dex
    assert miles_single.imf_slope == 1.3


def test_ssp_interp_wrong_shape(miles_ssp):
    with pytest.raises(ValueError):
        miles_ssp.interpolate(age=[10.0, 12.0] << u.Gyr, met=[-0.3, 0.0, 0.1] << u.dex)


def test_ssp_interp_array_age_met(miles_ssp):
    # As there are multiple IMFs available, this should fail
    with pytest.raises(ValueError):
        miles_ssp.interpolate(age=[10.0, 12.0] << u.Gyr, met=[-0.3, 0.0] << u.dex)


def test_ssp_interp_array_age_met_alpha(miles_ssp):
    # As there are multiple IMFs available, this should fail
    with pytest.raises(ValueError):
        miles_ssp.interpolate(
            age=[10.0, 12.0] << u.Gyr,
            met=[-0.3, 0.0] << u.dex,
            alpha=[0.1, 0.3] << u.dex,
        )


def test_ssp_interp_array_age_met_alpha_imf_fix(miles_ssp):
    out = miles_ssp.interpolate(
        age=[10.0, 12.0] << u.Gyr,
        met=[-0.3, 0.0] << u.dex,
        alpha=[0.1, 0.3] << u.dex,
        imf_slope=[1.2, 1.3] << u.dimensionless_unscaled,
    )
    assert np.array_equal(out.age, np.array([10.0, 12.0]) << u.Gyr)
    assert np.array_equal(out.met, np.array([-0.3, 0.0]) << u.dex)
    print(out.alpha)
    assert np.array_equal(out.alpha, np.array([0.1, 0.3]) << u.dex)
    assert np.array_equal(out.imf_slope, np.array([1.2, 1.3]))


def test_ssp_interp_array_age_met_alpha_imf(miles_ssp):
    out = miles_ssp.interpolate(
        age=[10.0, 12.0] << u.Gyr,
        met=[-0.3, 0.0] << u.dex,
        alpha=[0.1, 0.3] << u.dex,
        imf_slope=[1.25, 1.3] << u.dimensionless_unscaled,
    )
    assert np.array_equal(out.age, np.array([10.0, 12.0]) << u.Gyr)
    assert np.array_equal(out.met, np.array([-0.3, 0.0]) << u.dex)
    assert np.array_equal(out.alpha, np.array([0.1, 0.3]) << u.dex)
    assert np.array_equal(out.imf_slope, np.array([1.25, 1.3]))


def test_ssp_interp_array_age_met_imf(miles_ssp):
    out = miles_ssp.interpolate(
        age=[10.0, 11.0] << u.Gyr,
        met=[-0.3, 0.0] << u.dex,
        imf_slope=[1.2, 1.3] << u.dimensionless_unscaled,
    )
    assert np.array_equal(out.age, np.array([10.0, 11.0]) << u.Gyr)
    assert np.array_equal(out.met, np.array([-0.3, 0.0]) << u.dex)
    assert np.array_equal(out.imf_slope, np.array([1.2, 1.3]))


def test_ssp_interp_array_age_met_imf_mass(miles_ssp):
    out = miles_ssp.interpolate(
        age=[10.0, 11.0] << u.Gyr,
        met=[-0.3, 0.0] << u.dex,
        imf_slope=[1.2, 1.3] << u.dimensionless_unscaled,
        mass=[2.0, 3.0] << u.Msun,
    )
    assert np.array_equal(out.age, np.array([10.0, 11.0]) << u.Gyr)
    assert np.array_equal(out.met, np.array([-0.3, 0.0]) << u.dex)
    assert np.array_equal(out.imf_slope, np.array([1.2, 1.3]))
    assert np.array_equal(out.mass, np.array([2.0, 3.0]) << u.Msun)


# @pytest.fixture
# def miles_tuned(miles_single):
#     tab = ascii.read("./milespy/config_files/MUSE-WFM.lsf")
#     lsf_wave = tab["Lambda"]
#     lsf = tab["FWHM"] * 0.25

#     return miles_single.tune_spectra(
#         wave_lims=[4750.0, 5500.0],
#         dwave=1.0,
#         sampling="lin",
#         redshift=0.0,
#         lsf_flag=True,
#         lsf_wave=lsf_wave,
#         lsf=lsf,
#     )


# @pytest.mark.mpl_image_compare
# def test_tuned_spectra(miles_single, miles_tuned):
#     fig, ax = plt.subplots()
#     ax.plot(miles_single.wave, miles_single.spec)
#     ax.plot(miles_tuned.wave, miles_tuned.spec, "k")
#     ax.set_xlim(3500, 7500)
#     ax.set_ylim(0, 6e-5)
#     return fig


@pytest.mark.mpl_image_compare
def test_ssp_spectra(miles_single):
    fig, ax = plt.subplots()
    ax.plot(miles_single.spectral_axis, miles_single.flux)
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0, 6e-5)
    return fig


def test_ssp_mass(miles_ssp):
    light = miles_ssp.interpolate(
        age=5.7 * u.Gyr, met=-0.45 * u.dex, imf_slope=1.3, mass=1.0 * u.Msun
    )
    heavy = miles_ssp.interpolate(
        age=5.7 * u.Gyr, met=-0.45 * u.dex, imf_slope=1.3, mass=2.0 * u.Msun
    )
    ratio = light / heavy

    assert np.allclose(ratio.data, 0.5)
    assert light.mass.value == 1
    assert light.mass.unit == u.Msun
    assert heavy.mass.value == 2
    assert heavy.mass.unit == u.Msun


def test_ssp_units(miles_ssp):
    assert miles_ssp.models.age.unit == u.Gyr
    assert miles_ssp.models.alpha.unit == u.dex
    assert miles_ssp.models.met.unit == u.dex
    assert miles_ssp.models.imf_slope.unit == u.dimensionless_unscaled
