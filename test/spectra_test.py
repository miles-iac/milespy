# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from specutils.manipulation import spectral_slab


@pytest.fixture
def emiles_single(miles_single):
    # This is really just MILES but with a mock variable LSF for testing
    miles_single.meta["lsf_fwhm"] = (
        np.linspace(1.0, 25.0, len(miles_single.meta["lsf_wave"])) * u.AA
    )
    return miles_single


def plot_lsf(spec):
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(spec.spectral_axis, spec.flux, "k")
    ax2.plot(spec.lsf_wave, spec.lsf_fwhm, "r")
    ax2.yaxis.label.set_color("red")
    ax2.tick_params(axis="y", colors="red")
    ax2.set_ylim(0, 30)
    ax2.set_xlim(spec.spectral_axis.value.min(), spec.spectral_axis.value.max())

    return f


@pytest.mark.mpl_image_compare
def test_lsf_basic(emiles_single):
    return plot_lsf(emiles_single)


@pytest.mark.mpl_image_compare
def test_convolve_constant(emiles_single):
    spec = emiles_single.convolve(10 * u.AA)
    assert spec.lsf_wave.min() >= 10.0 * u.AA
    return plot_lsf(spec)


@pytest.mark.mpl_image_compare
def test_convolve_array(emiles_single):
    lsf_wave = np.linspace(3.5e3, 7e3, 100) * u.AA
    lsf_fwhm = 30 * np.exp(-(((lsf_wave - 5.5e3 * u.AA) / (1e3 * u.AA)) ** 2)) * u.AA
    spec = emiles_single.convolve(lsf_fwhm, lsf_wave)
    assert np.isclose(lsf_fwhm.max(), 30 * u.AA, atol=1e-2)
    return plot_lsf(spec)


@pytest.mark.mpl_image_compare
def test_trim_after_convolve(emiles_single):
    spec0 = emiles_single.convolve(10 * u.AA)
    spec = spectral_slab(spec0, 4e3 * u.AA, 6e3 * u.AA)
    assert spec.lsf_wave.min() >= 10.0 * u.AA
    return plot_lsf(spec)


@pytest.mark.mpl_image_compare
def test_convolve_after_trim(emiles_single):
    spec0 = spectral_slab(emiles_single, 4e3 * u.AA, 6e3 * u.AA)
    spec = spec0.convolve(10 * u.AA)
    assert spec.lsf_wave.min() >= 10.0 * u.AA
    return plot_lsf(spec)


@pytest.mark.mpl_image_compare
def test_resample(miles_single):
    s0 = miles_single
    s1 = s0.resample(np.linspace(4000, 6000, 200) << u.AA)
    s2 = s0.resample(np.logspace(np.log10(4000), np.log10(6000), 50) << u.AA)

    f, ax = plt.subplots()

    ax.plot(s0.spectral_axis, s0.flux, "k", alpha=0.5, label="Original")
    ax.plot(s1.spectral_axis, s1.flux, "b", alpha=0.5, label=r"10 $\AA$")
    ax.plot(s2.spectral_axis, s2.flux, "r.-", alpha=0.5, label="Log resample")

    ax.legend(loc=0)

    return f


@pytest.mark.mpl_image_compare
def test_resample_cube(miles_cube):
    s0 = miles_cube
    s1 = s0.resample(np.linspace(4000, 6000, 200) << u.AA)
    s2 = s0.resample(np.logspace(np.log10(4000), np.log10(6000), 50) << u.AA)

    f, ax = plt.subplots()

    ax.plot(
        s0.spectral_axis, s0.flux.mean(axis=(0, 1)), "k", alpha=0.5, label="Original"
    )
    ax.plot(
        s1.spectral_axis, s1.flux.mean(axis=(0, 1)), "b", alpha=0.5, label=r"10 $\AA$"
    )
    ax.plot(
        s2.spectral_axis,
        s2.flux.mean(axis=(0, 1)),
        "r.-",
        alpha=0.5,
        label="Log resample",
    )

    ax.legend(loc=0)

    return f
