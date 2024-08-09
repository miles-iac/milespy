# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pytest


@pytest.mark.skip(reason="logrebin is NotImplemented")
@pytest.mark.mpl_image_compare
def test_logrebin(miles_single):
    s0 = miles_single
    s1 = s0.log_rebin(velscale=500)
    s2 = s1.log_unbin()

    f, ax = plt.subplots()

    ax.plot(s0.spectral_axis, s0.flux, "b", alpha=0.5, label="Original")
    ax.plot(s1.spectral_axis, s1.flux, "k", alpha=0.5, label="Log_rebin velscale=500")
    ax.plot(s2.spectral_axis, s2.flux, "r", alpha=0.5, label="Log_unbin")

    ax.legend(loc=0)

    return f


@pytest.mark.skip(reason="logrebin is NotImplemented")
@pytest.mark.mpl_image_compare
def test_logrebin_cube(miles_cube):
    cube2 = miles_cube.log_rebin()
    _ = cube2.log_unbin()
