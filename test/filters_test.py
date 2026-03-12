# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pytest

import milespy.filter as flib


def test_nfilters():
    assert flib.nfilters == 153


@pytest.mark.mpl_image_compare
def test_plot_filter():
    filters = flib.get_filters(flib.search_filters("sdss"))
    fig, ax = plt.subplots()
    for f in filters:
        f.plot_transmissivity(ax)
    ax.legend()
    ax.set_ylabel("Transmissivity")
    ax.set_xlabel("Wavelength")
    return fig
