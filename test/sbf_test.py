# -*- coding: utf-8 -*-

import numpy as np
from astropy import units as u

from milespy import SSPLibrary
from milespy.filter import get_filters


def test_sbf_magnitudes_v92():
    miles = SSPLibrary(
        source="MILES_SSP",
        version="9.2",
        isochrone="P",
        imf_type="bi",
        load_variance=True,
    )
    spec = miles.interpolate(age=5.7 * u.Gyr, met=0.0 * u.dex, imf_slope=1.3)
    filters = get_filters(["Generic_Bessell.U", "CFHT_CFH12k.B"])
    mags = spec.sbf_magnitudes(filters=filters, zeropoint="VEGA")

    assert np.all(np.isfinite(mags["Generic_Bessell.U"]))
    assert np.all(np.isfinite(mags["CFHT_CFH12k.B"]))
