# -*- coding: utf-8 -*-
import pytest

from pymiles.ssp_models import ssp_models as ssp


@pytest.fixture
def miles_ssp():
    return ssp(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
    )


@pytest.fixture
def miles_single(miles_ssp):
    return miles_ssp.interpolate(age=5.7, met=-0.45, imf_slope=1.3)
