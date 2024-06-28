# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import ascii

import pymiles.filter as flib
from pymiles.ssp_models import ssp_models as ssp


@pytest.fixture
def miles_ssp():
    return ssp(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
        show_tree=False,
    )


@pytest.fixture
def miles_single(miles_ssp):
    return miles_ssp.get_ssp_by_params(age=5.7, met=-0.45, imf_slope=1.3)


@pytest.mark.mpl_image_compare
def test_ssp_by_params_alpha():
    # This test reproduces Fig 10 of Vazdekis et al. 2015
    miles_ssp = ssp(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="T",
        show_tree=False,
    )
    fig, ax = plt.subplots()

    # 12 Gyr
    enhanced = miles_ssp.get_ssp_by_params(age=12.0, met=0.0, imf_slope=1.3, alpha=0.4)
    base = miles_ssp.get_ssp_by_params(age=12.0, met=0.0, imf_slope=1.3)
    ax.plot(base.wave, enhanced.spec / base.spec, label="Age = 12 Gyr", c="k")

    # 2 Gyr
    enhanced = miles_ssp.get_ssp_by_params(age=2.0, met=0.0, imf_slope=1.3, alpha=0.4)
    base = miles_ssp.get_ssp_by_params(age=2.0, met=0.0, imf_slope=1.3)
    ax.plot(base.wave, enhanced.spec / base.spec, label="Age = 2 Gyr", c="g")

    ax.axhline(1.0, c="k")
    ax.legend(loc=0)
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0.9, 2.0)
    return fig


@pytest.mark.mpl_image_compare
def test_ssp_by_params_img(miles_ssp):
    miles_1 = miles_ssp.get_ssp_by_params(age=5.7, met=-0.45, imf_slope=1.3)
    # Also get the closest ones, which should be the base for the interpolation
    miles_vertices = miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, closest=True
    )
    fig, ax = plt.subplots()
    for i in range(miles_vertices.nspec):
        ax.plot(
            miles_vertices.wave,
            miles_vertices.spec[:, i],
            alpha=0.3,
            label=f"Vertex (age={miles_vertices.age[i]}, met={miles_vertices.met[i]})",
        )
    ax.plot(
        miles_1.wave,
        miles_1.spec,
        alpha=0.5,
        c="k",
        label=f"Interpolated (age={miles_1.age[0]}, met={miles_1.met[0]})",
    )
    ax.legend(loc=0)
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0, 6e-5)
    return fig


def test_ssp_in_range(miles_ssp):
    miles_1 = miles_ssp.get_ssp_in_range(age_lims=[17.0, 20.0], met_lims=[0.1, 0.5])
    assert miles_1.age.shape == (14,)
    # Is it ok that they all have the same age and metallicity?
    assert miles_1.age.min() == 17.7828
    assert miles_1.age.max() == 17.7828
    assert miles_1.met.max() == 0.22
    assert miles_1.met.min() == 0.22
    assert np.array_equal(
        miles_1.Mass_star_remn,
        np.array(
            [
                0.3354,
                0.3414,
                0.3983,
                0.4725,
                0.612,
                0.6999,
                0.8023,
                0.8508,
                0.901,
                0.924,
                0.9481,
                0.9594,
                0.9716,
                0.9775,
            ]
        ),
    )


def test_assert_alpha_in_fix(miles_ssp):
    with pytest.raises(ValueError):
        miles_ssp.get_ssp_in_list(
            age_list=[0.2512, 0.0708, 1.4125],
            met_list=[0.22, 0.0, -1.71],
            imf_slope_list=[1.3, 1.3, 1.3],
            alpha_list=[0.0, 0.4, 0.0],
        )


def test_ssp_in_list(miles_ssp):
    miles_1 = miles_ssp.get_ssp_in_list(
        age_list=[0.2512, 0.0708, 1.4125],
        met_list=[0.22, 0.0, -1.71],
        imf_slope_list=[1.3, 1.3, 1.3],
    )
    assert miles_1.age.shape == (3,)
    assert np.array_equal(miles_1.age, np.array([0.2512, 0.0708, 1.4125]))
    assert np.array_equal(miles_1.met, np.array([0.22, 0.0, -1.71]))
    assert np.array_equal(miles_1.imf_slope, np.array([1.3, 1.3, 1.3]))


def test_ssp_by_params(miles_single):
    assert miles_single.age == [5.7]
    assert miles_single.met == [-0.45]
    assert miles_single.imf_slope == [1.3]


@pytest.fixture
def miles_tuned(miles_single):
    tab = ascii.read("./pymiles/config_files/MUSE-WFM.lsf")
    lsf_wave = tab["Lambda"]
    lsf = tab["FWHM"] * 0.25

    return miles_single.tune_spectra(
        wave_lims=[4750.0, 5500.0],
        dwave=1.0,
        sampling="lin",
        redshift=0.0,
        lsf_flag=True,
        lsf_wave=lsf_wave,
        lsf=lsf,
    )


@pytest.mark.mpl_image_compare
def test_tuned_spectra(miles_single, miles_tuned):
    fig, ax = plt.subplots()
    ax.plot(miles_single.wave, miles_single.spec)
    ax.plot(miles_tuned.wave, miles_tuned.spec, "k")
    ax.set_xlim(3500, 7500)
    ax.set_ylim(0, 6e-5)
    return fig


def test_mags(miles_single):
    # Compute mags
    fnames = flib.search("sdss")
    filts = flib.get(fnames)
    outmags = miles_single.magnitudes(filters=filts, zeropoint="AB")
    assert np.allclose(outmags["OAJ_JPAS.gSDSS"], np.array([6.36344262]))
    assert np.allclose(outmags["OAJ_JPAS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPAS.rSDSS"], np.array([5.70916609]))
    assert np.allclose(outmags["OAJ_JPLUS.gSDSS"], np.array([6.34960724]))
    assert np.allclose(outmags["OAJ_JPLUS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPLUS.rSDSS"], np.array([5.7056477]))
    assert np.allclose(outmags["OAJ_JPLUS.zSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.g"], np.array([6.40127153]))
    assert np.allclose(outmags["SLOAN_SDSS.i"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.r"], np.array([5.72369018]))
    assert np.allclose(outmags["SLOAN_SDSS.u"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.z"], np.array([np.nan]), equal_nan=True)


def test_ls_indices(miles_single):
    outls = miles_single.compute_ls_indices()
    ref = {
        "Fe3619": np.array([4.2427661]),
        "Fe3631": np.array([2.09959643]),
        "Fe3646": np.array([1.65707798]),
        "Fe3683": np.array([1.64938502]),
        "Fe3706": np.array([1.73097357]),
        "Fe3741": np.array([14.24194305]),
        "UV_CN": np.array([0.13623793]),
        "H10Fe": np.array([3.49989936]),
        "CNB": np.array([0.13285228]),
        "MgH93838": np.array([7.92510864]),
        "CNO3862": np.array([4.56233091]),
        "CN3883": np.array([0.18729395]),
        "CaHK": np.array([18.23859878]),
        "CaIIH_K": np.array([20.70842364]),
        "CaK3933": np.array([9.00009823]),
        "H_K": np.array([0.32028138]),
        "CaH3968": np.array([8.48605792]),
        "FeBand": np.array([3.89174272]),
        "Fe4033": np.array([0.45146105]),
        "Fe4046": np.array([0.70069878]),
        "Fe4064": np.array([1.22391949]),
        "Sr4077": np.array([1.17867929]),
        "HdA": np.array([-0.21351792]),
        "HdF": np.array([1.27776455]),
        "CNO4175": np.array([-0.92661951]),
        "CN1": np.array([-0.0052772]),
        "CN2": np.array([0.03506814]),
        "Ca4227": np.array([1.09059413]),
        "G4300": np.array([4.99111535]),
        "HgA": np.array([-2.94598796]),
        "Fe4326": np.array([1.35030486]),
        "HgF": np.array([-0.08342484]),
        "Hg_sigma_275": np.array([0.11230075]),
        "Hg_sigma_200": np.array([0.49618757]),
        "Hg_sigma_125": np.array([1.02605582]),
        "Hg_sigma_130": np.array([1.20822842]),
        "Fe4383": np.array([3.32784955]),
        "Fe4457": np.array([2.12752336]),
        "Ca4455": np.array([1.24143643]),
        "Fe4531": np.array([3.01586018]),
        "Fe4592": np.array([2.03716249]),
        "FeII4550": np.array([1.25905272]),
        "Ca4592": np.array([1.44289693]),
        "CO4685": np.array([1.4817128]),
        "C2_4668": np.array([3.38011819]),
        "bTiO": np.array([0.01571478]),
        "Mg4780": np.array([0.53966877]),
        "Hbeta_o": np.array([3.16797002]),
        "Hbeta": np.array([2.21379742]),
        "Fe4920": np.array([0.96143232]),
        "Fe5015": np.array([4.60669416]),
        "Mg1": np.array([0.05179631]),
        "MgH": np.array([4.56594516]),
        "MgG": np.array([2.56348126]),
        "Mg2": np.array([0.14439579]),
        "Mgb": np.array([2.6147045]),
        "Fe5270": np.array([2.38518166]),
        "Fe5335": np.array([2.35667494]),
        "Fe5406": np.array([1.39125587]),
        "aTiO": np.array([0.01080768]),
        "Fe5709": np.array([0.77249581]),
        "Fe5782": np.array([0.62399652]),
        "NaD": np.array([2.04327609]),
        "TiO1": np.array([0.02854314]),
        "Ca6162": np.array([1.00257257]),
        "Fe6189": np.array([0.38147345]),
        "TiO2": np.array([0.05574065]),
        "TiO2sdss": np.array([0.06038641]),
        "CaH1": np.array([0.00748351]),
        "Fe6497": np.array([0.90521034]),
        "Halpha": np.array([2.41603984]),
        "Ha_Gregg94": np.array([2.0105494]),
        "CaH2": np.array([0.03270749]),
    }

    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outls[k], rtol=1e-5, err_msg=k)


def test_ml(miles_single):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = miles_single.mass_to_light(filters=filts, mass_in=["star+remn", "total"])
    ref = {
        "SLOAN_SDSS.g": np.array([1.98712619]),
        "SLOAN_SDSS.i": np.array([np.nan]),
        "SLOAN_SDSS.r": np.array([1.67578224]),
        "SLOAN_SDSS.u": np.array([np.nan]),
        "SLOAN_SDSS.z": np.array([np.nan]),
    }
    np.allclose(miles_single.Mass_star_remn, np.array([0.622357175092374]))
    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outmls["star+remn"][k], rtol=1e-5, err_msg=k)
