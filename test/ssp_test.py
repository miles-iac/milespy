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
    return miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, return_pars=False
    )


@pytest.mark.mpl_image_compare
def test_ssp_by_params_alpha(miles_ssp):
    miles_1 = miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, alpha=0.4, return_pars=False
    )
    miles_2 = miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, alpha=0.0, return_pars=False
    )
    fig, ax = plt.subplots()
    ax.plot(miles_1.wave, miles_1.spec, alpha=0.5, label="alpha=0.4")
    ax.plot(miles_2.wave, miles_2.spec, alpha=0.5, label="alpha=0")
    ax.legend()
    return fig


@pytest.mark.mpl_image_compare
def test_ssp_by_params_img(miles_ssp):
    miles_1 = miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, return_pars=False
    )
    # Also get the closest ones, which should be the base for the interpolation
    miles_vertices = miles_ssp.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, return_pars=False, closest=True
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
    ax.legend()
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
    return fig


def test_mags(miles_single):
    # Compute mags
    fnames = flib.search("sdss")
    filts = flib.get(fnames)
    outmags = miles_single.magnitudes(filters=filts, zeropoint="AB")
    assert np.allclose(outmags["OAJ_JPAS.gSDSS"], np.array([6.36259285]))
    assert np.allclose(outmags["OAJ_JPAS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPAS.rSDSS"], np.array([5.70746113]))
    assert np.allclose(outmags["OAJ_JPLUS.gSDSS"], np.array([6.348738]))
    assert np.allclose(outmags["OAJ_JPLUS.iSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["OAJ_JPLUS.rSDSS"], np.array([5.70393869]))
    assert np.allclose(outmags["OAJ_JPLUS.zSDSS"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.g"], np.array([6.4004899]))
    assert np.allclose(outmags["SLOAN_SDSS.i"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.r"], np.array([5.72204878]))
    assert np.allclose(outmags["SLOAN_SDSS.u"], np.array([np.nan]), equal_nan=True)
    assert np.allclose(outmags["SLOAN_SDSS.z"], np.array([np.nan]), equal_nan=True)


def test_ls_indices(miles_single):
    outls = miles_single.compute_ls_indices()
    ref = {
        "Fe3619": np.array([4.24374222]),
        "Fe3631": np.array([2.10202353]),
        "Fe3646": np.array([1.65740837]),
        "Fe3683": np.array([1.65097576]),
        "Fe3706": np.array([1.73122251]),
        "Fe3741": np.array([14.23980884]),
        "UV_CN": np.array([0.13594948]),
        "H10Fe": np.array([3.49896763]),
        "CNB": np.array([0.13234765]),
        "MgH93838": np.array([7.9226612]),
        "CNO3862": np.array([4.54468038]),
        "CN3883": np.array([0.18678602]),
        "CaHK": np.array([18.27913837]),
        "CaIIH_K": np.array([20.73305359]),
        "CaK3933": np.array([9.01743121]),
        "H_K": np.array([0.32064697]),
        "CaH3968": np.array([8.48854171]),
        "FeBand": np.array([3.90114633]),
        "Fe4033": np.array([0.45267352]),
        "Fe4046": np.array([0.70245179]),
        "Fe4064": np.array([1.22638068]),
        "Sr4077": np.array([1.18053771]),
        "HdA": np.array([-0.22762586]),
        "HdF": np.array([1.2717467]),
        "CNO4175": np.array([-0.89624335]),
        "CN1": np.array([-0.00476545]),
        "CN2": np.array([0.03557297]),
        "Ca4227": np.array([1.09357064]),
        "G4300": np.array([4.99391304]),
        "HgA": np.array([-2.95797546]),
        "Fe4326": np.array([1.35035966]),
        "HgF": np.array([-0.08956468]),
        "Hg_sigma_275": np.array([0.11016055]),
        "Hg_sigma_200": np.array([0.49385698]),  # noqa
        "Hg_sigma_125": np.array([1.02347192]),
        "Hg_sigma_130": np.array([1.20529933]),  # noqa
        "Fe4383": np.array([3.33428815]),
        "Fe4457": np.array([2.1305967]),
        "Ca4455": np.array([1.24400289]),
        "Fe4531": np.array([3.01967504]),
        "Fe4592": np.array([2.03947761]),
        "FeII4550": np.array([1.26062759]),
        "Ca4592": np.array([1.44691571]),
        "CO4685": np.array([1.48691463]),
        "C2_4668": np.array([3.3898653]),
        "bTiO": np.array([0.01574851]),
        "Mg4780": np.array([0.54044404]),
        "Hbeta_o": np.array([3.16459111]),
        "Hbeta": np.array([2.2108947]),
        "Fe4920": np.array([0.96259459]),
        "Fe5015": np.array([4.61379537]),
        "Mg1": np.array([0.05198452]),
        "MgH": np.array([4.57628813]),
        "MgG": np.array([2.56740612]),
        "Mg2": np.array([0.14471378]),
        "Mgb": np.array([2.61891892]),
        "Fe5270": np.array([2.38844946]),
        "Fe5335": np.array([2.35908341]),
        "Fe5406": np.array([1.39302161]),
        "aTiO": np.array([0.01084238]),
        "Fe5709": np.array([0.77513128]),
        "Fe5782": np.array([0.6242171]),
        "NaD": np.array([2.03584002]),
        "TiO1": np.array([0.02859881]),
        "Ca6162": np.array([1.00416356]),
        "Fe6189": np.array([0.38311829]),
        "TiO2": np.array([0.05588053]),
        "TiO2sdss": np.array([0.06053157]),
        "CaH1": np.array([0.00747431]),
        "Fe6497": np.array([0.90560345]),
        "Halpha": np.array([2.41228887]),
        "Ha_Gregg94": np.array([2.00872752]),
        "CaH2": np.array([0.03285235]),
    }
    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outls[k], rtol=1e-5, err_msg=k)


def test_ml(miles_single):
    fnames = flib.search("sloan")
    filts = flib.get(fnames)
    outmls = miles_single.mass_to_light(filters=filts, mass_in=["star+remn", "total"])
    ref = {
        "SLOAN_SDSS.g": np.array([1.98569615]),
        "SLOAN_SDSS.i": np.array([np.nan]),
        "SLOAN_SDSS.r": np.array([1.67325073]),
        "SLOAN_SDSS.u": np.array([np.nan]),
        "SLOAN_SDSS.z": np.array([np.nan]),
    }
    assert miles_single.Mass_star_remn == [0.622357175092374]
    for k in ref.keys():
        np.testing.assert_allclose(ref[k], outmls["star+remn"][k], rtol=1e-5, err_msg=k)
