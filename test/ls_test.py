# -*- coding: utf-8 -*-
import astropy.units as u
import numpy as np

import pymiles.ls_indices as lslib
from pymiles.spectra import Spectra


def test_cube_ls(miles_single):
    # Create a mock datacube
    wave = miles_single.spectral_axis
    flux = u.Quantity(np.random.random((30, 30, 4300)), unit=u.L_sun / u.AA)
    flux[...] = miles_single.flux

    cube = Spectra(flux=flux, spectral_axis=wave)

    names = lslib.search("Fe.*")
    indeces = lslib.get(names)
    outls = cube.line_strength(indeces)

    assert outls["Fe4033"].shape == cube.dim
    np.testing.assert_allclose(outls["Fe4033"], 0.45146105, rtol=1e-5)


def test_ls_indices(miles_single):
    lsnames = lslib.search(".*")
    indeces = lslib.get(lsnames)
    outls = miles_single.line_strength(indeces)
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
