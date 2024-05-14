# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

from pymiles.ssp_models import ssp_models as ssp

# ==============================================================================
if __name__ == "__main__":
    import logging

    logger = logging.getLogger("pymiles")

    logger.setLevel(logging.INFO)
    #
    # SSP MODELS EXAMPLES
    #

    # Initializing instance
    print("# Initializing instance")
    miles = ssp(
        source="MILES_SSP",
        version="9.1",
        imf_type="bi",
        isochrone="P",
        alp_type="fix",
        show_tree=False,
    )

    #  Get SSP in range
    miles_1 = miles.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, return_pars=False
    )
    plt.plot(miles_1.wave, miles_1.spec)
    plt.show()

    #  Get SSP in range
    print("# Get SSP in range")
    miles_1 = miles.get_ssp_in_range(
        age_lims=[17.0, 20.0], met_lims=[0.1, 0.5], verbose=False
    )
    print(np.amin(miles_1.age), np.amax(miles_1.age))
    print(miles_1.age.shape)
    print(miles_1.age)
    print(miles_1.met)
    print(miles_1.Mass_star_remn)
    #     exit()

    #  Get SSP in list
    print("# Get SSP in list")
    miles_2 = miles.get_ssp_in_list(
        age_list=[0.2512, 0.0708, 1.4125],
        met_list=[0.22, 0.0, -1.71],
        imf_slope_list=[1.3, 1.3, 1.3],
        alpha_list=[0.0, 0.4, 0.0],
    )

    #  Get SSP by params (gets interpolated spectrum)
    # ("# Get SSP by params (gets interpolated spectrum)")
    miles_1 = miles.get_ssp_by_params(
        age=5.7, met=-0.45, imf_slope=1.3, return_pars=False
    )
    print(miles_1.age)
    print(miles_1.filter_names)

    fnames = miles_1.find_filter("sloan")
    filts = miles_1.get_filters(fnames)
    outmls = miles_1.compute_ml(filters=filts, type="star+remn", verbose=False)

    print(
        miles_1.age,
        miles_1.met,
        miles_1.Mass_star_remn,
        outmls["SLOAN_SDSS.u"],
        outmls["SLOAN_SDSS.g"],
        outmls["SLOAN_SDSS.r"],
        outmls["SLOAN_SDSS.i"],
        outmls["SLOAN_SDSS.z"],
    )

    plt.plot(miles_1.wave, miles_1.spec)
    plt.show()
    # exit()

    #  Tune spectra
    tab = ascii.read("./pymiles/config_files/MUSE-WFM.lsf")
    lsf_wave = tab["Lambda"]
    lsf = tab["FWHM"] * 0.25

    tuned_miles = miles_1.tune_spectra(
        wave_lims=[4750.0, 5500.0],
        dwave=1.0,
        sampling="lin",
        redshift=0.0,
        lsf_flag=True,
        lsf_wave=lsf_wave,
        lsf=lsf,
        verbose=False,
    )

    plt.plot(miles_1.wave, miles_1.spec)
    plt.plot(tuned_miles.wave, tuned_miles.spec, "k")
    plt.show()

    print(tuned_miles.alpha)
    # Saving object to HDF5 file
    tuned_miles.save_object("spectra.hdf5", verbose=True)

    # Compute mags
    fnames = miles_1.find_filter("sdss")
    filts = miles_1.get_filters(fnames)
    outmags = miles_1.compute_save_mags(
        filters=filts, zeropoint="AB", saveCSV=False, verbose=True
    )

    # Compute LS indices
    outls = miles_1.compute_ls_indices(verbose=True, saveCSV=False)
    print(outls)

    # Compute solar mags
    # (this can be done with the Tuning Tools class. No need for SSP class)
    fnames = miles.find_filter("sloan")
    filts = miles.get_filters(fnames)
    outmags = miles.compute_mag_sun(filters=filts, verbose=True, zeropoint="AB")
    print(outmags)
    # exit()

    # Compute mass-to-light ratios
    print("# Computing M/Ls")
    fnames = miles_1.find_filter("sloan")
    filts = miles_1.get_filters(fnames)
    outmls = miles_1.compute_ml(filters=filts, type="star+remn", verbose=False)
