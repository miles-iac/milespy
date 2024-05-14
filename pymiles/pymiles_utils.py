# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
from scipy.interpolate import interp1d

# import scipy.interpolate

logger = logging.getLogger("pymiles.utils")
# ==============================================================================


def interp_weights(xyz, uvw, tri):
    # Creates a Delaunay triangulation and finds the vertices and weights of
    # points around a given location in parameter space

    d = len(uvw[0, :])
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)

    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


# ===============================================================================


def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs=None, fill=None):
    """
    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.
    Parameters

    Taken from:
    https://github.com/ACCarnall/SpectRes/blob/master/spectres/spectral_resampling.py
    ----------
    new_wavs : np.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.
    spec_wavs : np.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.
    spec_fluxes : np.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.
    spec_errs : np.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.
    fill : float (optional)
        Value for all new_fluxes and new_errs that fall outside the
        wavelength range in spec_wavs. These will be nan by default.
    Returns
    -------
    new_fluxes : np.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.
    new_errs : np.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Arrays of left hand sides and widths for the old and new bins
    old_lhs = np.zeros(old_wavs.shape[0])
    old_widths = np.zeros(old_wavs.shape[0])
    old_lhs = np.zeros(old_wavs.shape[0])
    old_lhs[0] = old_wavs[0]
    old_lhs[0] -= (old_wavs[1] - old_wavs[0]) / 2
    old_widths[-1] = old_wavs[-1] - old_wavs[-2]
    old_lhs[1:] = (old_wavs[1:] + old_wavs[:-1]) / 2
    old_widths[:-1] = old_lhs[1:] - old_lhs[:-1]
    old_max_wav = old_lhs[-1] + old_widths[-1]

    new_lhs = np.zeros(new_wavs.shape[0] + 1)
    new_widths = np.zeros(new_wavs.shape[0])
    new_lhs[0] = new_wavs[0]
    new_lhs[0] -= (new_wavs[1] - new_wavs[0]) / 2
    new_widths[-1] = new_wavs[-1] - new_wavs[-2]
    new_lhs[-1] = new_wavs[-1]
    new_lhs[-1] += (new_wavs[-1] - new_wavs[-2]) / 2
    new_lhs[1:-1] = (new_wavs[1:] + new_wavs[:-1]) / 2
    new_widths[:-1] = new_lhs[1:-1] - new_lhs[:-2]

    # Generate output arrays to be populated
    new_fluxes = np.zeros(old_fluxes[..., 0].shape + new_wavs.shape)

    if old_errs is not None:
        if old_errs.shape != old_fluxes.shape:
            raise ValueError(
                "If specified, spec_errs must be the same shape " "as spec_fluxes."
            )
        else:
            new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):
        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_lhs[j] < old_lhs[0]) or (new_lhs[j + 1] > old_max_wav):
            new_fluxes[..., j] = fill

            if spec_errs is not None:
                new_errs[..., j] = fill

            if j == 0:
                print(
                    "\nSpectres: new_wavs contains values outside the range "
                    "in spec_wavs. New_fluxes and new_errs will be filled "
                    "with the value set in the 'fill' keyword argument (nan "
                    "by default).\n"
                )
            continue

        # Find first old bin which is partially covered by the new bin
        while old_lhs[start + 1] <= new_lhs[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_lhs[stop + 1] < new_lhs[j + 1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = (old_lhs[start + 1] - new_lhs[j]) / (
                old_lhs[start + 1] - old_lhs[start]
            )

            end_factor = (new_lhs[j + 1] - old_lhs[stop]) / (
                old_lhs[stop + 1] - old_lhs[stop]
            )

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start : stop + 1] * old_fluxes[..., start : stop + 1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start : stop + 1])

            if old_errs is not None:
                e_wid = old_widths[start : stop + 1] * old_errs[..., start : stop + 1]

                new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
                new_errs[..., j] /= np.sum(old_widths[start : stop + 1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    if old_errs is not None:
        return new_fluxes, new_errs

    # Otherwise just return the new_fluxes spectrum array
    else:
        return new_fluxes


# ===============================================================================


def load_filters(filtersfile):
    # Reading the filters file and guessing the number of filters
    f = open(filtersfile, "r")
    data = f.readlines()
    nlines = len(data)
    f.close()

    nfilters = 0
    wave = np.zeros(nlines)
    flux = np.zeros(nlines)
    filter_names = []
    idx = []
    for i in range(len(data)):
        line = data[i].split()
        if line[0][0] == "#":
            nfilters += 1
            idx = np.append(idx, i)
            filter_names = np.append(filter_names, line[1])
        else:
            wave[i] = float(line[0])
            flux[i] = float(line[1])

    # Storing the values for each filter into a new recarray
    nmax = 9999
    print("- " + str(nfilters) + " filters found")
    idx = np.array(idx, dtype=int)
    filters = np.recarray((nfilters, nmax), dtype=[("wave", float), ("flux", float)])
    for i in range(nfilters):
        # Initializing vectors to force them to Zero
        filters[i].wave.put(range(nmax), 0.0)
        filters[i].flux.put(range(nmax), 0.0)
        if i < nfilters - 1:
            npt = idx[i + 1] - idx[i] - 1
            filters[i].wave.put(range(npt), wave[idx[i] + 1 : idx[i + 1]])
            filters[i].flux.put(
                range(npt),
                flux[idx[i] + 1 : idx[i + 1]] / np.amax(flux[idx[i] + 1 : idx[i + 1]]),
            )
        else:
            npt = len(wave[idx[i] + 1 :])
            filters[i].wave.put(range(npt), wave[idx[i] + 1 :])
            filters[i].flux.put(
                range(npt), flux[idx[i] + 1 :] / np.amax(flux[idx[i] + 1 :])
            )

    return filter_names, filters


# ===============================================================================


def load_zerofile(zeropoint):
    file = "./pymiles/config_files/vega_from_koo.sed"
    file = "./pymiles/config_files/vega.sed"
    data = ascii.read(file, comment=r"\s*#")
    npt = len(data["col1"])

    zerosed = np.recarray((npt,), dtype=[("wave", float), ("flux", float)])
    zerosed.wave = data["col1"]
    zerosed.flux = data["col2"]

    # If AB mags only need wavelength vector
    if zeropoint == "AB":
        zerosed.flux = 1.0 / np.power(zerosed.wave, 2)
        return zerosed

    elif zeropoint == "VEGA":
        # Normalizing the SED@ 5556.0\AA
        zp5556 = 3.44e-9  # erg cm^-2 s^-1 A^-1, Hayes 1985
        interp = interp1d(zerosed.wave, zerosed.flux)
        zerosed.flux *= zp5556 / interp(5556.0)

    return zerosed


# ===============================================================================


def compute_mags(wave, flux, filters, zerosed, zeropoint, sun=False):
    # Defining some variables
    cvel = 2.99792458e18  # Speed of light in Angstron/sec
    dl = 1e-5  # 10 pc in Mpc, z=0; for absolute magnitudes
    if sun:
        cfact = -5.0 * np.log10(4.84e-6 / 10.0)  # for absolute magnitudes
    else:
        cfact = 5.0 * np.log10(1.7684e8 * dl)  # from lum[erg/s/A] to flux [erg/s/A/cm2]

    # Getting info about filters
    nfilters = len(filters.keys())
    filter_names = list(filters.keys())
    outmag = np.zeros(nfilters) * np.nan
    interp_zp = interp1d(zerosed.wave, zerosed.flux)

    # Computing the magnitude for each filter
    for i in range(nfilters):
        # Finding the wavelength limits of the filters
        good = filters[filter_names[i]]["wave"] > 0.0
        wlow = np.amin(filters[filter_names[i]]["wave"][good])
        whi = np.amax(filters[filter_names[i]]["wave"][good])

        # Selecting the relevant pixels in the input spectrum
        w = (wave >= wlow) & (wave <= whi)
        tmp_wave = wave[w]
        tmp_flux = flux[w]
        if (np.amin(wave) > wlow) or (np.amax(wave) < whi):
            logger.warning(
                "Filter "
                + filter_names[i]
                + " ["
                + str(wlow)
                + ","
                + str(whi)
                + "] is outside of spectral range ["
                + str(np.amin(wave))
                + ","
                + str(np.amax(wave))
                + "]\t Returning nan"
            )
            continue

        # Identifying pixels with no flux
        bad = tmp_flux == 0.0
        if np.sum(bad) > 0:
            logger.warning(
                "Filter "
                + filter_names[i]
                + " ["
                + str(wlow)
                + ","
                + str(whi)
                + "] has zero flux\t Returning nan"
            )
            continue

        # Interpolate the filter response to data wavelength
        interp = interp1d(
            filters[filter_names[i]]["wave"][good],
            filters[filter_names[i]]["trans"][good],
        )
        response = interp(tmp_wave)

        # Calculating the magnitude in the desired system
        vega = interp_zp(tmp_wave)
        f = np.trapz(tmp_flux * response, x=tmp_wave)
        vega_f = np.trapz(vega * response, x=tmp_wave)
        mag = -2.5 * np.log10(f / vega_f)
        fmag = mag + cfact
        if zeropoint == "AB":
            fmag = fmag + 2.5 * np.log10(cvel) - 48.6  # oke & gunn 83

        outmag[i] = fmag

    return outmag


# ==============================================================================
#
# FUNCTION: sum_counts()
#


def sum_counts(ll, c, b1, b2):
    # Central full pixel range
    dw = ll[1] - ll[0]  # linear step size
    w = (ll >= b1 + dw / 2.0) & (ll <= b2 - dw / 2.0)
    s = np.sum(c[w])

    # First fractional pixel
    pixb = (ll < b1 + dw / 2.0) & (ll > b1 - dw / 2.0)
    if np.any(pixb):
        fracb = ((ll[pixb] + dw / 2.0) - b1) / dw
        s = s + c[pixb] * fracb

    # Last fractional pixel
    pixr = (ll < b2 + dw / 2.0) & (ll > b2 - dw / 2.0)
    if np.any(pixr):
        fracr = (b2 - (ll[pixr] - dw / 2.0)) / dw
        s = s + c[pixr] * fracr

    return s


# ==============================================================================
#
# FUNCTION: calc_index()
#


def calc_index(bands, name, ll, counts, plot):
    cb = sum_counts(ll, counts, bands[0], bands[1])
    cr = sum_counts(ll, counts, bands[4], bands[5])
    s = sum_counts(ll, counts, bands[2], bands[3])

    lb = (bands[0] + bands[1]) / 2.0
    lr = (bands[4] + bands[5]) / 2.0
    cb = cb / (bands[1] - bands[0])
    cr = cr / (bands[5] - bands[4])
    m = (cr - cb) / (lr - lb)
    c1 = (m * (bands[2] - lb)) + cb
    c2 = (m * (bands[3] - lb)) + cb
    cont = 0.5 * (c1 + c2) * (bands[3] - bands[2])

    if bands[6] == 1.0:
        # atomic index
        ind = (1.0 - (s / cont)) * (bands[3] - bands[2])
    elif bands[6] == 2.0:
        # molecular index
        ind = -2.5 * np.log10(s / cont)

    if plot > 0:
        minx = bands[0] - 0.05 * (bands[5] - bands[0])
        maxx = bands[5] + 0.05 * (bands[5] - bands[0])
        miny = np.amin(counts) - 0.05 * (np.amax(counts) - np.amin(counts))
        maxy = np.amax(counts) + 0.05 * (np.amax(counts) - np.amin(counts))
        plt.figure()
        plt.plot(ll, counts, "k")
        plt.xlabel(r"Wavelength ($\AA$)")
        plt.ylabel("Counts")
        plt.title(name)
        plt.xlim([minx, maxx])
        plt.ylim([miny, maxy])
        dw = ll[1] - ll[0]
        plt.plot([lb, lr], [c1 * dw, c2 * dw], "r")
        good = (ll >= bands[2]) & (ll <= bands[3])
        ynew = np.interp(ll, [lb, lr], [c1[0] * dw, c2[0] * dw])
        plt.fill_between(ll[good], counts[good], ynew[good], facecolor="green")
        for i in range(len(bands)):
            plt.plot([bands[i], bands[i]], [miny, maxy], "k--")
        plt.show()

    return ind


# ==============================================================================
# purpose : Measure line-strength indices
#
# input : ll    - wavelength vector; assumed to be in *linear steps*
#         flux  - counts as a function of wavelength
#         noise - noise spectrum
#         z, z_err - redshift and error (in km/s)
#         lickfile - file listing the index definitions
#
# keywords  debug  - more than 0 gives some basic info
#           plot   - plot spectra
#           sims   - number of simulations for the errors (default: 100)
#
# output : names       - index names
#          index       - index values
#          index_error - index error values
#
# author : J. Falcon-Barroso
#
# version : 1.0  IAC (08/07/16) A re-coding of H. Kuntschner's IDL routine into python
# version : 2.0  IAC (26/03/20) Increase efficient by computing indices
#                               within wavelength only
# ==============================================================================


def lsindex(ll, flux, noise, z, z_err, lickfile, plot=0, sims=0):
    # Deredshift spectrum to rest wavelength
    dll = (ll) / (z + 1.0)

    # Read index definition table
    tab = ascii.read(lickfile, comment=r"\s*#")
    names = tab["names"]
    bands = np.zeros((7, len(names)))
    bands[0, :] = tab["b1"]
    bands[1, :] = tab["b2"]
    bands[2, :] = tab["b3"]
    bands[3, :] = tab["b4"]
    bands[4, :] = tab["b5"]
    bands[5, :] = tab["b6"]
    bands[6, :] = tab["b7"]

    good = (bands[0, :] >= dll[0]) & (bands[5, :] <= dll[-1])
    num_ind = np.sum(good)
    names = names[good]
    bands = bands[:, good]

    # Measure line indices
    num_ind = len(bands[0, :])
    index = np.zeros(num_ind) * np.nan
    for i in range(num_ind):  # loop through all indices
        # calculate index value
        index0 = calc_index(bands[:, i], names[i], dll, flux, plot)
        index[i] = index0[0]

    # Calculate errors
    index_error = np.zeros(num_ind, dtype="D") * np.nan
    index_noise = np.zeros([num_ind, sims], dtype="D")
    if sims > 0:
        # Create redshift and sigma errors
        dz = np.random.randn(sims) * z_err

        # Loop through the simulations
        for i in range(sims):
            # resample spectrum according to noise
            ran = np.random.normal(0.0, 1.0, len(dll))
            flux_n = flux + ran * noise

            # loop through all indices
            for k in range(num_ind):
                # shift bands according to redshift error
                sz = z + dz[i]
                dll = ll / (sz + 1.0)
                bands2 = bands[:, k]
                if (dll[0] <= bands2[0]) and (dll[len(dll) - 1] >= bands2[5]):
                    tmp = calc_index(bands2, names[k], dll, flux_n, 0)
                    index_noise[k, i] = tmp
                else:
                    # index outside wavelength range
                    index_noise[k, i] = -999

        # Get STD of distribution (index error)
        index_error = np.std(index_noise, axis=1)

    return names, index, index_error


# ==============================================================================
