# -*- coding: utf-8 -*-
import logging

import numpy as np


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
