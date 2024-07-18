# -*- coding: utf-8 -*-
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

logger = logging.getLogger("pymiles.ls_indices")


class LineStrengthIndeces(dict):
    def write(self, output=sys.stdout, format="basic", **kwargs):
        """
        Save the line strength indices in the requested format

        Any extra keyword parameters are passed to astropy.io.ascrii.write

        Parameters
        ----------
        output: str
            Output filename. Defaults to sys.stdout
        format : str
            Any of the available format specifier for astropy.io.ascii.write:
            https://docs.astropy.org/en/stable/api/astropy.io.ascii.write.html#astropy.io.ascii.write
        """
        from astropy.table import Table

        tab = Table(data=dict(self))
        ascii.write(tab, output, format=format, **kwargs)


def _sum_counts(ll, c, b1, b2):
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


def _calc_index(bands, name, ll, counts, plot=False):
    cb = _sum_counts(ll, counts, bands[0], bands[1])
    cr = _sum_counts(ll, counts, bands[4], bands[5])
    s = _sum_counts(ll, counts, bands[2], bands[3])

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

    if plot:
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


def lsindex(ll, flux, noise, z, z_err, lickfile, plot=False, sims=100):
    """
    Measure line-strength indices

    Author: J. Falcon-Barroso

    Parameters
    ----------
    ll : np.ndarray
        wavelength vector; assumed to be in *linear steps*
    flux : np.ndarray
        counts as a function of wavelength
    noise :
        noise spectrum
    z : float
        redshift (in km/s)
    z_err : float
        redshift error (in km/s)
    lickfile : str
        file listing the index definitions
    plot : bool
        plot spectra
    sims : int
        number of simulations for the errors (default: 100)

    Returns
    -------
    names : [str]
        index names
    index :
        index values
    index_error :
        error values
    """
    # TODO: take into account the units
    ll = ll.to_value()
    z = z.to_value()

    # Deredshift spectrum to rest wavelength
    dll = ll / (z + 1.0)

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
        index0 = _calc_index(bands[:, i], names[i], dll, flux, plot)
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
                    tmp = _calc_index(bands2, names[k], dll, flux_n, 0)
                    index_noise[k, i] = tmp
                else:
                    # index outside wavelength range
                    index_noise[k, i] = -999

        # Get STD of distribution (index error)
        index_error = np.std(index_noise, axis=1)

    return names, index, index_error
