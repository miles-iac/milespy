# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import re
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import ascii

from .configuration import get_config_file

logger = logging.getLogger("milespy.ls_indices")

lsfile = get_config_file("ls_indices_full.def")
lsindex_names = ascii.read(lsfile, comment=r"\s*#")["names"]
logging.debug(f"Initialized line strength with {len(lsindex_names)} indeces.")


class LineStrengthIndex:
    """
    Line strength index information

    Attributes
    ----------
    name: str
        Name of the index.
    index_type: str
        Whether the index is for atomic ("A") or molecular ("M") lines.
    band_blue1:
        Left edge of the blue band
    band_blue2:
        Right edge of the blue band
    band_centre1:
        Left edge of the central bandpass
    band_centre2:
        Right edge of the central bandpass
    band_red1:
        Left edge of the red band
    band_red2:
        Right edge of the red band
    """

    def __init__(
        self,
        name: str,
        index_type: Literal["A", "M"],
        band_blue1: u.Quantity[u.AA],
        band_blue2: u.Quantity[u.AA],
        band_centre1: u.Quantity[u.AA],
        band_centre2: u.Quantity[u.AA],
        band_red1: u.Quantity[u.AA],
        band_red2: u.Quantity[u.AA],
    ):
        self.name = name
        self.type = index_type

        self.bands = np.empty(6)
        self.bands[0] = band_blue1.to(u.AA).value
        self.bands[1] = band_blue2.to(u.AA).value
        self.bands[2] = band_centre1.to(u.AA).value
        self.bands[3] = band_centre2.to(u.AA).value
        self.bands[4] = band_red1.to(u.AA).value
        self.bands[5] = band_red2.to(u.AA).value

    @staticmethod
    def from_database(name) -> LineStrengthIndex:
        """
        Create an index from the name in the database.

        This reads the information from the configuration files, so the
        name should match with a given existing index. This can be easily
        accomplished with :meth:`milespy.index.search`.

        Parameters
        ----------
        name : str
            Name of the index to be loaded

        Raises
        ------
        ValueError
            If there is no matching index in the database
        RuntimeError
            If there are multiple indeces that match the input name
        """
        tab = ascii.read(lsfile, comment=r"\s*#")
        names = tab["names"]
        if name in names:
            idx = np.argwhere(names == name)
            if len(idx) > 1:
                raise RuntimeError("Multiple matching filters")

            return LineStrengthIndex(
                name,
                tab["b7"][idx][0][0],
                tab["b1"][idx][0][0] << u.AA,
                tab["b2"][idx][0][0] << u.AA,
                tab["b3"][idx][0][0] << u.AA,
                tab["b4"][idx][0][0] << u.AA,
                tab["b5"][idx][0][0] << u.AA,
                tab["b6"][idx][0][0] << u.AA,
            )

        else:
            raise ValueError(f"The index {name} is not on the database")


def search(name) -> list[str]:
    """
    Searches for an index in database.

    Notes
    -----
    Search is case insensitive
    The filter seach does not have to be precise. Substrings within filter
    names are ok.  It uses the python package 're' for regular expressions
    matching

    Parameters
    ----------
    name:
        The search string to match index names

    Returns
    -------
    list[str]
        List of index names available matching the search string

    """

    reg = re.compile(name, re.IGNORECASE)
    filtered_filters = list(filter(reg.search, lsindex_names))

    if len(filtered_filters) == 0:
        logger.warning(
            "Cannot find filter in our database\n Available filters are:\n\n"
            + str(lsindex_names)
        )

    return filtered_filters


def get(lsindex_names: list[str]) -> list[LineStrengthIndex]:
    """
    Retrieves index from database

    Parameters
    ----------
    lsindex_names: list[str]
        The index names as given by :meth:`milespy.ls_indeces.search`

    Returns
    -------
    list[Filter]
    """
    indeces = [LineStrengthIndex.from_database(name) for name in lsindex_names]

    return indeces


class LineStrengthDict(dict):
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
    s = np.sum(c[..., w], axis=-1)

    # First fractional pixel
    pixb = (ll < b1 + dw / 2.0) & (ll > b1 - dw / 2.0)
    if np.any(pixb):
        fracb = ((ll[pixb] + dw / 2.0) - b1) / dw
        s = s + c[..., pixb][..., 0] * fracb

    # Last fractional pixel
    pixr = (ll < b2 + dw / 2.0) & (ll > b2 - dw / 2.0)
    if np.any(pixr):
        fracr = (b2 - (ll[pixr] - dw / 2.0)) / dw
        s = s + c[..., pixr][..., 0] * fracr

    return s


def _calc_index(bands, index_type, name, ll, counts, plot=False):
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

    if index_type == "A":
        # atomic index
        ind = (1.0 - (s / cont)) * (bands[3] - bands[2])
    elif index_type == "M":
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


def lsindex(
    indeces: [LineStrengthIndex],
    ll,
    flux,
    z,
    plot=False,
    noise=None,
    z_err=None,
    sims=0,
):
    """
    Measure line-strength indices

    Author: J. Falcon-Barroso

    Parameters
    ----------
    indeces : [LineStrengthIndex]
        indeces to be computed
    ll : np.ndarray
        wavelength vector; assumed to be in *linear steps*
    flux : np.ndarray
        counts as a function of wavelength
    z : float
        redshift (in km/s)
    plot : bool
        plot spectra
    noise :
        noise spectrum
    z_err : float
        redshift error (in km/s)
    sims : int
        number of simulations for the errors (default: 100)

    Returns
    -------
    LineStrengthDict
    """
    # TODO: take into account the units
    ll = ll.to_value()
    z = z.to_value()
    flux = flux.to_value()

    # Deredshift spectrum to rest wavelength
    dll = ll / (z + 1.0)

    outindex = LineStrengthDict(
        (ind.name, np.full(flux.shape[:-1], np.nan)) for ind in indeces
    )

    for ind in indeces:
        good = (ind.bands[0] >= dll[0]) & (ind.bands[5] <= dll[-1])
        if not good:
            logger.warning(
                f"Index {ind.name} [{ind.bands[0]}, {ind.bands[5]}] "
                f"is outside of the spectral range [{dll[0]}, {dll[-1]}]"
            )
            continue
        # calculate index value
        outindex[ind.name] = _calc_index(ind.bands, ind.type, ind.name, dll, flux, plot)

    if sims > 0:
        raise NotImplementedError
        """
        # Calculate errors
        index_error = np.zeros(num_ind, dtype="D") * np.nan
        index_noise = np.zeros([num_ind, sims], dtype="D")

        # Create redshift and sigma errors
        dz = np.random.randn(sims) * z_err

        # Loop through the simulations
        for i in range(sims):
            # resample spectrum according to noise
            ran = np.random.normal(0.0, 1.0, len(dll))
            flux_n = flux + ran * noise

            # loop through all indices
            for k, ind in enumerate(indeces):
                # shift bands according to redshift error
                sz = z + dz[i]
                dll = ll / (sz + 1.0)
                bands2 = ind.bands
                if (dll[0] <= bands2[0]) and (dll[len(dll) - 1] >= bands2[5]):
                    tmp = _calc_index(bands2, ind.name, dll, flux_n, 0)
                    index_noise[k, i] = tmp
                else:
                    # index outside wavelength range
                    index_noise[k, i] = -999

        # Get STD of distribution (index error)
        index_error = np.std(index_noise, axis=1)
        """

    return outindex
