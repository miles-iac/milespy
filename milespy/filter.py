# -*- coding: utf-8 -*-
from __future__ import annotations

import glob
import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

from .configuration import config_folder
from .configuration import get_config_file

logger = logging.getLogger("milespy.filter")


class Filter:
    """
    Filter information

    Attributes
    ----------
    wave: array
        Wavelength range of the filter
    trans: array
        Transmissivity for each wavelength
    name: str
        Name of the filter
    """

    def __init__(self, fname):
        """
        Create a filter from the name in the database.

        This reads the information from the configuration files, so the
        name should match with a given existing file. This can be easily
        accomplished with :meth:`milespy.filter.search`.

        Parameters
        ----------
        fname : str
            Name of the filter to be loaded
        """
        filename = get_config_file("filters/" + fname + ".dat")
        if not os.path.exists(filename):
            assert ValueError("Filter " + fname + " does not exist in database")
        else:
            tab = ascii.read(filename, names=["wave", "trans"])
            tab["trans"] /= np.amax(tab["trans"])
            self.wave = tab["wave"]
            self.name = fname
            self.trans = tab["trans"]

    def plot(self, ax) -> None:
        """
        Plot the filter transmissivity

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes where the plot is drawn
        """
        ax.fill_between(
            self.wave,
            self.trans,
            alpha=0.5,
            label=self.name,
            edgecolor="k",
        )


fnames = glob.glob(f"{config_folder.as_posix()}/filters/*.dat")
filter_names = np.sort([os.path.basename(x).split(".dat")[0] for x in fnames])
nfilters = len(filter_names)
logging.debug(f"Initialized library with {nfilters} filters")


def search(name) -> list[str]:
    """
    Searches for a filter in database.

    Notes
    -----
    Search is case insensitive
    The filter seach does not have to be precise. Substrings within filter
    names are ok.  It uses the python package 're' for regular expressions
    matching

    Parameters
    ----------
    name:
        The search string to match filter names

    Returns
    -------
    list[str]
        List of filter names available matching the search string

    """

    reg = re.compile(name, re.IGNORECASE)
    filtered_filters = list(filter(reg.search, filter_names))

    if len(filtered_filters) == 0:
        logger.warning(
            "Cannot find filter in our database\n Available filters are:\n\n"
            + str(filter_names)
        )

    return filtered_filters


def get(filter_names: list[str]) -> list[Filter]:
    """
    Retrieves filter from database

    Parameters
    ----------
    filter_names: list[str]
        The filter names as given by :meth:`milespy.filter.search`

    Returns
    -------
    list[Filter]
    """
    filters = [Filter(fname) for fname in filter_names]

    return filters


def plot(filter_names, legend=True):
    """
    Plot filters

    Parameters
    ----------
    filter_names: list[str]
        The filter names
    legend: bool
        Flag to turn on/off the legend

    """

    fig, ax = plt.subplots()

    _ = [Filter(fname).plot(ax) for fname in filter_names]

    if legend:
        plt.legend()

    plt.show()
