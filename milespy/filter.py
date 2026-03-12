# -*- coding: utf-8 -*-
"""Filter database access and transmissivity curves for photometric bandpasses."""
from __future__ import annotations

import glob
import logging
import os
import re
import warnings

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
        accomplished with :meth:`milespy.filter.search_filters`.

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

    def plot_transmissivity(self, ax) -> None:
        """
        Plot the filter transmissivity.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes where the plot is drawn.
        """
        ax.fill_between(
            self.wave,
            self.trans,
            alpha=0.5,
            label=self.name,
            edgecolor="k",
        )

    def plot(self, ax) -> None:
        """Plot the filter transmissivity (deprecated: use plot_transmissivity)."""
        warnings.warn(
            "Filter.plot is deprecated; use plot_transmissivity instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.plot_transmissivity(ax)


fnames = glob.glob(f"{config_folder.as_posix()}/filters/*.dat")
filter_names = np.sort([os.path.basename(x).split(".dat")[0] for x in fnames])
nfilters = len(filter_names)
logging.debug(f"Initialized library with {nfilters} filters")


def search_filters(name: str) -> list[str]:
    """
    Search for filters in the database by name (regex, case insensitive).

    Parameters
    ----------
    name : str
        The search string to match filter names (regex pattern).

    Returns
    -------
    list[str]
        List of filter names matching the search string.
    """
    reg = re.compile(name, re.IGNORECASE)
    filtered_filters = list(filter(reg.search, filter_names))

    if len(filtered_filters) == 0:
        logger.warning(
            "Cannot find filter in our database\n Available filters are:\n\n"
            + str(filter_names)
        )

    return filtered_filters


def search(name: str) -> list[str]:
    """Deprecated: use search_filters instead."""
    warnings.warn(
        "filter.search is deprecated; use search_filters instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return search_filters(name)


def get_filters(filter_name_list: list[str]) -> list[Filter]:
    """
    Retrieve Filter instances from the database by name.

    Parameters
    ----------
    filter_name_list : list[str]
        Filter names (e.g. as returned by search_filters).

    Returns
    -------
    list[Filter]
        List of Filter instances.
    """
    return [Filter(fname) for fname in filter_name_list]


def get(filter_names: list[str]) -> list[Filter]:
    """Deprecated: use get_filters instead."""
    warnings.warn(
        "filter.get is deprecated; use get_filters instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_filters(filter_names)


def plot_filters(filter_names: list[str], legend: bool = True) -> None:
    """
    Plot transmissivity curves for a set of filters.

    Parameters
    ----------
    filter_names : list[str]
        The filter names to plot.
    legend : bool, optional
        Whether to show the legend (default True).
    """
    fig, ax = plt.subplots()
    for fname in filter_names:
        Filter(fname).plot_transmissivity(ax)
    if legend:
        plt.legend()
    plt.show()


def plot(filter_names: list[str], legend: bool = True) -> None:
    """Deprecated: use plot_filters instead."""
    warnings.warn(
        "filter.plot is deprecated; use plot_filters instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    plot_filters(filter_names, legend)
