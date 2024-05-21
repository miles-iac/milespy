# -*- coding: utf-8 -*-
import glob
import logging
import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii

logger = logging.getLogger("pymiles.filter")
base_folder = pathlib.Path(__file__).parent.resolve() / "config_files" / "filters"


class Filter:
    def __init__(self, fname):
        filename = base_folder.as_posix() + "/" + fname + ".dat"
        if not os.path.exists(filename):
            assert ValueError("Filter " + fname + " does not exist in database")
        else:
            tab = ascii.read(filename, names=["wave", "trans"])
            tab["trans"] /= np.amax(tab["trans"])
            self.wave = tab["wave"]
            self.name = fname
            self.trans = tab["trans"]

    def plot(self, ax) -> None:
        ax.fill_between(
            self.wave,
            self.trans,
            alpha=0.5,
            label=self.name,
            edgecolor="k",
        )


fnames = glob.glob(f"{base_folder.as_posix()}/*.dat")
filter_names = np.sort([os.path.basename(x).split(".dat")[0] for x in fnames])
nfilters = len(filter_names)
logging.debug(f"Initialized library with {nfilters} filters")


def search(name):
    """
    Searches for a filter in database.

    Note
    _____
    Search is case insensitive
    The filter seach does not have to be precise. Substrings within filter
    names are ok.  It uses the python package 're' for regular expressions
    matching

    Arguments
    --------
    name: The search string to match filter names

    Return
    ------
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


def get(filter_names: [str]) -> [Filter]:
    """
    Retrieves filter from database

    Arguments
    --------
    filter_names: The filter names

    Return
    ------
    Object with filter's wavelength and normalised transmission

    """
    filters = [Filter(fname) for fname in filter_names]

    return filters


def plot(filter_names, legend=True):
    """
    Plot filters

    Arguments
    --------
    filter_names: The filter names
    legend:       Flag to turn on/off the legend

    Return
    ------
    Nothing

    """

    fig, ax = plt.subplots()

    _ = [Filter(fname).plot(ax) for fname in filter_names]

    if legend:
        plt.legend()

    plt.show()
