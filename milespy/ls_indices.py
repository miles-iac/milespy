# -*- coding: utf-8 -*-
"""Line-strength index definitions and computation (Lick/IDS-style indices)."""

from __future__ import annotations

import logging
import re
import sys
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import ascii
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator

from .configuration import get_config_file

logger = logging.getLogger("milespy.ls_indices")

lsfile = get_config_file("ls_indices_full.def")
lsindex_names = ascii.read(lsfile, comment=r"\s*#")["names"]
logging.debug(f"Initialized line strength with {len(lsindex_names)} indeces.")


class LineStrengthIndexConfig(BaseModel):
    """
    Validated configuration for a line-strength index from the database.

    Ensures index_type is "A" (atomic) or "M" (molecular), and that band edges
    are finite, ordered (left < right within each band), and convertible to Ångström.
    """

    name: str
    index_type: Literal["A", "M"]
    band_blue_left: float
    band_blue_right: float
    band_centre_left: float
    band_centre_right: float
    band_red_left: float
    band_red_right: float

    @field_validator("index_type", mode="before")
    @classmethod
    def check_index_type(cls, v):
        if v not in ("A", "M"):
            raise ValueError("index_type must be 'A' (atomic) or 'M' (molecular)")
        return v

    @field_validator(
        "band_blue_left",
        "band_blue_right",
        "band_centre_left",
        "band_centre_right",
        "band_red_left",
        "band_red_right",
    )
    @classmethod
    def check_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError(
                "band edges must be finite and convertible to length (e.g. Ångström)"
            )
        return float(v)

    @model_validator(mode="after")
    def check_ordered(self):
        if self.band_blue_left >= self.band_blue_right:
            raise ValueError("band_blue_left must be less than band_blue_right")
        if self.band_centre_left >= self.band_centre_right:
            raise ValueError("band_centre_left must be less than band_centre_right")
        if self.band_red_left >= self.band_red_right:
            raise ValueError("band_red_left must be less than band_red_right")
        return self


class LineStrengthIndex:
    """
    Line-strength index definition (Lick/IDS-style bandpasses).

    Attributes
    ----------
    name : str
        Name of the index.
    type : str
        Index type: "A" (atomic) or "M" (molecular). Atomic indices are in Ångström;
        molecular indices are in magnitudes.
    bands : np.ndarray
        Internal array of six band edges in Ångström: [band_blue_left, band_blue_right,
        band_centre_left, band_centre_right, band_red_left, band_red_right].
        Blue and red bands define the pseudo-continuum; the centre band is the feature.
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
        """
        Define a line-strength index by its bandpass edges.

        Parameters
        ----------
        name : str
            Index name (e.g. "Mg2", "Fe5015").
        index_type : "A" or "M"
            "A" for atomic (equivalent width in Ångström); "M" for molecular (magnitude).
        band_blue1, band_blue2 : quantity
            Left and right edges of the blue pseudo-continuum band (wavelength, e.g. u.AA).
        band_centre1, band_centre2 : quantity
            Left and right edges of the central feature band.
        band_red1, band_red2 : quantity
            Left and right edges of the red pseudo-continuum band.
        """
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
    def from_database(name: str) -> LineStrengthIndex:
        """
        Create an index from the name in the database.

        Reads from the configuration index file. Use :meth:`search_indices_in_database`
        to find index names. Validates database row with LineStrengthIndexConfig before
        building the instance.

        Parameters
        ----------
        name : str
            Name of the index to load.

        Returns
        -------
        LineStrengthIndex

        Raises
        ------
        ValueError
            If the index is not in the database or band data is invalid.
        RuntimeError
            If multiple indices match the name.
        """
        tab = ascii.read(lsfile, comment=r"\s*#")
        names = tab["names"]
        if name not in names:
            raise ValueError(f"The index {name} is not on the database")
        idx = np.argwhere(names == name)
        if len(idx) > 1:
            raise RuntimeError("Multiple matching filters")

        row = int(idx[0][0])
        cfg = LineStrengthIndexConfig(
            name=name,
            index_type=str(tab["b7"][row]).strip(),
            band_blue_left=float(tab["b1"][row]),
            band_blue_right=float(tab["b2"][row]),
            band_centre_left=float(tab["b3"][row]),
            band_centre_right=float(tab["b4"][row]),
            band_red_left=float(tab["b5"][row]),
            band_red_right=float(tab["b6"][row]),
        )
        return LineStrengthIndex(
            cfg.name,
            cfg.index_type,
            cfg.band_blue_left * u.AA,
            cfg.band_blue_right * u.AA,
            cfg.band_centre_left * u.AA,
            cfg.band_centre_right * u.AA,
            cfg.band_red_left * u.AA,
            cfg.band_red_right * u.AA,
        )


def search_indices_in_database(name: str) -> list[str]:
    """
    Search for line-strength indices in the database by name (regex, case insensitive).

    Parameters
    ----------
    name : str
        Search string (regex pattern) to match index names.

    Returns
    -------
    list[str]
        List of index names matching the search.
    """
    reg = re.compile(name, re.IGNORECASE)
    filtered = list(filter(reg.search, lsindex_names))
    if len(filtered) == 0:
        logger.warning(
            "Cannot find index in our database. Available indices are:\n\n"
            + str(lsindex_names)
        )
    return filtered


def search(name: str) -> list[str]:
    """Deprecated: use search_indices_in_database instead."""
    warnings.warn(
        "ls_indices.search is deprecated; use search_indices_in_database instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return search_indices_in_database(name)


def get_indices_from_database(index_name_list: list[str]) -> list[LineStrengthIndex]:
    """
    Load LineStrengthIndex instances from the database by name.

    Parameters
    ----------
    index_name_list : list[str]
        Index names (e.g. as returned by search_indices_in_database).

    Returns
    -------
    list[LineStrengthIndex]
    """
    return [LineStrengthIndex.from_database(n) for n in index_name_list]


def get(lsindex_names: list[str]) -> list[LineStrengthIndex]:
    """Deprecated: use get_indices_from_database instead."""
    warnings.warn(
        "ls_indices.get is deprecated; use get_indices_from_database instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_indices_from_database(lsindex_names)


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


def line_strength_index(
    indices: list[LineStrengthIndex],
    ll: u.Quantity[u.AA],
    flux: u.Quantity,
    z: u.Quantity[u.km / u.s] | u.Quantity[u.dimensionless_unscaled],
    plot: bool = False,
    noise: u.Quantity | None = None,
    z_err: u.Quantity | None = None,
    sims: int = 0,
) -> LineStrengthDict:
    """
    Measure line-strength indices on a spectrum.

    Wavelength and flux are converted to a canonical form: wavelength in Ångström
    (linear step assumed), flux in dimensionless counts. Redshift z must be in
    velocity units (e.g. km/s); it is used to deredshift the wavelength to rest frame.

    Parameters
    ----------
    indices : list[LineStrengthIndex]
        Index definitions to compute.
    ll : ~astropy.units.Quantity
        Wavelength vector (e.g. u.AA); assumed linear steps.
    flux : ~astropy.units.Quantity
        Flux or counts as a function of wavelength (any flux density unit).
    z : ~astropy.units.Quantity
        Redshift (adimensional) or velocity shift (e.g. km/s).
    plot : bool, optional
        Whether to plot the bands (default False).
    noise : ~astropy.units.Quantity, optional
        Noise spectrum (for error simulations; not yet implemented).
    z_err : ~astropy.units.Quantity, optional
        Redshift error in velocity (for sims; not yet implemented).
    sims : int, optional
        Number of Monte Carlo simulations for errors (default 0). If > 0,
        raises NotImplementedError.

    Returns
    -------
    LineStrengthDict
        Dictionary of index name -> value (array or scalar).
    """
    if sims > 0:
        raise NotImplementedError(
            "Monte Carlo error simulations (sims > 0) are not implemented"
        )

    # Convert to canonical units
    ll = ll.to_value(u.AA)
    flux = flux.to_value()

    # Deredshift: rest wavelength = obs / (1 + z). Accept z as dimensionless or velocity.
    try:
        from astropy.constants import c

        z_dimless = (z.to_value(u.km / u.s) / c).to_value(u.dimensionless_unscaled)
    except u.UnitConversionError:
        z_dimless = z
    dll = ll / (1.0 + z_dimless)

    outindex = LineStrengthDict(
        (ind.name, np.full(flux.shape[:-1], np.nan)) for ind in indices
    )

    for ind in indices:
        good = (ind.bands[0] >= dll[0]) & (ind.bands[5] <= dll[-1])
        if not good:
            logger.warning(
                f"Index {ind.name} [{ind.bands[0]}, {ind.bands[5]}] "
                f"is outside of the spectral range [{dll[0]}, {dll[-1]}]"
            )
            continue
        outindex[ind.name] = _calc_index(ind.bands, ind.type, ind.name, dll, flux, plot)

    return outindex


def lsindex(
    indeces: list[LineStrengthIndex],
    ll: u.Quantity,
    flux: u.Quantity,
    z: u.Quantity,
    plot: bool = False,
    noise=None,
    z_err=None,
    sims: int = 0,
) -> LineStrengthDict:
    """Deprecated: use line_strength_index instead."""
    warnings.warn(
        "ls_indices.lsindex is deprecated; use line_strength_index instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return line_strength_index(
        indeces, ll, flux, z, plot=plot, noise=noise, z_err=z_err, sims=sims
    )
