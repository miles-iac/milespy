# -*- coding: utf-8 -*-
"""Magnitude computation from spectra and filter response (AB and Vega zeropoints)."""
from __future__ import annotations

import logging
import sys
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.io import fits
from astropy.units import Quantity
from pydantic import BaseModel
from scipy.interpolate import interp1d

from .configuration import get_config_file
from .filter import Filter

logger = logging.getLogger("milespy.magnitudes")

solar_ref_spec = get_config_file("sun_mod_001.fits")


class MagnitudeComputationConfig(BaseModel):
    """Validates zeropoint and options for magnitude computation."""

    zeropoint: Literal["AB", "VEGA"]
    sun: bool = False


class Magnitude(dict):
    def write(self, output=sys.stdout, format="basic", **kwargs):
        """
        Save the magnitude data in the requested format

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

        to_save = dict(self)
        for k in to_save:
            if np.isscalar(to_save[k]):
                to_save[k] = np.array([to_save[k]])

        tab = Table(data=to_save)
        ascii.write(tab, output, format=format, **kwargs)


def _load_zerofile(zeropoint):
    file = get_config_file("vega.sed")
    data = ascii.read(file, comment=r"\s*#")

    sed = {"wave": np.array(data["col1"]), "flux": np.array(data["col2"])}

    # If AB mags only need wavelength vector
    if zeropoint == "AB":
        sed["flux"] = 1.0 / np.power(sed["wave"], 2)
    elif zeropoint == "VEGA":
        # Normalizing the SED@ 5556.0\AA
        zp5556 = 3.44e-9  # erg cm^-2 s^-1 A^-1, Hayes 1985
        interp = interp1d(sed["wave"], sed["flux"])
        sed["flux"] *= zp5556 / interp(5556.0)

    return sed


# Pre-load the sed for the different zero points
zerosed = {"AB": _load_zerofile("AB"), "VEGA": _load_zerofile("VEGA")}


def _find_wavelength_limits(filter_: Filter) -> tuple[float, float]:
    """Return (wlow, whi) in Ångström for the filter's valid response range."""
    good = filter_.wave > 0.0
    return float(np.amin(filter_.wave[good])), float(np.amax(filter_.wave[good]))


def _select_spectral_range(
    wave: np.ndarray, flux: np.ndarray, wlow: float, whi: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract wave and flux in [wlow, whi]; return None if spectrum does not cover the range."""
    w = (wave >= wlow) & (wave <= whi)
    if not np.any(w):
        return None
    return wave[w], flux[..., w]


def _interpolate_filter_response(filter_: Filter, wave_sub: np.ndarray) -> np.ndarray:
    """Interpolate filter transmissivity onto the given wavelength grid (Ångström)."""
    good = filter_.wave > 0.0
    interp = interp1d(filter_.wave[good], filter_.trans[good])
    return interp(wave_sub)


def _compute_reference_flux(
    zeropoint_sed: dict, response: np.ndarray, wave_sub: np.ndarray
) -> float:
    """Integrate zeropoint SED times response over wave_sub (trapezoid)."""
    ref_flux = np.interp(wave_sub, zeropoint_sed["wave"], zeropoint_sed["flux"])
    return float(np.trapezoid(ref_flux * response, x=wave_sub))


def _compute_magnitude_from_flux(
    flux_sub: np.ndarray,
    response: np.ndarray,
    wave_sub: np.ndarray,
    ref_flux: float,
    cfact: float,
    zeropoint: Literal["AB", "VEGA"],
) -> np.ndarray:
    """Compute magnitude from flux density, response, and zeropoint; apply cfact and AB term if needed."""
    cvel = 2.99792458e18  # Speed of light in Ångström/s
    f = np.trapezoid(flux_sub * response, x=wave_sub, axis=-1)
    mag = -2.5 * np.log10(f / ref_flux)
    fmag = mag + cfact
    if zeropoint == "AB":
        fmag = fmag + 2.5 * np.log10(cvel) - 48.6  # Oke & Gunn 83
    return fmag


def compute_mags(
    wave: Quantity,
    flux: Quantity,
    filters: list[Filter],
    zeropoint: str,
    sun: bool = False,
) -> Magnitude:
    """
    Compute magnitudes for a spectrum in the given filters.

    Wavelength and flux are converted to canonical units: wavelength in Ångström,
    flux in any spectral flux density unit (e.g. erg/s/cm²/Å or L_sun/Å). The
    spectrum must cover each filter's wavelength range.

    Parameters
    ----------
    wave : ~astropy.units.Quantity
        Wavelength (e.g. u.AA).
    flux : ~astropy.units.Quantity
        Spectral flux density (same length as wave on the last axis).
    filters : list[Filter]
        Filters as provided by :meth:`milespy.filter.get_filters`.
    zeropoint : str
        "AB" or "VEGA".
    sun : bool, optional
        If True, output is in absolute magnitudes (assuming input spectrum is
        absolute flux, e.g. solar). If False, output is in apparent magnitude
        (input interpreted as flux at observer; default).

    Returns
    -------
    Magnitude
        Dictionary-like mapping filter name -> magnitude (array or scalar).
    """
    cfg = MagnitudeComputationConfig(zeropoint=zeropoint, sun=sun)
    zeropoint = cfg.zeropoint
    sun = cfg.sun

    wave_q = Quantity(wave, u.AA)
    flux_q = Quantity(flux)
    flux_arr = flux_q.to_value()
    wave_arr = wave_q.to_value(u.AA)

    dl = 1e-5  # 10 pc in Mpc, z=0; for absolute magnitudes
    if sun:
        cfact = -5.0 * np.log10(4.84e-6 / 10.0)  # for absolute magnitudes
    else:
        cfact = 5.0 * np.log10(1.7684e8 * dl)  # from lum to flux [erg/s/A/cm2]

    outmags = Magnitude((f.name, np.full(flux_arr.shape[:-1], np.nan)) for f in filters)
    zp_sed = zerosed[zeropoint]

    for filt in filters:
        wlow, whi = _find_wavelength_limits(filt)
        selected = _select_spectral_range(wave_arr, flux_arr, wlow, whi)
        if selected is None:
            logger.warning(
                f"Filter {filt.name} [{wlow},{whi}] is outside of "
                f"the spectral range [{np.amin(wave_arr)}, {np.amax(wave_arr)}]"
            )
            continue
        tmp_wave, tmp_flux = selected
        if np.amin(wave_arr) > wlow or np.amax(wave_arr) < whi:
            logger.warning(
                f"Filter {filt.name} [{wlow},{whi}] is outside of "
                f"the spectral range [{np.amin(wave_arr)}, {np.amax(wave_arr)}]"
            )
            continue

        response = _interpolate_filter_response(filt, tmp_wave)
        ref_flux = _compute_reference_flux(zp_sed, response, tmp_wave)
        fmag = _compute_magnitude_from_flux(
            tmp_flux, response, tmp_wave, ref_flux, cfact, zeropoint
        )
        outmags[filt.name] = fmag

    return outmags


def vacuum2air(wave_vac: np.ndarray | Quantity) -> np.ndarray | Quantity:
    """
    Convert wavelength from vacuum to air (refractive index formula).

    Parameters
    ----------
    wave_vac : array or ~astropy.units.Quantity
        Wavelength in vacuum (e.g. Ångström). If Quantity, returned in same unit.

    Returns
    -------
    array or ~astropy.units.Quantity
        Wavelength in air (same type and unit as input).
    """
    if isinstance(wave_vac, Quantity):
        w = wave_vac.to_value(u.AA)
        out = w / (
            1.0 + 2.735182e-4 + 131.4182 / w**2 + 2.76249e8 / w**4
        )
        return Quantity(out, u.AA)
    wave_air = wave_vac / (
        1.0 + 2.735182e-4 + 131.4182 / wave_vac**2 + 2.76249e8 / wave_vac**4
    )
    return wave_air


def _load_solar_spectrum():
    """
    Loads the references solar spectrum

    Parameters
    ----------
    None

    Returns
    -------
    array
        Vector with wavelength in air system and flux

    """

    hdu = fits.open(solar_ref_spec)
    tab = hdu[1].data

    wave_air = Quantity(vacuum2air(tab["WAVELENGTH"]), unit=u.AA)
    flux = Quantity(tab["FLUX"], unit=u.erg / (u.cm**2 * u.s * u.AA))

    return wave_air, flux


def sun_magnitude(
    filters: list[Filter] | None = None, zeropoint: str = "AB"
) -> Magnitude:
    """
    Compute the Sun's absolute magnitudes in the desired filters.

    Parameters
    ----------
    filters : list[Filter], optional
        Filters as provided by :meth:`milespy.filter.get_filters`. Default empty list.
    zeropoint : str, optional
        "AB" or "VEGA" (default "AB").

    Returns
    -------
    Magnitude
        Dictionary with solar absolute magnitude for each filter.
    """
    if filters is None:
        filters = []
    MagnitudeComputationConfig(zeropoint=zeropoint, sun=True)
    logger.info("Computing solar absolute magnitudes...")
    wave, flux = _load_solar_spectrum()
    outmags = compute_mags(wave, flux, filters, zeropoint, sun=True)

    return outmags
