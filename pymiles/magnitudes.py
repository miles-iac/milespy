# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np
from astropy.io import ascii
from astropy.io import fits
from scipy.interpolate import interp1d

from pymiles import get_config_file
from pymiles.filter import Filter

logger = logging.getLogger("pymiles.magnitudes")

solar_ref_spec = get_config_file("sun_mod_001.fits")


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

        tab = Table(data=dict(self))
        ascii.write(tab, output, format=format, **kwargs)


def _load_zerofile(zeropoint):
    file = get_config_file("vega.sed")
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


# Pre-load the sed for the different zero points
zerosed = {"AB": _load_zerofile("AB"), "VEGA": _load_zerofile("VEGA")}


def compute_mags(wave, flux, filters: [Filter], zeropoint, sun=False) -> Magnitude:
    """
    Compute the magnitudes given a spectra and a set of fluxes

    Core implementation of the magnitude computation, including the checks for
    the filter wavelength limits

    Parameters
    ----------
    wave : array
        Wavelength of the spectrum
    flux : array
        Spectrum
    filters : list[Filter]
        List of filters for which the magnitude is to be computed
    zeropoint : str
        Type of zero point. Either AB or VEGA
    sun : bool
        Flag to determine if the output is absolute magnitudes (True) or in
        flux (False) ??

    Returns
    -------
    Magnitude
    """
    # Defining some variables
    cvel = 2.99792458e18  # Speed of light in Angstron/sec
    dl = 1e-5  # 10 pc in Mpc, z=0; for absolute magnitudes
    if sun:
        cfact = -5.0 * np.log10(4.84e-6 / 10.0)  # for absolute magnitudes
    else:
        cfact = 5.0 * np.log10(1.7684e8 * dl)  # from lum[erg/s/A] to flux [erg/s/A/cm2]

    # Default is nan to mark an invalid range/filter
    outmag = Magnitude((f.name, np.nan) for f in filters)
    interp_zp = interp1d(zerosed[zeropoint].wave, zerosed[zeropoint].flux)

    # Computing the magnitude for each filter
    for filt in filters:
        # Finding the wavelength limits of the filters
        good = filt.wave > 0.0
        wlow = np.amin(filt.wave[good])
        whi = np.amax(filt.wave[good])

        # Selecting the relevant pixels in the input spectrum
        w = (wave >= wlow) & (wave <= whi)
        tmp_wave = wave[w]
        tmp_flux = flux[w]
        if (np.amin(wave) > wlow) or (np.amax(wave) < whi):
            logger.warning(
                "Filter "
                + filt.name
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
                + filt.name
                + " ["
                + str(wlow)
                + ","
                + str(whi)
                + "] has zero flux\t Returning nan"
            )
            continue

        # Interpolate the filter response to data wavelength
        interp = interp1d(
            filt.wave[good],
            filt.trans[good],
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

        outmag[filt.name] = fmag

    return outmag


def vacuum2air(wave_vac):
    """
    Converts wavelength from vacuum to air

    Parameters
    ----------
    array
        Wavelength in vacuum system

    Returns
    -------
    array
        Vector with wavelength in air system

    """

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

    wave_air = vacuum2air(tab["WAVELENGTH"])
    flux = tab["FLUX"]

    return wave_air, flux


def sun_magnitude(filters: list[Filter] = [], zeropoint="AB") -> Magnitude:
    """
    Computes the magnitude of Sun in the desired filters

    Parameters
    ----------
    filters: list[Filter]
        Filters as provided by :meth:`pymiles.filter.get`
    zeropoint:
        Type of zero point. Valid inputs are AB/VEGA

    Returns
    -------
    Magnitude
        Dictionary with solar mags for each filter

    """
    logger.info("Computing solar absolute magnitudes...")

    wave, flux = _load_solar_spectrum()
    outmags = compute_mags(wave, flux, filters, zeropoint, sun=True)

    return outmags
