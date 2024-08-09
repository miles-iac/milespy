# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import typing
import warnings
from copy import copy

import numpy as np
from astropy import units as u
from astropy.io import ascii
from scipy import interpolate
from specutils import Spectrum1D

from .configuration import get_config_file
from .filter import Filter
from .ls_indices import LineStrengthDict
from .ls_indices import LineStrengthIndex
from .ls_indices import lsindex
from .magnitudes import compute_mags
from .magnitudes import Magnitude
from .magnitudes import sun_magnitude
from .misc import log_rebin
from .misc import log_unbinning

# ==============================================================================

logger = logging.getLogger("pymiles.spectra")


class Spectra(Spectrum1D):
    """
    Class for storing spectra in pymiles.

    This class inherits `Spectrum1D` from specutils, and can use
    all the methods of that class.

    The main difference is how we use the `meta` dictionary.
    We assume that all the values stored in meta have the same lengths as the
    number of spectra.  Thus, each value of each key univoquely refers to some
    information of a given spectra.  The exact keys in this dictionary will
    depend on the model creating this object.

    """

    warnings.filterwarnings("ignore")

    solar_ref_spec = get_config_file("sun_mod_001.fits")
    emiles_lsf = get_config_file("EMILES.lsf")

    @property
    def npix(self):
        if len(self.data.shape) > 1:
            return self.data.shape[1]
        else:
            return self.data.shape[0]

    @property
    def dim(self):
        return self.data.shape[:-1]

    @property
    def nspec(self):
        if len(self.data.shape) > 1:
            return np.prod(self.data.shape[:-1])
        else:
            return 1

    def __getitem__(self, item):
        out = super().__getitem__(item)
        for k in out.meta.keys():
            try:
                if len(out.meta[k]) > 1:
                    out.meta[k] = out.meta[k][item]
            except TypeError:
                pass
        return out

    # -----------------------------------------------------------------------------
    def compute_lsf(self):
        # This information could be given in the repository files!
        """
        Returns the line-spread function (LSF) given a source and wavelength from self

        Returns
        -------
        Spectra
            Object instance with LSF info included

        """

        cvel = 299792.458
        self.lsf_wave = self.wave
        if self.source == "MILES_SSP":
            self.lsf_fwhm = 2.51 * np.ones(self.npix)
            self.lsf_vdisp = cvel * (self.lsf_fwhm / 2.355) / self.wave

        elif self.source == "MILES_STARS":
            self.lsf_fwhm = 2.50 * np.ones(self.npix)
            self.lsf_vdisp = cvel * (self.lsf_fwhm / 2.355) / self.wave

        elif self.source == "CaT_SSP":
            self.lsf_fwhm = 1.50 * np.ones(self.npix)
            self.lsf_vdisp = cvel * (self.lsf_fwhm / 2.355) / self.wave

        elif self.source == "CaT_STARS":
            self.lsf_fwhm = 1.50 * np.ones(self.npix)
            self.lsf_vdisp = cvel * (self.lsf_fwhm / 2.355) / self.wave

        elif self.source == "EMILES_SSP":
            tab = ascii.read(self.emiles_lsf)
            wave = tab["col1"]
            fwhm = tab["col2"]
            sigma = tab["col3"]
            f_fwhm = interpolate.interp1d(wave, fwhm)
            f_vdisp = interpolate.interp1d(wave, sigma)
            self.lsf_fwhm = f_fwhm(self.wave)
            self.lsf_vdisp = f_vdisp(self.wave)

        else:
            raise ValueError(
                self.source
                + " is not a valid entry."
                + "Allowed values: MILES_SSP/MILES_STARS/CaT_SSP/CaT_STARS/EMILES"
            )

        return

    def redshift_spectra(self, redshift=None):
        # This may still be required because it also changes the LSF
        """
        Returns a copy of the instance with a redshifted wavelength vector,
        spectra and LSF

        Parameters
        ----------
        redshift:
            Desired redshift

        Returns
        -------
        Spectra
            Object instance with spectra redshifted and updated info

        """
        logger.info("Redshifting spectra ...")

        out = copy(self)
        wave = out.wave * (1.0 + redshift)
        # spec = out.spec / (1.0 + redshift)
        # out.update_basic_pars(wave, spec)
        out.redshift = redshift
        out.lsf_wave = wave
        out.lsf_fwhm = out.lsf_fwhm / (1.0 + redshift)
        out.lsf_vdisp = out.lsf_vdisp / (1.0 + redshift)

        return out

    def log_rebin(self, velscale=None):
        """
        Returns a logrebinned version of the spectra

        Notes
        -----
        Currently this functions is not supported and will raise a
        `NotImplementedError`

        Parameters
        ----------
        velscale:
            Desired velocity scale in km/s. Computed automatically if None.

        Returns
        -------
        Spectra
        """
        raise NotImplementedError
        logger.info("Ln-rebining the spectra")

        lamRange = [self.spectral_axis[0].value, self.spectral_axis[-1].value]
        outshape = self.flux.shape

        # Call once just to know the correct output size
        if velscale is not None:
            lspec, lwave, velscale = log_rebin(
                lamRange, np.empty(self.flux.shape[-1]), velscale=velscale, flux=False
            )
            outshape = self.flux.shape[:-1] + (len(lwave),)

        outflux = np.empty(outshape)
        for index in np.ndindex(outshape[:-1]):
            lspec, lwave, velscale = log_rebin(
                lamRange, self.flux[index].value, velscale=velscale, flux=False
            )
            s = slice(None)
            outflux[index + (s,)] = lspec

        # TODO: What about the LSF?

        return Spectra(
            spectral_axis=u.Quantity(np.exp(lwave), unit=self.spectral_axis.unit),
            flux=u.Quantity(outflux, unit=self.flux.unit),
            meta=self.meta,
        )

    def log_unbin(self):
        """
        Returns a un-logbinned version of the spectra

        Notes
        -----
        Currently this functions is not supported and will raise a
        `NotImplementedError`

        Returns
        -------
        Spectra
            Object instance with linearly binned spectra and updated info

        """
        raise NotImplementedError
        logger.info("Unbin ln spectra")

        lamRange = np.log([self.spectral_axis[0].value, self.spectral_axis[-1].value])
        outflux = np.empty(self.flux.shape)
        for index in np.ndindex(outflux.shape[:-1]):
            lspec, lwave = log_unbinning(lamRange, self.flux[index].value, flux=True)
            s = slice(None)
            outflux[index + (s,)] = lspec

        # TODO: What about the LSF?

        return Spectra(
            spectral_axis=u.Quantity(lwave, unit=self.spectral_axis.unit),
            flux=u.Quantity(outflux, unit=self.flux.unit),
            meta=self.meta,
        )

    # -----------------------------------------------------------------------------
    def convolve(self, lsf_wave=None, lsf=None, mode="FWHM"):
        """
        Returns a convolved version of the spectra

        Notes
        -----
        this assumes SAMPLING='lin'
        If output LSF < input LSF setting bad values to input LSF

        Parameters
        ----------
        lsf_wave:
            Wavelength vector of output LSF
        lsf:
            LSF vector
        mode:
            FWHM/VDISP. First one in Angstroms. Second one in km/s

        Returns
        -------
        Spectra
            Object instance with convolved spectra and updated info

        """

        logger.info("# Convolving spectra ...")

        out = copy(self)
        if mode == "FWHM":
            f_fwhm = interpolate.interp1d(lsf_wave, lsf)
            out_lsf = f_fwhm(out.wave)
            in_lsf = out.lsf_fwhm / 2.35
            out.lsf_fwhm = out_lsf
        elif mode == "VDISP":
            f_vdisp = interpolate.interp1d(lsf_wave, lsf)
            out_lsf = f_vdisp(out.wave)
            in_lsf = out.lsf_vdisp
            out.lsf_vdisp = out_lsf
        else:
            raise ValueError(
                "Mode " + mode + " not a valid entry. Allowed values are FWHM/VDISP"
            )

        sigma = np.sqrt(out_lsf**2 - in_lsf**2) / out.dwave
        bad = np.isnan(sigma)
        sigma[bad] = 1e-10

        out_spec = np.zeros_like(out.spec)
        for i in range(out.nspec):
            out_spec[:, i] = Spectra._gaussian_filter1d(out.spec[:, i], sigma)

        out.spec = out_spec

        return out

    # -----------------------------------------------------------------------------
    def tune_spectra(
        self,
        wave_lims=None,
        dwave=None,
        sampling=None,
        redshift=None,
        lsf_flag=False,
        lsf_mode="FWHM",
        lsf_wave=None,
        lsf=None,
    ):
        # This wrapper should be defined at the flask application layer
        """
        Returns the a tuned to desired input parameters

        Parameters
        ----------
        wave_lims:
            Wavelength limits in Angstroms
        dwave:
            Step in wavelength (in Angstroms)
        sampling:
            Type of sampling of the spectra. Valid inputs are lin/ln.
            Default: lin
        redshift:
            Desired redshift
        lsf_flag:
            Boolean flag to do LSF correction
        lsf_wave:
            Wavelength vector of output LSF
        lsf:
            LSF vector
        lsf_mode:
            FWHM/VDISP. First one in Angstroms. Second one in km/s

        Returns
        -------
        Spectra
            Object instance with tuned spectra and updated info

        """

        logger.debug("Tuning spectra ----------------------")

        out = copy(self)
        if lsf_wave is None:
            lsf_wave = out.wave

        # Resampling the spectra if necessary
        # to be done with specutils
        if (
            (wave_lims[0] != out.wave_init)
            or (wave_lims[1] != out.wave_last)
            or (dwave != self.dwave)
        ):
            pass
        #    out = self.resample_spectra(wave_lims=wave_lims, dwave=dwave)

        # Redshift spectra if necessary
        # to be done with specutils
        if redshift != out.redshift:
            pass
        #    out = out.redshift_spectra(redshift=redshift)

        # Convolving spectra is necessary
        if lsf is not None:
            out = out.convolve(lsf_wave=lsf_wave, lsf=lsf, mode="FWHM")

        # Log-rebinning spectra if necessary
        # is this really needed?
        if sampling == "ln":
            out = out.logrebin_spectra()

        return out

    def magnitudes(
        self,
        filters: list[Filter] = [],
        zeropoint="AB",
    ) -> Magnitude:
        """
        Returns the magnitudes of the input spectra given a list of filters in a file

        Parameters
        ----------
        filters: list[Filter]
            Filters as provided by :meth:`pymiles.filter.get`
        zeropoint:
            Type of zero point. Valid inputs are AB/VEGA

        Returns
        -------
        Magnitude
            Dictionary with output magnitudes for each spectra for each filter

        """
        logger.info("Computing absolute magnitudes")

        outmags = compute_mags(self.spectral_axis, self.flux, filters, zeropoint)

        return outmags

    # -----------------------------------------------------------------------------
    def line_strength(self, indeces: list[LineStrengthIndex]) -> LineStrengthDict:
        """
        Returns the LS indices of the input spectra given a list of index definitions

        Parameters
        ----------
        indeces: list[LineStrenghtIndex]
            Indeces as provided by :meth:`pymiles.ls_indices.get`

        Returns
        -------
        LineStrengthDict
            Dictionary with output LS indices for each spectra for each index

        """
        logger.info("Computing Line-Strength indices")
        outls = lsindex(
            indeces,
            self.spectral_axis,
            self.flux,
            self.redshift,
        )

        """
        nls = len(indices)
        indices = np.zeros((nls, self.nspec))

        for i in tqdm(range(self.nspec), delay=3.):
            names, indices[:, i], dummy = lsindex(
                self.spectral_axis,
                self.flux,
                self.flux * 0.1,
                self.redshift,
                0.0,
                self.lsfile,
            )

        outls = LineStrengthDict((names[i], indices[i, :]) for i in range(nls))
        """

        return outls

    @staticmethod
    def _gaussian_filter1d(spec, sig):
        """
        Convolve a spectrum by a Gaussian with different sigma for every pixel.

        If all sigma are the same this routine produces the same output as
        scipy.ndimage.gaussian_filter1d, except for the border treatment.
        Here the first/last p pixels are filled with zeros.
        When creating a template library for SDSS data, this implementation
        is 60x faster than a naive for loop over pixels.

        Parameters
        ----------
        spec:
            vector with the spectrum to convolve
        sig:
            vector of sigma values (in pixels) for every pixel

        Returns
        -------
        spec: ndarray
            Spectrum convolved with a Gaussian with dispersion sig

        """
        sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
        p = int(np.ceil(np.max(3 * sig)))
        m = 2 * p + 1  # kernel size
        x2 = np.linspace(-p, p, m) ** 2

        n = spec.size
        a = np.zeros((m, n))
        for j in range(m):  # Loop over the small size of the kernel
            a[j, p:-p] = spec[j : n - m + j + 1]

        gau = np.exp(-x2[:, None] / (2 * sig**2))
        gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

        conv_spectrum = np.sum(a * gau, 0)

        return conv_spectrum

    def mass_to_light(
        self, filters: list[Filter], mass_in: typing.Union[str, list[str]] = "star+remn"
    ) -> dict:
        """
        Computes the mass-to-light ratios of models in the desired filters

        Parameters
        ----------
        filters: list[Filter]
            Filters as provided by the method 'get_filters"
        mass_in: str | list[str]
            What mass to take into account for the ML. It can be given as a list,
            so that it returns a dictionary for each type.
            Valid values are: total, star, remn, star+remn, gas

        Returns
        -------
        dict
            Dictionary with mass-to-light ratios for each SSP model and filter.
            If mass_in is a list, the first key is the type of ML.

        """
        logger.info("Computing mass-to-light ratios")

        if type(mass_in) is str:
            mass_in = [mass_in]

        #  We need to choose a system. For M/Ls this is irrelevant
        zeropoint = "AB"
        mags = self.magnitudes(filters=filters, zeropoint=zeropoint)
        msun = sun_magnitude(filters=filters, zeropoint=zeropoint)

        outmls = {}
        logger.debug(f"{self.meta.keys()}")
        for m in mass_in:
            if m == "total":
                mass = self.meta["Mass_total"]
            elif m == "remn":
                mass = self.meta["Mass_remn"]
            elif m == "star":
                mass = self.meta["Mass_star"]
            elif m == "star+remn":
                mass = self.meta["Mass_star_remn"]
            elif m == "gas":
                mass = self.meta["Mass_gas"]
            else:
                raise ValueError(
                    "Mass type not allowed. "
                    "Valid options are total, star, remn, star+remn, gas"
                )

            outmls[m] = self._single_type_mass_to_light(filters, mass, mags, msun)

        # If only a single mass is requested we omit the information in the
        # returned dictionary
        if len(mass_in) == 1:
            return outmls[mass_in[0]]
        else:
            return outmls

    def _single_type_mass_to_light(self, filters: list[Filter], mass, mags, msun):
        outmls = {}
        for f in filters:
            outmls[f.name] = (mass / 1.0) * 10 ** (
                -0.40 * (msun[f.name] - mags[f.name])
            )
        return outmls

    def _assign_mass(self, mass):
        if np.ndim(mass) == 0:
            m = mass
        else:
            m = mass[:, np.newaxis]

        out = self.multiply(m, handle_meta="first_found")
        out.meta["mass"] = mass
        return out
