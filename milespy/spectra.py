# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import typing
import warnings
from copy import copy

import numpy as np
from astropy import units as u
from astropy.constants import c
from spectres import spectres
from specutils import Spectrum
from tqdm import tqdm

from .configuration import get_config_file
from .filter import Filter
from .ls_indices import LineStrengthDict
from .ls_indices import LineStrengthIndex
from .ls_indices import lsindex
from .magnitudes import compute_mags
from .magnitudes import Magnitude
from .magnitudes import sun_magnitude

# ==============================================================================

logger = logging.getLogger("milespy.spectra")


class Spectra(Spectrum):
    """
    Class for storing spectra in milespy.

    This class inherits `Spectrum` from specutils, and can use
    all the methods of that class.

    The main difference is how we use the `meta` dictionary.
    We assume that all the values stored in meta have the same lengths as the
    number of spectra.  Thus, each value of each key univoquely refers to some
    information of a given spectra.  The exact keys in this dictionary will
    depend on the model creating this object.

    """

    warnings.filterwarnings("ignore")

    solar_ref_spec = get_config_file("sun_mod_001.fits")

    def __init__(self, *args, **kwargs):
        # We force the default behaviour of specutils < 2 of having the spectral
        # axis in the last index
        if "flux" in kwargs.keys():
            kwargs["spectral_axis_index"] = -1
        super().__init__(*args, **kwargs)

    @property
    def properties(self):
        return list(self.meta.keys())

    def __getattr__(self, attr):
        if attr not in ["meta", "_meta"]:
            if attr in self.meta.keys():
                if (
                    np.ndim(self.meta[attr]) > 0
                    and hasattr(self.meta[attr], "__len__")
                    and len(self.meta[attr]) == 1
                ):
                    return self.meta[attr][0]
                else:
                    return self.meta[attr]
            else:
                raise AttributeError

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
                if len(out.meta[k]) == self.nspec:
                    try:
                        out.meta[k] = out.meta[k][item]
                    except IndexError:
                        # This can happen now because specutils>=2.0 passes a
                        # list of `slice(None, None, None)` rather than an
                        # Ellipsis to allow for the spectral axis to be in any
                        # index.
                        out.meta[k] = out.meta[k][item[self.spectral_axis_index]]
                else:
                    out.meta[k] = self.meta[k]
            except TypeError:
                pass
        return out

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
        # wave = out.wave * (1.0 + redshift)
        # spec = out.spec / (1.0 + redshift)
        # out.update_basic_pars(wave, spec)
        out.redshift = redshift
        out.lsf_fwhm = out.lsf_fwhm / (1.0 + redshift)

        return out

    # @u.quantity_input # For some reason this does not work...
    def velocity_shift(self, v_los: u.Quantity[u.km / u.s]) -> Spectra:
        if v_los.isscalar:
            # Simply move the spectral axis
            return Spectra(
                spectral_axis=self.spectral_axis * (1.0 + v_los / c),
                flux=self.flux,
                meta=copy(self.meta),
            )
        else:
            if len(v_los) != self.nspec:
                raise ValueError(
                    "The size of the input velocity shift should"
                    "match the number of spectra"
                )

            # Preallocate for performance
            new_flux = np.empty(self.flux.shape)
            ref_axis = self.spectral_axis.to_value(u.AA)
            new_axis = np.empty_like(self.spectral_axis.to_value(u.AA))

            for i, vi in tqdm(enumerate(v_los), delay=3.0, total=self.nspec):
                # NB: This would be ideal to conserve flux, but it is very
                #   computationally expensive.
                # new_flux[i, :] = spectres(
                #     self.spectral_axis.to_value(u.AA),
                #     self.spectral_axis.to_value(u.AA) * (1. + vi / c),
                #     self.flux[i, :],
                #     fill=None,
                # ) << self.flux.unit
                fac = 1.0 + vi / c
                new_axis[:] = ref_axis[:] * fac
                new_flux[i, :] = np.interp(
                    ref_axis,
                    new_axis,
                    self.flux[i, :],
                )

            out = Spectra(
                spectral_axis=self.spectral_axis,
                flux=new_flux << self.flux.unit,
                meta=copy(self.meta),
            )
            return out

    def resample(self, new_wave: u.Quantity):
        """
        Resample the spectra


        Parameters
        ----------
        new_wave
            Spectral axis with the desired sampling for the spectra. These values
            are the positions of the bin centers.

        Notes
        -----
        This functions makes use of :func:`spectres.spectres` [1]_.
        Care must be taken when resampling near the boundaries of the spectral
        axis, which can cause `nan` values.

        .. [1] A. C. Carnall, "SpectRes: A Fast Spectral Resampling Tool in Python",
           arXiv:1705.05165

        """
        new_flux = spectres(
            new_wave.to_value(u.AA),
            self.spectral_axis.to_value(u.AA),
            self.flux,
            fill=None,
        )
        out = Spectra(
            spectral_axis=new_wave,
            flux=new_flux << self.flux.unit,
            meta=copy(self.meta),
        )
        return out

    def convolve(self, lsf: u.Quantity = u.Quantity(1, unit=u.AA), lsf_wave=None):
        """
        Returns a convolved version of the spectra. It does the convolution
        using the FWHM given in the input line spread function (LSF).

        Notes
        -----
        If output LSF < input LSF the sigma used for the convolution is set to
        very small values.

        Parameters
        ----------
        lsf: `~astropy.units.Quantity`
            Line spread function as a function of `lsf_wave`. This is the
            FWMH to be used in the convolution, thus, should be provide in units
            of wavelength. It accepts a scalar value, that is assumend constant
            for all wavelenghts.
        lsf_wave: `~astropy.units.Quantity`
            Associated wavelenghts to the values of `lsf`.

        Returns
        -------
        Spectra
            Object instance with convolved spectra

        """

        logger.info("Convolving spectra")

        if np.isscalar(lsf.to_value(u.AA)):
            out_lsf = np.full(self.npix, lsf) * lsf.unit
        else:
            out_lsf = np.interp(self.spectral_axis, lsf_wave, lsf)

        # In most cases this interpolation is trivial, but allows the
        # flexibility to do the convolution after trimming the spectra
        in_lsf = np.interp(
            self.spectral_axis, self.meta["lsf_wave"], self.meta["lsf_fwhm"]
        )

        # NB: there is a factor 2.355 between the FWHM and the \sigma of a gaussian
        sigma = np.sqrt(out_lsf**2 - in_lsf**2) / 2.355
        bad = np.isnan(sigma)
        sigma[bad] = 1e-10 * u.AA
        out_lsf[bad] = in_lsf[bad]

        if self.nspec == 1:
            outflux = Spectra._gaussian_filter1d(
                self.flux.value, sigma.to_value(self.spectral_axis.unit)
            )
        else:
            outshape = self.flux.shape
            outflux = np.empty(outshape)
            for index in np.ndindex(outshape[:-1]):
                flux = Spectra._gaussian_filter1d(self.flux[index].value, sigma)
                s = slice(None)
                outflux[index + (s,)] = flux

        out = Spectra(
            spectral_axis=self.spectral_axis,
            flux=outflux * self.flux.unit,
            meta=copy(self.meta),
        )
        out.meta["lsf_fwhm"] = out_lsf
        out.meta["lsf_wave"] = self.spectral_axis
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
            Filters as provided by :meth:`milespy.filter.get`
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
            Indeces as provided by :meth:`milespy.ls_indices.get`

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

    def _update_mass(self, mass):
        if "mass" in self.meta.keys():
            previous_mass = self.meta["mass"]
        else:
            previous_mass = np.ones(mass.shape) << u.Msun
            self.meta["mass"] = previous_mass

        ratio = mass.to_value(u.Msun) / previous_mass.to_value(u.Msun)
        if np.ndim(ratio) == 0:
            if ratio <= 0.0:
                raise ValueError("Trying to set zero or negative mass to an spectra")
        else:
            # This can happens for spectra in a SFH that have a total contribution
            # of zero
            ratio[np.isnan(ratio)] = 0.0

        self.meta["mass"] *= ratio
        self.meta["Mass_total"] *= ratio
        self.meta["Mass_remn"] *= ratio
        self.meta["Mass_star"] *= ratio
        self.meta["Mass_star_remn"] *= ratio
        self.meta["Mass_gas"] *= ratio

    def _apply_mass(self, mass):
        if np.ndim(mass) == 0:
            m = mass
        else:
            m = mass[:, np.newaxis]

        out = Spectra(
            flux=self.flux * m, spectral_axis=self.spectral_axis, meta=copy(self.meta)
        )
        out._update_mass(mass)
        return out
