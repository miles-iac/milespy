# -*- coding: utf-8 -*-
"""Spectrum container with metadata, resampling, convolution, and magnitude/index computation."""

from __future__ import annotations

import logging
import typing
from copy import copy

import numpy as np
from astropy import units as u
from astropy.constants import c
from spectres import spectres
from specutils import Spectrum
from tqdm import tqdm

from .configuration import get_config_file
from .filter import Filter
from .line_strength_indices import line_strength_index
from .line_strength_indices import LineStrengthDict
from .line_strength_indices import LineStrengthIndex
from .magnitudes import compute_mags
from .magnitudes import compute_sbf_mags
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
    information of a given spectra.      The exact keys in this dictionary will
    depend on the model creating this object.

    """

    solar_ref_spec = get_config_file("sun_mod_001.fits")

    def __init__(self, *args, **kwargs):
        # We force the default behaviour of specutils < 2 of having the spectral
        # axis in the last index
        if "flux" in kwargs.keys():
            kwargs["spectral_axis_index"] = -1
        super().__init__(*args, **kwargs)

    @property
    def properties(self) -> list:
        """List of metadata keys (e.g. age, metallicity) attached to each spectrum."""
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
    def npix(self) -> int:
        """Number of spectral pixels (length of the spectral axis)."""
        if len(self.data.shape) > 1:
            return self.data.shape[1]
        return self.data.shape[0]

    @property
    def dim(self) -> tuple:
        """Shape of the spectrum array excluding the spectral axis (e.g. (n_models,) or (ny, nx))."""
        return self.data.shape[:-1]

    @property
    def nspec(self) -> int:
        """Total number of spectra (product of non-spectral dimensions)."""
        if len(self.data.shape) > 1:
            return int(np.prod(self.data.shape[:-1]))
        return 1

    @property
    def has_variance(self) -> bool:
        """True if variance spectra are attached in meta['flux_var']."""
        return "flux_var" in self.meta

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

    def redshift_spectra(self, redshift=None) -> Spectra:
        """
        Return a copy with redshift and line-spread function (LSF) metadata updated.

        Does not change the spectral axis or flux; only sets the redshift
        attribute and scales the LSF FWHM by 1/(1+z) for use in later processing.

        Parameters
        ----------
        redshift : float or ~astropy.units.Quantity, optional
            Desired redshift (dimensionless or velocity).

        Returns
        -------
        Spectra
            Copy with updated redshift and lsf_fwhm metadata.
        """
        logger.info("Redshifting spectra ...")
        out = copy(self)
        out._redshift = redshift
        out.lsf_fwhm = out.lsf_fwhm / (1.0 + redshift)
        return out

    def velocity_shift(self, v_los: u.Quantity[u.km / u.s]) -> Spectra:
        """
        Apply a line-of-sight velocity shift to the spectrum (Doppler shift).

        Parameters
        ----------
        v_los : ~astropy.units.Quantity
            Line-of-sight velocity (e.g. km/s); scalar or array of length nspec.

        Returns
        -------
        Spectra
            New instance with shifted spectral axis (and flux if per-spectrum velocities).
        """
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

    def resample(self, new_wave: u.Quantity) -> Spectra:
        """
        Resample the spectra onto a new wavelength grid.

        Uses flux-conserving resampling via spectres. Values at bin centers
        are interpolated; boundaries may yield NaN.

        Parameters
        ----------
        new_wave : ~astropy.units.Quantity
            New spectral axis (e.g. wavelength in u.AA); bin centers.

        Returns
        -------
        Spectra
            New instance with the resampled flux and spectral_axis.

        Notes
        -----
        Uses :func:`spectres.spectres` [1]_. Care at boundaries may cause NaN.

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

    def convolve(
        self, lsf: u.Quantity = u.Quantity(1, unit=u.AA), lsf_wave=None
    ) -> Spectra:
        """
        Convolve the spectra with a Gaussian kernel given by the line-spread function (LSF).

        The LSF is specified as FWHM (full width at half maximum) in wavelength.
        Convolution sigma is derived from the difference between target and current LSF.
        If target LSF is smaller than the current one, a very small sigma is used.

        Parameters
        ----------
        lsf : ~astropy.units.Quantity
            LSF FWHM (wavelength); scalar (constant with wavelength) or array.
        lsf_wave : ~astropy.units.Quantity, optional
            Wavelengths for each LSF value (required if lsf is not scalar).

        Returns
        -------
        Spectra
            New instance with convolved flux and updated lsf_fwhm/lsf_wave metadata.
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
    ) -> Spectra:
        """
        Return a copy tuned to desired wavelength range, sampling, redshift, and LSF.

        Parameters
        ----------
        wave_lims : array-like, optional
            Wavelength limits in Ångström.
        dwave : float, optional
            Wavelength step (Ångström).
        sampling : str, optional
            'lin' or 'ln' (log).
        redshift : float, optional
            Desired redshift.
        lsf_flag : bool, optional
            Whether to apply LSF (line-spread function) correction.
        lsf_wave : ~astropy.units.Quantity, optional
            Wavelength vector for the output LSF.
        lsf : ~astropy.units.Quantity, optional
            LSF FWHM (or velocity dispersion if lsf_mode='VDISP').
        lsf_mode : str, optional
            'FWHM' (wavelength) or 'VDISP' (km/s).

        Returns
        -------
        Spectra
            Tuned spectra (resampling/redshift may be no-ops; convolution applied if lsf given).
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
        filters: list[Filter] | None = None,
        zeropoint: str = "AB",
    ) -> Magnitude:
        """
        Compute magnitudes in the given filters (AB or Vega zeropoint).

        Parameters
        ----------
        filters : list[Filter], optional
            Filters as provided by :meth:`milespy.filter.get_filters`. Default empty.
        zeropoint : str, optional
            'AB' or 'VEGA'.

        Returns
        -------
        Magnitude
            Dictionary mapping filter name to magnitude (per spectrum).
        """
        if filters is None:
            filters = []
        logger.info("Computing absolute magnitudes")

        outmags = compute_mags(self.spectral_axis, self.flux, filters, zeropoint)

        return outmags

    def sbf(self) -> Spectra:
        """
        Return the surface brightness fluctuation (SBF) spectrum (variance / mean).

        Returns
        -------
        Spectra
            New instance with flux = flux_var / flux.

        Raises
        ------
        ValueError
            If variance spectra are not available.
        """
        if not self.has_variance:
            raise ValueError(
                "Variance spectra not available. Load the library with "
                "load_variance=True."
            )
        meta = copy(self.meta)
        flux_var = meta.pop("flux_var")
        return Spectra(
            spectral_axis=self.spectral_axis,
            flux=flux_var / self.flux,
            meta=meta,
        )

    def sbf_magnitudes(
        self,
        filters: list[Filter] | None = None,
        zeropoint: str = "AB",
    ) -> Magnitude:
        """
        Compute SBF magnitudes (Vazdekis et al. 2020, eq. 8).

        Parameters
        ----------
        filters : list[Filter], optional
            Filters as provided by :meth:`milespy.filter.get_filters`.
        zeropoint : str, optional
            'AB' or 'VEGA'.

        Returns
        -------
        Magnitude
            Dictionary mapping filter name to SBF magnitude (per spectrum).
        """
        if not self.has_variance:
            raise ValueError(
                "Variance spectra not available. Load the library with "
                "load_variance=True."
            )
        if filters is None:
            filters = []
        logger.info("Computing SBF magnitudes")
        return compute_sbf_mags(
            self.spectral_axis,
            self.flux,
            self.meta["flux_var"],
            filters,
            zeropoint,
        )

    def line_strength(self, indeces: list[LineStrengthIndex]) -> LineStrengthDict:
        """
        Compute line-strength (Lick/IDS) indices for the spectra.

        Parameters
        ----------
        indeces : list[LineStrengthIndex]
            Index definitions from :meth:`milespy.ls_indices.get_indices_from_database`.

        Returns
        -------
        LineStrengthDict
            Dictionary mapping index name to value (array or scalar per spectrum).
        """
        logger.info("Computing Line-Strength indices")
        outls = line_strength_index(
            indeces,
            self.spectral_axis,
            self.flux,
            self.redshift,
        )
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
        Compute mass-to-light (M/L) ratios in the desired filters.

        Uses solar absolute magnitudes and the chosen mass component (total, stellar,
        remnant, star+remnant, or gas).

        Parameters
        ----------
        filters : list[Filter]
            Filters as provided by :meth:`milespy.filter.get_filters`.
        mass_in : str or list[str], optional
            Mass component: 'total', 'star', 'remn', 'star+remn', 'gas'. If a list,
            returns a dict keyed by type.

        Returns
        -------
        dict
            Mass-to-light ratio per filter (and per mass type if mass_in is a list).
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

    def _single_type_mass_to_light(
        self, filters: list[Filter], mass, mags: Magnitude, msun: Magnitude
    ) -> dict:
        """Compute mass-to-light for one mass type and all filters (M/L = mass * 10^(-0.4*(Msun - mag)))."""
        outmls = {}
        for f in filters:
            outmls[f.name] = (mass / 1.0) * 10 ** (
                -0.40 * (msun[f.name] - mags[f.name])
            )
        return outmls

    def _update_mass(self, mass: u.Quantity) -> None:
        """Update metadata mass fields (total, star, remn, etc.) by the given ratio."""
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

    def _apply_mass(self, mass: u.Quantity) -> Spectra:
        """Scale flux by mass and update mass-related metadata; returns new Spectra."""
        if np.ndim(mass) == 0:
            m = mass
        else:
            m = mass[:, np.newaxis]

        out = Spectra(
            flux=self.flux * m, spectral_axis=self.spectral_axis, meta=copy(self.meta)
        )
        if "flux_var" in self.meta:
            out.meta["flux_var"] = self.meta["flux_var"] * (m**2)
        out._update_mass(mass)
        return out
