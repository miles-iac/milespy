# -*- coding: utf-8 -*-
import logging
import warnings
from typing import List
from typing import Optional

import h5py
import numpy as np
from astropy import units as u
from astropy.io import ascii
from astropy.units import Quantity
from scipy.spatial import Delaunay
from tqdm import tqdm

from .configuration import get_config_file
from .misc import interp_weights
from .repository import Repository
from .sfh import SFH
from .spectra import Spectra

logger = logging.getLogger("milespy.ssp")

_ssp_attr_units = {
    "age": u.Gyr,
    "alpha": u.dex,
    "met": u.dex,
    "imf_slope": u.dimensionless_unscaled,
}


class SSPLibrary(Repository):
    """
    Single stellar population (SSP) model library.

    This class is used to generate SSP spectra from the repository files.

    Attributes
    ----------
    models: Spectra
        Spectra of all the SSP that form the loaded library
    avail_alphas: list
        Available alpha/Fe values in the loaded library
    avail_imfs: list
        Available initial mass function (IMF) slopes in the loaded library
    source: str
        Name of input models being used
    version: str
        Version number of the models
    """

    warnings.filterwarnings("ignore")

    emiles_lsf = get_config_file("EMILES.lsf")

    # -----------------------------------------------------------------------------
    def __init__(
        self,
        source="MILES_SSP",
        version="9.1",
        isochrone="P",
        imf_type="ch",
    ):
        """
        Creates an instance of the class

        Parameters
        ----------
        source: str, default: "MILES_SSP"
            Name of input models to use.
            If none of the options below, it will assume that `source` is the path
            to an external model repository file.
            Valid inputs are
            MILES_SSP/CaT_SSP/EMILES_SSP
        version: str, default: "9.1"
            version number of the models
        isochrone: str, default: "P"
            Type of isochrone to use. Valid inputs are P/T for Padova+00
            and BaSTI isochrones respectively
        imf_type: str, default: "ch"
            Type of IMF shape. Valid inputs are ch/ku/kb/un/bi

        Notes
        -----
        We limit the choice of models to a given isochrone and imf_type for
        effective loading. Otherwise it can take along time to upload the entire
        dataset

        Returns
        -------
        SSPLibrary

        """
        repo_filename = self._get_repository(source, version)
        self._assert_repository_file(repo_filename)

        # Opening the relevant file in the repository and selecting the desired
        # models at init
        f = h5py.File(repo_filename, "r")
        total_nspec = len(f["age"])

        idx = np.logical_and(
            np.equal(f["imf_type"][...], imf_type.encode()),
            np.equal(f["isochrone"][...], isochrone.encode()),
        )
        self.nspec = np.sum(idx)
        if self.nspec == 0:
            raise ValueError("No cases found with those specs.")

        avail_alphas = np.unique(f["alpha"][idx])
        self.fixed_alpha = len(avail_alphas) == 1
        self.avail_alphas = avail_alphas[~np.isnan(avail_alphas)]

        avail_imfs = np.unique(f["imf_slope"][idx])
        self.avail_imfs = avail_imfs[~np.isnan(avail_imfs)]

        logger.debug(f"{self.nspec} / {total_nspec} cases found")
        logger.debug(f"Fixed_alpha: {self.fixed_alpha}")

        if self.fixed_alpha:
            logger.debug(f"Available alphas: {self.avail_alphas}")

        wave = np.array(f["wave"])
        spec = np.array(f["spec/" + isochrone + "/" + imf_type + "/data"])[...]

        self.source = source
        self.version = version

        fullpath = np.array(np.array(f["filename"][idx]), dtype="str")[0]
        filename = []
        for fullpath in np.array(np.array(f["filename"][idx]), dtype="str"):
            filename.append((fullpath.split("/"))[-1].split(".fits")[0])

        meta = {
            "index": np.arange(self.nspec),
            "isochrone": np.array(np.array(f["isochrone"][idx]), dtype="str"),
            "imf_type": np.array(np.array(f["imf_type"][idx]), dtype="str"),
            "filename": np.array(filename),
        }
        # Standard set of keys available in the repository that are taken
        # care of manually
        base_keys = ["spec", "wave", "isochrone", "imf_type", "filename"]
        # Other information in the repository, store them as arrays
        for k in f.keys():
            if k not in base_keys:
                meta[k] = np.array(f[k])[idx]
                if k in _ssp_attr_units:
                    meta[k] <<= _ssp_attr_units[k]

        f.close()

        super().__init__(
            Spectra(
                spectral_axis=Quantity(wave, unit=u.AA),
                flux=Quantity(spec.T, unit=u.L_sun / u.M_sun / u.AA),
                meta=meta,
            )
        )

        lsf_wave, lsf_fhwm = self._compute_lsf(source)
        self._models.meta["lsf_wave"] = lsf_wave
        self._models.meta["lsf_fwhm"] = lsf_fhwm

        logger.info(source + " models loaded")

    @u.quantity_input
    def in_range(
        self,
        age_lims: u.Quantity[u.Gyr],
        met_lims: u.Quantity[u.dex],
        alpha_lims: Optional[u.Quantity[u.dex]] = None,
        imf_slope_lims: Optional[u.Quantity[u.dimensionless_unscaled]] = None,
        mass: u.Quantity[u.Msun] = u.Quantity(1, unit=u.Msun),
    ) -> Spectra:
        """
        Extracts all SSP models within selected limits

        Parameters
        ----------
        age_lims:
            tuple with age limits
        met_lims:
            tuple with metallicity limits
        alpha_lims:
            tuple with alpha limits
        imf_slope_lims:
            tuple with IMF slope limits
        mass:
            mass of each SSP

        Raises
        ------
        ValueError
            If there is no matching SSP.

        Returns
        -------
        Spectra
            Spectra in the selected ranges

        """

        logger.debug("Searching for models within parameters range")

        if imf_slope_lims is None:
            imf_slope_lims = np.array([0.0, 5.0])

        if self.fixed_alpha or alpha_lims is None:
            idx = (
                (self.models.meta["age"] >= age_lims[0])
                & (self.models.meta["age"] <= age_lims[1])
                & (self.models.meta["met"] >= met_lims[0])
                & (self.models.meta["met"] <= met_lims[1])
                & (self.models.meta["imf_slope"] >= imf_slope_lims[0])
                & (self.models.meta["imf_slope"] <= imf_slope_lims[1])
            )
        else:
            idx = (
                (self.models.meta["age"] >= age_lims[0])
                & (self.models.meta["age"] <= age_lims[1])
                & (self.models.meta["met"] >= met_lims[0])
                & (self.models.meta["met"] <= met_lims[1])
                & (self.models.meta["alpha"] >= alpha_lims[0])
                & (self.models.meta["alpha"] <= alpha_lims[1])
                & (self.models.meta["imf_slope"] >= imf_slope_lims[0])
                & (self.models.meta["imf_slope"] <= imf_slope_lims[1])
            )

        ncases = np.sum(idx)
        logger.debug(f"{ncases} cases found")

        if ncases == 0:
            raise ValueError("No matching SSPs")

        out = Spectra.__getitem__(self.models, idx)._apply_mass(mass)

        return out

    @u.quantity_input
    def in_list(
        self,
        age: u.Quantity[u.Gyr],
        met: u.Quantity[u.dex],
        alpha: Optional[u.Quantity[u.dex]] = None,
        imf_slope: Optional[u.Quantity[u.dimensionless_unscaled]] = None,
        mass: Optional[u.Quantity[u.Msun]] = None,
    ) -> Spectra:
        """
        Extracts a selected set of models available from the library.

        Parameters
        ----------
        age :
            list of ages to extract
        met :
            list of metallicities to extract
        alpha :
            list of alphas to extract
        imf_slope :
            list of IMF slopes to extract
        mass : default 1 solar mass
            mass of each SSP

        Notes
        -----
        All lists must have the same length.
        This function does not perform any interpolation.
        Values in the inputs have to be valid ages, mets, alpha, and imf_slopes
        for the input isochrone and imf_type.

        Raises
        ------
        ValueError
            If inputs do not have the same shape or there is no resulting SSP.

        Warns
        -----
        If the number of output spectra is different that the input values.

        Returns
        -------
        Spectra

        """
        if alpha is not None and self.fixed_alpha:
            raise ValueError(
                "This repository does not provide variable alpha:\n"
                + "Source: "
                + self.source
                + "Isochrone: "
                + self.models.meta["isochrone"][0]
                + "IMF: "
                + self.models.meta["imf_type"][0]
            )

        # Treating the input list
        age = np.array(age).ravel()
        met = np.array(met).ravel()
        alpha = np.array(alpha).ravel()
        imf_slope = np.array(imf_slope).ravel()

        # Checking they have the same number of elements
        if self.fixed_alpha or alpha is None:
            if len(set(map(len, (age, met, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")
        else:
            if len(set(map(len, (age, met, alpha, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")

        ncases = len(age)
        logger.debug("Searching for {ncases} SSPs")

        if mass is None:
            mass = Quantity(value=np.ones(ncases), unit=u.Msun)

        id = []
        for i in range(ncases):
            if self.fixed_alpha or alpha is None:
                idx = (
                    (np.array(self.models.meta["age"]) == age[i])
                    & (np.array(self.models.meta["met"]) == met[i])
                    & (np.array(self.models.meta["imf_slope"]) == imf_slope[i])
                )
            else:
                idx = (
                    (np.array(self.models.meta["age"]) == age[i])
                    & (np.array(self.models.meta["met"]) == met[i])
                    & (np.array(self.models.meta["alpha"]) == alpha[i])
                    & (np.array(self.models.meta["imf_slope"]) == imf_slope[i])
                )

            if np.any(idx):
                id = np.append(id, self.models.meta["index"][idx])

        good = np.array(id, dtype=int)
        ngood = len(good)
        logger.debug(f"{ngood} cases found")

        if ncases != ngood:
            if ngood == 0:
                raise ValueError(
                    "Check those values exist for isochrone: "
                    + self.models.meta["isochrone"][0]
                    + " and IMF_Type: "
                    + self.models.meta["imf_type"][0]
                    + ". Please check this agrees with init instance."
                )
            else:
                logger.warning(
                    f"Asked for {ncases} SSPs, but found only {ngood} matching ones"
                )

        out = Spectra.__getitem__(self.models, good)._apply_mass(mass)

        return out

    @u.quantity_input
    def closest(
        self,
        age: u.Quantity[u.Gyr],
        met: u.Quantity[u.dex],
        alpha: Optional[u.Quantity[u.dex]] = None,
        imf_slope: Optional[u.Quantity[u.dimensionless_unscaled]] = None,
        mass: u.Quantity[u.Msun] = Quantity(value=1.0, unit=u.Msun),
    ) -> Spectra:
        """
        Retrieve the closest SSP avaiable in the library

        Parameters
        ----------
        age:
            Desired age
        met:
            Desired metallicity
        alpha:
            Desired alpha
        imf_slope:
            Desired IMF slope
        mass:
            Mass of the SSP

        Returns
        -------
        Spectra
            Closest spectra from the repository.

        Raises
        ------
        RuntimeError
            If the values are out of the grid.
        ValueError
            If the provided parameters do not have the same shape.
        """
        return self.interpolate(
            age=age, met=met, alpha=alpha, imf_slope=imf_slope, mass=mass, closest=True
        )

    @u.quantity_input
    def interpolate(
        self,
        age: u.Quantity[u.Gyr],
        met: u.Quantity[u.dex],
        alpha: Optional[u.Quantity[u.dex]] = None,
        imf_slope: Optional[u.Quantity[u.dimensionless_unscaled]] = None,
        mass: u.Quantity[u.Msun] = Quantity(value=1.0, unit=u.Msun),
        closest: bool = False,
        simplex: bool = False,
        force_interp: List = [],
    ) -> Spectra:
        """
        Interpolates SSP models for certain parameters using Delaunay triangulation

        Parameters
        ----------
        age:
            Desired age
        met:
            Desired metallicity
        alpha:
            Desired alpha
        imf_slope:
            Desired IMF slope
        mass:
            Mass of the SSP
        closest:
            Return the closest spectra, rather than performing the interpolation.
            If only one interpolation is performed, all the spectra in the simplex
            vertices are returned.
        simplex:
            If only one set of input parameters is given, return all the spectra
            that form part of the simplex used for the interpolation. These spectra
            have the weights information in their `meta` dictionary.
        force_interp:
            Force the interpolation over the indicated variables, even if the
            asked alpha/imf_slope is sampled in the repository. Valid values
            are "alpha" and "imf_slope".

        Returns
        -------
        Spectra
            Interpolated spectrum.
            If closest == True, return the closest spectra from the repository,
            rather than doing the interpolation.

        Raises
        ------
        RuntimeError
            If the values are out of the grid.
        ValueError
            If the provided parameters do not have the same shape.
        """
        # Preprocess the input
        single_alpha = alpha is not None and np.ndim(alpha) == 0
        nan_alpha = alpha is None
        single_imf_slope = imf_slope is not None and np.ndim(imf_slope) == 0
        nan_imf = imf_slope is None

        age = np.atleast_1d(age)
        met = np.atleast_1d(met)
        alpha = np.atleast_1d(alpha)
        # For the IMF slope we can have floats being passed, so we make sure
        # that it is a Quantity object
        imf_slope = np.atleast_1d(imf_slope)
        if not nan_imf:
            imf_slope <<= u.dimensionless_unscaled

        wrong_shape = age.shape != met.shape
        if not nan_alpha:
            wrong_shape |= age.shape != alpha.shape
        if not nan_imf:
            wrong_shape |= age.shape != imf_slope.shape
        if wrong_shape:
            raise ValueError("The input parameters should all have the same shape")

        ninterp = len(age)

        # Checking input point is within the grid
        def _compute_bounds(val, model):
            min_model = np.amin(model)
            max_model = np.amax(model)
            if np.any(val < min_model) or np.any(val > max_model):
                raise RuntimeError(
                    f"Input parameters {val} is outside of model grid: "
                    f"({min_model},{max_model})"
                )

        _compute_bounds(age, self.models.meta["age"])
        _compute_bounds(met, self.models.meta["met"])

        if not nan_imf:
            _compute_bounds(imf_slope, self.models.meta["imf_slope"])

        if not self.fixed_alpha and not nan_alpha:
            _compute_bounds(alpha, self.models.meta["alpha"])

        ndims = 2
        extra_dim = 0

        if nan_imf and len(self.avail_imfs) > 1:
            raise ValueError(f"An IMF slope should be provided from: {self.avail_imfs}")

        # If there is a single available imf slope, there is no need to interpolate
        # over it. Also, if the users gives an imf_slope that **exactly** matches
        # one of the database, we fix the imf_slope for the interpolation
        interp_fix_imf_slope = len(self.avail_imfs) == 1 or (
            single_imf_slope and imf_slope in self.avail_imfs
        )
        # But this can be overruled
        interp_fix_imf_slope &= "imf_slope" not in force_interp
        if interp_fix_imf_slope:
            if nan_imf:
                imf_slope = self.avail_imfs[0]

            imf_mask = np.equal(self.models.meta["imf_slope"], imf_slope[0])
        else:
            imf_mask = np.full(self.models.meta["imf_slope"].shape, True)
            ndims += 1

        logger.debug(f"Fixed imf during the interpolation? {interp_fix_imf_slope}")

        # Same as above for the imf_slope, however, now we can have nan values in
        # the repository in the case of "base" models.
        interp_fix_alpha = (
            self.fixed_alpha
            or nan_alpha
            or (single_alpha and alpha in self.avail_alphas)
        )
        if self.fixed_alpha and not nan_alpha:
            logger.warning("There is no alpha-enhanced SSPs with this model choice")
        interp_fix_alpha &= "alpha" not in force_interp
        if interp_fix_alpha:
            if nan_alpha:
                alpha_mask = np.isnan(self.models.meta["alpha"])
            elif alpha[0] in self.avail_alphas:
                alpha_mask = np.equal(self.models.meta["alpha"], alpha[0])
            else:
                # There is a single alpha in all the repository, so no need to mask
                alpha_mask = np.full(self.models.meta["alpha"].shape, True)
        else:
            # Remove the base alpha (i.e., nans) from the interpolation
            alpha_mask = ~np.isnan(self.models.meta["alpha"])
            ndims += 1

        logger.debug(f"Fixed alpha during the interpolation? {interp_fix_alpha}")

        idx = alpha_mask & imf_mask
        # This is required for future extrapolation, e.g., in the SFH module
        self.idx = idx

        n_avail = np.sum(idx)

        logger.debug(f"Interpolating over a grid of {n_avail} spectra")
        logger.debug(f"Creating {ndims}-D Delaunay triangulation")

        self.params = np.empty((n_avail, ndims))
        self.params[:, 0] = self.models.meta["age"][idx]
        self.params[:, 1] = self.models.meta["met"][idx]

        if not interp_fix_imf_slope:
            self.params[:, 2 + extra_dim] = self.models.meta["imf_slope"][idx]
            extra_dim += 1

        if not interp_fix_alpha:
            self.params[:, 2 + extra_dim] = self.models.meta["alpha"][idx]

        self.tri = Delaunay(self.params, qhull_options="QJ")

        if closest:
            closest_idx = np.empty(ninterp, dtype=int)
        else:
            wave = self.models.spectral_axis
            spec = Quantity(
                value=np.empty((ninterp, self.models.data.shape[1])),
                unit=self.models.flux.unit,
            )

            new_meta = {
                "imf_type": np.full(ninterp, self.models.meta["imf_type"][0]),
                "met": met,
                "age": age,
                "lsf_wave": self.models.meta["lsf_wave"],
                "lsf_fwhm": self.models.meta["lsf_fwhm"],
            }
            if interp_fix_alpha:
                new_meta["alpha"] = np.full(ninterp, alpha)
                if not nan_alpha:
                    new_meta["alpha"] <<= u.dex
            if interp_fix_imf_slope:
                new_meta["imf_slope"] = np.full(ninterp, imf_slope)

            base_keys = list(new_meta.keys())
            for k in self.models.meta.keys():
                if k not in new_meta.keys():
                    if len(self.models.meta[k]) > 1:
                        if "U" not in self.models.meta[k].dtype.kind:
                            new_meta[k] = np.empty(
                                ninterp, dtype=self.models.meta[k].dtype
                            )
                            if hasattr(self.models.meta[k], "unit"):
                                new_meta[k] <<= self.models.meta[k].unit

        for i in tqdm(range(ninterp), delay=3.0):
            input_pt = [age[i].to_value(u.Gyr), met[i].to_value(u.dex)]
            if not interp_fix_imf_slope:
                input_pt.append(imf_slope[i].value)
            if not interp_fix_alpha:
                input_pt.append(alpha[i].value)

            vtx, wts = interp_weights(self.params, np.atleast_2d(input_pt), self.tri)
            vtx, wts = vtx.ravel(), wts.ravel()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Searching for the simplex around: {input_pt}")
                logger.debug(f"Simplex ids: {self.models.meta['index'][idx][vtx]}")
                logger.debug(f"Simple ages: {self.models.meta['age'][idx][vtx]}")
                logger.debug(f"Simplex met: {self.models.meta['met'][idx][vtx]}")
                logger.debug(f"Simplex weights: {wts}, norm: {np.sum(wts)}")

            if closest:
                if simplex and ninterp == 1:
                    out = Spectra.__getitem__(
                        self.models, self.models.meta["index"][idx][vtx]
                    )._apply_mass(mass)
                    return out
                else:
                    closest_idx[i] = vtx[np.argmax(wts)]

            else:
                spec[i, :] = np.dot(self.models.flux[idx, :][vtx].T, wts)

                # Interpolate the rest of the meta if possible
                for k in new_meta:
                    if len(self.models.meta[k]) > 1:
                        if k not in base_keys:
                            new_meta[k][i] = np.dot(self.models.meta[k][idx][vtx], wts)

        if closest:
            out = Spectra.__getitem__(self.models, closest_idx)._apply_mass(mass)
        else:
            out = Spectra(spectral_axis=wave, flux=spec, meta=new_meta)._apply_mass(
                mass
            )

        # Select the first and only spectra so that the users does not need to
        # do this all the time, manually
        if ninterp == 1:
            out = out[0]

        return out

    def _compute_lsf(self, source: str):
        # This information could be given in the repository files!
        """
        Returns the line-spread function (LSF) given a source

        Parameters
        ----------
        source:
            Name of input models to use. Valid inputs are MILES_SSP/CaT_SSP/EMILES_SSP


        Returns
        -------
        lsf_wave: Quantity
            Wavelenghts of the LSF
        lsf_fwhm: Quantity
            Value of the LSF in terms of full-width half-maximum
        """

        lsf_wave = self.models.spectral_axis
        npix = self.models.npix
        if source == "MILES_SSP":
            lsf_fwhm = 2.51 * np.ones(npix)

        elif source == "MILES_STARS":
            lsf_fwhm = 2.50 * np.ones(npix)

        elif source == "CaT_SSP":
            lsf_fwhm = 1.50 * np.ones(npix)

        elif source == "CaT_STARS":
            lsf_fwhm = 1.50 * np.ones(npix)

        elif source == "EMILES_SSP":
            tab = ascii.read(self.emiles_lsf)
            wave = tab["col1"]
            fwhm = tab["col2"]
            lsf_fwhm = np.interp(lsf_wave.to_value(u.AA), wave, fwhm)

        else:
            raise ValueError(
                self.source
                + " is not a valid entry."
                + "Allowed values: MILES_SSP/MILES_STARS/CaT_SSP/CaT_STARS/EMILES"
            )

        return lsf_wave, lsf_fwhm * u.AA

    def from_sfh(self, sfh: SFH) -> Spectra:
        """
        Compute the spectra derived from the input SFH.

        Parameters
        ----------
        sfh:
            Star formation history

        Returns
        -------
        Spectra
        """

        # We make an initial call to interpolate (ssp_model_class) to
        # obtain the values of the triangulation to be used extensively below
        interp_specs = self.interpolate(
            age=sfh.time,
            met=sfh.met,
            imf_slope=sfh.imf,
            mass=sfh.time_weights,
            alpha=sfh.alpha,
            force_interp=["alpha", "imf_slope"],
        )

        new_meta = {
            "lsf_wave": self.models.meta["lsf_wave"],
            "lsf_fwhm": self.models.meta["lsf_fwhm"],
        }
        for k in self.models.meta.keys():
            if len(self.models.meta[k]) > 1:
                # Skip the interpolation of string data, e.g., filenames
                kind = self.models.meta[k].dtype.kind
                if "f" in kind:
                    new_meta[k] = np.zeros(1)
                    if hasattr(self.models.meta[k], "unit"):
                        new_meta[k] <<= self.models.meta[k].unit

        # We iterate now over all the age bins in the SFH
        for t, date in tqdm(enumerate(sfh.time), delay=3.0):
            # Update quantities
            for k in new_meta.keys():
                if len(interp_specs.meta[k]) == interp_specs.nspec:
                    new_meta[k] += interp_specs.meta[k][t] * (
                        sfh.time_weights[t] / sfh.mass
                    )

        # The final spectrum is also the mass-weighted one, but the mass was
        # already taking into account when doing the interpolation
        ospec = np.sum(interp_specs.flux, axis=0)

        return Spectra(
            spectral_axis=self.models.spectral_axis, flux=ospec, meta=new_meta
        )
