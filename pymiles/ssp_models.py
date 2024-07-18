# -*- coding: utf-8 -*-
import logging
import warnings

import h5py
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from scipy.spatial import Delaunay

import pymiles.misc as misc
from pymiles.repository import repository
from pymiles.spectra import spectra

logger = logging.getLogger("pymiles.ssp")


class ssp_models(repository):
    """
    Single stellar population (SSP) model library.

    This class is used to generate SSP spectra from the repository files.

    Attributes
    ----------
    models: spectra
        Spectra of all the SSP that form the loaded library
    avail_alphas: list
        Available alpha/Fe values in the loaded library
    avail_imfs: list
        Available initial mass function (IMF) slopes in the loaded library
    """

    warnings.filterwarnings("ignore")

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
            Name of input models to use. Valid inputs are
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
        ssp_models

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
            "alpha": np.array(f["alpha"][idx]),
            "filename": np.array(filename),
        }
        # Standard set of keys available in the repository that are taken
        # care of manually
        base_keys = ["spec", "wave", "isochrone", "imf_type", "alpha", "filename"]
        # Other information in the repository, store them as arrays
        for k in f.keys():
            if k not in base_keys:
                meta[k] = np.array(f[k])[idx]

        f.close()

        self.models = spectra(
            spectral_axis=Quantity(wave, unit=u.AA),
            flux=Quantity(spec.T, unit=u.L_sun / u.M_sun / u.AA),
            meta=meta,
        )

        logger.info(source + " models loaded")

    def in_range(
        self,
        age_lims=[0.0, 20.0],
        met_lims=[-5.0, 1.0],
        alpha_lims=None,
        imf_slope_lims=[0.0, 5.0],
        mass=Quantity(value=1.0, unit=u.Msun),
    ) -> spectra:
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
        mass: Quantity, default: 1 solar mass
            mass of each SSP

        Raises
        ------
        ValueError
            If there is no matching SSP.

        Returns
        -------
        spectra
            Spectra in the selected ranges

        """

        logger.debug("Searching for models within parameters range")

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

        out = spectra.__getitem__(self.models, idx)._assign_mass(mass)

        return out

    def in_list(
        self, age=None, met=None, alpha=None, imf_slope=None, mass=None
    ) -> spectra:
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
        spectra

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

        out = spectra.__getitem__(self.models, good)._assign_mass(mass)

        return out

    def interpolate(
        self,
        age=None,
        met=None,
        alpha=None,
        imf_slope=None,
        mass=Quantity(value=1.0, unit=u.Msun),
        closest=False,
        force_interp=[],
    ) -> spectra:
        """
        Interpolates SSP models for certain parameters using Delaunay triangulation

        Parameters
        ----------
        age: float
            Desired age
        met: float
            Desired metallicity
        alpha: float
            Desired alpha
        img_slope: float
            Desired IMF slope
        mass: Quantity
            Mass of the SSP
        closest: bool
            Return the closest spectra, rather than performing the interpolation.
        force_interp: list
            Force the interpolation over the indicated variables, even if the
            asked alpha/imf_slope is sampled in the repository. Valid values
            are "alpha" and "imf_slope".

        Returns
        -------
        spectra
            Interpolated spectrum.
            If closest == True, return the closest spectra from the repository,
            rather than doing the interpolation.

        Raises
        ------
        RuntimeError
            If the values are out of the grid.
        """
        # Checking input point is within the grid
        in_age_lim = (age >= np.amin(self.models.meta["age"])) & (
            age <= np.amax(self.models.meta["age"])
        )
        in_met_lim = (met >= np.amin(self.models.meta["met"])) & (
            met <= np.amax(self.models.meta["met"])
        )
        in_imf_lim = (imf_slope >= np.amin(self.models.meta["imf_slope"])) & (
            imf_slope <= np.amax(self.models.meta["imf_slope"])
        )

        good = in_age_lim & in_met_lim & in_imf_lim
        if not self.fixed_alpha and alpha is not None:
            in_alpha_lim = (alpha >= np.amin(self.avail_alphas)) & (
                alpha <= np.amax(self.avail_alphas)
            )
            good &= in_alpha_lim

        if not good:
            raise RuntimeError("Desired point outside model grid")

        ndims = 2
        extra_dim = 0

        # If there is a single available imf slope, there is no need to interpolate
        # over it. Also, if the users gives an imf_slope that **exactly** matches
        # one of the database, we fix the imf_slope for the interpolation
        interp_fix_imf_slope = len(self.avail_imfs) == 1 or imf_slope in self.avail_imfs
        # But this can be overruled
        interp_fix_imf_slope &= not ("imf_slope" in force_interp)
        if interp_fix_imf_slope:
            if imf_slope is None:
                imf_slope = self.avail_imfs[0]

            imf_mask = np.equal(self.models.meta["imf_slope"], imf_slope)
        else:
            imf_mask = np.full(self.models.meta["imf_slope"].shape, True)
            ndims += 1

        logger.debug(f"Fixed imf during the interpolation? {interp_fix_imf_slope}")

        # Same as above for the imf_slope, however, now we can have nan values in
        # the repository in the case of "base" models.
        interp_fix_alpha = (
            self.fixed_alpha or alpha is None or alpha in self.avail_alphas
        )
        if self.fixed_alpha and alpha is not None:
            logger.warning("There is no alpha-enhanced SSPs with this model choice")
        interp_fix_alpha &= not ("alpha" in force_interp)
        if interp_fix_alpha:
            if alpha is None:
                alpha_mask = np.isnan(self.models.meta["alpha"])
            elif alpha in self.avail_alphas:
                alpha_mask = np.equal(self.models.meta["alpha"], alpha)
            else:
                # There is a single alpha in all the repository, so no need to mask
                alpha_mask = np.full(self.models.meta["alpha"].shape, True)
        else:
            # Remove the base alpha (i.e., nans) from the interpolation
            alpha_mask = ~np.isnan(self.models.meta["alpha"])
            ndims += 1

        logger.debug(f"Fixed alpha during the interpolation? {interp_fix_alpha}")

        idx = alpha_mask & imf_mask
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

        input_pt = [age, met]
        if not interp_fix_imf_slope:
            input_pt.append(imf_slope)
        if not interp_fix_alpha:
            input_pt.append(alpha)

        logger.info(
            f"Searching for the simplex that surrounds the desired point: {input_pt}"
        )

        vtx, wts = misc.interp_weights(
            self.params, np.array(input_pt, ndmin=2), self.tri
        )
        vtx, wts = vtx.ravel(), wts.ravel()
        logger.debug(
            f"Simplex formed by the ids: {self.models.meta['index'][idx][vtx]}"
        )
        logger.debug(f"Age of simplex vertices: {self.models.meta['age'][idx][vtx]}")
        logger.debug(
            f"Metallicity of simplex vertices: {self.models.meta['met'][idx][vtx]}"
        )
        logger.debug(f"Simplex weights: {wts}, norm: {np.sum(wts)}")

        # Save which spectra has been used for building the tesselation
        self.idx = idx

        if closest:
            logger.info("Getting closest spectra")
            out = spectra.__getitem__(
                self.models, self.models.meta["index"][idx][vtx]
            )._assign_mass(mass)
            return out
        else:
            logger.info("Interpolating spectra")
            wave = self.models.spectral_axis
            spec = np.dot(self.models.flux[idx, :][vtx].T, wts)
            new_meta = {
                "imf_type": np.array([self.models.meta["imf_type"]]),
                "imf_slope": np.array([imf_slope]),
                "met": np.array([met]),
                "age": np.array([age]),
                "alpha": np.array([alpha]),
            }

            # Interpolate the rest of the meta if possible
            for k in self.models.meta.keys():
                if k not in new_meta.keys():
                    if len(self.models.meta[k]) > 1:
                        # Skip the interpolation of string data, e.g., filenames
                        if "U" not in self.models.meta[k].dtype.kind:
                            new_meta[k] = np.dot(self.models.meta[k][idx][vtx], wts)

            out = spectra(spectral_axis=wave, flux=spec, meta=new_meta)._assign_mass(
                mass
            )

            return out
