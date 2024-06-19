# -*- coding: utf-8 -*-
import logging
import sys
import typing
import warnings
from copy import copy
from itertools import compress

import h5py
import numpy as np
from scipy.spatial import Delaunay
from typing_extensions import Self

import pymiles.misc as misc
from pymiles.filter import Filter
from pymiles.magnitudes import sun_magnitude
from pymiles.repository import repository
from pymiles.spectra import spectra

logger = logging.getLogger("pymiles.ssp")


class ssp_models(spectra, repository):
    warnings.filterwarnings("ignore")

    # -----------------------------------------------------------------------------
    def __init__(
        self,
        source="MILES_SSP",
        version="9.1",
        isochrone="P",
        imf_type="ch",
        show_tree=False,
    ):
        """
        Creates an instance of the class

        Parameters
        ----------
        source:
            Name of input models to use. Valid inputs are
                   MILES_SSP/CaT_SSP/EMILES_SSP
        version:
            version number of the models
        isochrone:
            Type of isochrone to use. Valid inputs are P/T for Padova+00
                   and BaSTI isochrones respectively (Default: T)
        imf_type:
            Type of IMF shape. Valid inputs are ch/ku/kb/un/bi (Default: ch)
        show_tree:
            Bool that shows the variables available with the instance

        Notes
        -----
        We limit the choice of models to a given isochrone and imf_type for
        effective loading. Otherwise it can take along time to upload the entire
        dataset

        Returns
        -------
        ssp_models
            Object instance

        """
        repo_filename = self._get_repository(source, version)
        self._assert_repository_file(repo_filename)

        # Opening the relevant file in the repository and selecting the desired
        # models at init
        f = h5py.File(repo_filename, "r")
        nspec = len(f["age"])

        idx = np.logical_and(
            np.equal(f["imf_type"][...], imf_type.encode()),
            np.equal(f["isochrone"][...], isochrone.encode()),
        )
        if np.sum(idx) == 0:
            logger.error("No cases found with those specs. Returning NoneType object")
            return

        avail_alphas = np.unique(f["alpha"][idx])
        self.fixed_alpha = len(avail_alphas) == 1
        self.avail_alphas = avail_alphas[~np.isnan(avail_alphas)]

        avail_imfs = np.unique(f["imf_slope"][idx])
        self.avail_imfs = avail_imfs[~np.isnan(avail_imfs)]

        logger.debug(str(np.sum(idx)) + "/" + str(nspec) + " cases found")
        logger.debug(f"Fixed_alpha: {self.fixed_alpha}")

        if self.fixed_alpha:
            logger.debug(f"Available alphas: {self.avail_alphas}")

        # If requested, list instance attributes
        if show_tree:
            logger.info("Showing instance attributes...")
            for item in list(f.keys()):
                logger.info(" - ", item)
            sys.exit

        # Extracting relevant info
        # ------------------------------
        for key, val in f.items():
            if (
                (key != "spec")
                and (key != "wave")
                and (key != "isochrone")
                and (key != "imf_type")
                and (key != "alpha")
                and (key != "filename")
            ):
                setattr(self, key, np.array(val)[idx])

        self.wave = np.array(f["wave"])
        self.spec = np.array(f["spec/" + isochrone + "/" + imf_type + "/data"])

        for key, val in vars(self).items():
            if (key != "spec") and (key != "wave"):
                setattr(self, key, np.array(val))

        self.spec = self.spec[:]
        self.nspec = self.spec.shape[1]
        self.index = np.arange(self.nspec)
        self.source = source
        self.version = version

        fullpath = np.array(np.array(f["filename"][idx]), dtype="str")[0]
        self.route = "/".join(fullpath.split("/")[:-1]) + "/"
        self.filename = []
        for fullpath in np.array(np.array(f["filename"][idx]), dtype="str"):
            self.filename.append((fullpath.split("/"))[-1].split(".fits")[0])

        self.isochrone = np.array(np.array(f["isochrone"][idx]), dtype="str")
        self.imf_type = np.array(np.array(f["imf_type"][idx]), dtype="str")
        self.alpha = np.array(f["alpha"][idx])
        # ------------------------------
        f.close()

        self.main_keys = list(self.__dict__.keys())

        # Inheriting the spectra class
        spectra.__init__(self, source=self.source, wave=self.wave, spec=self.spec)
        #        sfh.__init__(self)

        #        super().__init__(source=self.source,wave=self.wave,spec=self.spec)

        logger.info(source + " models loaded")

    # -----------------------------------------------------------------------------

    def set_item(self, idx):
        """
        Creates a copy of input instance and slices the arrays for input indices

        Parameters
        ----------
        idx:
            integer or boolean array indicating the elements to be extracted

        Returns
        -------
        ssp_models
            Object instance for selected items
        """

        nspec_in = self.nspec
        if len(idx) == nspec_in:
            # Mask array
            nspec_out = np.sum(idx)
        else:
            # Array of indices
            nspec_out = len(idx)
        out = copy(self)
        keys = list(out.main_keys)
        out.wave = np.array(self.wave)
        out.spec = np.array(self.spec[:, idx], ndmin=2)
        out.nspec = nspec_out
        out.filename = list(compress(self.filename, idx))

        for i in range(len(keys)):
            if (keys[i] == "wave") or (keys[i] == "spec") or (keys[i] == "nspec"):
                continue
            val = np.array(getattr(out, keys[i]))
            if np.ndim(val) == 0:
                continue
            if val.shape[0] == nspec_in:
                setattr(out, keys[i], val[idx])

        return out

    # -----------------------------------------------------------------------------
    # GET_SSP_IN_RANGE
    #
    # Extracts SSP models within selected limits for a give instance
    # -----------------------------------------------------------------------------
    def get_ssp_in_range(
        self,
        age_lims=[0.0, 20.0],
        met_lims=[-5.0, 1.0],
        alpha_lims=None,
        imf_slope_lims=[0.0, 5.0],
    ):
        #    def get_ssp_in_range(self, age_lims=[0.0,20.0], met_lims=[-5.0,1.0],
        #                         alpha_lims=[-1.0,1.0],
        #                         imf_slope_lims=[0.0,5.0]):
        # print(
        #    "NO VEO QUE NI LIST NI RANGE SSP DEVUELVAN CORRECTAMENTE LAS EDADES
        #    Y METALICIDADES: VER"
        # )
        """
        Extracts SSP models within selected limits

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

        Returns
        -------
        ssp_models
            Object instance for items in selected ranges

        """

        logger.info("# Searching for models within parameters range")

        if self.fixed_alpha or alpha_lims is None:
            idx = (
                (self.age >= age_lims[0])
                & (self.age <= age_lims[1])
                & (self.met >= met_lims[0])
                & (self.met <= met_lims[1])
                & (self.imf_slope >= imf_slope_lims[0])
                & (self.imf_slope <= imf_slope_lims[1])
            )
        else:
            idx = (
                (self.age >= age_lims[0])
                & (self.age <= age_lims[1])
                & (self.met >= met_lims[0])
                & (self.met <= met_lims[1])
                & (self.alpha >= alpha_lims[0])
                & (self.alpha <= alpha_lims[1])
                & (self.imf_slope >= imf_slope_lims[0])
                & (self.imf_slope <= imf_slope_lims[1])
            )

        ncases = np.sum(idx)
        logger.debug(" - " + str(ncases) + " cases found")

        if ncases == 0:
            logger.warning("get_ssp_in_range returning NoneType object")
            return

        out = self.set_item(idx)

        logger.info("DONE")
        return out

    # -----------------------------------------------------------------------------
    def get_ssp_in_list(
        self,
        age_list=None,
        met_list=None,
        alpha_list=None,
        imf_slope_list=None,
    ):
        """
        Extracts a selected set of models from init instance.

        Parameters
        ----------
        age_list (length N):
            list of ages to extract
        met_list (length N):
            list of metallicities to extract
        alpha_list (length N):
            list of alphas to extract
        imf_slope_list (length N):
            list of IMF slopes to extract

        Notes
        -----
        All lists must have the same length.
        Numbers in the list have to be valid ages, mets, alpha, and imf_slopes
        for the input isochrone and imf_type

        Returns
        -------
        ssp_models
            Object instance with selected list items

        """
        # print(
        #    "NO VEO QUE NI LIST NI RANGE SSP DEVUELVAN CORRECTAMENTE LAS
        #    EDADES Y METALICIDADES: VER"
        # )
        if alpha_list is not None and self.fixed_alpha:
            raise ValueError(
                "This repository does not provide variable alpha:\n"
                + "Source: "
                + self.source
                + "Isochrone: "
                + self.isochrone[0]
                + "IMF: "
                + self.imf_type[0]
            )

        # Treating the input list
        age = np.array(age_list).ravel()
        met = np.array(met_list).ravel()
        alpha = np.array(alpha_list).ravel()
        imf_slope = np.array(imf_slope_list).ravel()

        # Checking they have the same number of elements
        if self.fixed_alpha or alpha_list is None:
            if len(set(map(len, (age, met, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")
        else:
            if len(set(map(len, (age, met, alpha, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")

        logger.info("# Searching for selected cases")

        ncases = len(age)
        id = []
        for i in range(ncases):
            if self.fixed_alpha or alpha_list is None:
                idx = (
                    (np.array(self.age) == age[i])
                    & (np.array(self.met) == met[i])
                    & (np.array(self.imf_slope) == imf_slope[i])
                )
            else:
                idx = (
                    (np.array(self.age) == age[i])
                    & (np.array(self.met) == met[i])
                    & (np.array(self.alpha) == alpha[i])
                    & (np.array(self.imf_slope) == imf_slope[i])
                )

            if np.any(idx):
                id = np.append(id, self.index[idx])

        good = np.array(id, dtype=int)
        ncases = len(good)
        logger.debug(" - " + str(ncases) + " cases found")

        assert ncases > 0, (
            "Check those values exist for isochrone: "
            + self.isochrone[0]
            + " and IMF_Type: "
            + self.imf_type[0]
            + ". Please check this agrees with init instance."
        )

        out = self.set_item(good)

        logger.info("DONE")

        return out

    def get_ssp_by_params(
        self,
        age=None,
        met=None,
        alpha=None,
        imf_slope=None,
        closest=False,
    ) -> Self:
        """
        Interpolates SSP models for certain params using Delaunay triangulation

        Parameters
        ----------
        age:
            Desired age
        met:
            Desired metallicity
        alpha:
            Desired alpha
        img_slope:
            Desired IMF slope
        closest: bool
            Return the closest spectra, rather than performing the interpolation.

        Returns
        -------
        ssp_models
            Wavelength and interpolated spectrum.
            If closest == True, return the closest spectra from the repository,
            rather than doing the interpolation.

        Note
        ----
        It raises a RuntimeError if the values are out of the grid.
        """
        # Checking input point is within the grid
        in_age_lim = (age >= np.amin(self.age)) & (age <= np.amax(self.age))
        in_met_lim = (met >= np.amin(self.met)) & (met <= np.amax(self.met))
        in_imf_lim = (imf_slope >= np.amin(self.imf_slope)) & (
            imf_slope <= np.amax(self.imf_slope)
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
        if interp_fix_imf_slope:
            if imf_slope is None:
                imf_slope = self.avail_imfs[0]

            imf_mask = np.equal(self.imf_slope, imf_slope)
        else:
            imf_mask = np.full(self.imf_slope.shape, True)
            extra_dim += 1

        logger.debug(f"Fixed imf during the interpolation? {interp_fix_imf_slope}")

        # Same as above for the imf_slope, however, now we can have nan values in
        # the repository in the case of "base" models.
        interp_fix_alpha = (
            self.fixed_alpha or alpha is None or alpha in self.avail_alphas
        )
        if interp_fix_alpha:
            if alpha is None:
                alpha_mask = np.isnan(self.alpha)
            elif alpha in self.avail_alphas:
                alpha_mask = np.equal(self.alpha, alpha)
            else:
                # There is a single alpha in all the repository, so no need to mask
                alpha_mask = np.full(self.alpha.shape, True)
        else:
            # Remove the base alpha (i.e., nans) from the interpolation
            alpha_mask = ~np.isnan(self.alpha)
            extra_dim += 1

        logger.debug(f"Fixed alpha during the interpolation? {interp_fix_alpha}")

        idx = alpha_mask & imf_mask
        n_avail = np.sum(idx)

        logger.debug(f"Interpolating over a grid of {n_avail} spectra")
        logger.debug(f"Creating {ndims}-D Delaunay triangulation")

        self.params = np.empty((n_avail, ndims))
        self.params[:, 0] = self.age[idx]
        self.params[:, 1] = self.met[idx]

        if not interp_fix_imf_slope:
            self.params[:, 2 + extra_dim] = self.imf_slope[idx]
            extra_dim += 1

        if not interp_fix_alpha:
            self.params[:, 2 + extra_dim] = self.alpha[idx]

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
        logger.debug(f"Simplex formed by the ids: {self.index[idx][vtx]}")
        logger.debug(f"Age of simplex vertices: {self.age[idx][vtx]}")
        logger.debug(f"Metallicity of simplex vertices: {self.met[idx][vtx]}")
        logger.debug(f"Simplex weights: {wts}, norm: {np.sum(wts)}")

        if closest:
            logger.info("Getting closest spectra")
            out = self.set_item(self.index[idx][vtx])
            return out
        else:
            logger.info("Interpolating spectra")
            wave = self.wave
            spec = np.dot(self.spec[:, idx][:, vtx], wts)
            # Saving all the new info into an object
            out = self.create_new_object(
                age, met, alpha, imf_slope, wave, spec, self.index[idx][vtx], wts
            )
            return out

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
        for m in mass_in:
            if m == "total":
                mass = self.Mass_total
            elif m == "remn":
                mass = self.Mass_remn
            elif m == "star":
                mass = self.Mass_star
            elif m == "star+remn":
                mass = self.Mass_star_remn
            elif m == "gas":
                mass = self.Mass_gas
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

    def create_new_object(
        self, age, met, alpha, imf_slope, wave, spec, indices, weights
    ):
        """
        Creates a new object using the info from the get_ssp_by_params method

        Parameters
        ----------
        age:
            Interpolated age
        met:
            Interpolated metallicity
        alpha:
            Interpolates alpha
        imf_slope:
            Interpolated imf slope
        wave:
            Input wavelength
        spec:
            Interpolated spectrum
        indices:
            Elements of the original object to do the interpolation
        weights:
            Weights for each of the elements

        Returns
        -------
        dict
            Dictionary with mass-to-light ratios for each SSP model and filter

        """

        # Copying basic info
        out = copy(self)
        nspec_in = self.nspec
        out.wave = wave
        out.spec = np.array(spec, ndmin=2).T
        out.imf_type = [self.imf_type[0]]
        out.imf_slope = [imf_slope]
        out.met = [met]
        out.age = [age]
        out.alpha = [alpha]
        out.nspec = 1
        out.isochrone = [self.isochrone[0]]
        out.imf_type = [self.imf_type[0]]
        out.imf_slope = [imf_slope]
        out.index = np.nan
        # Creating filenames
        # Filename's string with the source and IMF
        filename_imf = out.source[0] + out.imf_type[0] + "%.2f" % out.imf_slope[0]

        # Filename's string with the metallicity
        if met < 0:
            met_sign = "m"
            abs_met = abs(met)
        if met >= 0:
            met_sign = "p"
            abs_met = met
        filename_met = "Z" + met_sign + "%.2f" % abs_met

        # Filename's string with the age
        if age < 10.0:
            filename_age = "T0%.4f" % age
        if age >= 10.0:
            filename_age = "T%.4f" % age

        # Filename's string with the isochrone and alpha enhancement
        filename_iso = "_i" + out.isochrone[0] + "p0.00_baseFe"
        #       filename_iso = '_i'+out.isochrone[0]+'p0.00_'+alpha

        # Joining all filename string parts
        out.filename = [
            filename_imf + filename_met + filename_age + filename_iso
        ]  # + linear? + FWHM? se ponen en la web de miles
        ###

        # Interpolating other parameters
        # NOTE: The problem with this is that we are hard-coding a few attributes
        #       but it is probably not too bad. We hard code only minimum necessary
        keys = list(out.main_keys)
        for i in range(len(out.main_keys)):
            if (
                (keys[i] == "wave")
                or (keys[i] == "spec")
                or (keys[i] == "nspec")
                or (keys[i] == "age")
                or (keys[i] == "met")
                or (keys[i] == "alpha")
                or (keys[i] == "imf_slope")
                or (keys[i] == "filename")
                or (keys[i] == "<")
                or (keys[i] == "imf_type")
                or (keys[i] == "index")
            ):
                continue
            val = np.array(getattr(out, keys[i]))

            if np.ndim(val) == 0:
                continue
            if val.shape[0] == nspec_in:
                setattr(out, keys[i], np.dot(val[indices], weights))

        # Instaitiating the spectra class
        #     super().__init__(source=out.source,wave=out.wave,spec=out.spec)
        return out
