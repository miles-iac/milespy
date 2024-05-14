import os
import sys
import glob
import logging
import h5py
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pymiles.pymiles_utils as utils
import pymiles.misc_functions as misc
from astropy.io import ascii, fits
from astropy import constants
from scipy.spatial import Delaunay
from copy import copy
from pymiles.tuning_tools import tuning_tools
from itertools import compress

logger = logging.getLogger('pymiles.ssp')

# from ipdb import set_trace as stop
# ==============================================================================


class ssp_models(tuning_tools):

    warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
    def __init__(self, source='MILES_SSP', version='9.1', isochrone='P', imf_type='ch', alp_type='fix', show_tree=False, verbose=False):
        """
        Creates an instance of the class 

        Keywords
        --------
        source:    Name of input models to use. Valid inputs are MILES_SSP/CaT_SSP/EMILES_SSP
        version:   version number of the models
        isochrone: Type of isochrone to use. Valid inputs are P/T for Padova+00 and BaSTI isochrones respectively (Default: T)
        imf_type:  Type of IMF shape. Valid inputs are ch/ku/kb/un/bi (Default: ch)
        alp_type:  Type of [alpha/Fe]. Valid inputs are fix/variable (Default: fix). Variable [alpha/Fe] predictions are only
                   available for BaSTI isochrones
        show_tree: Bool that shows the variables available with the instance
        verbose:   Flag to verbose mode

        Note
        ----
        We limit the choice of models to a given isochrone and imt_type for effective loading
        Otherwise it can take along time to upload the entire dataset

        Return
        ------
        Object instance

        """
        repo_filename = "./pymiles/repository/" + source + "_v" + version + ".hdf5"

        if verbose:
            logger.info("# Loading models in " + repo_filename + " for isochrone: " + isochrone + " and IMF_type: " + imf_type)
            filesizeGB = os.path.getsize(repo_filename) / (1024 * 1024 * 1024)
            if (filesizeGB >= 0.05):
                logger.info(" [Be patient. The file is " + '{:4.2f}'.format(filesizeGB) + "Gb in size]")

        # Opening the relevant file in the repository and selecting the desired models at init
        f = h5py.File(repo_filename, "r")
        nspec = len(f['age'])
        tmp_iso = np.array(np.array(f['isochrone']), dtype='str')
        tmp_imf_type = np.array(np.array(f['imf_type']), dtype='str')

        if (alp_type == 'variable') and (isochrone == 'P'):
            raise ValueError("Variable [alpha/Fe] predictions are only available for BaSTI isochrones")

        idx = (tmp_iso == isochrone) & (tmp_imf_type == imf_type)
        if (np.sum(idx) == 0):
            logger.error("No cases found with those specs. Returning NoneType object")
            return

        logger.debug(" - " + str(np.sum(idx)) + "/" + str(nspec) + " cases found")

        # If requested, list instance attributes
        if show_tree == True:
            logger.info("# Showing instance attributes...")
            for item in list(f.keys()):
                logger.info(" - ", item)
            sys.exit

        # Extracting relevant info
        # ------------------------------
        for key, val in f.items():
            if (key != 'spec') and (key != 'wave') and (key != 'isochrone') and (key != 'imf_type') and (key != 'filename'):
                setattr(self, key, np.array(val)[idx])

        self.wave = np.array(f['wave'])
        self.spec = np.array(f['spec/' + isochrone + '/' + imf_type + '/data'])

        # Selecting only either base or alpha variable models
        if alp_type == 'variable':
            agood = np.isfinite(self.alpha)
        else:
            agood = np.isnan(self.alpha)

        for key, val in vars(self).items():
            if (key != 'spec') and (key != 'wave'):
                setattr(self, key, np.array(val)[agood])

        self.spec = self.spec[:, agood]
        self.nspec = self.spec.shape[1]
        self.index = np.arange(self.nspec)
        self.source = source
        self.version = version

        fullpath = np.array(np.array(f['filename'][idx]), dtype='str')[0]
        self.route = '/'.join(fullpath.split('/')[:-1]) + '/'
        self.filename = []
        for fullpath in np.array(np.array(f['filename'][idx]), dtype='str'):
            self.filename.append((fullpath.split('/'))[-1].split('.fits')[0])

        self.isochrone = np.array(np.array(f['isochrone'][idx]), dtype='str')
        self.imf_type = np.array(np.array(f['imf_type'][idx]), dtype='str')
        # ------------------------------
        f.close()
        # Flagging if all elements of alpha are NaNs
        if (alp_type == 'fix'):
            self.alpha_flag = 1
        else:
            self.alpha_flag = 0

        self.main_keys = list(self.__dict__.keys())

        # Inheriting the tuning_tools class
        tuning_tools.__init__(self, source=self.source, wave=self.wave, spec=self.spec)
#        sfh.__init__(self)

#        super().__init__(source=self.source,wave=self.wave,spec=self.spec)

        logger.info(source + " models loaded")

# -----------------------------------------------------------------------------
    def set_item(self, idx):
        """
        Creates a copy of input instance and slices the arrays for input indices

        Arguments
        ---------
        idx: integer or boolean array indicating the elements to be extracted

        Returns
        -------

        Object instance for selected items

        """

        nspec_in = self.nspec
        nspec_out = np.sum(idx)
        out = copy(self)
        keys = list(out.main_keys)
        out.wave = np.array(self.wave)
        out.spec = np.array(self.spec[:, idx], ndmin=2)
        out.nspec = nspec_out
        out.filename = list(compress(self.filename, idx))

        for i in range(len(keys)):
            if (keys[i] == 'wave') or (keys[i] == 'spec') or (keys[i] == 'nspec'):
                continue
            val = np.array(getattr(out, keys[i]))
            if (np.ndim(val) == 0):
                continue
            if (val.shape[0] == nspec_in):
                setattr(out, keys[i], val[idx])

        return out

# -----------------------------------------------------------------------------
# GET_SSP_IN_RANGE
#
# Extracts SSP models within selected limits for a give instance
# -----------------------------------------------------------------------------
    def get_ssp_in_range(self, age_lims=[0.0, 20.0], met_lims=[-5.0, 1.0], alpha_lims=[-1.0, 1.0],
                         imf_slope_lims=[0.0, 5.0], verbose=False):
        #    def get_ssp_in_range(self, age_lims=[0.0,20.0], met_lims=[-5.0,1.0], alpha_lims=[-1.0,1.0],
        #                         imf_slope_lims=[0.0,5.0],verbose=False):
        print('NO VEO QUE NI LIST NI RANGE SSP DEVUELVAN CORRECTAMENTE LAS EDADES Y METALICIDADES: VER')

        """
       Extracts SSP models within selected limits

       Keywords
       --------
       age_lims:       tuple with age limits
       met_lims:       tuple with metallicity limits
       alpha_lims:     tuple with alpha limits
       imf_slope_lims: tuple with IMF slope limits
       verbose:        Flag for verbose

       Return
       ------
       Object instance for items in selected ranges

       """

        logger.info("# Searching for models within parameters range")

        if self.alpha_flag == 1:
            idx = (self.age >= age_lims[0]) & (self.age <= age_lims[1]) & \
                  (self.met >= met_lims[0]) & (self.met <= met_lims[1]) & \
                  (self.imf_slope >= imf_slope_lims[0]) & (self.imf_slope <= imf_slope_lims[1])
        else:
            idx = (self.age >= age_lims[0]) & (self.age <= age_lims[1]) & \
                  (self.met >= met_lims[0]) & (self.met <= met_lims[1]) & \
                  (self.alpha >= alpha_lims[0]) & (self.alpha <= alpha_lims[1]) & \
                  (self.imf_slope >= imf_slope_lims[0]) & (self.imf_slope <= imf_slope_lims[1])

        ncases = np.sum(idx)
        logger.debug(" - " + str(ncases) + " cases found")

        if (ncases == 0):
            logger.warning("get_ssp_in_range returning NoneType object")
            return

        out = self.set_item(idx)

        logger.info("DONE")
        return out

# -----------------------------------------------------------------------------
    def get_ssp_in_list(self, age_list=None, met_list=None, alpha_list=None,
                        imf_slope_list=None, verbose=False):
        """
        Extracts a selected set of models from init instance.

        Keywords
        --------
        age_list (length N): list of ages to extract
        met_list (length N): list of metallicities to extract
        alpha_list (length N): list of alphas to extract
        imf_slope_list (length N): list of IMF slopes to extract

        Note
        ----
        All lists must have the same length.
        Numbers in the list have to be valid ages, mets, alpha, and imf_slopes for the input isochrone and imf_type

        Return
        ------
        Object instance with selected list items

        """
        print('NO VEO QUE NI LIST NI RANGE SSP DEVUELVAN CORRECTAMENTE LAS EDADES Y METALICIDADES: VER')

        # Treating the input list
        age = np.array(age_list).ravel()
        met = np.array(met_list).ravel()
        alpha = np.array(alpha_list).ravel()
        imf_slope = np.array(imf_slope_list).ravel()

        # Checking they have the same number of elements
        if self.alpha_flag == 1:
            if len(set(map(len, (age, met, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")
        else:
            if len(set(map(len, (age, met, alpha, imf_slope)))) != 1:
                raise ValueError("Input list values do not have the same length.")

        logger.info("# Searching for selected cases")

        ncases = len(age)
        id = []
        for i in range(ncases):
            if self.alpha_flag == 1:
                idx = (np.array(self.age) == age[i]) & \
                      (np.array(self.met) == met[i]) & \
                      (np.array(self.imf_slope) == imf_slope[i])
            else:
                idx = (np.array(self.age) == age[i]) & \
                      (np.array(self.met) == met[i]) & \
                      (np.array(self.alpha) == alpha[i]) & \
                      (np.array(self.imf_slope) == imf_slope[i])

            if np.any(idx):
                id = np.append(id, self.index[idx])

        good = np.array(id, dtype=int)
        ncases = len(good)
        logger.debug(" - " + str(ncases) + " cases found")

        assert ncases > 0, "Check those values exist for isochrone: " + self.isochrone[0] + " and IMF_Type: " + self.imf_type[0] + ". Please check this agrees with init instance."

        out = self.set_item(good)

        logger.info("DONE")

        return out

# -----------------------------------------------------------------------------
# GET_SSP_BY_PARAMS
#
# Uses Delaunay triangulation to interpolate the SSP models for certain params
# NOTE: It raises an ERROR if desired point is outside the grid
# -----------------------------------------------------------------------------
    def get_ssp_by_params(self, age=None, met=None, alpha=None, imf_slope=None, return_pars=False, verbose=False):
        """
        Interpolates SSP models for certain params using Delaunay triangulation

        Keywords
        --------
        age:  Desired age
        met:  Desired metallicity
        alpha: Desired alpha
        img_slope: Desired IMF slope
        return_pars: If True, returns more information about interpolation
        verbose: Flag for verbose

        Return
        ------
        If return_pars == False, returns wavelength and interpolated spectrum
        If return_pars == True, besides above it returns all spectra used for interpolation, indices, and weights

        """
        # Checking input point is within the grid
        if self.alpha_flag == 1:
            good = (age >= np.amin(self.age)) & (age <= np.amax(self.age)) & \
                   (met >= np.amin(self.met)) & (met <= np.amax(self.met)) & \
                   (imf_slope >= np.amin(self.imf_slope)) & (imf_slope <= np.amax(self.imf_slope))
        else:
            good = (age >= np.amin(self.age)) & (age <= np.amax(self.age)) & \
                   (met >= np.amin(self.met)) & (met <= np.amax(self.met)) & \
                   (alpha >= np.amin(self.alpha)) & (alpha <= np.amax(self.alpha)) & \
                   (imf_slope >= np.amin(self.imf_slope)) & (imf_slope <= np.amax(self.imf_slope))

        if np.sum(good) == 0:
            raise RuntimeError("Desired point outside model grid")

        # Checking that the IMF_SLOPE desired is available
        uimf_slope = np.unique(self.imf_slope)
        nimf_slope = len(uimf_slope)
        ok = (nimf_slope == 1)

        # Creating Delaunay triangulation of parameters
        logger.info("# Creating Delaunay triangulation")
        if self.alpha_flag == 1:
            if ok == True:
                self.params = np.empty((len(self.age), 2))
                self.params[:, 0] = self.age
                self.params[:, 1] = self.met
            else:
                self.params = np.empty((len(self.age), 3))
                self.params[:, 0] = self.age
                self.params[:, 1] = self.met
                self.params[:, 2] = self.imf_slope
        else:
            if ok == True:
                self.params = np.empty((len(self.age), 3))
                self.params[:, 0] = self.age
                self.params[:, 1] = self.met
                self.params[:, 2] = self.alpha
            else:
                self.params = np.empty((len(self.age), 4))
                self.params[:, 0] = self.age
                self.params[:, 1] = self.met
                self.params[:, 2] = self.imf_slope
                self.params[:, 3] = self.alpha

        self.tri = Delaunay(self.params, qhull_options='QJ')

       # Searching for the simplex that surrounds the desired point in parameter space
        logger.info("# Searching for the simplex that surrounds the desired point in parameter space")

        if self.alpha_flag == 1:
            if ok == True:
                input_pt = np.array([age, met], ndmin=2)
            else:
                input_pt = np.array([age, met, imf_slope], ndmin=2)
        else:
            if ok == True:
                input_pt = np.array([age, met, alpha], ndmin=2)
            else:
                input_pt = np.array([age, met, imf_slope, alpha], ndmin=2)

        vtx, wts = utils.interp_weights(self.params, input_pt, self.tri)
        vtx, wts = vtx.ravel(), wts.ravel()

        logger.info("# Interpolating spectra around the desired point")
        wave = self.wave
        spec = np.dot(self.spec[:, vtx], wts)
        # Saving all the new info into an object
        out = self.create_new_object(age, met, alpha, imf_slope, wave, spec, vtx, wts)
        return out
#        if return_pars == True:
#            return wave, spec, all_spec, self.params[vtx,:], wts
#        else:
#            return wave, spec

# -----------------------------------------------------------------------------
# COMPUTE_ML
#
# Computes the mass to light ratio for a set of SSP models and filters
# -----------------------------------------------------------------------------
    def compute_ml(self, filters, type='star+remn', verbose=False):
        """
        Computes the mass-to-light ratios of models in the desired filters 

        Arguments
        --------
        filters:   Filters as provided by the method 'get_filters"
        verbose:   Flag for verbose

        Return
        ------
        Dictionary with mass-to-light ratios for each SSP model and filter

        """
        logger.info("# Computing mass-to-light ratios -----------------")

        zeropoint = 'AB'  #  We need to choose a system. For M/Ls this is irrelevant
        nfilters = len(filters.keys())
        mags = self.compute_save_mags(filters=filters, zeropoint=zeropoint, saveCSV=False, verbose=verbose)
        msun = self.compute_mag_sun(filters=filters, zeropoint=zeropoint, verbose=verbose)

        if type == 'total':
            mass = self.Mass_total
        elif type == 'remn':
            mass = self.Mass_remn
        elif type == 'star':
            mass = self.Mass_star
        elif type == 'star+remn':
            mass = self.Mass_star_remn
        elif type == 'gas':
            mass = self.Mass_gas
        else:
            raise ValueError("Mass type not allowed. Valid options are total/star/remn/star+remn/gas.")

        outmls = {}
        fnames = list(filters.keys())
        for i in range(nfilters):
            outmls[fnames[i]] = (mass / 1.0) * 10**(-0.40 * (msun[fnames[i]] - mags[fnames[i]]))

        return outmls

# -----------------------------------------------------------------------------
# CREATE_NEW_OBJECT
#
# Creates a new object from an interpolated spectra in get_ssp_by_params
# -----------------------------------------------------------------------------
    def create_new_object(self, age, met, alpha, imf_slope, wave, spec, indices, weights):
        """
        Creates a new object using the info from the get_ssp_by_params method 

        Arguments
        --------
        age:       Interpolated age
        met:       Interpolated metallicity
        alpha:     Interpolates alpha
        imf_slope: Interpolated imf slope
        wave:      Input wavelength
        spec:      Interpolated spectrum
        indices:   Elements of the original object to do the interpolation
        weights:   Weights for each of the elements

        Return
        ------
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
        filename_imf = out.source[0] + out.imf_type[0] + '%.2f' % out.imf_slope[0]

        # Filename's string with the metallicity
        if (met < 0):
            met_sign = 'm'
            abs_met = abs(met)
        if (met >= 0):
            met_sign = 'p'
            abs_met = met
        filename_met = 'Z' + met_sign + '%.2f' % abs_met

        # Filename's string with the age
        if (age < 10.):
            filename_age = 'T0%.4f' % age
        if (age >= 10.):
            filename_age = 'T%.4f' % age

        # Filename's string with the isochrone and alpha enhancement
        filename_iso = '_i' + out.isochrone[0] + 'p0.00_baseFe'
#       filename_iso = '_i'+out.isochrone[0]+'p0.00_'+alpha

        # Joining all filename string parts
        out.filename = [filename_imf + filename_met + filename_age + filename_iso]  # + linear? + FWHM? se ponen en la web de miles
        ###

        # Interpolating other parameters
        # NOTE: The problem with this is that we are hard-coding a few attributes
        #       but it is probably not too bad. We hard code only minimum necessary
        keys = list(out.main_keys)
        for i in range(len(out.main_keys)):
            if (keys[i] == 'wave') or (keys[i] == 'spec') or (keys[i] == 'nspec') or \
               (keys[i] == 'age') or (keys[i] == 'met') or (keys[i] == 'alpha') or  \
               (keys[i] == 'imf_slope') or (keys[i] == 'filename') or (keys[i] == '<') or \
               (keys[i] == 'imf_type') or (keys[i] == 'index'):
                continue
            val = np.array(getattr(out, keys[i]))

            if (np.ndim(val) == 0):
                continue
            if (val.shape[0] == nspec_in):
                setattr(out, keys[i], np.dot(val[indices], weights))

        # Instaitiating the tuning_tools class
#     super().__init__(source=out.source,wave=out.wave,spec=out.spec)
        return out
