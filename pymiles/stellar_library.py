# -*- coding: utf-8 -*-
import logging

import h5py
import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.units import Quantity
from scipy.spatial import Delaunay

import pymiles.misc as misc
from pymiles.repository import repository
from pymiles.spectra import spectra

# ==============================================================================

logger = logging.getLogger("pymiles.lib")


class stellar_library(repository):
    # -----------------------------------------------------------------------------
    def __init__(self, source="MILES_STARS", version="9.1"):
        """
        Creates an instance of the class

        Parameters
        ----------
        source:
            Name of input models to use. Valid inputs are
                 MILES_STARS/CaT_STARS/EMILES_STARS
        version:
            Version number of the models

        """
        repo_filename = self._get_repository(source, version)
        self._assert_repository_file(repo_filename)

        # Opening the relevant file in the repository
        f = h5py.File(repo_filename, "r")
        # ------------------------------
        meta = {
            "index": np.array(f["index"]),
            "teff": np.array(f["teff"]),
            "logg": np.array(f["logg"]),
            "FeH": np.array(f["FeH"]),
            "MgFe": np.array(f["MgFe"]),
            "starname": np.array([n.decode() for n in f["starname"]]),
            "filename": np.array([n.decode() for n in f["filename"]]),
            "id": np.array([np.int32(n.decode()) for n in f["id"]]),
        }
        wave = np.array(f["wave"])
        spec = np.array(f["spec"])
        self.source = source
        self.version = version
        # ------------------------------
        f.close()

        # Flagging if all elements of MgFe are NaNs
        self.MgFe_flag = 0
        if np.nansum(meta["MgFe"]) == 0:
            self.MgFe_flag = 1

        # Creating Delaunay triangulation of parameters for future searches and
        # interpolations
        if self.MgFe_flag == 1:
            idx = (
                np.isfinite(meta["teff"])
                & np.isfinite(meta["logg"])
                & np.isfinite(meta["FeH"])
            )
            ngood = np.sum(idx)
            self.params = np.empty((ngood, 3))
            self.params[:, 0] = np.log10(meta["teff"])[idx]
            self.params[:, 1] = meta["logg"][idx]
            self.params[:, 2] = meta["FeH"][idx]
        else:
            idx = (
                np.isfinite(meta["teff"])
                & np.isfinite(meta["logg"])
                & np.isfinite(meta["FeH"])
                & np.isfinite(meta["MgFe"])
            )
            ngood = np.sum(idx)
            self.params = np.empty((ngood, 4))
            self.params[:, 0] = np.log10(meta["teff"])[idx]
            self.params[:, 1] = meta["logg"][idx]
            self.params[:, 2] = meta["FeH"][idx]
            self.params[:, 3] = meta["MgFe"][idx]

        self.tri = Delaunay(self.params)
        self.index = meta["index"][idx]
        self.main_keys = list(self.__dict__.keys())

        self.models = spectra(
            spectral_axis=Quantity(wave, unit=u.AA),
            flux=Quantity(spec.T, unit=None),
            meta=meta,
        )

    # -----------------------------------------------------------------------------
    def search_by_id(self, id=None):
        """
        Searches a star in database for a given ID

        Parameters
        ----------
        id:
            integer with the star ID in database

        Returns
        -------
        Object instance for selected items

        """

        idx = self._id_to_idx(id)

        out = spectra.__getitem__(self.models, idx)

        return out

    def _id_to_idx(self, lib_id: npt.ArrayLike) -> np.ndarray:
        id_arr = np.array(lib_id, ndmin=1)
        common, _, idx = np.intersect1d(
            id_arr, self.models.meta["id"], assume_unique=True, return_indices=True
        )
        if len(common) != len(id_arr):
            raise ValueError("No star with that ID")
        return idx

    # -----------------------------------------------------------------------------

    def get_starname(self, id=None):
        """
        Gets a starname in database for a given ID

        Parameters
        ----------
        id:
            integer with the star ID in database

        Returns
        -------
        Star name

        """

        idx = self._id_to_idx(id)
        logger.debug(f"{type(idx[0])}")

        return self.models.meta["starname"][idx]

    # -----------------------------------------------------------------------------

    def in_range(self, teff_lims=None, logg_lims=None, FeH_lims=None, MgFe_lims=None):
        """
        Gets set of stars with parameters range

        Parameters
        ----------
        teff_lims:
            Limits in Teff
        logg_lims:
            Limits in Log(g)
        FeH_lims:
            Limits in [Fe/H]
        MgFe_lims:
            Limits in [Mg/Fe]

        Returns
        -------
        stellar_library
            Object instance for stars within parameters range

        """

        if self.MgFe_flag == 1:
            idx = (
                (self.models.meta["teff"] >= teff_lims[0])
                & (self.models.meta["teff"] <= teff_lims[1])
                & (self.models.meta["logg"] >= logg_lims[0])
                & (self.models.meta["logg"] <= logg_lims[1])
                & (self.models.meta["FeH"] >= FeH_lims[0])
                & (self.models.meta["FeH"] <= FeH_lims[1])
            )
        else:
            idx = (
                (self.models.meta["teff"] >= teff_lims[0])
                & (self.models.meta["teff"] <= teff_lims[1])
                & (self.models.meta["logg"] >= logg_lims[0])
                & (self.models.meta["logg"] <= logg_lims[1])
                & (self.models.meta["FeH"] >= FeH_lims[0])
                & (self.models.meta["FeH"] <= FeH_lims[1])
                & (self.models.meta["MgFe"] >= MgFe_lims[0])
                & (self.models.meta["MgFe"] <= MgFe_lims[1])
            )

        out = spectra.__getitem__(self.models, idx)

        return out

    # -----------------------------------------------------------------------------
    def search_by_params(self, teff=None, logg=None, FeH=None, MgFe=None):
        """
        Gets closest star in database for given set of parameters

        Parameters
        ----------
        teff:
            Desired Teff
        logg:
            Desired Log(g)
        FeH:
            Desired [Fe/H]
        MgFe:
            Desired [Mg/Fe]

        Returns
        -------
        stellar_library
            Object instance for closest star

        """

        # Searching for the simplex that surrounds the desired point in parameter space
        if self.MgFe_flag == 1:
            input_pt = np.array([np.log10(teff), logg, FeH], ndmin=2)
        else:
            input_pt = np.array([np.log10(teff), logg, FeH, MgFe], ndmin=2)

        vtx, wts = misc.interp_weights(self.params, input_pt, self.tri)
        vtx, wts = vtx.ravel(), wts.ravel()

        # Deciding on the closest vertex and extracting info
        idx = np.argmax(wts)
        new_idx = self.index[vtx[idx]]
        out = spectra.__getitem__(self.models, new_idx)

        return out

    # -----------------------------------------------------------------------------
    def interpolate(self, teff=None, logg=None, FeH=None, MgFe=None):
        """
        Interpolates a star spectrum for given set of parameters using Delaunay
        triangulation

        Parameters
        ----------
        teff:
            Desired Teff
        logg:
            Desired Log(g)
        FeH:
            Desired [Fe/H]
        MgFe:
            Desired [Mg/Fe]

        Returns
        -------
        wave:
            wavelength of output spectrum
        spec:
            interpolated spectrum

        """

        # Searching for the simplex that surrounds the desired point in parameter space
        if self.MgFe_flag == 1:
            input_pt = np.array([np.log10(teff), logg, FeH], ndmin=2)
        else:
            input_pt = np.array([np.log10(teff), logg, FeH, MgFe], ndmin=2)

        vtx, wts = misc.interp_weights(self.params, input_pt, self.tri)
        vtx, wts = vtx.ravel(), wts.ravel()

        idx = self.index[vtx]

        wave = self.models.spectral_axis
        spec = np.dot(self.models.flux[idx, :].T, wts)

        new_meta = {
            "teff": np.array([teff]),
            "logg": np.array([logg]),
            "FeH": np.array([FeH]),
            "MgFe": np.array([MgFe]),
        }

        # Interpolate the rest of the meta if possible
        for k in self.models.meta.keys():
            if k not in new_meta.keys():
                if len(self.models.meta[k]) > 1:
                    # Skip the interpolation of string data, e.g., filenames
                    if "U" not in self.models.meta[k].dtype.kind:
                        new_meta[k] = np.dot(self.models.meta[k][idx], wts)

        out = spectra(spectral_axis=wave, flux=spec, meta=new_meta)

        return out
