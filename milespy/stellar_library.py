# -*- coding: utf-8 -*-
import logging
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.units import Quantity
from scipy.spatial import Delaunay
from tqdm import tqdm

from .misc import interp_weights
from .repository import Repository
from .spectra import Spectra


logger = logging.getLogger("milespy.stellarlib")


class StellarLibrary(Repository):
    """
    Single stars library.

    Attributes
    ----------
    models: Spectra
        Spectra of all the stars that form the loaded library
    source: str
        Name of input library being used
    version: str
        Version number of the library
    """

    def __init__(self, source="MILES_STARS", version="9.1"):
        """
        Creates an instance of the class

        Parameters
        ----------
        source:
            Name of input models to use.
            If none of the options below, it will assume that `source` is the path
            to an external model repository file.
            Valid inputs are
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
            "teff": np.array(f["teff"]) << u.K,
            "logg": np.array(f["logg"]) << u.dex,
            "FeH": np.array(f["FeH"]) << u.dex,
            "MgFe": np.array(f["MgFe"]) << u.dex,
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
        self.MgFe_nan = False
        if np.nansum(meta["MgFe"]) == 0:
            self.MgFe_nan = True

        # Creating Delaunay triangulation of parameters for future searches and
        # interpolations
        if self.MgFe_nan:
            idx = (
                np.isfinite(meta["teff"])
                & np.isfinite(meta["logg"])
                & np.isfinite(meta["FeH"])
            )
            ngood = np.sum(idx)
            self.params = np.empty((ngood, 3))
            self.params[:, 0] = np.log10(meta["teff"].to_value(u.K))[idx]
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

        super().__init__(
            Spectra(
                spectral_axis=Quantity(wave, unit=u.AA),
                flux=Quantity(spec.T, unit=None),
                meta=meta,
            )
        )

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

        out = Spectra.__getitem__(self.models, idx)

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

    @u.quantity_input
    def in_range(
        self,
        teff_lims: u.Quantity[u.K],
        logg_lims: u.Quantity[u.dex],
        FeH_lims: u.Quantity[u.dex],
        MgFe_lims: Optional[u.Quantity[u.dex]] = None,
    ):
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
        StellarLibrary
            Object instance for stars within parameters range

        """

        if self.MgFe_nan:
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

        out = Spectra.__getitem__(self.models, idx)

        return out

    @u.quantity_input
    def closest(
        self,
        teff: u.Quantity[u.K],
        logg: u.Quantity[u.dex],
        FeH: u.Quantity[u.dex],
        MgFe: Optional[u.Quantity[u.dex]] = None,
    ):
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
        Spectra
            Spectrum from the closest star in the library.

        """
        return self.interpolate(teff, logg, FeH, MgFe, closest=True)

    @u.quantity_input
    def interpolate(
        self,
        teff: u.Quantity[u.K],
        logg: u.Quantity[u.dex],
        FeH: u.Quantity[u.dex],
        MgFe: Optional[u.Quantity[u.dex]] = None,
        closest: bool = False,
        simplex: bool = False,
    ):
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
        closest:
            Return the closest spectra, rather than performing the interpolation.
            If only one interpolation is performed, all the spectra in the simplex
            vertices are returned.
        simplex:
            If only one set of input parameters is given, return all the spectra
            that form part of the simplex used for the interpolation. These spectra
            have the weights information in their `meta` dictionary.

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

        teff = np.atleast_1d(teff)
        logg = np.atleast_1d(logg)
        FeH = np.atleast_1d(FeH)
        MgFe = np.atleast_1d(MgFe)

        wrong_shape = teff.shape != logg.shape
        wrong_shape |= teff.shape != FeH.shape
        if not self.MgFe_nan:
            wrong_shape |= teff.shape != MgFe.shape
        if wrong_shape:
            raise ValueError("The input parameters should all have the same shape")

        ninterp = len(teff)

        # Checking input point is within the grid
        def _compute_bounds(val, model):
            min_model = np.amin(model)
            max_model = np.amax(model)
            if np.any(val < min_model) or np.any(val > max_model):
                raise RuntimeError(
                    f"Input parameters {val} is outside of model grid: "
                    f"({min_model},{max_model})"
                )

        _compute_bounds(teff, self.models.meta["teff"])
        _compute_bounds(logg, self.models.meta["logg"])
        _compute_bounds(FeH, self.models.meta["FeH"])
        if not self.MgFe_nan:
            _compute_bounds(MgFe, self.models.meta["MgFe"])

        if closest:
            closest_idx = np.empty(ninterp, dtype=int)
        else:
            wave = self.models.spectral_axis
            spec = Quantity(
                value=np.empty((ninterp, self.models.data.shape[1])),
                unit=self.models.flux.unit,
            )
            new_meta = {
                "teff": teff,
                "logg": logg,
                "FeH": FeH,
                "MgFe": MgFe,
            }
            base_keys = list(new_meta.keys())
            for k in self.models.meta.keys():
                if k not in new_meta.keys():
                    if len(self.models.meta[k]) > 1:
                        if "U" not in self.models.meta[k].dtype.kind:
                            new_meta[k] = np.empty(
                                ninterp, dtype=self.models.meta[k].dtype
                            )

        for i in tqdm(range(ninterp), delay=3.0):
            if self.MgFe_nan:
                input_pt = np.atleast_2d(
                    [np.log10(teff[i].to_value(u.K)), logg[i].value, FeH[i].value]
                )
            else:
                input_pt = np.atleast_2d(
                    [
                        np.log10(teff[i].to_value(u.K)),
                        logg[i].value,
                        FeH[i].value,
                        MgFe[i].value,
                    ]
                )

            vtx, wts = interp_weights(self.params, input_pt, self.tri)
            vtx, wts = vtx.ravel(), wts.ravel()

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Simplex ids: {self.models.meta['index'][vtx]}")
                logger.debug(f"Simple ages: {self.models.meta['teff'][vtx]}")
                logger.debug(f"Simplex met: {self.models.meta['logg'][vtx]}")
                logger.debug(f"Simplex met: {self.models.meta['FeH'][vtx]}")
                logger.debug(f"Simplex weights: {wts}, norm: {np.sum(wts)}")

            idx = self.index[vtx]

            if closest:
                if simplex and ninterp == 1:
                    out = Spectra.__getitem__(
                        self.models, self.models.meta["index"][idx]
                    )
                    return out
                else:
                    closest_idx[i] = idx[np.argmax(wts)]
            else:
                spec[i, :] = np.dot(self.models.flux[idx, :].T, wts)

                # Interpolate the rest of the meta if possible
                for k in new_meta:
                    if len(self.models.meta[k]) > 1:
                        if k not in base_keys:
                            new_meta[k] = np.dot(self.models.meta[k][idx], wts)
        if closest:
            out = Spectra.__getitem__(self.models, closest_idx)
        else:
            out = Spectra(spectral_axis=wave, flux=spec, meta=new_meta)
        # Select the first and only spectra so that the users does not need to
        # do this all the time, manually
        if ninterp == 1:
            return out[0]
        else:
            return out
