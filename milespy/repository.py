# -*- coding: utf-8 -*-
"""HDF5-based model repository management and optional download of MILES/EMILES/CaT data."""

import logging
import os
from pathlib import Path

import astropy.units as u
import h5py
import pooch
from specutils.manipulation import spectral_slab

from .configuration import config
from .configuration import def_repo_folder

logger = logging.getLogger("milespy.repository")


ZENODO_DOI = "10.5281/zenodo.19741547"

repository_files = {
    "MILES_STARS_v9.1": "MILES_STARS_v9.1.hdf5",
    "MILES_SSP_v9.1": "MILES_SSP_v9.1.hdf5",
    "sMILES_SSP_v9.1": "sMILES_SSP_v9.1.hdf5",
    "EMILES_SSP_v9.1": "EMILES_SSP_v9.1.hdf5",
    "CaT_STARS_v9.1": "CaT_STARS_v9.1.hdf5",
    "CaT_SSP_v9.1": "CaT_SSP_v9.1.hdf5",
}

repository_hashes = {
    "MILES_STARS_v9.1": "md5:8488c82fbf083cf4a519a3a9fe2e58ac",
    "MILES_SSP_v9.1": "md5:f925504bedeb8bd64670132b67f9dbf1",
    "sMILES_SSP_v9.1": "md5:648cb9ecc27b5d9f5d6d959d3d9d10ec",
    "EMILES_SSP_v9.1": "md5:5ab72bdf97cf887bcb2dbab90976e2a1",
    "CaT_STARS_v9.1": " md5:03ec120ac215c280bd4a9af991884744",
    "CaT_SSP_v9.1": " md5:86e96758188d5f29813ff54310b10da3",
}


class Repository:
    """
    Base class for HDF5-based model repositories (SSP or stellar library).

    Manages the ``models`` attribute (a Spectra-like object), optional download
    of repository files from configured URLs, and validation that each repository
    file is a readable HDF5 with a ``wave`` dataset. Subclasses (e.g. SSPLibrary)
    load and attach models after resolving the repository path via _get_repository.
    """

    def __init__(self, models):
        self._models = models

    def _assert_repository_file(self, file_path: str) -> None:
        """
        Verify that the path points to a readable HDF5 file containing a 'wave' dataset.

        Raises
        ------
        AssertionError
            If the file cannot be opened or does not contain the required structure.
        """
        try:
            with h5py.File(file_path, "r") as f:
                _ = f["wave"]
        except (OSError, KeyError) as e:
            raise AssertionError("Repository file is unreadable") from e

    def _download_repository(self, base_name: str, output_path: str) -> None:
        """
        Download the repository file for base_name from Zenodo to output_path.

        Parameters
        ----------
        base_name : str
            Key in repository_files (e.g. "MILES_SSP_v9.1").
        output_path : str
            Local path where the HDF5 file will be written.
        """
        file_name = repository_files[base_name]
        known_hash = repository_hashes.get(base_name)
        if known_hash is None:
            logger.warning(
                f"No known hash for {base_name}; downloading without hash validation."
            )
        repo_dir = Path(output_path).parent
        repo_dir.mkdir(parents=True, exist_ok=True)
        fetched_path = pooch.retrieve(
            url=f"doi:{ZENODO_DOI}/{file_name}",
            known_hash=known_hash,
            fname=file_name,
            path=repo_dir,
            progressbar=True,
        )
        # Keep backwards-compatible behavior: return file in the expected location.
        if Path(fetched_path).as_posix() != Path(output_path).as_posix():
            os.replace(fetched_path, output_path)

        logger.debug(
            f"Downloaded {base_name} from Zenodo DOI {ZENODO_DOI} to {output_path}"
        )

    def _get_repository(self, source: str, version: str) -> str:
        """
        Resolve the path to the repository HDF5 file, downloading if missing and allowed.

        If source + version matches a known key in repository_files, the file is
        looked up in the configured repository folder (or default); if missing,
        it may be downloaded when auto_download is True or the user confirms.
        Otherwise, source is returned as-is (treated as a local path).

        Parameters
        ----------
        source : str
            Model source name (e.g. "MILES_SSP") or path to a local file.
        version : str
            Version string (e.g. "9.1") used to form base_name = source + "_v" + version.

        Returns
        -------
        str
            Path to the repository HDF5 file, or source if not a known repository.
        """
        base_name = source + "_v" + version
        if "repository_folder" in config:
            repo_filename = config["repository_folder"] + base_name + ".hdf5"
        else:
            repo_filename = def_repo_folder.as_posix() + "/" + base_name + ".hdf5"

        logger.debug(f"Loading models in {repo_filename}")

        if base_name in repository_files.keys():
            if not os.path.exists(repo_filename):
                logger.warning("Unable to locate repository")
                if "auto_download" in config.keys() and config["auto_download"]:
                    self._download_repository(base_name, repo_filename)
                else:
                    opt = input(
                        f"Do you want to download the {base_name} repository? [y/n]: "
                    )
                    if opt == "y":
                        self._download_repository(base_name, repo_filename)
        else:
            logger.debug(f"Not known URL for {base_name}, trying to load it as a file")
            return source

        return repo_filename

    @property
    def models(self):
        """The spectra (or spectrum-like) object holding the loaded models."""
        return self._models

    def trim(self, lower: u.Quantity, upper: u.Quantity) -> None:
        """
        Restrict the spectral range of all models to [lower, upper].

        Updates :attr:`models` in place with the trimmed spectra; metadata is preserved.

        Parameters
        ----------
        lower : ~astropy.units.Quantity
            Lower wavelength bound (e.g. u.AA).
        upper : ~astropy.units.Quantity
            Upper wavelength bound.
        """
        trimmed = spectral_slab(self.models, lower, upper)
        trimmed.meta = self.models.meta
        self._models = trimmed

    def resample(self, new_wave: u.Quantity) -> None:
        """
        Resample all models onto the given wavelength grid.

        Updates :attr:`models` in place; metadata is preserved.

        Parameters
        ----------
        new_wave : ~astropy.units.Quantity
            New spectral axis (e.g. wavelength in u.AA).

        See Also
        --------
        :meth:`milespy.spectra.Spectra.resample`
        """
        resample = self.models.resample(new_wave)
        resample.meta = self.models.meta
        self._models = resample
