# -*- coding: utf-8 -*-
"""HDF5-based model repository management and optional download of MILES/EMILES/CaT data."""
import logging
import os

import astropy.units as u
import h5py
import requests
from specutils.manipulation import spectral_slab
from tqdm import tqdm

from .configuration import config
from .configuration import def_repo_folder

logger = logging.getLogger("milespy.repository")


repository_url = {
    "MILES_STARS_v9.1": "https://cloud.iac.es/index.php/s/TKEwKfSiaZePYsx/download/MILES_STARS_v9.1.hdf5",  # noqa
    "MILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/wz3xS9jj7zDe7Hs/download/MILES_SSP_v9.1.hdf5",  # noqa
    "sMILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/KsJFXKB7LLmGrxN/download/sMILES_SSP_v9.1.hdf5",  # noqa
    "EMILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/2CqEBsreXdeK2Pd/download/EMILES_SSP_v9.1.hdf5",  # noqa
    "CaT_STARS_v9.1": "https://cloud.iac.es/index.php/s/jCt2TzD8DMFXXdZ/download/CaT_STARS_v9.1.hdf5",  # noqa
    "CaT_SSP_v9.1": "https://cloud.iac.es/index.php/s/ex3Ep9jA5eG6Pwt/download/CaT_SSP_v9.1.hdf5",  # noqa
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
        Download the repository file for base_name from the configured URL to output_path.

        Parameters
        ----------
        base_name : str
            Key in repository_url (e.g. "MILES_SSP_v9.1").
        output_path : str
            Local path where the HDF5 file will be written.
        """
        response = requests.get(repository_url[base_name], stream=True)

        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(output_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Unable to download file")

        logger.debug(f"Downloaded {base_name} repository to {output_path}")

    def _get_repository(self, source: str, version: str) -> str:
        """
        Resolve the path to the repository HDF5 file, downloading if missing and allowed.

        If source + version matches a known key in repository_url, the file is
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

        if base_name in repository_url.keys():
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
