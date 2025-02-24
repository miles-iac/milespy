# -*- coding: utf-8 -*-
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
    "EMILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/2CqEBsreXdeK2Pd/download/EMILES_SSP_v9.1.hdf5",  # noqa
    "CaT_STARS_v9.1": "https://cloud.iac.es/index.php/s/jCt2TzD8DMFXXdZ/download/CaT_STARS_v9.1.hdf5",  # noqa
    "CaT_SSP_v9.1": "https://cloud.iac.es/index.php/s/ex3Ep9jA5eG6Pwt/download/CaT_SSP_v9.1.hdf5",  # noqa
}


class Repository:
    def __init__(self, models):
        self._models = models

    def _assert_repository_file(self, file_path):
        try:
            with h5py.File(file_path) as f:
                _ = f["wave"]
        except:  # noqa
            raise AssertionError("Repository file is unreadable")

    def _download_repository(self, base_name, output_path):
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

        logger.debug(f"Dowloaded {base_name} repository in {output_path}")

    def _get_repository(self, source, version) -> str:
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
        return self._models

    def trim(self, lower: u.Quantity, upper: u.Quantity):
        """
        Trim all the models in the library

        Parameters
        ----------
        lower : Quantity
        upper : Quantity
        """
        trimmed = spectral_slab(self.models, lower, upper)
        trimmed.meta = self.models.meta
        self._models = trimmed

    def resample(self, new_wave: u.Quantity):
        """
        Resample all the models in the library

        Parameters
        ----------
        new_wave
            Spectral axis with the desired sampling for the spectra

        See Also
        --------
        :meth:`milespy.spectra.resample`
        """
        resample = self.models.resample(new_wave)
        resample.meta = self.models.meta
        self._models = resample
