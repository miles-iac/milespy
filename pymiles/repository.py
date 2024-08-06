# -*- coding: utf-8 -*-
import logging
import os

import h5py
import requests
from tqdm import tqdm

from .configuration import config
from .configuration import def_repo_folder

logger = logging.getLogger("pymiles.repository")


repository_url = {
    "MILES_STARS_v9.1": "https://cloud.iac.es/index.php/s/TKEwKfSiaZePYsx/download/MILES_STARS_v9.1.hdf5",  # noqa
    "MILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/wz3xS9jj7zDe7Hs/download/MILES_SSP_v9.1.hdf5",  # noqa
    "EMILES_SSP_v9.1": "https://cloud.iac.es/index.php/s/2CqEBsreXdeK2Pd/download/EMILES_SSP_v9.1.hdf5",  # noqa
    "CaT_STARS_v9.1": "https://cloud.iac.es/index.php/s/jCt2TzD8DMFXXdZ/download/CaT_STARS_v9.1.hdf5",  # noqa
    "CaT_SSP_v9.1": "https://cloud.iac.es/index.php/s/ex3Ep9jA5eG6Pwt/download/CaT_SSP_v9.1.hdf5",  # noqa
}


class Repository:
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

        logger.debug(f"# Loading models in {repo_filename}")

        if not os.path.exists(repo_filename):
            logger.warning("Unable to locate repository")

            if base_name in repository_url.keys():
                if "auto_download" in config.keys() and config["auto_download"]:
                    self._download_repository(base_name, repo_filename)
                else:
                    opt = input(
                        f"Do you want to download the {base_name} repository? [y/n]: "
                    )
                    if opt == "y":
                        self._download_repository(base_name, repo_filename)
            else:
                raise ValueError(f"No known URL for {base_name}")

        return repo_filename
