# -*- coding: utf-8 -*-
import logging
import pathlib

from rcfile import rcfile


def _setup_logger(config):
    logger = logging.getLogger("milespy")
    logger.setLevel(logging.WARNING)

    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(ch)

    if "log" in config.keys():
        logger.setLevel(config["log"])


config = rcfile("milespy", {})
config_folder = pathlib.Path(__file__).parent.resolve() / "config_files"
def_repo_folder = pathlib.Path(__file__).parent.resolve()

logger = _setup_logger(config)


def get_config_file(name):
    return config_folder.as_posix() + "/" + name
