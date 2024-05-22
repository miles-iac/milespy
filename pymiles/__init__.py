# -*- coding: utf-8 -*-
import logging

from rcfile import rcfile


logger = logging.getLogger("pymiles")
logger.setLevel(logging.WARNING)

for existing_handler in list(logger.handlers):
    logger.removeHandler(existing_handler)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logger.addHandler(ch)

config = rcfile(__name__, {})
