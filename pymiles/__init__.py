# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger("pymiles")
logger.setLevel(logging.DEBUG)

for existing_handler in list(logger.handlers):
    logger.removeHandler(existing_handler)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(name)s: %(message)s"))
logger.addHandler(ch)
