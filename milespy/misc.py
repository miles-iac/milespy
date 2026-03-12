# -*- coding: utf-8 -*-
"""Deprecated: use milespy.interpolation_utils instead."""
import warnings

warnings.warn(
    "milespy.misc is deprecated; use milespy.interpolation_utils instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .interpolation_utils import interp_weights  # noqa: E402

__all__ = ["interp_weights"]
