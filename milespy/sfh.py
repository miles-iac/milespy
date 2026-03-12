# -*- coding: utf-8 -*-
"""Deprecated: use milespy.star_formation_histories and StarFormationHistories instead."""
import warnings

from .star_formation_histories import StarFormationHistories


class SFH(StarFormationHistories):
    """Deprecated alias for StarFormationHistories; use StarFormationHistories instead."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SFH is deprecated; use StarFormationHistories instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["SFH", "StarFormationHistories"]
