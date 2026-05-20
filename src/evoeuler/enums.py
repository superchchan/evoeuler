"""Enums."""

from __future__ import annotations

from enum import Enum


class Algorithm(Enum):
    """Available algorithms."""

    NSGA_II = "NSGA-II"
    NSGA_III = "NSGA-III"
