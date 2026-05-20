"""A non-dominated solution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon


@dataclass
class Solution:
    """A non-dominated solution.

    Attributes:
        idx: An integer representing the index of the non-dominated solution.
        x: An NDArray[np.float64] representing the genotype parameters.
        f: An NDArray[np.float64] representing the objective function values.
        set_labels: A set of strings representing the set labels.
        param_counts: A dictionary mapping shape names to parameter counts.
        zones: A set of strings representing the target zones.
        shape: A string representing the shape type for all curves.
        symmetric: A boolean indicating whether the diagram should be
        rotationally symmetric.
        congruent: A boolean indicating whether all shapes should be congruent.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.
    """

    idx: int
    x: NDArray[np.float64]
    f: NDArray[np.float64]
    set_labels: set[str]
    param_counts: dict[str, int]
    zones: set[str]
    shape: str
    symmetric: bool = False
    congruent: bool = False
    custom_shapes: dict[str, Polygon] | None = None
