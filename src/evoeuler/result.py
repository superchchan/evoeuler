"""Result of an optimization run."""

from __future__ import annotations

from typing import Iterator

from pymoo.core.result import Result as PymooResult
from shapely.geometry import Polygon

from evoeuler.config import DEFAULT_N_GEN, DEFAULT_N_PROCESS
from evoeuler.solution import Solution


class Result:
    """Result of an optimization run.

    Attributes:
        res: A Pymoo Result containing the optimization result.
        set_labels: A set of strings representing the set labels.
        param_counts: A dictionary mapping shape names to parameter counts.
        zones: A set of strings representing the target zones.
        shape: A string representing the shape type for all curves.
        algorithm: A string representing the optimization algorithm.
        n_gen: An integer representing the number of generations.
        n_process: An integer representing the number of processes for
        parallelization.
        time: An optional integer representing the time in seconds.
        symmetric: A boolean indicating whether the diagram should be
        rotationally symmetric.
        congruent: A boolean indicating whether all shapes should be congruent.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.
    """

    def __init__(
        self,
        res: PymooResult,
        set_labels: set[str],
        param_counts: dict[str, int],
        zones: set[str],
        shape: str,
        algorithm: str,
        n_gen: int = DEFAULT_N_GEN,
        n_process: int = DEFAULT_N_PROCESS,
        time: int | None = None,
        symmetric: bool = False,
        congruent: bool = False,
        custom_shapes: dict[str, Polygon] | None = None,
    ) -> None:
        """Initialize a Result.

        Args:
            res: A Pymoo Result containing the optimization result.
            set_labels: A set of strings representing the set labels.
            param_counts: A dictionary mapping shape names to parameter counts.
            zones: A set of strings representing the target zones.
            shape: A string representing the shape type for all curves.
            algorithm: A string representing the optimization algorithm name.
            n_gen: An integer representing the number of generations.
            n_process: An integer representing the number of processes for
            parallelization.
            time: An optional integer representing the optimization time
            in seconds.
            symmetric: A boolean indicating whether the diagram should be
            rotationally symmetric.
            congruent: A boolean indicating whether all shapes should be
            congruent.
            custom_shapes: An optional dictionary mapping custom shape names to
            Shapely Polygons.
        """
        self.res = res
        self.set_labels = set_labels
        self.param_counts = param_counts
        self.zones = zones
        self.shape = shape
        self.algorithm = algorithm
        self.n_gen = n_gen
        self.n_process = n_process
        self.time = time
        self.symmetric = symmetric
        self.congruent = congruent
        self.custom_shapes = custom_shapes

    def __len__(self) -> int:
        """Return the number of non-dominated solutions.

        Returns:
            An integer representing the number of non-dominated solutions.
        """
        return len(self.res.X)

    def __getitem__(self, idx: int) -> Solution:
        """Return the non-dominated solution at the specified index.

        Args:
            idx: An integer representing the index of the non-dominated
            solution.

        Returns:
            A Solution at the specified index.
        """
        return Solution(
            idx,
            self.res.X[idx],
            self.res.F[idx],
            self.set_labels,
            self.param_counts,
            self.zones,
            self.shape,
            symmetric=self.symmetric,
            congruent=self.congruent,
            custom_shapes=self.custom_shapes,
        )

    def __iter__(self) -> Iterator[Solution]:
        """Iterate over the set of non-dominated solutions.

        Yields:
            A Solution.
        """
        for i in range(len(self)):
            yield self[i]
