"""Optimization-related functions."""

from __future__ import annotations

import multiprocessing
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection
from pymoo.util.ref_dirs import get_reference_directions
from shapely.affinity import rotate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

from evoeuler.config import (
    ANGLE_LB,
    ANGLE_UB,
    BUILTIN_SHAPE_PARAM_COUNTS,
    CR_LB,
    CR_UB,
    CX_LB,
    CX_UB,
    CY_LB,
    CY_UB,
    DEFAULT_N_GEN,
    DEFAULT_N_PROCESS,
    H_LB,
    H_UB,
    N_OBJ,
    NSGA_III_N_PARTITIONS,
    NSGA_III_REF_DIR_GEN_ALG,
    POP_SIZE,
    R_LB,
    R_UB,
    REGULAR_POLYGONS,
    RX_LB,
    RX_UB,
    RY_LB,
    RY_UB,
    SX_LB,
    SX_UB,
    SY_LB,
    SY_UB,
    W_LB,
    W_UB,
)
from evoeuler.enums import Algorithm
from evoeuler.result import Result
from evoeuler.shapes.creation import create_shape
from evoeuler.utils import seconds_to_time
from evoeuler.zone import (
    all_zone_labels,
    compute_zone_geoms,
    extract_set_labels_from_zones,
    get_connectivity_status,
)


def get_param_counts(
    shape: str, custom_shapes: dict[str, Polygon] | None = None
) -> dict[str, int]:
    """Return the number of parameters required for a shape.

    Args:
        shape: A string representing the shape name.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.

    Returns:
        A dictionary mapping the shape name to its parameter count.

    Raises:
        ValueError: If the shape is unsupported.
    """
    if custom_shapes and shape in custom_shapes:
        return {shape: 5}
    if shape not in BUILTIN_SHAPE_PARAM_COUNTS:
        raise ValueError(f"unsupported shape: {shape}")
    return {shape: BUILTIN_SHAPE_PARAM_COUNTS[shape]}


def get_bounds(
    shape: str, n_set: int, custom_shapes: dict[str, Polygon] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute lower and upper bounds for the genotype.

    Args:
        shape: A string representing the shape name.
        n_set: An integer representing the number of sets.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.

    Returns:
        A tuple of two NDArray[np.float64].

    Raises:
        ValueError: If the shape is unsupported.
    """
    lb, ub = [], []
    for _ in range(n_set):
        if custom_shapes and shape in custom_shapes:
            lb += [CX_LB, CY_LB, SX_LB, SY_LB, ANGLE_LB]
            ub += [CX_UB, CY_UB, SX_UB, SY_UB, ANGLE_UB]
        elif shape == "circle":
            lb += [CX_LB, CY_LB, R_LB]
            ub += [CX_UB, CY_UB, R_UB]
        elif shape == "ellipse":
            lb += [CX_LB, CY_LB, RX_LB, RY_LB, ANGLE_LB]
            ub += [CX_UB, CY_UB, RX_UB, RY_UB, ANGLE_UB]
        elif shape in {"rectangle", "rounded_rectangle"}:
            lb += [CX_LB, CY_LB, W_LB, H_LB, ANGLE_LB]
            ub += [CX_UB, CY_UB, W_UB, H_UB, ANGLE_UB]
        elif shape in REGULAR_POLYGONS:
            lb += [CX_LB, CY_LB, CR_LB, ANGLE_LB]
            ub += [CX_UB, CY_UB, CR_UB, ANGLE_UB]
        else:
            raise ValueError(f"unsupported shape: {shape}")
    return np.array(lb, float), np.array(ub, float)


def enforce_symmetry(
    shape: str, params_per_set: list[NDArray[np.float64]]
) -> list[NDArray[np.float64]]:
    """Enforce rotational symmetry constraint on shape parameters.

    Args:
        shape: A string representing the shape type name.
        params_per_set: A list of NDArray[np.float64], where each
        NDArray[np.float64] represents shape parameters for one set.

    Returns:
        A list of NDArray[np.float64], where each NDArray[np.float64]
        represents shape parameters for one set.
    """
    n_set = len(params_per_set)
    if n_set < 2:
        return params_per_set
    # Set up parameters for enforcing rotational symmetry.
    rot_center = (0.0, 0.0)
    cx, cy = rot_center
    angle_step = 360.0 / n_set
    # Extract shape parameters from the first set.
    params = params_per_set[0]
    base_cx = params[0]
    base_cy = params[1]
    base_angle = 0.0
    if shape != "circle":
        base_angle = params[-1]
    # For each subsequent set, update its center coordinates and optionally the
    # angle if angle is part of its parameter.
    for i in range(1, n_set):
        new_center = rotate(
            Point(base_cx, base_cy), i * angle_step, origin=(cx, cy)
        )
        params_per_set[i][0] = new_center.x
        params_per_set[i][1] = new_center.y
        if shape != "circle":
            params_per_set[i][-1] = base_angle + i * angle_step
    return params_per_set


class EulerProblem(ElementwiseProblem):
    """Optimization problem for Euler diagram generation.

    Attributes:
        lb: A NDArray[np.float64] representing lower bounds for genotype
        parameters.
        ub: A NDArray[np.float64] representing upper bounds for genotype
        parameters.
        set_labels: A set of strings representing the set labels.
        n_set: An integer representing the number of sets.
        param_counts: A dictionary mapping shape names to parameter counts.
        n_obj: An integer representing the number of objectives.
        runner: A StarmapParallelization.
        zones: A set of strings representing the target zones.
        shape: A string representing the shape type for all curves.
        symmetric: A boolean indicating whether the diagram should be
        rotationally symmetric.
        congruent: A boolean indicating whether all shapes should be
        congruent.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.
    """

    def __init__(
        self,
        lb: NDArray[np.float64],
        ub: NDArray[np.float64],
        set_labels: set[str],
        param_counts: dict[str, int],
        n_obj: int,
        runner: StarmapParallelization,
        zones: set[str],
        shape: str,
        symmetric: bool = False,
        congruent: bool = False,
        custom_shapes: dict[str, Polygon] | None = None,
    ) -> None:
        """Initialize the optimization problem.

        Args:
            lb: A NDArray[np.float64] representing lower bounds for genotype
            parameters.
            ub: A NDArray[np.float64] representing upper bounds for genotype
            parameters.
            set_labels: A set of strings representing the set labels.
            param_counts: A dictionary mapping shape names to parameter counts.
            n_obj: An integer representing the number of objectives.
            runner: A StarmapParallelization.
            zones: A set of strings representing the target zones.
            shape: A string representing the shape type for all curves.
            symmetric: A boolean indicating whether the diagram should be
            rotationally symmetric.
            congruent: A boolean indicating whether all shapes should be
            congruent.
            custom_shapes: An optional dictionary mapping custom shape names to
            Shapely Polygons.
        """
        super().__init__(
            n_var=len(lb),
            n_obj=n_obj,
            n_constr=0,
            xl=lb,
            xu=ub,
            elementwise_runner=runner,
        )
        self.set_labels = set_labels
        self.n_set = len(set_labels)
        self.param_counts = param_counts
        self.n_obj = n_obj
        self.zones = zones
        self.shape = shape
        self.symmetric = symmetric
        self.congruent = congruent
        self.custom_shapes = custom_shapes

    def _evaluate(
        self,
        x: NDArray[np.float64],
        out: dict[str, NDArray[np.float64]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Evaluate a solution by computing objective function values.

        Args:
            x: An NDArray[np.float64] representing the genotype parameters.
            out: A dictionary mapping "F" to objective function values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        params_per_set = split_params(
            x, self.shape, self.param_counts, self.n_set
        )
        f = evaluate(
            self.set_labels,
            params_per_set,
            self.shape,
            self.zones,
            symmetric=self.symmetric,
            congruent=self.congruent,
            custom_shapes=self.custom_shapes,
        )
        out["F"] = f


def create_nsga2() -> NSGA2:
    """Create an NSGA2.

    Returns:
        An NSGA2.
    """
    return NSGA2(pop_size=POP_SIZE)


def create_nsga3() -> NSGA3:
    """Create an NSGA3.

    Returns:
        An NSGA3.
    """
    ref_dirs = get_reference_directions(
        NSGA_III_REF_DIR_GEN_ALG, N_OBJ, n_partitions=NSGA_III_N_PARTITIONS
    )
    return NSGA3(pop_size=POP_SIZE, ref_dirs=ref_dirs)


algorithm_dispatch = {
    Algorithm.NSGA_II: create_nsga2,
    Algorithm.NSGA_III: create_nsga3,
}


def create_algorithm(algorithm: str) -> NSGA2 | NSGA3:
    """Create an optimization algorithm.

    Args:
        algorithm: A string representing the algorithm name.

    Returns:
        An NSGA2 or an NSGA3.

    Raises:
        ValueError: If the algorithm is unsupported.
    """
    try:
        algorithm_enum = Algorithm(algorithm)
    except ValueError:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    create_algorithm_function = algorithm_dispatch.get(algorithm_enum)
    if create_algorithm_function is None:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    return create_algorithm_function()


def enforce_congruency(
    shape: str, params_per_set: list[NDArray[np.float64]]
) -> list[NDArray[np.float64]]:
    """Enforce congruency constraint on shape parameters.

    Args:
        shape: A string representing the shape type name.
        params_per_set: A list of NDArray[np.float64], where each
        NDArray[np.float64] represents shape parameters for one set.

    Returns:
        A list of NDArray[np.float64], where each NDArray[np.float64]
        represents shape parameters for one set.
    """
    if not params_per_set or any(len(params) < 3 for params in params_per_set):
        return params_per_set
    params = params_per_set[0]
    # Copy size-related parameters based on shape type from the first set to
    # each set.
    for i in range(1, len(params_per_set)):
        params_i = params_per_set[i]
        if shape == "circle":
            params_i[2] = params[2]
        elif shape == "ellipse":
            params_i[2], params_i[3] = params[2], params[3]
        elif shape in {"rectangle", "rounded_rectangle"}:
            params_i[2], params_i[3] = params[2], params[3]
        elif shape in REGULAR_POLYGONS:
            params_i[2] = params[2]
        else:
            params_i[2], params_i[3] = params[2], params[3]
    return params_per_set


def split_params(
    x: NDArray[np.float64],
    shape: str,
    param_counts: dict[str, int],
    n_set: int,
) -> list[NDArray[np.float64]]:
    """Split the genotype into shape parameter arrays for each set.

    Args:
        x: An NDArray[np.float64] representing the genotype parameters.
        shape: A string representing the shape type name.
        param_counts: A dictionary mapping shape names to parameter counts.
        n_set: An integer representing the number of sets.

    Returns:
        A list of NDArray[np.float64], where each NDArray[np.float64]
        represents shape parameters for one set.
    """
    params_per_set = []
    param_count = param_counts[shape]
    for i in range(n_set):
        start_idx = i * param_count
        params_per_set.append(x[start_idx : start_idx + param_count].copy())
    return params_per_set


def optimize(
    zones: set[str],
    shape: str,
    algorithm: str,
    n_gen: int = DEFAULT_N_GEN,
    n_process: int = DEFAULT_N_PROCESS,
    time: int | None = None,
    symmetric: bool = False,
    congruent: bool = False,
    custom_shapes: dict[str, Polygon] | None = None,
) -> Result:
    """Start an optimization.

    Args:
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

    Returns:
        A Result.

    Raises:
        ValueError: If not zones.
        ValueError: If n_gen <= 0.
        ValueError: If n_process <= 0.
        ValueError: If time is not None and time <= 0:
    """
    # Validate inputs.
    if not zones:
        raise ValueError("zones cannot be empty")
    if n_gen <= 0:
        raise ValueError("n_gen must be positive")
    if n_process <= 0:
        raise ValueError("n_process must be positive")
    if time is not None and time <= 0:
        raise ValueError("time must be positive or none")
    # Symmetry implies congruency.
    if symmetric:
        congruent = True
    # Rename zone labels by sorting their tokens (set labels) alphabetically.
    # For example, zone "B A" will become "A B".
    sorted_zones = set()
    for zone in zones:
        zone_label_tokens = zone.strip().split()
        if not zone_label_tokens:
            continue
        sorted_zone_label = " ".join(sorted(zone_label_tokens))
        sorted_zones.add(sorted_zone_label)
    zones = sorted_zones
    # Extract attributes for initializing an EulerProblem.
    set_labels = extract_set_labels_from_zones(zones)
    n_set = len(set_labels)
    param_counts = get_param_counts(shape, custom_shapes)
    lb, ub = get_bounds(shape, n_set, custom_shapes)
    algo = create_algorithm(algorithm)
    pool = multiprocessing.Pool(n_process)
    try:
        runner = StarmapParallelization(pool.starmap)
        # Initialize an EulerProblem.
        problem = EulerProblem(
            lb,
            ub,
            set_labels,
            param_counts,
            N_OBJ,
            runner,
            zones,
            shape,
            symmetric=symmetric,
            congruent=congruent,
            custom_shapes=custom_shapes,
        )
        # Set up termination criteria (based on generations and/or time).
        n_gen_termination = get_termination("n_gen", n_gen)
        if time is None:
            termination = n_gen_termination
        else:
            time_str = seconds_to_time(time)
            time_termination = get_termination("time", time_str)
            termination = TerminationCollection(
                n_gen_termination,
                time_termination,
            )
        # Run optimization.
        res = minimize(problem, algo, termination=termination, verbose=False)
    finally:
        pool.close()
        pool.join()
    return Result(
        res,
        set_labels,
        param_counts,
        zones,
        shape,
        algorithm,
        n_gen=n_gen,
        n_process=n_process,
        time=time,
        symmetric=symmetric,
        congruent=congruent,
        custom_shapes=custom_shapes,
    )


def evaluate(
    set_labels: set[str],
    params_per_set: list[NDArray[np.float64]],
    shape: str,
    zones: set[str],
    symmetric: bool = False,
    congruent: bool = False,
    custom_shapes: dict[str, Polygon] | None = None,
) -> NDArray[np.float64]:
    """Evaluate a solution by computing all objective function values.

    Computes 5 objectives:
    - Missing zones.
    - Unwanted zones.
    - Triple points.
    - Concurrency.
    - Disconnected zones.

    Args:
        set_labels: A set of strings representing set labels.
        params_per_set: A list of NDArray[np.float64], where each
        NDArray[np.float64] represents shape parameters for one set.
        zones: A set of strings representing target zones.
        shape: A string representing the shape type name.
        symmetric: A boolean indicating whether the diagram should be
        rotationally symmetric.
        congruent: A boolean indicating whether all shapes should be congruent.
        custom_shapes: An optional dictionary mapping shape names to custom
        Shapely Polygons.

    Returns:
        A NDArray[np.float64] of 5 floats representing the objective values.
    """
    shapes = {}
    # Enforce congruency constraint on shape parameters if enabled.
    if congruent:
        params_per_set = enforce_congruency(shape, params_per_set)
    # Enforce symmetry constraint on shape parameters if enabled.
    if symmetric:
        params_per_set = enforce_symmetry(shape, params_per_set)
    # Create shapes
    sorted_set_labels = sorted(set_labels)
    for set_label, params in zip(sorted_set_labels, params_per_set):
        shapes[set_label] = create_shape(shape, params, custom_shapes)
    # Compute the geometry of all possible zones.
    zone_labels = all_zone_labels(set_labels)
    zone_geoms = compute_zone_geoms(shapes, zone_labels)
    # Compute missing_zone, unwanted_zone, and disconnected_zone.
    missing_zone = 0
    unwanted_zone = 0
    disconnected_zone = 0
    for zone_label in zone_labels:
        zone_geom = zone_geoms.get(zone_label)
        is_empty = (
            zone_geom is None or zone_geom.is_empty or zone_geom.area == 0.0
        )
        if zone_label in zones and is_empty:
            missing_zone += 1
        if zone_label not in zones and not is_empty:
            unwanted_zone += 1
        if zone_geom is not None:
            connectivity_status = get_connectivity_status(zone_geom)
            if connectivity_status is None:
                continue
            if connectivity_status is False:
                disconnected_zone += 1
    # Compute triple_point, and concurrency.
    triple_point = 0
    concurrency = 0
    shared_points: dict[Point, set[str]] = {}
    # Loop through all pairs of curves.
    for set_label_a, set_label_b in combinations(shapes.keys(), 2):
        shape_a, shape_b = shapes[set_label_a], shapes[set_label_b]
        boundary_intersection = shape_a.boundary.intersection(shape_b.boundary)
        if (
            isinstance(boundary_intersection, LineString)
            and not boundary_intersection.is_empty
        ):
            concurrency += 1
        elif (
            isinstance(boundary_intersection, MultiLineString)
            and not boundary_intersection.is_empty
        ):
            for line_string in boundary_intersection.geoms:
                if (
                    isinstance(line_string, LineString)
                    and not line_string.is_empty
                ):
                    concurrency += 1
        elif (
            isinstance(boundary_intersection, Point)
            and not boundary_intersection.is_empty
        ):
            shared_points.setdefault(boundary_intersection, set()).update(
                [set_label_a, set_label_b]
            )
        elif (
            isinstance(boundary_intersection, MultiPoint)
            and not boundary_intersection.is_empty
        ):
            for point in boundary_intersection.geoms:
                if isinstance(point, Point) and not point.is_empty:
                    shared_points.setdefault(point, set()).update(
                        [set_label_a, set_label_b]
                    )
        elif (
            isinstance(boundary_intersection, GeometryCollection)
            and not boundary_intersection.is_empty
        ):
            for geom in boundary_intersection.geoms:
                if isinstance(geom, LineString) and not geom.is_empty:
                    concurrency += 1
                elif isinstance(geom, MultiLineString) and not geom.is_empty:
                    for line_string in geom.geoms:
                        if (
                            isinstance(line_string, LineString)
                            and not line_string.is_empty
                        ):
                            concurrency += 1
                elif isinstance(geom, Point) and not geom.is_empty:
                    shared_points.setdefault(geom, set()).update(
                        [set_label_a, set_label_b]
                    )
                elif isinstance(geom, MultiPoint) and not geom.is_empty:
                    for point in geom.geoms:
                        if isinstance(point, Point) and not point.is_empty:
                            shared_points.setdefault(point, set()).update(
                                [set_label_a, set_label_b]
                            )
    for _, sets in shared_points.items():
        if len(sets) >= 3:
            triple_point += 1
    return np.array(
        [
            missing_zone,
            unwanted_zone,
            triple_point,
            concurrency,
            disconnected_zone,
        ],
        dtype=float,
    )
