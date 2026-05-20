"""Shape transformation functions."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from evoeuler.config import BUFFER_DISTANCE, CHAIKIN_DEFAULT_ITERS


def chaikin(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Apply one iteration of Chaikin's corner-cutting algorithm.

    Args:
        coords: A list of tuples of two floats representing the coordinates of
        vertices forming a closed polygon.

    Returns:
        A list of tuples of two floats representing the coordinates of vertices
        forming a smoothed, closed polygon.

    Raises:
        ValueError: If len(coords) < 4.
        ValueError: If coords[0] != coords[-1].
    """
    if len(coords) < 4:
        raise ValueError("len(coords) must be >= 4")
    if coords[0] != coords[-1]:
        raise ValueError(
            "first and last coordinates must match to form a closed polygon"
        )
    new_coords = []
    for i in range(len(coords) - 1):
        coords_a = np.array(coords[i], dtype=float)
        coords_b = np.array(coords[i + 1], dtype=float)
        q = 0.75 * coords_a + 0.25 * coords_b
        r = 0.25 * coords_a + 0.75 * coords_b
        new_coords.append((float(q[0]), float(q[1])))
        new_coords.append((float(r[0]), float(r[1])))
    # Close the polygon by adding the first coords.
    new_coords.append(new_coords[0])
    return new_coords


def round_polygon(polygon: Polygon) -> Polygon:
    """Round the corners of a polygon.

    If the original or rounded polygon is not a simple polygon, return the
    original polygon. Otherwise, return the rounded polygon.

    Args:
        polygon: A Shapely Polygon.

    Returns:
        A Shapely Polygon representing the original or rounded polygon.
    """
    # Adapted from "https://gis.stackexchange.com/questions/93213/can-i-convert
    # -the-sharp-edges-of-a-polygon-easily-to-round-edges"
    if (
        polygon is None
        or polygon.is_empty
        or polygon.area == 0.0
        or not isinstance(polygon, Polygon)
        or not polygon.is_valid
    ):
        return polygon
    rounded_polygon = polygon.buffer(-BUFFER_DISTANCE).buffer(BUFFER_DISTANCE)
    if (
        rounded_polygon is None
        or rounded_polygon.is_empty
        or rounded_polygon.area == 0.0
        or not isinstance(rounded_polygon, Polygon)
        or not rounded_polygon.is_valid
    ):
        return polygon
    else:
        return rounded_polygon


def smooth_polygon(
    polygon: Polygon, iterations: int = CHAIKIN_DEFAULT_ITERS
) -> Polygon:
    """Smooth a polygon.

    Apply Chaikin's corner-cutting algorithm with the specified number of
    iterations. If the original or smoothed polygon is not a simple
    polygon, return the original polygon. Otherwise, return the smoothed
    polygon.

    Args:
        polygon: A Shapely Polygon.
        iterations: An integer representing the number of iterations.

    Returns:
        A Shapely Polygon representing the original or smoothed polygon.

    Raises:
        ValueError: If iterations <= 0.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if (
        polygon is None
        or polygon.is_empty
        or polygon.area == 0.0
        or not isinstance(polygon, Polygon)
        or not polygon.is_valid
    ):
        return polygon
    coords = list(zip(*polygon.exterior.coords.xy))
    for _ in range(iterations):
        coords = chaikin(coords)
    smoothed_polygon = Polygon(coords)
    if (
        smoothed_polygon is None
        or smoothed_polygon.is_empty
        or smoothed_polygon.area == 0.0
        or not isinstance(smoothed_polygon, Polygon)
        or not smoothed_polygon.is_valid
    ):
        return polygon
    return smoothed_polygon
