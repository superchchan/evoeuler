"""Shape creation functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from shapely import set_precision
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Point, Polygon

from evoeuler.config import GRID_SIZE, RESOLUTION
from evoeuler.shapes.transforms import round_polygon


def create_regular_polygon(
    cx: float, cy: float, cr: float, angle: float, n_sides: int
) -> Polygon:
    """Create a regular polygon with the specified number of sides.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.
        n_sides: An integer representing the number of sides.

    Returns:
        A Shapely Polygon representing the regular polygon.

    Raises:
        ValueError: If n_sides < 3.
        ValueError: If cr <= 0.
    """
    if n_sides < 3:
        raise ValueError("n_sides must be >= 3")
    if cr <= 0:
        raise ValueError("cr must be positive")
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    points = []
    for a in angles:
        points.append((np.cos(a) * cr, np.sin(a) * cr))
    regular_polygon = Polygon(points)
    regular_polygon = rotate(regular_polygon, angle)
    regular_polygon = translate(regular_polygon, xoff=cx, yoff=cy)
    return regular_polygon


def create_ellipse(
    cx: float, cy: float, rx: float, ry: float, angle: float
) -> Polygon:
    """Create an ellipse.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        rx: A float representing the radius along the x-axis.
        ry: A float representing the radius along the y-axis.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the ellipse.

    Raises:
        ValueError: If rx <= 0.
        ValueError: If ry <= 0.
    """
    if rx <= 0:
        raise ValueError("rx must be positive")
    if ry <= 0:
        raise ValueError("ry must be positive")
    ellipse = Point(0, 0).buffer(1.0, resolution=RESOLUTION)
    ellipse = scale(ellipse, xfact=rx, yfact=ry)
    ellipse = rotate(ellipse, angle)
    ellipse = translate(ellipse, xoff=cx, yoff=cy)
    return ellipse


def create_circle(cx: float, cy: float, r: float) -> Polygon:
    """Create a circle.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        r: A float representing the radius.

    Returns:
        A Shapely Polygon representing the circle.

    Raises:
        ValueError: If r <= 0.
    """
    if r <= 0:
        raise ValueError("r must be positive")
    circle = Point(0, 0).buffer(1.0, resolution=RESOLUTION)
    circle = scale(circle, xfact=r, yfact=r)
    circle = translate(
        circle,
        xoff=cx,
        yoff=cy,
    )
    return circle


def create_equilateral_triangle(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create an equilateral triangle.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the equilateral triangle.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=3)


def create_square(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a square.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the square.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=4)


def create_pentagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a pentagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the pentagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=5)


def create_hexagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a hexagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the hexagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=6)


def create_heptagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a heptagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the heptagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=7)


def create_octagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create an octagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the octagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=8)


def create_nonagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a nonagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the nonagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=9)


def create_decagon(cx: float, cy: float, cr: float, angle: float) -> Polygon:
    """Create a decagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the decagon.
    """
    return create_regular_polygon(cx, cy, cr, angle, n_sides=10)


def create_rectangle(
    cx: float, cy: float, w: float, h: float, angle: float
) -> Polygon:
    """Create a rectangle.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        w: A float representing the width.
        h: A float representing the height.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rectangle.

    Raises:
        ValueError: If w <= 0.
        ValueError: If h <= 0.
    """
    if w <= 0:
        raise ValueError("w must be positive")
    if h <= 0:
        raise ValueError("h must be positive")
    rectangle = Polygon(
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    )
    rectangle = rotate(rectangle, angle)
    rectangle = translate(rectangle, xoff=cx, yoff=cy)
    return rectangle


def create_rounded_equilateral_triangle(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded equilateral triangle.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded equilateral triangle.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=3))


def create_rounded_square(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded square.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded square.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=4))


def create_rounded_pentagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded pentagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded pentagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=5))


def create_rounded_hexagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded hexagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded hexagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=6))


def create_rounded_heptagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded heptagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded heptagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=7))


def create_rounded_octagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded octagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded octagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=8))


def create_rounded_nonagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded nonagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded nonagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=9))


def create_rounded_decagon(
    cx: float, cy: float, cr: float, angle: float
) -> Polygon:
    """Create a rounded decagon.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        cr: A float representing the circumradius.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded decagon.
    """
    return round_polygon(create_regular_polygon(cx, cy, cr, angle, n_sides=10))


def create_rounded_rectangle(
    cx: float, cy: float, w: float, h: float, angle: float
) -> Polygon:
    """Create a rounded rectangle.

    Args:
        cx: A float representing the x-coordinate of the center.
        cy: A float representing the y-coordinate of the center.
        w: A float representing the width.
        h: A float representing the height.
        angle: A float representing the rotation angle in degrees.

    Returns:
        A Shapely Polygon representing the rounded rectangle.

    Raises:
        ValueError: If w <= 0.
        ValueError: If h <= 0.
    """
    if w <= 0:
        raise ValueError("w must be positive")
    if h <= 0:
        raise ValueError("h must be positive")
    rounded_rectangle = Polygon(
        [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    )
    rounded_rectangle = rotate(rounded_rectangle, angle)
    rounded_rectangle = translate(rounded_rectangle, xoff=cx, yoff=cy)
    rounded_rectangle = round_polygon(rounded_rectangle)
    return rounded_rectangle


shape_dispatch: dict[str, Callable[..., Polygon]] = {
    "circle": create_circle,
    "ellipse": create_ellipse,
    "rectangle": create_rectangle,
    "rounded_rectangle": create_rounded_rectangle,
    "equilateral_triangle": create_equilateral_triangle,
    "rounded_equilateral_triangle": create_rounded_equilateral_triangle,
    "square": create_square,
    "rounded_square": create_rounded_square,
    "pentagon": create_pentagon,
    "rounded_pentagon": create_rounded_pentagon,
    "hexagon": create_hexagon,
    "rounded_hexagon": create_rounded_hexagon,
    "heptagon": create_heptagon,
    "rounded_heptagon": create_rounded_heptagon,
    "octagon": create_octagon,
    "rounded_octagon": create_rounded_octagon,
    "nonagon": create_nonagon,
    "rounded_nonagon": create_rounded_nonagon,
    "decagon": create_decagon,
    "rounded_decagon": create_rounded_decagon,
}


def create_shape(
    shape: str,
    params: NDArray[np.float64],
    custom_shapes: dict[str, Polygon] | None = None,
) -> Polygon:
    """Create a shape based on shape name and parameters.

    Create a built-in shape or transform an existing custom shape using the
    provided parameters.

    Args:
        shape: A string representing the shape name.
        params: A NDArray[np.float64] representing the parameters for the
        shape.
        custom_shapes: An optional dictionary mapping custom shape names to
        Shapely Polygons.

    Returns:
        A Shapely Polygon representing the created shape.

    Raises:
        ValueError: If len(params) != 5 for custom shapes.
        ValueError: If sx <= 0 for custom shapes.
        ValueError: If sy <= 0 for custom shapes.
        ValueError: If setting precision on a custom shape does not return a
        valid polygon.
        ValueError: If the shape name is unsupported.
        ValueError: If setting precision on a built-in shape does not return a
        valid polygon.
    """
    if custom_shapes and shape in custom_shapes:
        custom_shape = custom_shapes[shape]
        if len(params) != 5:
            raise ValueError("custom shapes require 5 parameters")
        # cx: A float representing the x-coordinate of the center.
        # cy: A float representing the y-coordinate of the center.
        # sx: A float representing the scale factor along the x-axis.
        # sy: A float representing the scale factor along the y-axis.
        # angle: A float representing the rotation angle in degrees.
        cx, cy, sx, sy, angle = params
        if sx <= 0:
            raise ValueError("sx must be positive")
        if sy <= 0:
            raise ValueError("sy must be positive")
        current_centroid = custom_shape.centroid
        custom_shape = translate(
            custom_shape, xoff=-current_centroid.x, yoff=-current_centroid.y
        )
        custom_shape = scale(custom_shape, xfact=sx, yfact=sy, origin=(0, 0))
        custom_shape = rotate(custom_shape, angle, origin=(0, 0))
        custom_shape = translate(custom_shape, xoff=cx, yoff=cy)
        custom_shape = set_precision(custom_shape, GRID_SIZE)
        if (
            custom_shape is None
            or not isinstance(custom_shape, Polygon)
            or custom_shape.is_empty
            or not custom_shape.is_valid
            or custom_shape.area == 0.0
        ):
            raise ValueError(
                f"setting precision on a custom {shape} does not return a "
                f"valid polygon"
            )
        return custom_shape
    create_shape_function = shape_dispatch.get(shape)
    if not create_shape_function:
        raise ValueError(f"unsupported shape: {shape}")
    built_in_shape = set_precision(create_shape_function(*params), GRID_SIZE)
    if (
        built_in_shape is None
        or not isinstance(built_in_shape, Polygon)
        or built_in_shape.is_empty
        or not built_in_shape.is_valid
        or built_in_shape.area == 0.0
    ):
        raise ValueError(
            f"setting precision on a/an {shape} does not return a valid polygon"
        )
    return built_in_shape
