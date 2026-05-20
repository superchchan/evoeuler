"""Utility functions."""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure as PltFigure
from PIL import Image
from shapely.geometry import Polygon

from evoeuler.config import (
    DPI,
    SHAPE_FIGSIZE,
    SHAPE_XLIM,
    SHAPE_YLIM,
    TAB10,
    TAB20,
    TITLE_FONTSIZE,
)


def get_colors(n_set: int) -> list[tuple[float, float, float, float]]:
    """Generate a list of distinct colors.

    Args:
        n_set: An integer representing the number of sets that require distinct
        colors.

    Returns:
        A list of tuples of 4 floats representing the RGBA colors.
    """
    if n_set <= 0:
        return []
    elif n_set <= 10:
        # Use TAB10 for up to 10 sets.
        return TAB10[:n_set]
    elif n_set <= 20:
        # Use TAB20 for up to 20 sets.
        return TAB20[:n_set]
    else:
        # Use TAB20 for the first 20 sets and additional colors from the turbo
        # colormap for the remaining sets.
        n_extra_colors = n_set - 20
        cmap = plt.get_cmap("turbo")
        extra_colors = []
        for i in np.linspace(0, 1, n_extra_colors):
            extra_colors.append(cmap(i))
        return TAB20 + extra_colors


def figure_to_png(fig: PltFigure) -> bytes:
    """Convert a Matplotlib Figure to bytes of a PNG image.

    Args:
        fig: A Matplotlib Figure.

    Returns:
        Bytes representing a PNG image.
    """
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=DPI)
    return png_buf.getvalue()


def figure_to_svg(fig: PltFigure) -> str:
    """Convert a Matplotlib Figure to an SVG string.

    Args:
        fig: A Matplotlib Figure.

    Returns:
        A string representing an SVG image.
    """
    svg_buf = io.StringIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight", dpi=DPI)
    return svg_buf.getvalue()


def png_to_html_img(png_bytes: bytes, height: int) -> str:
    """Convert bytes of a PNG image to an HTML img tag.

    Args:
        png_bytes: Bytes representing a PNG image.
        height: An integer representing the target height in pixels.

    Returns:
        A string representing an HTML img tag.
    """
    img = Image.open(io.BytesIO(png_bytes))
    width = int(img.width * height / img.height)
    resized_img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized_img.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return (
        f"<img src='data:image/png;base64,{b64}' "
        f"style='height:{height}px; width:100%; "
        f"object-fit:contain; display:block;'/>"
    )


def seconds_to_time(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format.

    Args:
        seconds: An integer representing the number of seconds.

    Returns:
        A string representing time in HH:MM:SS format.

    Raises:
        ValueError: If seconds <= 0.
    """
    if seconds <= 0:
        raise ValueError("seconds must be a positive integer")
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def xy_to_svg_path(x: list[float], y: list[float]) -> str:
    """Convert x and y coordinates to a closed SVG path string.

    Args:
        x: A list of floats representing the x-coordinates.
        y: A list of floats representing the y-coordinates.

    Returns:
        A string representing a closed SVG path.

    Raises:
        ValueError: If not x or not y.
        ValueError: If len(x) != len(y):
    """
    if not x or not y:
        raise ValueError("x and y must be non-empty")
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    # Start the path by moving to the first point.
    path = f"M {x[0]} {y[0]}"
    # For each iteration, draw a line connecting the previous point to the
    # current point.
    for i in range(1, len(x)):
        path += f" L {x[i]} {y[i]}"
    # Close the path.
    path += " Z"
    return path


def plot_shape(shape: Polygon, title: str) -> PltFigure:
    """Plot a shape with the specified title.

    Args:
        shape: A Shapely Polygon.
        title: A string representing the title for the plot.

    Returns:
        A Matplotlib Figure showing the plotted shape.
    """
    fig, ax = plt.subplots(figsize=SHAPE_FIGSIZE)
    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        ax.plot(x, y)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(SHAPE_XLIM[0], SHAPE_XLIM[1])
    ax.set_ylim(SHAPE_YLIM[0], SHAPE_YLIM[1])
    ax.axis("off")
    ax.set_aspect("equal")
    return fig
