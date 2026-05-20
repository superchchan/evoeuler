"""Public APIs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from matplotlib.figure import Figure as PltFigure
from plotly.graph_objects import Figure as GoFigure
from shapely.geometry import MultiPolygon, Polygon

from evoeuler.config import (
    DEFAULT_N_GEN,
    DEFAULT_N_PROCESS,
    DIAGRAM_FIGSIZE,
    DIAGRAM_LINEWIDTH,
    FILL_COLOR,
    LEGEND_BBOX_TO_ANCHOR,
    LEGEND_LOC,
    OBJECTIVE_NAMES,
    PCP_BOTTOM_MARGIN,
    PCP_FONT_COLOR,
    PCP_FONT_SIZE,
    PCP_HEIGHT,
    PCP_LEFT_MARGIN,
    PCP_RIGHT_MARGIN,
    PCP_TEMPLATE,
    REGULAR_POLYGONS,
    SHADE_COLOR,
    ZONE_LABEL_FONTSIZE,
)
from evoeuler.optimization import enforce_congruency, enforce_symmetry
from evoeuler.optimization import optimize as _optimize
from evoeuler.optimization import split_params
from evoeuler.result import Result
from evoeuler.shapes.creation import create_shape
from evoeuler.solution import Solution
from evoeuler.utils import get_colors
from evoeuler.zone import (
    all_zone_labels,
    compute_zone_geoms,
    format_connectivity_status,
)


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
    """
    return _optimize(
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


def pcp(result: Result) -> GoFigure:
    """Create a parallel coordinates plot visualizing the solutions.

    Args:
        result: A Result.

    Returns:
        A Plotly Figure showing the parallel coordinates plot.
    """
    # Create a DataFrame containing objective values for all solutions from the
    # set of non-dominated solutions.
    df = pd.DataFrame(data=result.res.F, columns=OBJECTIVE_NAMES)
    # Add an "Index" column so each solution can be traced.
    df["Index"] = df.index
    dimensions = ["Index"] + OBJECTIVE_NAMES
    fig = px.parallel_coordinates(
        data_frame=df,
        dimensions=dimensions,
        labels={col: col for col in dimensions},
    )
    for i, col in enumerate(dimensions):
        if col != "Index":
            max_val = df[col].max()
            range_max = max(max_val, 1)
        else:
            max_val = df[col].max()
            range_max = max_val
        fig.data[0].dimensions[i].range = [0, range_max]
        if range_max <= 10:
            tickvals = list(range(0, int(range_max) + 1))
            ticktext = list(map(str, range(0, int(range_max) + 1)))
            fig.data[0].dimensions[i].tickvals = tickvals
            fig.data[0].dimensions[i].ticktext = ticktext
    fig.update_layout(
        template=PCP_TEMPLATE,
        font=dict(size=PCP_FONT_SIZE, color=PCP_FONT_COLOR),
        margin=dict(l=PCP_LEFT_MARGIN, r=PCP_RIGHT_MARGIN, b=PCP_BOTTOM_MARGIN),
        height=PCP_HEIGHT,
    )
    return fig


def obj_values(solution: Solution) -> dict[str, int]:
    """Extract objective function values from a solution.

    Args:
        solution: A Solution from the set of non-dominated solutions.

    Returns:
        A dictionary mapping objective names to their values.
    """
    result: dict[str, int] = {}
    for name, value in zip(OBJECTIVE_NAMES, solution.f):
        result[name] = int(value)
    return result


def shape_params(solution: Solution) -> pd.DataFrame:
    """Extract shape parameters from a solution as a pandas DataFrame.

    Args:
        solution: A Solution from the set of non-dominated solutions.

    Returns:
        A pandas DataFrame with each row showing shape parameters for a set.
    """
    set_labels = solution.set_labels
    shape = solution.shape
    n_set = len(set_labels)
    # Get original shape parameters per set.
    params_per_set = split_params(
        solution.x, shape, solution.param_counts, n_set
    )
    # Enforce congruency constraint on shape parameters if enabled.
    if solution.congruent:
        params_per_set = enforce_congruency(shape, params_per_set)
    # Enforce symmetry constraint on shape parameters if enabled.
    if solution.symmetric:
        params_per_set = enforce_symmetry(shape, params_per_set)
    # Each row describes the shape type and shape parameters for a set.
    rows = []
    sorted_set_labels = sorted(set_labels)
    for set_label, params in zip(sorted_set_labels, params_per_set):
        row = {"Set": set_label, "Shape": shape}
        if shape == "circle":
            row.update(
                {
                    "Center X": params[0],
                    "Center Y": params[1],
                    "Radius": params[2],
                }
            )
        elif shape == "ellipse":
            row.update(
                {
                    "Center X": params[0],
                    "Center Y": params[1],
                    "Radius X": params[2],
                    "Radius Y": params[3],
                    "Rotation Angle (in degree)": params[4],
                }
            )
        elif shape in {"rectangle", "rounded_rectangle"}:
            row.update(
                {
                    "Center X": params[0],
                    "Center Y": params[1],
                    "Width": params[2],
                    "Height": params[3],
                    "Rotation Angle (in degree)": params[4],
                }
            )
        elif shape in REGULAR_POLYGONS:
            row.update(
                {
                    "Center X": params[0],
                    "Center Y": params[1],
                    "Circumradius": params[2],
                    "Rotation Angle (in degree)": params[3],
                }
            )
        else:
            row.update(
                {
                    "Center X": params[0],
                    "Center Y": params[1],
                    "Scale Factor X": params[2],
                    "Scale Factor Y": params[3],
                    "Rotation Angle (in degree)": params[4],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def zone_info(solution: Solution) -> pd.DataFrame:
    """Extract zone information from a solution as a pandas DataFrame.

    Args:
        solution: A Solution from the set of non-dominated solutions.

    Returns:
        A pandas DataFrame containing zone information.
    """
    set_labels = solution.set_labels
    shape = solution.shape
    n_set = len(set_labels)
    # Get original shape parameters per set.
    params_per_set = split_params(
        solution.x, shape, solution.param_counts, n_set
    )
    # Enforce congruency constraint on shape parameters if enabled.
    if solution.congruent:
        params_per_set = enforce_congruency(shape, params_per_set)
    # Enforce symmetry constraint on shape parameters if enabled.
    if solution.symmetric:
        params_per_set = enforce_symmetry(shape, params_per_set)
    # Create shapes.
    shapes = {}
    sorted_set_labels = sorted(set_labels)
    for set_label, params in zip(sorted_set_labels, params_per_set):
        shapes[set_label] = create_shape(
            shape, params, custom_shapes=solution.custom_shapes
        )
    # Compute geometry of zones.
    zone_geoms = compute_zone_geoms(shapes, all_zone_labels(set_labels))
    # Each row describes the information of a zone.
    rows = []
    for zone_label in sorted(
        all_zone_labels(set_labels),
        key=lambda zone_label: (len(zone_label.split()), zone_label),
    ):
        zone_geom = zone_geoms.get(zone_label)
        is_wanted = zone_label in solution.zones
        is_shown = (
            zone_geom is not None
            and not zone_geom.is_empty
            and zone_geom.area > 0.0
        )
        if zone_geom is not None:
            connectivity = format_connectivity_status(zone_geom)
        else:
            connectivity = "N/A"
        rows.append(
            {
                "Zone": zone_label,
                "IsWanted": is_wanted,
                "IsShown": is_shown,
                "Connectivity": connectivity,
            }
        )
    return pd.DataFrame(rows)


def diagram(
    solution: Solution,
    show_zone_labels: bool = False,
    shade_unwanted_zones: bool = True,
) -> PltFigure:
    """Generate a Matplotlib figure of the diagram.

    Args:
        solution: A Solution from the set of non-dominated solutions.
        show_zone_labels: A boolean indicating whether to display zone labels
        on the diagram.
        shade_unwanted_zones: A boolean indicating whether to shade unwanted
        zones on the diagram.

    Returns:
        A Matplotlib Figure showing the diagram.
    """
    set_labels = solution.set_labels
    shape = solution.shape
    n_set = len(set_labels)
    # Get original shape parameters per set.
    params_per_set = split_params(
        solution.x, shape, solution.param_counts, n_set
    )
    # Enforce congruency constraint on shape parameters if enabled.
    if solution.congruent:
        params_per_set = enforce_congruency(shape, params_per_set)
    # Enforce symmetry constraint on shape parameters if enabled.
    if solution.symmetric:
        params_per_set = enforce_symmetry(shape, params_per_set)
    # Create shapes.
    shapes = {}
    sorted_set_labels = sorted(set_labels)
    for set_label, params in zip(sorted_set_labels, params_per_set):
        shapes[set_label] = create_shape(
            shape, params, custom_shapes=solution.custom_shapes
        )
    # Compute geometry of zones.
    zone_geoms = compute_zone_geoms(shapes, all_zone_labels(set_labels))
    # Set up the plot.
    fig, ax = plt.subplots(figsize=DIAGRAM_FIGSIZE)
    # Get the colors for each shape.
    colors = get_colors(len(shapes))
    # Plot shapes.
    for (set_label, shape_geom), color in zip(shapes.items(), colors):
        ax.plot(
            *shape_geom.exterior.xy,
            color=color,
            linewidth=DIAGRAM_LINEWIDTH,
            label=set_label,
        )
    # Shade unwanted zones if enabled.
    if shade_unwanted_zones:
        for zone_label, zone_geom in zone_geoms.items():
            if zone_geom.is_empty:
                continue
            is_unwanted = (
                zone_label not in solution.zones and zone_geom.area > 0.0
            )
            if is_unwanted:
                if isinstance(zone_geom, Polygon):
                    x, y = zone_geom.exterior.xy
                    ax.fill(x, y, color=SHADE_COLOR)
                    for hole in zone_geom.interiors:
                        hole_x, hole_y = hole.xy
                        ax.fill(hole_x, hole_y, color=FILL_COLOR)
                elif isinstance(zone_geom, MultiPolygon):
                    for region_geom in zone_geom.geoms:
                        x, y = region_geom.exterior.xy
                        ax.fill(x, y, color=SHADE_COLOR)
                        for hole in region_geom.interiors:
                            hole_x, hole_y = hole.xy
                            ax.fill(hole_x, hole_y, color=FILL_COLOR)
    # Plot zone labels if enabled.
    if show_zone_labels:
        for zone_label, zone_geom in zone_geoms.items():
            if zone_geom.is_empty or zone_geom.area == 0:
                continue
            representative_point = zone_geom.representative_point()
            ax.text(
                representative_point.x,
                representative_point.y,
                zone_label,
                ha="center",
                va="center",
                fontsize=ZONE_LABEL_FONTSIZE,
            )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR)
    return fig
