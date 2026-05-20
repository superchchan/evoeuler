"""Tabs for the Streamlit application."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Polygon

from evoeuler.api import diagram, pcp, zone_info
from evoeuler.config import (
    ALGORITHM_CHOICES,
    AUTOSIZE,
    BUILTIN_SHAPES,
    CLICK_MODE,
    DEFAULT_N_GEN,
    DEFAULT_N_PROCESS,
    DIAGRAM_HEIGHT,
    DRAG_MODE,
    DRAW_REDRAW_HEIGHT,
    DRAW_REDRAW_WIDTH,
    PNG_DIAGRAM_NAME,
    ROUND_SMOOTH_DELETE_SHAPE_HEIGHT,
    SHAPE_XLIM,
    SHAPE_YLIM,
    SVG_DIAGRAM_NAME,
    VIEW_SHAPE_PARAMS,
    ZONE_INPUT_EXAMPLE,
    ZONE_INPUT_HEIGHT,
    ZONE_TABLE_HEIGHT,
)
from evoeuler.optimization import optimize
from evoeuler.shapes.creation import shape_dispatch
from evoeuler.shapes.transforms import round_polygon, smooth_polygon
from evoeuler.utils import (
    figure_to_png,
    figure_to_svg,
    plot_shape,
    png_to_html_img,
    xy_to_svg_path,
)
from evoeuler.zone import parse_zone_input


def get_next_shape_name() -> str:
    """Generate a unique shape name."""
    st.session_state.setdefault("shape_id", 0)
    st.session_state["shape_id"] += 1
    return f"shape_{st.session_state['shape_id']}"


def get_shape_list() -> list[str]:
    """Get list of custom shape names."""
    return list(st.session_state.get("custom_shapes", {}).keys())


def get_selected_shape(tab_name: str, shapes: list[str]) -> str:
    """Get or initialize the selected shape for a given tab.

    Args:
        tab_name: A string representing the name of the tab.
        shapes: A list of available shape names.

    Returns:
        A string representing the currently selected shape name.
    """
    key = f"{tab_name}_selection"
    if key not in st.session_state or st.session_state[key] not in shapes:
        st.session_state[key] = shapes[0]
    return str(st.session_state[key])


def validate_polygon(
    coords: list[tuple[float, float]],
) -> tuple[Polygon | None, str | None]:
    """Validate coordinates and create a polygon.

    Args:
        coords: A list of tuples of two floats representing the coordinates of
        points.

    Returns:
        If the polygon is valid, return a tuple of Shapely Polygon and None.
        Otherwise, return a tuple of None and a string representing the error
        message.
    """
    if len(coords) < 3:
        return None, "Need at least 3 points to form a simple, closed shape."
    polygon = Polygon(coords)
    if not polygon.is_valid:
        return None, "Shape has to be simple and closed."
    return polygon, None


def manage_shape_tab() -> None:
    """Render the "Draw, Redraw, Round, or Smooth Shape" tab."""
    tab1, tab2, tab3, tab4 = st.tabs(["Draw", "Redraw", "Round", "Smooth"])
    with tab1:
        draw_shape_tab()
    with tab2:
        redraw_shape_tab()
    with tab3:
        round_shape_tab()
    with tab4:
        smooth_shape_tab()


def draw_shape_tab() -> None:
    """Render the "Draw" tab for creating new shapes."""
    with st.columns([3, 1, 3])[1]:
        fig = go.Figure()
        fig.update_layout(
            autosize=AUTOSIZE,
            dragmode=DRAG_MODE,
            clickmode=CLICK_MODE,
            xaxis=dict(range=[SHAPE_XLIM[0], SHAPE_XLIM[1]]),
            yaxis=dict(range=[SHAPE_YLIM[0], SHAPE_YLIM[1]]),
            width=DRAW_REDRAW_WIDTH,
            height=DRAW_REDRAW_HEIGHT,
        )
        st.plotly_chart(
            fig, use_container_width=False, key="draw", on_select="rerun"
        )
    event_data = st.session_state.get("draw", {})
    lasso_data = event_data.get("selection", {}).get("lasso", [])
    if not lasso_data:
        st.info("Draw a polygon.")
        st.button("Create", use_container_width=True, disabled=True)
        return
    coords = list(zip(lasso_data[0]["x"], lasso_data[0]["y"]))
    polygon, error = validate_polygon(coords)
    if error:
        st.error(error)
        st.button("Create", use_container_width=True, disabled=True)
        return
    if st.button("Create", use_container_width=True):
        name = get_next_shape_name()
        st.session_state["custom_shapes"][name] = polygon
        st.success(f"{name} is created.")
        st.rerun()


def redraw_shape_tab() -> None:
    """Render the "Redraw" tab for modifying existing shapes."""
    shapes = get_shape_list()
    if not shapes:
        st.info("No shapes to redraw.")
        return
    selected = get_selected_shape("redraw", shapes)
    current_idx = shapes.index(selected)
    shape_to_redraw = st.selectbox(
        "Select a shape to redraw:",
        shapes,
        index=current_idx,
        key="redraw_selectbox",
    )
    if shape_to_redraw != st.session_state.get("redraw_selection"):
        st.session_state["redraw_selection"] = shape_to_redraw
    with st.columns([3, 1, 3])[1]:
        shape = st.session_state["custom_shapes"][shape_to_redraw]
        x, y = shape.exterior.xy
        path_string = xy_to_svg_path(x, y)
        fig = go.Figure()
        fig.update_layout(
            autosize=AUTOSIZE,
            dragmode=DRAG_MODE,
            clickmode=CLICK_MODE,
            xaxis=dict(range=[SHAPE_XLIM[0], SHAPE_XLIM[1]]),
            yaxis=dict(range=[SHAPE_YLIM[0], SHAPE_YLIM[1]]),
            width=DRAW_REDRAW_WIDTH,
            height=DRAW_REDRAW_HEIGHT,
        )
        fig.add_selection(type="path", path=path_string)
        st.plotly_chart(
            fig, use_container_width=False, key="redraw", on_select="rerun"
        )
    event_data = st.session_state.get("redraw", {})
    lasso_data = event_data.get("selection", {}).get("lasso", [])
    if not lasso_data:
        st.info("Redraw a polygon.")
        return
    coords = list(zip(lasso_data[0]["x"], lasso_data[0]["y"]))
    polygon, error = validate_polygon(coords)
    if error:
        st.error(error)
        return
    if st.button("Create", use_container_width=True, key="redraw_create"):
        name = get_next_shape_name()
        st.session_state["custom_shapes"][name] = polygon
        st.success(f"{name} is created.")
        st.rerun()


def transform_shape_tab(
    tab_name: str,
    action_name: str,
    transform_function: Callable[[Polygon], Polygon],
    fig_title: str,
) -> None:
    """Render a tab for transforming shapes.

    Allow users to select an existing shape and transform it.

    Args:
        tab_name: A string representing the name of the tab.
        action_name: A string representing the transformation action.
        transform_function: A callable that takes a Shapely Polygon and
        returns a transformed Shapely Polygon.
        fig_title: A string representing the figure title for the
        transformed shape.
    """
    shapes = get_shape_list()
    if not shapes:
        st.info(f"No shapes to {action_name}.")
        return
    selected = get_selected_shape(tab_name, shapes)
    current_idx = shapes.index(selected)
    shape_to_transform = st.selectbox(
        f"Select a shape to {action_name}:",
        shapes,
        index=current_idx,
        key=f"{tab_name}_selectbox",
    )
    if shape_to_transform != st.session_state.get(f"{tab_name}_selection"):
        st.session_state[f"{tab_name}_selection"] = shape_to_transform
    shape = st.session_state["custom_shapes"][shape_to_transform]
    transformed = transform_function(shape)
    fig1 = plot_shape(shape, "Original Shape")
    png1 = figure_to_png(fig1)
    plt.close(fig1)
    fig2 = plot_shape(transformed, f"{fig_title} Shape")
    png2 = figure_to_png(fig2)
    plt.close(fig2)
    original_col, transformed_col = st.columns(2)
    with original_col:
        st.markdown(
            png_to_html_img(png1, ROUND_SMOOTH_DELETE_SHAPE_HEIGHT),
            unsafe_allow_html=True,
        )
    with transformed_col:
        st.markdown(
            png_to_html_img(png2, ROUND_SMOOTH_DELETE_SHAPE_HEIGHT),
            unsafe_allow_html=True,
        )
    if st.button("Create", use_container_width=True, key=f"{tab_name}_create"):
        name = get_next_shape_name()
        st.session_state["custom_shapes"][name] = transformed
        st.success(f"{name} is created.")
        st.rerun()


def round_shape_tab() -> None:
    """Render the "Round" tab for rounding shapes."""
    transform_shape_tab(
        tab_name="round",
        action_name="round",
        transform_function=round_polygon,
        fig_title="Rounded",
    )


def smooth_shape_tab() -> None:
    """Render the "Smooth" tab for smoothing shapes."""
    transform_shape_tab(
        tab_name="smooth",
        action_name="smooth",
        transform_function=smooth_polygon,
        fig_title="Smoothed",
    )


def view_shape_tab() -> None:
    """Render the "View Shape" tab for displaying all available shapes."""
    custom_shapes = st.session_state["custom_shapes"]
    all_shapes: list[str] = list(BUILTIN_SHAPES) + list(custom_shapes.keys())
    preview_cols = st.columns(5)
    for idx, shape_name in enumerate(all_shapes):
        preview_col = preview_cols[idx % len(preview_cols)]
        if shape_name in custom_shapes:
            shape = custom_shapes[shape_name]
        else:
            params = VIEW_SHAPE_PARAMS.get(shape_name)
            create_shape_function = shape_dispatch.get(shape_name)
            if params is None or create_shape_function is None:
                continue
            shape = create_shape_function(*params)
        with preview_col:
            fig = plot_shape(shape, shape_name)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


def generate_diagram_tab() -> None:
    """Render the "Generate Diagram" tab."""
    custom_shapes = st.session_state["custom_shapes"]
    all_shapes: list[str] = list(BUILTIN_SHAPES) + list(custom_shapes.keys())
    settings_col, results_col = st.columns([0.5, 2.0])
    with settings_col:
        st.markdown("Settings")
        zone_input = st.text_area(
            "Zone(s):", height=ZONE_INPUT_HEIGHT, value=ZONE_INPUT_EXAMPLE
        )
        zones = parse_zone_input(zone_input)
        if "ellipse" in all_shapes:
            default_shape = "ellipse"
        else:
            default_shape = all_shapes[0]
        selected_shape = st.selectbox(
            "Shape for All Sets:",
            all_shapes,
            index=all_shapes.index(default_shape),
        )
        algorithm = st.radio("Algorithm:", ALGORITHM_CHOICES, horizontal=True)
        n_gen = st.number_input(
            "Number of Generations:",
            min_value=1,
            value=DEFAULT_N_GEN,
            step=1,
        )
        n_process = st.number_input(
            "Number of Processes:",
            min_value=1,
            max_value=DEFAULT_N_PROCESS,
            value=DEFAULT_N_PROCESS,
            step=1,
        )
        st.session_state.setdefault("symmetric", False)
        st.session_state.setdefault("congruent", False)
        st.checkbox(
            "Make diagram symmetric",
            key="symmetric",
        )
        if st.session_state["symmetric"]:
            st.session_state["congruent"] = True
        st.checkbox(
            "Make shapes congruent",
            key="congruent",
            disabled=st.session_state["symmetric"],
        )
        if st.button("Start New Optimization", use_container_width=True):
            if not zones:
                st.error(
                    "Please specify at least one zone before "
                    "starting a new optimization."
                )
            else:
                result = optimize(
                    zones,
                    selected_shape,
                    algorithm,
                    n_gen=n_gen,
                    n_process=n_process,
                    symmetric=st.session_state["symmetric"],
                    congruent=st.session_state["congruent"],
                    custom_shapes=custom_shapes,
                )
                st.session_state["result"] = result
                st.session_state["selected_idx_sb"] = 0
                st.session_state["selected_idx"] = 0
    non_dominated_sols_col, diagram_col = results_col.columns([6, 4])
    with non_dominated_sols_col:
        st.markdown("Visualization of Non-Dominated Solutions")
        if "result" in st.session_state:
            ids = list(range(len(st.session_state["result"].res.X)))
            selected_idx = st.selectbox(
                "Solution Index:",
                ids,
                index=st.session_state.get("selected_idx_sb", 0),
                key="selected_idx_sb",
            )
            st.session_state["selected_idx"] = selected_idx
            st.session_state["pcp"] = pcp(st.session_state["result"])
            st.plotly_chart(st.session_state["pcp"], use_container_width=True)
        else:
            st.info(
                "Visualization of Non-Dominated Solutions will appear after optimization."
            )
    with diagram_col:
        st.markdown("Diagram")
        if "result" in st.session_state:
            (
                show_zone_labels_col,
                shade_unwanted_zones_col,
                export_png_col,
                export_svg_col,
            ) = st.columns([5, 5, 4, 4], gap="small")
            with show_zone_labels_col:
                show_zone_labels = st.checkbox(
                    "Show zone labels", key="show_zone_labels_cb", value=False
                )
            with shade_unwanted_zones_col:
                shade_unwanted_zones = st.checkbox(
                    "Shade unwanted zones",
                    key="shade_unwanted_zones_cb",
                    value=True,
                )
            solution = st.session_state["result"][
                st.session_state["selected_idx"]
            ]
            st.session_state["zone_info"] = zone_info(solution)
            fig = diagram(
                solution,
                show_zone_labels=show_zone_labels,
                shade_unwanted_zones=shade_unwanted_zones,
            )
            st.session_state["png"] = figure_to_png(fig)
            st.session_state["svg"] = figure_to_svg(fig)
            png = st.session_state["png"]
            st.markdown(
                png_to_html_img(png, DIAGRAM_HEIGHT), unsafe_allow_html=True
            )
            with export_png_col:
                st.download_button(
                    "Export PNG",
                    data=st.session_state["png"],
                    file_name=PNG_DIAGRAM_NAME,
                )
            with export_svg_col:
                st.download_button(
                    "Export SVG",
                    data=st.session_state["svg"],
                    file_name=SVG_DIAGRAM_NAME,
                )
        else:
            st.info("Diagram will appear after optimization.")
    zone_info_container = results_col.container()
    with zone_info_container:
        st.markdown("Zone Information")
        if "zone_info" in st.session_state:
            st.dataframe(
                st.session_state["zone_info"],
                hide_index=True,
                use_container_width=True,
                height=ZONE_TABLE_HEIGHT,
            )
        else:
            st.info("Zone Information will appear after optimization.")
