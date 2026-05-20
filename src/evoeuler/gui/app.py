"""EvoEuler Streamlit application."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import streamlit as st

from evoeuler.gui.tabs import (
    generate_diagram_tab,
    manage_shape_tab,
    view_shape_tab,
)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="EvoEuler", layout="wide")
    st.title(
        "EvoEuler: Evolving Euler and Venn Diagrams with Specific Properties"
    )
    st.session_state.setdefault("custom_shapes", {})
    tab1, tab2, tab3 = st.tabs(
        [
            "Draw, Redraw, Round, or Smooth Shape",
            "View Shape",
            "Generate Diagram",
        ]
    )
    with tab1:
        manage_shape_tab()
    with tab2:
        view_shape_tab()
    with tab3:
        generate_diagram_tab()


if __name__ == "__main__":
    main()
