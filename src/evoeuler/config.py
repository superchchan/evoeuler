"""Constants."""

from __future__ import annotations

import multiprocessing

import matplotlib.pyplot as plt

from evoeuler.enums import Algorithm

# Precision grid size
GRID_SIZE = 1e-3

# Number of parameters for each built-in shape
BUILTIN_SHAPE_PARAM_COUNTS = {
    "circle": 3,
    "ellipse": 5,
    "rectangle": 5,
    "rounded_rectangle": 5,
    "equilateral_triangle": 4,
    "rounded_equilateral_triangle": 4,
    "square": 4,
    "rounded_square": 4,
    "pentagon": 4,
    "rounded_pentagon": 4,
    "hexagon": 4,
    "rounded_hexagon": 4,
    "heptagon": 4,
    "rounded_heptagon": 4,
    "octagon": 4,
    "rounded_octagon": 4,
    "nonagon": 4,
    "rounded_nonagon": 4,
    "decagon": 4,
    "rounded_decagon": 4,
}

# Available built-in shapes
BUILTIN_SHAPES = list(BUILTIN_SHAPE_PARAM_COUNTS.keys())

# Available regular polygons
REGULAR_POLYGONS = {
    "equilateral_triangle",
    "rounded_equilateral_triangle",
    "square",
    "rounded_square",
    "pentagon",
    "rounded_pentagon",
    "hexagon",
    "rounded_hexagon",
    "heptagon",
    "rounded_heptagon",
    "octagon",
    "rounded_octagon",
    "nonagon",
    "rounded_nonagon",
    "decagon",
    "rounded_decagon",
}

# Objective names
OBJECTIVE_NAMES = [
    "Missing Zone",
    "Unwanted Zone",
    "Triple Point",
    "Concurrency",
    "Disconnected Zone",
]

# General algorithm hyperparameters
DEFAULT_N_GEN = 100
DEFAULT_N_PROCESS = multiprocessing.cpu_count()
POP_SIZE = 100
N_OBJ = 5

# NSGA-III hyperparameters
NSGA_III_N_PARTITIONS = 4
NSGA_III_REF_DIR_GEN_ALG = "das-dennis"

# Shape parameter bounds for optimization
ANGLE_LB = 0.0
ANGLE_UB = 360.0
CR_LB = 0.1
CR_UB = 2.0
CX_LB = -2.0
CX_UB = 2.0
CY_LB = -2.0
CY_UB = 2.0
H_LB = 0.1
H_UB = 2.0
R_LB = 0.1
R_UB = 2.0
RX_LB = 0.1
RX_UB = 2.0
RY_LB = 0.1
RY_UB = 2.0
SX_LB = 0.1
SX_UB = 2.0
SY_LB = 0.1
SY_UB = 2.0
W_LB = 0.1
W_UB = 2.0

# Shape visualization settings
SHAPE_FIGSIZE = (2, 2)
SHAPE_XLIM = (-2, 2)
SHAPE_YLIM = (-2, 2)
TITLE_FONTSIZE = 8

# Resolution for creating circles and ellipses
RESOLUTION = 128

# General figure settings
DPI = 300

# Draw and redraw shape settings
AUTOSIZE = False
CLICK_MODE = "event+select"
DRAG_MODE = "lasso"
DRAW_REDRAW_HEIGHT = 300
DRAW_REDRAW_WIDTH = 150

# Round, smooth, and delete shape settings
ROUND_SMOOTH_DELETE_SHAPE_HEIGHT = 250

# Round shape settings
BUFFER_DISTANCE = 0.1

# Smooth shape settings
CHAIKIN_DEFAULT_ITERS = 5

# View shape settings
VIEW_SHAPE_PARAMS = {
    "circle": (0.0, 0.0, 1.0),
    "ellipse": (0.0, 0.0, 1.0, 0.6, 0.0),
    "rectangle": (0.0, 0.0, 1.5, 1.0, 0.0),
    "rounded_rectangle": (0.0, 0.0, 1.5, 1.0, 0.0),
    "equilateral_triangle": (0.0, 0.0, 1.0, 90.0),
    "rounded_equilateral_triangle": (0.0, 0.0, 1.0, 90.0),
    "square": (0.0, 0.0, 1.0, 45.0),
    "rounded_square": (0.0, 0.0, 1.0, 45.0),
    "pentagon": (0.0, 0.0, 1.0, 18.0),
    "rounded_pentagon": (0.0, 0.0, 1.0, 18.0),
    "hexagon": (0.0, 0.0, 1.0, 0.0),
    "rounded_hexagon": (0.0, 0.0, 1.0, 0.0),
    "heptagon": (0.0, 0.0, 1.0, 38.0),
    "rounded_heptagon": (0.0, 0.0, 1.0, 38.0),
    "octagon": (0.0, 0.0, 1.0, 0.0),
    "rounded_octagon": (0.0, 0.0, 1.0, 0.0),
    "nonagon": (0.0, 0.0, 1.0, 10.0),
    "rounded_nonagon": (0.0, 0.0, 1.0, 10.0),
    "decagon": (0.0, 0.0, 1.0, 0.0),
    "rounded_decagon": (0.0, 0.0, 1.0, 0.0),
}

# Zone specification settings
ZONE_INPUT_EXAMPLE = """A
B
C
A B
A C
B C
A B C
"""
ZONE_INPUT_HEIGHT = 70

# Available algorithms
ALGORITHM_CHOICES = [algorithm.value for algorithm in Algorithm]

# PCP settings
PCP_BOTTOM_MARGIN = 20
PCP_FONT_COLOR = "black"
PCP_FONT_SIZE = 12
PCP_HEIGHT = 250
PCP_LEFT_MARGIN = 45
PCP_RIGHT_MARGIN = 45
PCP_TEMPLATE = "plotly_white"

# Diagram visualization settings
DIAGRAM_FIGSIZE = (6, 6)
FILL_COLOR = "white"
LEGEND_BBOX_TO_ANCHOR = (1.0, 0.5)
LEGEND_LOC = "center left"
PNG_DIAGRAM_NAME = "diagram.png"
SVG_DIAGRAM_NAME = "diagram.svg"
SHADE_COLOR = "gray"
DIAGRAM_HEIGHT = 250
DIAGRAM_LINEWIDTH = 1
ZONE_LABEL_FONTSIZE = 10

# Curve colors for diagram visualization
tab10 = plt.get_cmap("tab10")
TAB10 = [tab10(i) for i in range(10)]
tab20 = plt.get_cmap("tab20")
TAB20_DARK = [tab20(2 * i) for i in range(10)]
TAB20_LIGHT = [tab20(2 * i + 1) for i in range(10)]
TAB20 = TAB20_DARK + TAB20_LIGHT

# Zone information settings
ZONE_TABLE_HEIGHT = 180
