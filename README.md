# EvoEuler: Evolving Euler and Venn Diagrams with Specific Properties
EvoEuler searches for Euler and Venn diagrams with a simple closed shape using a multi-objective evolutionary algorithm. Both a web interface and an application programming interface (API) are available.

EvoEuler allows users to:
- specify optional constraints (symmetric diagram and congruent shape) before optimization.
- obtain a set of non-dominated solutions (diagrams), each evaluated with five objectives (missing zone, unwanted zone, triple point, concurrency, and disconnected zone) that quantify the violation of well-matchedness or well-formedness properties.
- explore trade-offs among non-dominated solutions via parallel coordinates plots that visualize the objective values of all solutions.

## Installation
Run the following command to install EvoEuler:
```bash
pip install evoeuler
```

## Web Interface
Run the following command to launch the web interface:
```bash
evoeuler
```

![EvoEuler Web Interface](https://raw.githubusercontent.com/superchchan/evoeuler/main/EvoEuler.png)
Compared to the API, the web interface additionally allows users to draw, redraw, round, smooth, or view custom simple closed shapes, which can then be used for diagram generation.

## API
The example below shows the use of all functions provided by the API.

The supported algorithms for the API are: `NSGA-II` and `NSGA-III`.

The supported shapes for the API are: `circle`, `ellipse`, `rectangle`, `rounded_rectangle`, `equilateral_triangle`, `rounded_equilateral_triangle`, `square`, `rounded_square`, `pentagon`, `rounded_pentagon`, `hexagon`, `rounded_hexagon`, `heptagon`, `rounded_heptagon`, `octagon`, `rounded_octagon`, `nonagon`, `rounded_nonagon`, `decagon`, and `rounded_decagon`.

Note that if `symmetric` is set to `True`, `congruent` will be automatically set to `True`.
```python
"""Search for symmetric Venn-3 diagrams with congruent ellipses."""

import matplotlib.pyplot as plt

from evoeuler import diagram, obj_values, optimize, pcp, shape_params, zone_info

if __name__ == "__main__":
    # Start an optimization.
    result = optimize(
        zones={"A", "B", "C", "A B", "A C", "B C", "A B C"},
        shape="ellipse",
        algorithm="NSGA-II",
        n_gen=100,
        n_process=1,
        time=None,
        symmetric=True,
        congruent=True,
    )

    # Print the number of non-dominated solutions.
    print(f"Number of non-dominated solutions: {len(result)}")

    # Show the parallel coordinates plot that visualizes the objective values
    # of all non-dominated solutions.
    fig = pcp(result)
    fig.show()

    solution_idx = 0  # Solution index

    # Show the diagram of a non-dominated solution.
    diagram(
        result[solution_idx], show_zone_labels=False, shade_unwanted_zones=True
    )
    plt.show()

    # Print the objective values of a non-dominated solution.
    print("Objective values:")
    print(obj_values(result[solution_idx]))

    # Print the zone information of a non-dominated solution.
    print("Zone info:")
    print(zone_info(result[solution_idx]))

    # Print the shape parameters of a non-dominated solution.
    print("Shape params:")
    print(shape_params(result[solution_idx]))

```

## Examples
Some example scripts for searching Euler or Venn diagrams are in `./examples`.

## Contact
If you have any questions, please contact Chi Ho Chan at [c.chan3@napier.ac.uk](mailto:c.chan3@napier.ac.uk).