"""Search for symmetric Venn-2 diagrams with congruent ellipses."""

import matplotlib.pyplot as plt

from evoeuler import diagram, obj_values, optimize, pcp, shape_params, zone_info

if __name__ == "__main__":
    # Start an optimization.
    result = optimize(
        zones={"A", "B", "A B"},
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
