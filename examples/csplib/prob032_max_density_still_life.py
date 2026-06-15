"""
Maximum density still life problem in cpmpy.

Problem 032 on CSPlib
https://www.csplib.org/Problems/prob032/

This problem, arising from Conway's Game of Life, seeks to find the most densely
populated stable pattern (a "still life") on an n x n grid. A still life is a
pattern of live and dead cells that does not change from one generation to the next.
The grid is assumed to be surrounded by an infinite expanse of dead cells.

The rules for a pattern to be a still life are:
1. A live cell must have exactly 2 or 3 live neighbors to remain alive.
2. A dead cell must not have exactly 3 live neighbors (otherwise it would become alive).

The goal is to maximize the number of live cells.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_032_max_density_still_life/csplib_032_max_density_still_life.cpmpy.py)
"""

from cpmpy import *


def max_density_still_life(n=6, m=6):
    grid = boolvar(shape=(n, m), name="grid")

    model = Model()

    # Valid still-life combinations of (neighbor_count, cell_state)
    still_life_table = []
    # Rule for dead cells: neighbor count can be anything except 3.
    for i in range(9):
        if i != 3:
            still_life_table.append((i, 0))
    # Rule for live cells: neighbor count must be 2 or 3.
    still_life_table.append((2, 1))
    still_life_table.append((3, 1))

    for i in range(n):
        for j in range(m):
            # Collect all valid neighbor cells. Off-board neighbors are implicitly dead (0)
            # and do not need to be added to the sum.
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = i + dx, j + dy
                    if 0 <= nx < n and 0 <= ny < m:
                        neighbors.append(grid[nx, ny])
            num_neighbors = sum(neighbors)

            # Enforce that the combination of the neighbor count and the cell's own state
            # is a valid still-life configuration.
            model += Table([num_neighbors, grid[i, j]], still_life_table)

    # Symmetry breaking for square boards
    if n == m:
        model += grid[0, 0] >= grid[n - 1, m - 1]
        model += grid[0, m - 1] >= grid[n - 1, 0]

    model.maximize(sum(grid))

    return model, (grid,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=6, help="Number of rows")
    parser.add_argument("-m", type=int, default=6, help="Number of columns")

    args = parser.parse_args()

    model, (grid,) = max_density_still_life(args.n, args.m)

    if model.solve():
        print(f"Maximum live cells: {int(model.objective_value())}")
        for row in grid.value():
            print("".join(['#' if cell else '.' for cell in row]))
    else:
        raise ValueError("Model is unsatisfiable")
