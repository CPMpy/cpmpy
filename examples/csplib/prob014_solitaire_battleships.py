"""
Solitaire Battleships puzzle in cpmpy.

Problem 014 on CSPlib
https://www.csplib.org/Problems/prob014/

The Battleships puzzle is played on a grid. The fleet consists of battleships
(four grid squares in length), cruisers (three grid squares long), destroyers
(two squares long) and submarines (one square each). The ships may be oriented
horizontally or vertically, and no two ships will occupy adjacent grid squares,
not even diagonally. The digits along the right side of and below the grid
indicate the number of grid squares in the corresponding rows and columns that
are occupied by vessels.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_014_solitaire_battleships/csplib_014_solitaire_battleships.cpmpy.py)
"""

from cpmpy import *

WATER = 0
CIRCLE = 2  # submarine
LEFT = 3
RIGHT = 4
TOP = 5
BOTTOM = 6
MIDDLE = 7


def solitaire_battleships(rows=10, cols=10, rowsum=None, colsum=None,
                          fleet_counts=None, hints=None):
    if rowsum is None:
        rowsum = [0, 2, 3, 1, 2, 4, 2, 1, 2, 3]
    if colsum is None:
        colsum = [1, 3, 3, 1, 5, 1, 2, 4, 0, 0]
    if fleet_counts is None:
        fleet_counts = [(4, 1), (3, 2), (2, 3), (1, 4)]
    if hints is None:
        hints = [(7, 1, CIRCLE)]

    fleet_counts = dict(fleet_counts)

    grid = intvar(0, 7, shape=(rows, cols), name="grid")

    model = Model()

    # Hints constraint
    for r, c, v in hints:
        model += grid[r, c] == v

    # Row and column sums
    for i in range(rows):
        model += sum(grid[i, :] > WATER) == rowsum[i]
    for j in range(cols):
        model += sum(grid[:, j] > WATER) == colsum[j]

    # Adjacency and connectivity
    for r in range(rows):
        for c in range(cols):
            # No diagonal or corner-touching ships
            diag_is_water = []
            if r > 0 and c > 0:
                diag_is_water.append(grid[r - 1, c - 1] == WATER)
            if r > 0 and c < cols - 1:
                diag_is_water.append(grid[r - 1, c + 1] == WATER)
            if r < rows - 1 and c > 0:
                diag_is_water.append(grid[r + 1, c - 1] == WATER)
            if r < rows - 1 and c < cols - 1:
                diag_is_water.append(grid[r + 1, c + 1] == WATER)
            model += (grid[r, c] > WATER).implies(all(diag_is_water))

            # Orthogonal water neighbors
            ortho_is_water = []
            if r > 0:
                ortho_is_water.append(grid[r - 1, c] == WATER)
            if r < rows - 1:
                ortho_is_water.append(grid[r + 1, c] == WATER)
            if c > 0:
                ortho_is_water.append(grid[r, c - 1] == WATER)
            if c < cols - 1:
                ortho_is_water.append(grid[r, c + 1] == WATER)

            # A CIRCLE must be entirely surrounded by water
            model += (grid[r, c] == CIRCLE).implies(all(ortho_is_water))

            # LEFT piece
            model += (grid[r, c] == LEFT).implies(
                ((grid[r, c + 1] == MIDDLE) | (grid[r, c + 1] == RIGHT) if c < cols - 1 else False) &
                (grid[r, c - 1] == WATER if c > 0 else True) &
                (grid[r - 1, c] == WATER if r > 0 else True) &
                (grid[r + 1, c] == WATER if r < rows - 1 else True)
            )

            # RIGHT piece
            model += (grid[r, c] == RIGHT).implies(
                ((grid[r, c - 1] == MIDDLE) | (grid[r, c - 1] == LEFT) if c > 0 else False) &
                (grid[r, c + 1] == WATER if c < cols - 1 else True) &
                (grid[r - 1, c] == WATER if r > 0 else True) &
                (grid[r + 1, c] == WATER if r < rows - 1 else True)
            )

            # TOP piece
            model += (grid[r, c] == TOP).implies(
                ((grid[r + 1, c] == MIDDLE) | (grid[r + 1, c] == BOTTOM) if r < rows - 1 else False) &
                (grid[r - 1, c] == WATER if r > 0 else True) &
                (grid[r, c - 1] == WATER if c > 0 else True) &
                (grid[r, c + 1] == WATER if c < cols - 1 else True)
            )

            # BOTTOM piece
            model += (grid[r, c] == BOTTOM).implies(
                ((grid[r - 1, c] == MIDDLE) | (grid[r - 1, c] == TOP) if r > 0 else False) &
                (grid[r + 1, c] == WATER if r < rows - 1 else True) &
                (grid[r, c - 1] == WATER if c > 0 else True) &
                (grid[r, c + 1] == WATER if c < cols - 1 else True)
            )

            # MIDDLE piece must be either horizontal or vertical
            is_hor_middle = (((grid[r, c - 1] == LEFT) | (grid[r, c - 1] == MIDDLE)) &
                             ((grid[r, c + 1] == RIGHT) | (grid[r, c + 1] == MIDDLE)) &
                             (grid[r - 1, c] == WATER if r > 0 else True) &
                             (grid[r + 1, c] == WATER if r < rows - 1 else True)) if 0 < c < cols - 1 else False
            is_ver_middle = (((grid[r - 1, c] == TOP) | (grid[r - 1, c] == MIDDLE)) &
                             ((grid[r + 1, c] == BOTTOM) | (grid[r + 1, c] == MIDDLE)) &
                             (grid[r, c - 1] == WATER if c > 0 else True) &
                             (grid[r, c + 1] == WATER if c < cols - 1 else True)) if 0 < r < rows - 1 else False
            model += (grid[r, c] == MIDDLE).implies(is_hor_middle | is_ver_middle)

    # Fleet composition
    model += sum(grid == CIRCLE) == fleet_counts[1]

    num_horizontal_ships = sum(grid == LEFT)
    num_vertical_ships = sum(grid == TOP)
    model += num_horizontal_ships == sum(grid == RIGHT)
    model += num_vertical_ships == sum(grid == BOTTOM)

    total_long_ships = sum(count for size, count in fleet_counts.items() if size > 1)
    model += num_horizontal_ships + num_vertical_ships == total_long_ships

    expected_middles = sum((size - 2) * count for size, count in fleet_counts.items() if size > 2)
    model += sum(grid == MIDDLE) == expected_middles

    return model, (grid,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-rows", type=int, default=10, help="Number of rows")
    parser.add_argument("-cols", type=int, default=10, help="Number of columns")

    args = parser.parse_args()
    model, (grid,) = solitaire_battleships(rows=args.rows, cols=args.cols)

    if model.solve():
        # print pretty board with symbols
        def pretty_print(grid):
            print("\n--- Solution Board ---")
            symbols = {
                WATER: '~', CIRCLE: 'O', LEFT: '<',
                RIGHT: '>', TOP: '^', BOTTOM: 'v', MIDDLE: '#'
            }
            for r in range(args.rows):
                print(" ".join(symbols[grid[r, c].value()] for c in range(args.cols)))
        pretty_print(grid)
        # get all solutions by restricting the grid to be different from previous solutions
        for _s in range(10):  # limit to 10 solutions
            model += sum(grid != grid.value()) > 0  # at least one cell must differ
            if model.solve():
                pretty_print(grid)
            else:
                print("No more solutions.")
                break
    else:
        raise ValueError("Model is unsatisfiable")
