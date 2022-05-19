"""
Perfect squares problem in cpmpy.

CSPLib prob 009: Perfect square placements
http://www.cs.st-andrews.ac.uk/~ianm/CSPLib/prob/prob009/index.html
'''
The perfect square placement problem (also called the squared square
problem) is to pack a set of squares with given integer sizes into a
bigger square in such a way that no squares overlap each other and all
square borders are parallel to the border of the big square. For a
perfect placement problem, all squares have different sizes. The sum of
the square surfaces is equal to the surface of the packing square, so
that there is no spare capacity. A simple perfect square placement
problem is a perfect square placement problem in which no subset of
the squares (greater than one) are placed in a rectangle.
'''

Model created by Ignace Bleukx, inspired by implementation of Vessel Packing problem (008)
"""
import sys
import numpy as np
from cpmpy import *
from cpmpy.expressions.utils import all_pairs


def perfect_squares(base, sides, num_sols=0):
    model = Model()

    squares = range(len(sides))

    # Ensure that the squares cover the base exactly
    assert np.square(sides).sum() == base ** 2, "Squares do not cover the base exactly!"

    # variables
    x_coords = intvar(0, base, shape=len(squares), name="x_coords")
    y_coords = intvar(0, base, shape=len(squares), name="y_coords")

    # squares must be in bounds of big square
    model += x_coords + sides <= base
    model += y_coords + sides <= base

    # no overlap between coordinates
    for a, b in all_pairs(squares):
        model += (
            (x_coords[a] + sides[a] <= x_coords[b]) |
            (x_coords[b] + sides[b] <= x_coords[a]) |
            (y_coords[a] + sides[a] <= y_coords[b]) |
            (y_coords[b] + sides[b] <= y_coords[a])
        )

    return model, (x_coords, y_coords)


def get_data(name):
    if name == "problem1":
        base, sides =  4, [2,2,2,2]
    elif name == "problem2":
        base, sides = 6, [3, 3, 3, 2, 1, 1, 1, 1, 1]
    elif name == "problem3":
        # tricky problem: should give no solution
        # (we can not fit 2 3x3 squares in a 5x5 square)
        base, sides = 5, [3, 3, 2, 1, 1, 1]
    elif name == "problem4":
        # Problem from Sam Loyd
        # http://squaring.net/history_theory/sam_loyd.html
        base, sides = 13, [1, 1, 2, 2, 2, 3, 3, 4, 6, 6, 7]
    elif name == "problem5":
        # Problem from
        # http://www.maa.org/editorial/mathgames/mathgames_12_01_03.html
        base, sides = 14,  [1, 1, 1, 1, 2, 3, 3, 3, 5, 6, 6, 8]
    elif name == "problem6":
        # Problem from
        # http://www.maa.org/editorial/mathgames/mathgames_12_01_03.html
        base, sides = 30,[1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 7, 8, 8, 9, 9, 10, 10, 11, 13]
    elif name == "problem7":
        base, sides = 112, [2, 4, 6, 7, 8, 9, 11, 15, 16, 17, 18, 19, 24, 25, 27, 29, 33, 35, 37, 42, 50]
    elif name == "problem8":
        base, sides = 110, [2, 3, 4, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 26, 27, 28, 50, 60]

    return base, np.array(sides)

def print_sol(base, sides, x_coords,y_coords, alpha=False):
    x_coords, y_coords = x_coords.value(), y_coords.value()
    big_square = np.zeros(dtype=int, shape=(base, base))
    for i, (x,y) in enumerate(zip(x_coords, y_coords)):
        big_square[x:x+sides[i],y:y+sides[i]] = i
    if alpha:
        big_square = np.array([
            chr(val) for row in big_square+65 for val in row
        ], dtype=str).reshape(big_square.shape)
    print(big_square)

if __name__ == "__main__":
    num_sols = 1
    problem_number = 4

    if len(sys.argv) > 1:
        problem_number = int(sys.argv[1])

    data = get_data(f"problem{problem_number}")
    model, vars = perfect_squares(*data, num_sols=num_sols)

    n = model.solveAll(
        solution_limit=num_sols,
        display=lambda : print_sol(*data,*vars, alpha=True)
    )

    print(f"Found {n} solutions")

