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
Inspired by implementation of Vessel Packing problem (008)
Model created by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import sys
import numpy as np
from cpmpy import *
from cpmpy.expressions.utils import all_pairs


def perfect_squares(base, sides, **kwargs):
    model = Model()
    sides = np.array(sides)

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

def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name '{pname}' not found, use --list-instances to get the full list.")


def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "base", "sides", "note"], na_rep='-')
    print(df_str)


if __name__ == "__main__":

    import argparse
    import json
    import requests

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob009_perfect_squares.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', nargs='?', default="problem7", help="Name of the problem instance found in file 'filename'")
    parser.add_argument('-filename', nargs='?', default=url, help="File containing problem instances, can be local file or url")
    parser.add_argument('--list-instances', help='List all problem instances', action='store_true')

    args = parser.parse_args()

    if "http" in args.filename:
        problem_data = requests.get(args.filename).json()
    else:
        with open(args.filename, "r") as f:
            problem_data = json.load(f)

    if args.list_instances:
        _print_instances(problem_data)
        exit(0)

    problem_params = _get_instance(problem_data, args.instance)
    print("Problem name:", problem_params["name"])

    model, (x_coords, y_coords) = perfect_squares(**problem_params)

    if model.solve():
        np.set_printoptions(linewidth=problem_params['base']*5, threshold=np.inf)

        base, sides = problem_params['base'], problem_params['sides']
        x_coords, y_coords = x_coords.value(), y_coords.value()
        big_square = np.zeros(dtype=str, shape=(base, base))
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            big_square[x:x + sides[i], y:y + sides[i]] = chr(i+65)

        print(np.array2string(big_square,formatter={'str_kind': lambda v: v}))

    else:
        raise ValueError(f"Problem is unsatisfiable")

