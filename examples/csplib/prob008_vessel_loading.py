"""
    Vessel loading in CPMpy
    Problem 008 in CSPlib

    Model inspired by Essence implementation on CSPlib

    Created by Ignace Bleukx

"""

import sys
import requests
import json

import numpy as np

from cpmpy import *
from cpmpy.expressions.utils import all_pairs

def vessel_loading(deck_width, deck_length, n_containers, width, length, classes, separation, **kwargs):

    # setup data
    containers = list(range(n_containers))
    classes = np.array(classes)
    separation = np.array(separation)

    model = Model()

    # layout of containers
    left = intvar(0, deck_width, shape=n_containers, name="left")
    right = intvar(0, deck_width, shape=n_containers, name="right")
    top = intvar(0, deck_length, shape=n_containers, name="top")
    bottom = intvar(0, deck_length, shape=n_containers, name="bottom")

    # set shape of containers
    model += (
            (((right - left) == width) & ((top - bottom) == length)) # along shipdeck
                                        |
            (((right - left) == length) & ((top - bottom) == width)) # accross shipdeck
    )


    # no overlap between containers
    for x,y in all_pairs(containers):
        c1,c2 = classes[[x,y]]
        sep = separation[c1-1,c2-1]
        model += (
                (right[x] + sep <= left[y]) | # x at least sep left of y or
                (left[x] >= right[y] + sep) | # x at least sep right of y or
                (top[x] + sep <= bottom[y]) | # x at least sep under y or
                (bottom[x] >= top[y] + sep)   # x at least sep above y
        )

    return model, (left, right, top, bottom)

def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name {pname} not found, use --list-instances to get the full list.")


def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "deck_width", "deck_length", "n_containers", "n_classes"], na_rep='-')
    print(df_str)


if __name__ == "__main__":
    import argparse
    import json
    import requests

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob008_vessel_loading.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', default="easy", help="Name of the problem instance found in file 'filename'")
    parser.add_argument('-filename', default=url, help="File containing problem instances, can be local file or url")
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

    model, (left, right, top, bottom) = vessel_loading(**problem_params)

    # solve the model
    if model.solve():
        container_map = np.zeros(shape=(problem_params["deck_length"], problem_params["deck_width"]), dtype=int)
        l, r, t, b = left.value(), right.value(), top.value(), bottom.value()
        for c in range(problem_params["n_containers"]):
            container_map[b[c]:t[c], l[c]:r[c]] = c + 1

        print("Shipdeck layout (0 means no container in that spot):")
        print(np.flip(container_map, axis=0))

    else:
        raise ValueError("Model is unsatisfiable")