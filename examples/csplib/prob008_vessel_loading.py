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

def get_data(data, pname):

    for entry in data:
        if pname in entry["name"]:
            return entry

if __name__ == "__main__":

    fname = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob008_vessel_loading.json"
    problem_name = "easy"

    data = None

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        with open(fname, "r") as f:
            data = json.load(f)

    if len(sys.argv) > 2:
        problem_name = sys.argv[2]

    if data is None:
        data = requests.get(fname).json()

    params = get_data(data, problem_name)

    data = get_data(data, problem_name)
    model, (left, right, top, bottom) = vessel_loading(**params)

    # solve the model
    if model.solve():
        container_map = np.zeros(shape=(data["deck_length"], data["deck_width"]), dtype=int)
        l, r, t, b = left.value(), right.value(), top.value(), bottom.value()
        for c in range(data["n_containers"]):
            container_map[b[c]:t[c], l[c]:r[c]] = c + 1

        print("Shipdeck layout (0 means no container in that spot):")
        print(np.flip(container_map, axis=0))

    else:
        print("Model is unsatisfiable")