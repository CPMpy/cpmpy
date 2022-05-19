"""
    Vessel loading in CPMpy
    Problem 008 in CSPlib

    Model inspired by Essence implementation on CSPlib

    Created by Ignace Bleukx

"""

import sys

from cpmpy import *
import numpy as np
from cpmpy.expressions.utils import all_pairs

def vessel_loading(deck_width, deck_length, n_containers, n_classes, width, length, classes, separation):

    containers = list(range(n_containers))

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

def get_data(name):

    if name == "easy":
        return {
            "deck_width": 5,
            "deck_length" : 5,
            "n_containers" : 3,
            "n_classes" : 2,

            "width" : np.array([5,2,3]),
            "length" :np.array([1,4,4]),

            "classes" : np.array([1,1,1]),
            "separation" : np.array([[0,0],[0,0]])
        }

    if name == "hard":
        return {
            "deck_width" : 16,
            "deck_length" : 16,
            "n_containers" : 10,
            "n_classes" : 3,

            "width" : np.array([6, 4, 4, 4, 4, 4, 4, 4, 4, 4]),
            "length" : np.array([8, 6, 4, 4, 4, 6, 8, 8, 6, 6]),

            "classes" : np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
            "separation" : np.array([[0, 0, 0],
                                     [0, 0, 2],
                                     [0, 2, 0]])
        }

if __name__ == "__main__":
    name = "hard"
    if len(sys.argv) > 1:
        name = sys.argv[1]

    data = get_data(name)
    model, (left, right, top, bottom) = vessel_loading(**data)

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