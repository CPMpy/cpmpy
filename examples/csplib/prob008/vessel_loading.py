import sys

from cpmpy import *
import numpy as np
from cpmpy.expressions.utils import all_pairs

def get_data(name):

    if name == "easy":
        deck_width = 5
        deck_length = 5
        n_containers = 3
        n_classes = 2

        width = [5,2,3]
        length=[1,4,4]

        classes = [1,1,1]
        separation = [[0,0],[0,0]]

    elif name == "hard":
        deck_width = 16
        deck_length = 16
        n_containers = 10
        n_classes = 3

        width = [6, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        length = [8, 6, 4, 4, 4, 6, 8, 8, 6, 6]

        classes = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

        separation = [[0, 0, 0],
                      [0, 0, 2],
                      [0, 2, 0]]

    return deck_width, deck_length, n_containers, n_classes, np.array(width), np.array(length), np.array(classes), np.array(separation)

def vessel_loading(deck_width, deck_length, n_containers, n_classes, width, length, classes, separation):

    print(f"{deck_width=}")
    print(f"{deck_length=}")
    print(f"{n_containers=}")
    print(f"{n_classes=}")
    print(f"{width=}")
    print(f"{length=}")
    print(f"{classes=}")
    print(f"{separation=}")

    containers = list(range(n_containers))

    model = Model()

    # layout of containers
    left = intvar(0, deck_width, shape=n_containers, name="left")
    right = intvar(0, deck_width, shape=n_containers, name="right")
    top = intvar(0, deck_length, shape=n_containers, name="top")
    bottom = intvar(0, deck_length, shape=n_containers, name="bottom")

    all_points = np.stack([left, right, top, bottom], axis=0)

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


    # solve the model
    if model.solve(solver="ortools"):
        container_map = np.zeros(shape=(deck_length, deck_width),dtype=int)
        l, r, t, b = left.value(), right.value(), top.value(), bottom.value()
        for c in containers:
            container_map[b[c]:t[c],l[c]:r[c]] = c+1

        print("Shipdeck layout (0 means no container in that spot):")
        print(np.flip(container_map, axis=0))

    else:
        print("Model is unsatisfiable")



if __name__ == "__main__":
    name = "hard"

    if len(sys.argv) > 1:
        name = sys.argv[1]

    data = get_data(name)

    vessel_loading(*data)
