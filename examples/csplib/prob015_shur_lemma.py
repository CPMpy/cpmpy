"""
Problem 015 on CSPLib

Given n balls, labelled 1 to n, put them into c boxes such that for any triple of balls (x1,x2,...,xc) with sum(x1,x2,...,xc-1) = xc, not all are in the same box.


Adapted from Numberjack implementation https://github.com/csplib/csplib/blob/master/Problems/prob015/models/SchursLemma.py

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import numpy as np

from cpmpy import *

def shur_lemma(n, c):
    # balls[i] = j iff ball i is in box j
    balls = intvar(1, c, shape=n, name="balls")

    model = Model()

    for i in range(1, n + 1):
        for j in range(1, n - i + 1):
            model += (balls[i - 1] != balls[j - 1]) | \
                     (balls[i - 1] != balls[i + j - 1]) | \
                     (balls[j - 1] != balls[i + j - 1])

    return model, (balls,)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-balls", type=int, default=12, help="Number of balls")
    parser.add_argument("-boxes", type=int, default=3, help="Number of boxes")

    args = parser.parse_args()

    model, (balls,) = shur_lemma(args.balls, args.boxes)

    if model.solve():
        balls = balls.value()
        print("Balls:", balls)
        for b in range(args.boxes):
            print(f"Box {b+1}:", np.where(balls == b+1)[0].tolist())
    else:
        raise ValueError("Model is unsatisfiable")