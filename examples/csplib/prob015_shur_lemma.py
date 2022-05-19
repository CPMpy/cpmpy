"""
Problem 015 on CSPLib

Given n balls, labelled 1 to n, put them into c boxes such
that for any triple of balls (x1,x2,...,xc) with sum(x1,x2,...,xc-1) = xc, not all are in the same box.


Adapted from Numberjack implementation https://github.com/csplib/csplib/blob/master/Problems/prob015/models/SchursLemma.py

Modified by Ignace Bleukx
"""

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

    n, c = 12, 3  # n is the number of balls -- d is the number of boxes

    model, (balls,) = shur_lemma(n,c)

    if model.solve():
        print("Balls:")
        print(balls.value())
    else:
        print("Model is unsatisfiable")