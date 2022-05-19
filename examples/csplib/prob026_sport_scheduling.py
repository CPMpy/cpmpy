"""
Problem 026 on CSPLib
Sport scheduling in CPMpy

Model created by Ignace Bleukx
"""

from cpmpy import *
from cpmpy.expressions.utils import all_pairs

import numpy as np

if __name__ == "__main__":

    n_teams = 8
    n_weeks, n_periods, n_matches = n_teams - 1, n_teams // 2, (n_teams - 1) * n_teams // 2

    home = intvar(1,n_teams, shape=(n_weeks, n_periods), name="home")
    away = intvar(1,n_teams, shape=(n_weeks, n_periods), name="away")

    model = Model()

    # teams cannot play each other
    model += home != away

    # every teams plays once a week
    # can be written cleaner, see issue #117
    # model += AllDifferent(np.append(home, away, axis=1), axis=0)
    for w in range(n_weeks):
        model += AllDifferent(np.append(home[w], away[w]))

    # every team plays each other
    for t1, t2 in all_pairs(range(1,n_teams+1)):
        model += (sum((home == t1) & (away == t2)) + sum((home == t2) & (away == t1))) >= 1

    # every team plays at most twice in the same period
    for t in range(1, n_teams + 1):
        # can be written cleaner, see issue #117
        # sum((home == t) | (away == t), axis=1) <= 2
        for p in range(n_periods):
            model += sum((home[p] == t) | (away[p] == t)) <= 2

    if model.solve():
        print(" " * 12, end ="")
        print(("{:^7} "*n_weeks).format(*[f"Week {w+1}" for w in range(n_weeks)]))

        for p in range(n_periods):
            print(f"Period {p+1}:", end=" || ")
            for w in range(n_weeks):
                print(f"{home.value()[w,p]} v {away.value()[w,p]}", end=" | ")
            print()