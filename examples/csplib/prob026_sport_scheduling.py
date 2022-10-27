"""
Problem 026 on CSPLib
Sport scheduling in CPMpy

Model created by Ignace Bleukx
"""
import pandas as pd

from cpmpy import *
from cpmpy.expressions.utils import all_pairs

import numpy as np

def sport_scheduling(n_teams):

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

    return model, (home, away)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n_teams", type=int, default=8, help="Number of teams to schedule")

    args = parser.parse_args()

    n_teams = args.n_teams
    n_weeks, n_periods, n_matches = n_teams - 1, n_teams // 2, (n_teams - 1) * n_teams // 2

    model, (home, away) = sport_scheduling(n_teams)

    if model.solve():
        import pandas as pd
        home, away = home.value(), away.value()
        matches = [[f"{h} v {a}" for h,a in zip(home[w], away[w])] for w in range(n_weeks)]
        print(matches)
        df = pd.DataFrame(matches,
                          index=[f"Week {w+1}" for w in range(n_weeks)],
                          columns=[f"Period {p+1}" for p in range(n_periods)])
        print(df.T.to_string(col_space=8, justify="center"))

    else:
        raise ValueError("Model is unsatisfiable")