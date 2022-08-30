"""
    Progressive Party Problem (PPP) in CPMpy

    Problem 013 on CSPlib:
    https://www.csplib.org/Problems/prob013/

    The problem is to timetable a party at a yacht club.
    Certain boats are to be designated hosts, and the crews of the remaining boats in turn visit the host boats for several successive half-hour periods.
    The crew of a host boat remains on board to act as hosts while the crew of a guest boat together visits several hosts.
    Every boat can only hold a limited number of people at a time (its capacity) and crew sizes are different.
    The total number of people aboard a boat, including the host crew and guest crews, must not exceed the capacity.
    A guest boat cannot not revisit a host and guest crews cannot meet more than once.
    The problem facing the rally organizer is that of minimizing the number of host boats.

    Model by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
import sys
import requests
import json

from cpmpy import *
from cpmpy.expressions.utils import all_pairs


def progressive_party(n_boats, n_periods, capacity, crew_size, **kwargs):

    is_host = boolvar(shape=n_boats, name="is_host")
    visits = intvar(lb=0, ub=n_boats-1, shape=(n_periods,n_boats), name="visits")

    model = Model()

    # crews of host boats stay on boat
    for boat in range(n_boats):
        model += (is_host[boat]).implies(all(visits[:,boat] == boat))

    # number of visitors can never exceed capacity of boat
    for slot in range(n_periods):
        for boat in range(n_boats):
            model += sum((visits[slot] == boat) * crew_size) <= capacity[boat]

    # guests cannot visit a boat twice
    for boat in range(n_boats):
        # Alldiff must be decomposed in v0.9.8, see issue #105 on github
        model += (~is_host[boat]).implies(all((AllDifferent(visits[:,boat]).decompose())))

    # non-host boats cannot be visited
    for boat in range(n_boats):
        model += (~is_host[boat]).implies(all(visits != boat))

    # crews cannot meet more than once
    for c1, c2 in all_pairs(range(n_boats)):
        model += sum(visits[:,c1] == visits[:,c2]) <= 1

    # minimize number of hosts needed
    model.minimize(sum(is_host))

    return model, (visits,is_host)



# Helper functions
def _get_instance(data, pname):
    for entry in data:
        if pname == entry["name"]:
            return entry
    raise ValueError(f"Problem instance with name '{pname}' not found, use --list-instances to get the full list.")

def _print_instances(data):
    import pandas as pd
    df = pd.json_normalize(data)
    df_str = df.to_string(columns=["name", "n_boats", "n_periods", "note"], na_rep='-')
    print(df_str)

if __name__ == "__main__":
    import argparse
    import json
    import requests

    # argument parsing
    url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob013_progressive_party.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-instance', nargs='?', default="csplib_example", help="Name of the problem instance found in file 'filename'")
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

    model, (visits, is_host) = progressive_party(**problem_params)

    if model.solve():
        n_periods, n_boats = visits.shape
        visits, is_host = visits.value(), is_host.value()
        fmt = "{:<4}"
        for crew in range(n_boats):
            print("Crew {:<3}".format(crew), end=" ")
            print("(host) " if is_host[crew] else "(guest)", end=" ")
            print("visits boats", end=" ")
            print((fmt * n_periods).format(*visits[:, crew]))

    else:
        raise ValueError("Model is unsatisfiable!")