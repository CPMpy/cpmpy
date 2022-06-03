"""
    Progressive Party Problem (PPP) in CPMpy

    Problem 013 on CSPlib:
    https://www.csplib.org/Problems/prob013/

    The problem is to timetable a party at a yacht club.
    Certain boats are to be designated hosts, and the crews of the remaining boats in turn visit the host boats for several successive half-hour periods.
    The crew of a host boat remains on board to act as hosts while the crew of a guest boat together visits several hosts.
    Every boat can only hold a limited number of people at a time (its capacity) and crew sizes are different.
    The total number of people aboard a boat, including the host crew and guest crews, must not exceed the capacity.
    A table with boat capacities and crew sizes can be found below; there were six time periods.
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

def get_data(data, pname):
  for entry in data:
    if pname in entry["name"]:
      return entry


def print_solution(visits, hosts):
    n_periods, n_boats = visits.shape
    fmt = "{:<4}"
    for crew in range(n_boats):
        print("Crew {:<3}".format(crew), end=" ")
        print("(host) " if hosts[crew] else "(guest)", end=" ")
        print("visits boats", end=" ")
        print((fmt*n_periods).format(*visits[:,crew]))


if __name__ == "__main__":
    # get data
    fname = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob013_progressive_party.json"
    problem_name = "6"

    data = None

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        with open(fname,"r") as f:
          data = json.load(f)

    if len(sys.argv) > 2:
        problem_name = sys.argv[2]

    if data is None:
        data = requests.get(fname).json()

    params = get_data(data, problem_name)

    model, (visits, is_host) = progressive_party(**params)

    if model.solve():
        print_solution(visits.value(), is_host.value())

    else:
        print("Model is unsatisfiable!")