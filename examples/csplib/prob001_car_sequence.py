"""
Car sequencing in CPMpy (prob001 in CSPlib)
https://www.csplib.org/Problems/prob001/

Based on the Minizinc model car.mzn.

Data format compatible with both variations of model (with and without block constraints)
Model was created by Ignace Bleukx.
"""
import sys
import json
import requests

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cpmpy import *



def car_sequence(n_cars, n_options, n_classes, n_cars_p_class, options, capacity=None, block_size=None, **kwargs):

  # build model
  model = Model()

  # decision variables
  slots = intvar(0, n_classes-1, shape=n_cars, name="slots")
  setup = boolvar(shape=(n_cars, n_options), name="setup")

  # convert options to cpm_array
  options = cpm_array(options)

  # satisfy demand
  model += [sum(slots == c) == n_cars_p_class[c] for c in range(n_classes)]

  # car has correct options
  # This can be written cleaner, see issue #117 on github
  # m += [setup[s] == options[slots[s]] for s in range(n_cars)]
  for s in range(n_cars):
    model += [setup[s,o] == options[slots[s],o] for o in range(n_options)]

  if capacity is not None:
    # satisfy block capacity
    for o in range(n_options):
      setup_seq = setup[:,o]
      # get all setups within block size of each other
      blocks = sliding_window_view(setup_seq, block_size[o])
      for block in blocks:
        model += sum(block) <= capacity[o]


  return model, (slots, setup)


def get_data(data, pname):
  for entry in data:
    if pname in entry["name"]:
      return entry


if __name__ == "__main__":
  import argparse

  url = "https://raw.githubusercontent.com/CPMpy/cpmpy/csplib/examples/csplib/prob001_car_sequence.json"
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--instance', default="Problem 4/72", help="Name of the problem instance found in file 'filename'")
  parser.add_argument('--url', help="URL to a set of instances, serves the same purpose as the --filename argument", default=url)
  parser.add_argument('--filename', help="File containing problem instances, overrides --url argument")

  args = parser.parse_args()

  if args.filename is None:
    problem_data = requests.get(args.url)
  else:
    with open(args.filename,"r") as f:
      problem_data = json.load(f)

  params = get_data(problem_data, args.instance)

  model, (slots, setup) = car_sequence(**params)

  # solve the model
  if model.solve():
    print("Class", "Options req.", sep="\t")
    for i in range(len(slots)):
      print(slots.value()[i],
            setup.value()[i].astype(int),
            sep="\t\t")
  else:
    print("Model is unsatisfiable!")
