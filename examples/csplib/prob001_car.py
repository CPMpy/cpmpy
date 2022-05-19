"""
Car sequencing in CPMpy (prob001 in CSPlib)

Based on the Minizinc model car.mzn.

Data format compatible with both variations of model (with and without block constraints)
Model was created by Ignace Bleukx.
"""

from cpmpy import *
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import sys

def car_sequence(n_cars, n_options, n_classes, capacity, block_size, n_cars_p_class, options):

  # build model
  model = Model()

  # decision variables
  slots = intvar(0, n_classes-1, shape=n_cars, name="slots")
  setup = boolvar(shape=(n_cars, n_options), name="setup")
  
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
      blocks = sliding_window_view(setup_seq, block_size[o])
      for block in blocks:
        model += sum(block) <= capacity[o]


  return model, slots, setup


def get_data(fname):
  # read data
  with open(fname, "r") as data:
    n_cars, n_options, n_classes = map(int, data.readline().split(" "))

    n_cars_p_class = np.zeros(shape=n_classes, dtype=int)
    options = np.zeros(shape=(n_classes, n_options), dtype=bool)

    line = list(map(int, data.readline().split(" ")))
    if line[0] == 0:
      # no block constraints
      capacity, block_size = None, None

      c_idx, n, *opt = line
      n_cars_p_class[c_idx] = n
      options[c_idx] = opt

    else:
      # block constraints
      capacity = np.array(line)
      block_size = np.array(list(map(int, data.readline().split(" "))))

    while 1:
      line = data.readline()
      if len(line) == 0:
        break

      c_idx, n, *opt = map(int, line.split(" "))

      n_cars_p_class[c_idx] = n
      options[c_idx] = opt

    options = cpm_array(options)

    return n_cars, n_options, n_classes, \
           capacity, block_size,\
           n_cars_p_class, options

if __name__ == "__main__":
  # get data
  data = get_data(sys.argv[1])
  model, slots, setup = car_sequence(*data)

  # solve the model
  if model.solve():
    print("Class", "Options req.", sep="\t")
    for i in range(len(slots)):
      print(slots.value()[i],
            setup.value()[i].astype(int),
            sep="\t\t")
  else:
    print("Model is unsatisfiable!")
