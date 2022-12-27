import json

import numpy as np
import re
import sys
from pprint import pprint
import pprint


def parse_data(fname):

  with open(fname, "r") as data:
    # skip data
    for _ in range(7):
      data.readline()

    flags = np.array([
      "params", "capacity", "blocks", "options"
    ])

    for line in data:
      if flags[0] == "params":
        if "problem" in line.lower():
          # found problem name
          name = re.search("Problem.*\n",line).group()[:-1]
          json_entry = {"name": name,
                        "n_cars": None,
                        "n_options": None,
                        "n_classes": None,
                        "n_cars_p_class": [],
                        "options": []}

        elif re.match("#(-)+", line):
          # skip line
          continue
        elif re.match("[0-9]+ [0-9]+ [0-9]+", line):
          # problem parameters
          n_cars, n_options, n_classes = map(int, line.strip("\n").split(" "))
          json_entry |= {"n_cars":n_cars, "n_options":n_options, "n_classes":n_classes}
          flags = np.roll(flags,-1)
        else:
          json_entry["note"] = line.strip("# \n")

      elif flags[0] == "capacity":
        if line.startswith("0"):
          flags = np.roll(flags,-1)
        else:
          json_entry["capacity"] = list(map(int,line.strip("\n").split(" ")))
        flags = np.roll(flags,-1)

      if flags[0] == "blocks":
        json_entry["blocks"] = list(map(int,line.strip("\n").split(" ")))
        flags = np.roll(flags,-1)

      if flags[0] == "options":
        if len(line.strip(" \n")) == 0:
          # end of this problem, yield
          flags = np.roll(flags, -1)
          # convert options and cars_p_class to numpy
          yield json_entry

        else:
          c_idx, n, *opt = map(int, line.split(" "))
          json_entry["options"] += [opt]
          json_entry["n_cars_p_class"] += [n]


if __name__ == "__main__":
  fname = "data.txt"
  out = "prob001_car_sequence.json"
  if len(sys.argv) > 1:
    fname = sys.argv[1]
  if len(sys.argv) > 2:
    out = sys.argv[2]

  problems = list(parse_data(fname))

  outstring = pprint.pformat(problems, indent=4, sort_dicts=False)
  outstring = outstring.replace("'",'"')

  with open(out, "w") as outfile:
    outfile.write(outstring)