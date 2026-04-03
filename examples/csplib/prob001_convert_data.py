"""
This file just reads a csp001 instance textfile and converts it into JSON format
See `prob001_car_sequence.py` for the actual model that uses the JSON data file
"""
import json
import os

import numpy as np
import re
import sys


class CompactJSONEncoder(json.JSONEncoder):
  """A JSON Encoder that puts small lists on single lines."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.indentation_level = 0

  def encode(self, o):
    """Encode JSON object *o* with respect to single line lists."""

    if isinstance(o, (list, tuple)):
      if self._is_single_line_list(o):
        return "[" + ", ".join(json.dumps(el) for el in o) + "]"
      elif self._is_2d_list(o):
        return "[" + (",\n" + self.indentation_level * self.indent_str).join(json.dumps(lst) for lst in o) + "]"
      else:
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    elif isinstance(o, dict):
      self.indentation_level += 1
      output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
      self.indentation_level -= 1
      return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    else:
      return json.dumps(o)

  def _is_single_line_list(self, o):
    if isinstance(o, (list, tuple)):
      a = not any(isinstance(el, (list, tuple, dict)) for el in o)
      b = len(str(o)) <= 150
      return a or b

  def _is_2d_list(self, o):
    if isinstance(o, (list, tuple)):
      return any(self._is_single_line_list(el) for el in o)
    return False

  @property
  def indent_str(self) -> str:
    return " " * self.indentation_level * self.indent

  def iterencode(self, o, **kwargs):
    """Required to also work with `json.dump`."""
    return self.encode(o)

def parse_data(fname):

  with open(fname, "r") as data:
    # skip data
    while 1:
      line = data.readline()
      if line[0].isnumeric() or line[0] == "#" or line[0] == "\n":
        break

    flags = np.array([
      "params", "capacity", "blocks", "options"
    ])

    for line in data:
      print(line[:-1])

      if flags[0] == "params":
        if line == "\n":
          continue

        if "problem" in line.lower():
          # found problem name
          name = re.search("Problem.*\n",line).group()[:-1]
          if name == "Problem 60-01":
            print("breakpoint")
          json_entry = {"name": name,
                        "n_cars": None,
                        "n_options": None,
                        "n_classes": None,
                        "n_cars_p_class": [],
                        "options": []}
          continue

        elif re.match("#(-)+", line):
          # comment line
          continue
        elif re.match("[0-9]+ [0-9]+ [0-9]+", line):
          # problem parameters
          n_cars, n_options, n_classes = map(int, line.strip("\n").split(" "))
          json_entry |= {"n_cars":n_cars, "n_options":n_options, "n_classes":n_classes}
          flags = np.roll(flags,-1)
          continue
        else:
          json_entry["note"] = line.strip("# \n")

      elif flags[0] == "capacity":
        if line.startswith("0"):
          flags = np.roll(flags,-2)
        else:
          json_entry["capacity"] = list(map(int,line.strip("\n").split(" ")))
          flags = np.roll(flags,-1)
          continue

      elif flags[0] == "blocks":
        json_entry["blocks"] = list(map(int,line.strip("\n").split(" ")))
        flags = np.roll(flags,-1)
        continue

      if flags[0] == "options":
        if len(line.strip(" \n")) == 0:
          # end of this problem, yield
          flags = np.roll(flags, -1)
          # convert options and cars_p_class to numpy
          yield json_entry
          json_entry = None

        else:
          c_idx, n, *opt = map(int, line.split(" "))
          json_entry["options"] += [opt]
          json_entry["n_cars_p_class"] += [n]

    if json_entry is not None:
      yield json_entry

if __name__ == "__main__":
  fname = "data.txt"
  out = "prob001_car_sequence_new.json"
  if len(sys.argv) > 1:
    fname = sys.argv[1]
  if len(sys.argv) > 2:
    out = sys.argv[2]

  # if fname file does not exist, end with a warning
  if not os.path.exists(fname):
    print(f"File {fname} does not exist. No data to convert.")
  else:
    problems = list(parse_data(fname))

    with open(out, "w") as outfile:
      json.dump(problems, outfile, cls=CompactJSONEncoder, indent=4)

    print(f"Converted {len(problems)} problems to {out}")
