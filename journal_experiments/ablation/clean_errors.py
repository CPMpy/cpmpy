#!/usr/bin/env python
"""Remove result JSON files where the error field is not null."""
import json
import os
import sys

results_dir = sys.argv[1] if len(sys.argv) > 1 else "run-ablation"
if not os.path.isdir(results_dir):
    sys.exit("not a directory: {}".format(results_dir))

removed = 0
for name in sorted(os.listdir(results_dir)):
    if not name.endswith(".json"):
        continue
    path = os.path.join(results_dir, name)
    with open(path) as f:
        if json.load(f).get("error") is not None:
            os.remove(path)
            removed += 1
            print(path)

print("removed {} error result(s) from {}".format(removed, results_dir), file=sys.stderr)
