"""
Quasigroup existence problem (QG3) in cpmpy.

Problem 003 on CSPlib
https://www.csplib.org/Problems/prob003/

An order m quasigroup is a Latin square of size m. That is, a $m \\times m$ multiplication table in which each element
occurs once in every row and column. For example,

```
1    2   3   4
4    1   2   3
3    4   1   2
2    3   4   1
```

is an order 4 quasigroup. A quasigroup can be specified by a set and a binary multiplication operator, \\* defined over
this set. Quasigroup existence problems determine the existence or non-existence of quasigroups of a given size with
additional properties. Certain existence problems are of sufficient interest that a naming scheme has been invented
for them. We define two new relations, \\*321 and \\*312 by $a \\*321 b = c$ iff $c\\*b=a$ and
$a \\*312 b = c$ iff $b\\*c=a$.

QG3.m problems are order m quasigroups for which $(a\\*b)\\*(b\\*a) = a$.
We only consider the QG3.m problem for this task.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_003_quasigroup_existence/csplib_003_quasigroup_existence.cpmpy.py)
"""

import cpmpy as cp

def quasigroup_existence(m=8):
    quasigroup = cp.intvar(1, m, shape=(m, m), name="quasigroup")

    model = cp.Model()

    # Each element occurs once in every row and column
    model += [cp.AllDifferent(row) for row in quasigroup]
    model += [cp.AllDifferent(col) for col in quasigroup.T]

    # QG3.m property: (a*b)*(b*a) = a
    for a in range(m):
        for b in range(m):
            model += quasigroup[
                quasigroup[a, b] - 1,
                quasigroup[b, a] - 1,
            ] == a + 1

    return model, (quasigroup,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", type=int, default=8, help="Order of the quasigroup")

    m = parser.parse_args().m

    model, (quasigroup,) = quasigroup_existence(m)

    if model.solve():
        print(quasigroup.value())
    else:
        raise ValueError("Model is unsatisfiable")
