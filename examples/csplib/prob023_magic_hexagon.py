"""
Magic hexagon problem in cpmpy.

Problem 023 on CSPlib
https://www.csplib.org/Problems/prob023/

A magic hexagon of diameter d (the number of cells in the longest row) arranges
the numbers 1 to N in a hexagonal pattern such that every row in all three
directions sums to the same magic constant.

For diameter 5 (edge length 3, N=19 cells), the magic constant is 38.
The hexagonal grid:
      A, B, C
     D, E, F, G
    H, I, J, K, L
     M, N, O, P
      Q, R, S

The goal is to find an assignment of numbers to the variables A through S that
satisfies all the sum constraints.

Model from DCP-Bench-Open (https://github.com/DCP-Bench/DCP-Bench-Open/blob/main/dataset/csplib_023_magic_hexagon/csplib_023_magic_hexagon.cpmpy.py)
"""

import cpmpy as cp


def _hex_coords(n):
    """Return axial coordinates (q, r) for a hexagon of edge length n.

    Coordinates are ordered row by row (top to bottom), left to right within
    each row. This matches the traditional hex printing layout.
    """
    coords = []
    for r in range(-(n - 1), n):  # rows from top to bottom
        for q in range(-(n - 1), n):  # left to right within row
            if -(n - 1) <= q + r <= n - 1:
                coords.append((q, r))
    return coords


def magic_hexagon(n=3):
    """Build a magic hexagon model for edge length n.

    Args:
        n: Edge length of the hexagon (default 3).

    Returns:
        (model, vars) where vars is a flat list of decision variables.
    """
    num_cells = 3 * n * n - 3 * n + 1
    magic_sum = num_cells * (num_cells + 1) // (2 * (2 * n - 1))

    coords = _hex_coords(n)
    assert len(coords) == num_cells

    LD = cp.intvar(1, num_cells, shape=num_cells, name="LD")

    # Build lookup: coordinate -> variable
    var_at = {c: v for c, v in zip(coords, LD)}

    model = cp.Model()
    model += cp.AllDifferent(LD)

    # Horizontal rows: group by r coordinate
    for r in range(-(n - 1), n):
        row = [var_at[(q, r)] for q in range(-(n - 1), n)
               if (q, r) in var_at]
        model += cp.sum(row) == magic_sum

    # Diagonals (top-left to bottom-right): group by (q + r) coordinate
    for s in range(-(n - 1) * 2, (n - 1) * 2 + 1):
        diag = [var_at[(q, r)] for q in range(-(n - 1), n)
                for r in range(-(n - 1), n)
                if (q, r) in var_at and q + r == s]
        if diag:
            model += cp.sum(diag) == magic_sum

    # Diagonals (top-right to bottom-left): group by q coordinate
    for q in range(-(n - 1), n):
        diag = [var_at[(q, r)] for r in range(-(n - 1), n)
                if (q, r) in var_at]
        if diag:
            model += cp.sum(diag) == magic_sum

    return model, (LD,)


def print_hexagon(LD, n):
    coords = _hex_coords(n)
    vals = LD.value()
    current_r = None
    for (q, r), v in zip(coords, vals):
        if r != current_r:
            if current_r is not None:
                print()
            # Indent: rows further from center need more padding
            indent = abs(r)
            print(" " * indent, end="")
            current_r = r
        print(f"{v:2d} ", end="")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=3, help="Edge length of the hexagon (default: 3)")
    args = parser.parse_args()

    model, (LD,) = magic_hexagon(n=args.n)

    if model.solve():
        print_hexagon(LD, n=args.n)
    else:
        raise ValueError("Model is unsatisfiable")
