import cpmpy as cp
import numpy as np
from setuptools.namespaces import flatten

# This cpmpy example solves a sudoku by marty_sears, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000JUJ

# sudoku cells
cells = cp.intvar(1,6, shape=(6,6))
# duplicates in rows and columns
duplicate_rs = cp.intvar(1,6, shape=6)
duplicate_cs = cp.intvar(1,6, shape=6)
# removals in rows and columns
removals_rs = cp.intvar(1,6, shape=6)
removals_cs = cp.intvar(1,6, shape=6)

def white_kropki(a, b):
    # digits separated by a white dot differ by 1
    return abs(a-b) == 1

def black_kropki(a, b):
    # digits separated by a black dot are in a 1:2 ratio
    return  cp.any([a * 2 == b, a == 2 * b])

def zipper(args):
    # equidistant cells from the middle, sum to the value in the value in the middle
    assert len(args) % 2 == 1
    mid = len(args) // 2
    return cp.all([args[i] + args[len(args)-1-i] == args[mid] for i in range(mid)])

def X(a, b):
    # digits separated by an X sum to 10
    return a + b == 10

def renban(args):
    # digits on a pink renban form a set of consecutive non repeating digits
    return cp.all([cp.AllDifferent(args), cp.max(args) - cp.min(args) == len(args) - 1])

def cage(args, total):
    # digits in a cage sum to the given total
    # no all different necessary in this puzzle
    return cp.sum(args) == total

def duplicate(array, doppel):
    # every row and column has one duplicated digit
    all_triplets = [[(a, b, c), (a, c, b), (b,c,a)] for idx, a in enumerate(array) for idx2, b in enumerate(array[idx + 1:]) for c in array[idx+idx2+2:]]
    all_triplets = flatten(all_triplets)
    # any vars in a pair cannot be equal to a third var
    # pairs implying a decision var blocks multiple pairs from existing
    # we know there will be at least one duplicate because of the removal constraints
    return cp.all([(var1 == var2).implies(cp.all([var1==doppel, var1 != var3])) for var1, var2, var3 in all_triplets])

def missing(array, removed):
    # every row and column has one missing digit
    return cp.all([removed != elm for elm in array])

m = cp.Model(
    # zipper lines
    zipper(np.concatenate((cells[:,1], [cells[5,0]]))),
    zipper(cells[1,3:]),
    # kropki dots
    black_kropki(cells[0,0], cells[1,0]),
    white_kropki(cells[1,0], cells[2,0]),
    black_kropki(cells[2,0], cells[3,0]),
    black_kropki(cells[5,0], cells[4,0]),
    white_kropki(cells[3,0], cells[3,1]),
    white_kropki(cells[5,1], cells[5,2]),
    white_kropki(cells[3,2], cells[3,3]),
    white_kropki(cells[3,4], cells[3,5]),
    black_kropki(cells[4,2], cells[5,2]),
    black_kropki(cells[3,3], cells[4,3]),
    black_kropki(cells[2,4], cells[2,5]),
    # killer cages
    cage(cells[:,5], 19),
    cage(cells[4:,4], 7),
    # Xes
    X(cells[0,1], cells[0,2]),
    X(cells[0,2], cells[1,2]),
    X(cells[0,3], cells[1,3]),
    X(cells[0,3], cells[0,4]),
    # renban
    renban(cells[2, 1:]),
    # one duplicate in each row and column
    cp.all(duplicate(cells[i, :], duplicate_rs[i]) for i in range(duplicate_rs.shape[0])),
    cp.all(duplicate(cells[:,i], duplicate_cs[i]) for i in range(duplicate_cs.shape[0])),
    # one removed from each row and column
    cp.all(missing(cells[i,:], removals_rs[i]) for i in range(removals_rs.shape[0])),
    cp.all(missing(cells[:,i], removals_cs[i]) for i in range(removals_cs.shape[0])),
    # all removals and repeats should be unique for each row and column
    cp.AllDifferent(duplicate_rs),
    cp.AllDifferent(duplicate_cs),
    cp.AllDifferent(removals_rs),
    cp.AllDifferent(removals_cs)

)

sol = m.solve()
print("The solution is:")
print(cells.value())
print("The duplicates in rows are:")
print(duplicate_rs.value())
print("The duplicates in columns are:")
print(duplicate_cs.value())
print("The removed digits in rows are:")
print(removals_rs.value())
print("The removed digits in columns are:")
print(removals_cs.value())