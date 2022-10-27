"""
Social golfer problem in cpmpy.

CSPLib problem 10:
http://www.csplib.org/prob/prob010/index.html
'''
The coordinator of a local golf club has come to you with the following
problem. In her club, there are 32 social golfers, each of whom play golf
once a week, and always in groups of 4. She would like you to come up
with a schedule of play for these golfers, to last as many weeks as
possible, such that no golfer plays in the same group as any other golfer
on more than one occasion.

Possible variants of the above problem include: finding a 10-week schedule
with ``maximum socialisation''; that is, as few repeated pairs as possible
(this has the same solutions as the original problem if it is possible
to have no repeated pairs), and finding a schedule of minimum length
such that each golfer plays with every other golfer at least once
(``full socialisation'').

The problem can easily be generalized to that of scheduling m groups of
n golfers over p weeks, such that no golfer plays in the same group as any
other golfer twice (i.e. maximum socialisation is achieved).
'''


This model is a translation of the OPL code from
http://www.dis.uniroma1.it/~tmancini/index.php?currItem=research.publications.webappendices.csplib2x.problemDetails&problemid=010



This cpmpy model was written by Hakan Kjellerstrand (hakank@gmail.com)
See also my cpmpy page: http://hakank.org/cpmpy/

Modified by Ignace Bleukx, ignace.bleukx@kuleuven.be
"""
from cpmpy import *
from cpmpy.expressions.utils import all_pairs
import numpy as np

def social_golfers(n_weeks, n_groups, group_size, **kwargs):

    n_golfers = n_groups * group_size
    print("n_golfers:", n_golfers, "n_weeks:", n_weeks, "group_size:", group_size, "groups:", n_groups)

    golfers = np.arange(n_golfers)
    weeks = np.arange(n_weeks)
    groups = np.arange(n_groups)

    #Possible configurations
    assign = intvar(0, n_groups - 1, shape=(n_golfers, n_weeks), name="assign")

    model = Model()

    # C1: Each group has exactly groupSize players
    for gr in groups:
        # can be written cleaner, see issue #117
        # model += sum(assign == gr, axis=1) == groupSize
        for w in weeks:
            model += sum(assign[:,w] == gr) == group_size

    # C2: Each pair of players only meets at most once
    for g1, g2 in all_pairs(golfers):
        model += sum(assign[g1] == assign[g2]) <= 1


    # SBSA: Symmetry-breaking by selective assignment
    # On the first week, the first groupSize golfers play in group 1, the
    # second groupSize golfers play in group 2, etc. On the second week,
    # golfer 1 plays in group 1, golfer 2 plays in group 2, etc.
    model += [assign[:,0] == (golfers // group_size)]

    for g in golfers:
        if g < group_size:
            model += [assign[g, 1] == g]

    # First golfer always in group 0
    model += [assign[0, :] == 0]

    return model, (assign,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n_weeks", type=int, default=4, help="Number of weeks")
    parser.add_argument("-n_groups", type=int, default=3, help="Number of groups")
    parser.add_argument("-group_size", type=int, default=3, help="Size of groups")

    args = parser.parse_args()

    n_golfers = args.n_groups * args.group_size

    model, (assign,) = social_golfers(**args.__dict__)

    if model.solve():
        assign = assign.value()
        # print schedule
        print("Schedule:")
        for w in range(args.n_weeks):
            print("week:", w + 1)
            for gr in range(args.n_groups):
                gs = np.where(assign[:, w] == gr)[0] + 1
                print(f"golfers in group {gr+1}:", "".join([f"{g:3d}" for g in gs]))
        print()

        # check which golfers meet each other
        meets = {g: [] for g in range(n_golfers)}
        for g1, g2 in all_pairs(range(n_golfers)):
            if sum(assign[g1] == assign[g2]) >= 1:
                meets[g1] += [g2 + 1]
                meets[g2] += [g1 + 1]

        for g in range(n_golfers):
            print(f"Golfer {g + 1} meets:", sorted(meets[g]))

    else:
        raise ValueError("Problem is unsatisfiable")