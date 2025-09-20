import time
import cpmpy as cp
import numpy as np

from cpmpy.tools.explain import optimal_mus, mus

solver = "exact"


# This cpmpy example solves a sudoku by Zanno, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N39

# sudoku cells
cells = cp.intvar(1,9, shape=(9,9), name="values")
# path indices 0 if not on path else the var indicates at what point the rat passes this cell
path = cp.intvar(0, 81, shape=(9,9), name="path")
# list of cells (before, after) on path with indices and value (Due to the structure of the puzzle, the max amount of neighbours is 20, before running into another emitter. It is more efficient to reduce the amount like this. This amount can also be proven with a cpmpy model that runs before this one.)
sequence = cp.intvar(-1, 9, shape=(9,9,6,20), name="sequence")  # for each cell: before value, before row, before column, after value, after row, after column | -1 if not applicable

# inducers for induced C_WALLS
inducers = cp.intvar(0, 80, shape=(8,8), name="inducers")

# givens
# vertical walls
V_WALLS = np.array([[0,0,0,0,0,0,1,0],
                    [0,0,0,1,0,1,0,1],
                    [0,0,0,0,1,1,1,0],
                    [1,0,1,0,1,0,0,1],
                    [1,0,0,0,1,0,0,0],
                    [0,0,0,1,0,0,0,0],
                    [0,1,0,0,0,1,0,0],
                    [0,1,0,0,0,0,0,1],
                    [0,0,0,0,0,0,1,0]])
# horizontal walls
H_WALLS = np.array([[0,1,0,0,0,0,0,0,0],
                    [0,0,1,1,0,0,0,0,0],
                    [0,0,1,0,0,0,0,1,0],
                    [0,0,0,0,0,1,0,0,0],
                    [0,0,0,0,0,0,0,1,1],
                    [0,0,0,1,0,0,1,1,0],
                    [0,0,0,0,0,0,0,0,0],
                    [0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0]])
# corners walls (to block diagonal movement and future proofing for puzzles that only include walls just made out of a corner)
C_WALLS = np.array([[1,1,0,1,0,1,1,1],
                    [1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1],
                    [1,1,1,0,1,1,0,1],
                    [1,0,1,1,1,0,1,1],
                    [0,1,1,1,0,1,1,1],
                    [0,1,0,0,0,1,0,1],
                    [0,1,1,1,1,0,1,1]])

# emitters 1 = renban, 2 = nabner, 3 = modular, 4 = entropy, 5 = region_sum, 6 = ten_sum
EMITTERS = {(0, 5): 6, (2, 1): 5, (2, 4): 3, (2, 8): 1, (4, 2): 4, (4, 6): 6, (5, 0): 3, (5, 6): 2, (6, 5): 2, (6, 6): 4, (7, 5): 5, (8, 2): 1}

def gate(idx1, idx2):
    # the path can not pass from idx2 to idx1, and the value in idx1 must be greater than the value in idx2
    r1, c1 = idx1
    r2, c2 = idx2
    return cp.all([cells[r1,c1] > cells[r2,c2], path[r1,c1] - 1 != path[r2,c2]])

def get_reachable_neighbours(row, column):
    # from a cell get the indices of its reachable neighbours
    reachable_neighbours = []
    if row != 0:
        if H_WALLS[row-1, column] == 0:
            reachable_neighbours.append([row - 1, column])
        if column != 0:
            if C_WALLS[row-1, column-1] == 0:
                reachable_neighbours.append([row - 1, column - 1])
        if column != 8:
            if C_WALLS[row-1, column] == 0:
                reachable_neighbours.append([row - 1, column + 1])
    if row != 8:
        if H_WALLS[row, column] == 0:
            reachable_neighbours.append([row + 1, column])
        if column != 0:
            if C_WALLS[row, column-1] == 0:
                reachable_neighbours.append([row + 1, column - 1])
        if column != 8:
            if C_WALLS[row, column] == 0:
                reachable_neighbours.append([row + 1, column + 1])
    if column != 0:
        if V_WALLS[row, column-1] == 0:
            reachable_neighbours.append([row, column - 1])
    if column != 8:
        if V_WALLS[row, column] == 0:
            reachable_neighbours.append([row, column + 1])
    return reachable_neighbours

def path_valid(path):
    constraints = []
    for r in range(path.shape[0]):
        for c in range(path.shape[0]):
            neighbours = get_reachable_neighbours(r, c)
            non_emitter_neighbours = [n for n in neighbours if tuple(n) not in EMITTERS]
            emitter_neighbours = [n for n in neighbours if tuple(n) in EMITTERS]
            if (r,c) == (2,8):
                # The path starts on emitter. It doesn't have any previous cells so the first 3 vectors in the sequence must be fully -1. As for the the last 3, they are taken from the neighbour. Vice versa also applies.
                constraints.append(cp.all([cp.all(sequence[r,c,:3].flatten() == -1)]))
                constraints.append(cp.any([cp.all([path[nr,nc] == 2, sequence[r,c,3,0] == cells[nr,nc], sequence[r,c,4,0] == nr, sequence[r,c,5,0] == nc, cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), sequence[nr,nc,0,0] == cells[r,c], sequence[nr,nc,1,0] == r, sequence[nr,nc,2,0] == c, cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19])]) for nr, nc in neighbours]))
            elif (r,c) == (6,6):
                # nothing comes after this emitter on the path
                constraints.append(cp.all([cp.all(sequence[r,c,3:].flatten() == -1)]))
                constraints.append(path[r,c] == cp.max(path))
            else:
                # for any pathcell, the next pathcell must always be reachable
                constraints.append((path[r,c] != 0).implies(cp.any([path[neighbour[0], neighbour[1]] == path[r,c] + 1 for neighbour in neighbours])))
                if (r,c) not in EMITTERS:
                    # carry over sequence values from non-emitter to non-emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies(cp.all([sequence[r,c,3,0] == cells[nr,nc], sequence[r,c,4,0] == nr, sequence[r,c,5,0] == nc, cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), sequence[nr,nc,0,0] == cells[r,c], sequence[nr,nc,1,0] == r, sequence[nr,nc,2,0] == c, cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19])])) for nr,nc in non_emitter_neighbours]))
                    # carry over sequence values from non-emitter to emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies(cp.all([sequence[r,c,3,0] == cells[nr,nc], sequence[r,c,4,0] == nr, sequence[r,c,5,0] == nc, cp.all(sequence[r,c,3,1:] == -1), cp.all(sequence[r,c,4,1:] == -1), cp.all(sequence[r,c,5,1:] == -1), sequence[nr,nc,0,0] == cells[r,c], sequence[nr,nc,1,0] == r, sequence[nr,nc,2,0] == c, cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19])])) for nr,nc in emitter_neighbours]))
                else:
                    # carry over sequence values from emitter to non-emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies(cp.all([sequence[r,c,3,0] == cells[nr,nc], sequence[r,c,4,0] == nr, sequence[r,c,5,0] == nc, cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), sequence[nr,nc,0,0] == cells[r,c], sequence[nr,nc,1,0] == r, sequence[nr,nc,2,0] == c, cp.all(sequence[nr,nc,0,1:] == -1), cp.all(sequence[nr,nc,1,1:] == -1), cp.all(sequence[nr,nc,2,1:] == -1)])) for nr,nc in non_emitter_neighbours]))
                    # carry over sequence values from emitter to emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies(cp.all([sequence[r,c,3,0] == cells[nr,nc], sequence[r,c,4,0] == nr, sequence[r,c,5,0] == nc, cp.all(sequence[r,c,3,1:] == -1), cp.all(sequence[r,c,4,1:] == -1), cp.all(sequence[r,c,5,1:] == -1), sequence[nr,nc,0,0] == cells[r,c], sequence[nr,nc,1,0] == r, sequence[nr,nc,2,0] == c, cp.all(sequence[nr,nc,0,1:] == -1), cp.all(sequence[nr,nc,1,1:] == -1), cp.all(sequence[nr,nc,2,1:] == -1)])) for nr,nc in emitter_neighbours]))

            # for any non-pathcell, its sequence must be fully -1
            constraints.append((path[r,c] == 0).implies(cp.all(sequence[r,c] == -1)))

            # if the path moves diagonally, it induces a C_WALL
            for nr, nc in neighbours:
                if abs(nr - r) == 1 and abs(nc - c) == 1:
                    wall_r = min(r, nr)
                    wall_c = min(c, nc)
                    constraints.append((cp.all([path[r,c] != 0, path[r,c] + 1 == path[nr,nc]])).implies(inducers[wall_r, wall_c] == path[r,c])) # this forces no diagonal crossings
    return constraints

def same_region(r1, c1, r2, c2):
    # check if two cells are in the same 3x3 region
    return cp.all([(r1 // 3 == r2 // 3), (c1 // 3 == c2 // 3)])

def renban(array, rs, cs):
    # the line must form a set of consecutive non repeating digits
    cons = []
    renban_emitters = [k for k, v in EMITTERS.items() if v == 1]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in renban_emitters])))  # if the next cell is -1, the current cell can't be a renban emitter
    cons.append(cp.all([cp.AllDifferentExceptN(array, -1), cp.max(array) - cp.min([a + 10*(a == -1) for a in array]) + 1 == len(array) - cp.Count(array, -1)]))
    # print("Renban constraints:", cons)
    return cons

def nabner(array, rs, cs):
    # no two digits are consecutive and no digit repeats
    cons = []
    nabner_emitters = [k for k, v in EMITTERS.items() if v == 2]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in nabner_emitters])))  # if the next cell is -1, the current cell can't be a nabner emitter

    for i in range(len(array) - 1):
        cons.append((array[i] != -1).implies(cp.all([cp.abs(array[i] - array[j]) > 1 for j in range(i+1,len(array))])))
    cons.append(cp.AllDifferentExceptN(array, -1))
    # print("Nabner constraints:", cons)
    return cons

def modular(array, rs, cs):
    # every set of 3 consecutive digits must have a different value mod 3
    cons = []
    modular_emitters = [k for k, v in EMITTERS.items() if v == 3]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in modular_emitters])))  # if the next cell is -1, the current cell can't be a modular emitter
    arr = cp.intvar(-1, 2, shape=len(array))
    cons.append(cp.all([cp.all(arr == (array % 3)), cp.all([cp.AllDifferentExceptN(arr[i:i+3], -1) for i in range(len(arr) - 2)])]))
    # print("Modular constraints:", cons)
    return cons

def entropy(array, rs, cs):
    # every set of 3 consecutive digit must have 1 low digit (1-3) and 1 middle digit (4-6) and 1 high digit (7-9)
    cons = []
    entropy_emitters = [k for k, v in EMITTERS.items() if v == 4]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in entropy_emitters])))  # if the next cell is -1, the current cell can't be an entropy emitter
    cons.append(cp.all([cp.AllDifferentExceptN([(a+2) // 3 for a in array[i:i+3]], 0) for i in range(len(array) - 2)]))
    # print("Entropy constraints:", cons)
    return cons

def region_sum(array, rs, cs):
    # box borders divide the line into segments of the same sum
    running_sums = cp.intvar(-1, 45, shape=len(array), name=f"running_sums_{rs[0]}_{cs[0]}")
    region_sum = cp.intvar(1, 45, name=f"region_sum_{rs[0]}_{cs[0]}")
    cons = []
    region_sum_emitters = [k for k, v in EMITTERS.items() if v == 5]

    cons.append(region_sum == running_sums[0])
    cons.append(running_sums[-1] == array[-1])
    for i in range(len(array)-1):
        cons.append((cp.all([array[i+1] != -1, ~same_region(rs[i], cs[i], rs[i+1], cs[i+1])])).implies(cp.all([running_sums[i] == array[i], region_sum == running_sums[i+1]])))
        cons.append((cp.all([(array[i+1] == -1), (array[i] != -1)])).implies(running_sums[i] == array[i]))
        cons.append((cp.all([array[i+1] != -1, same_region(rs[i], cs[i], rs[i+1], cs[i+1])])).implies(cp.all([running_sums[i] == running_sums[i+1] + array[i]])))
        cons.append((array[i] != -1).implies(running_sums[i] == array[i]))  # if the next cell is -1, the current cell can't be a region_sum emitter
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in region_sum_emitters])))  # if the next cell is -1, the current cell can't be a region_sum emitter
    cons.append(cp.Count(running_sums, region_sum) >= 2)  # at least two segments of the same sum
    # print("RS constraints:", cons)
    return cons

def ten_sum(array, rs, cs):
    # the line can be divided into one or more non-overlapping segments that each sum to 10
    running_sums = cp.intvar(-1, 45, shape=len(array), name=f"ten_running_sums_{rs[0]}_{cs[0]}")
    splits = cp.boolvar(shape=len(array)-1, name=f"splits_{rs[0]}_{cs[0]}")
    region_sum = 10
    cons = []
    ten_sum_emitters = [k for k, v in EMITTERS.items() if v == 6]

    cons.append(region_sum == running_sums[0])
    cons.append(running_sums[-1] == array[-1])
    for i in range(len(array)-1):
        cons.append((cp.all([array[i+1] != -1, splits[i]])).implies(cp.all([running_sums[i] == array[i], region_sum == running_sums[i+1]])))
        cons.append((cp.all([(array[i+1] == -1), (array[i] != -1)])).implies(running_sums[i] == array[i]))
        cons.append((cp.all([array[i+1] != -1, ~splits[i]])).implies(cp.all([running_sums[i] == running_sums[i+1] + array[i]])))
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in ten_sum_emitters])))  # if the next cell is -1, the current cell can't be a ten_sum emitter
    # print("TenSum constraints:", cons)
    return cons

def activate_lines(sequence):
    # once the path has been fully traversed, apply the line constraints
    constraints = []

    for er,ec in EMITTERS:
        line = EMITTERS[(er,ec)]
        before = np.concatenate(([cells[er,ec]], sequence[er,ec,0,:]))  # before emitter
        after = np.concatenate(([cells[er,ec]], sequence[er,ec,3,:]))   # after emitter
        before_rs = np.concatenate(([er], sequence[er,ec,1,:]))  # before emitter rows
        before_cs = np.concatenate(([ec], sequence[er,ec,2,:]))  # before emitter columns
        after_rs = np.concatenate(([er], sequence[er,ec,4,:]))   # after emitter rows
        after_cs = np.concatenate(([ec], sequence[er,ec,5,:]))   # after emitter columns

        if line == 1:
            cons_before = renban(before, before_rs, before_cs)
            cons_after = renban(after, after_rs, after_cs)
        elif line == 2:
            cons_before = nabner(before, before_rs, before_cs)
            cons_after = nabner(after, after_rs, after_cs)
        elif line == 3:
            cons_before = modular(before, before_rs, before_cs)
            cons_after = modular(after, after_rs, after_cs)
        elif line == 4:
            cons_before = entropy(before, before_rs, before_cs)
            cons_after = entropy(after, after_rs, after_cs)
        elif line == 5:
            cons_before = region_sum(before, before_rs, before_cs)
            cons_after = region_sum(after, after_rs, after_cs)
        elif line == 6:
            cons_before = ten_sum(before, before_rs, before_cs)
            cons_after = ten_sum(after, after_rs, after_cs)
        for c in cons_before:
            constraints.append((path[er,ec] > 1).implies(c))
        for c in cons_after:
            constraints.append((cp.all([0 < path[er,ec],path[er,ec] < cp.max(path)])).implies(c))
    return constraints

m = cp.Model(

    # path givens
    path[2,8] == 1,
    cp.max(path) == path[6,6],

    # all totals different
    cp.AllDifferentExcept0(path),

    # gates
    gate((0,4), (1,4)),
    gate((2,3), (2,2)),
    gate((6,6), (7,6)),
)


# Add these constraints one-by-one to the model to be able to find a more fine-grained MUS
cons = activate_lines(sequence)
for c in cons:
    m += c
# m += cons

cons = path_valid(path)
for c in cons:
    m += c
# m += cons

def regroup_to_blocks(grid):
    # Create an empty list to store the blocks
    blocks = [[] for _ in range(9)]

    for row_index in range(9):
        for col_index in range(9):
            # Determine which block the current element belongs to
            block_index = (row_index // 3) * 3 + (col_index // 3)
            # Add the element to the appropriate block
            blocks[block_index].append(grid[row_index][col_index])

    return blocks


blocks = regroup_to_blocks(cells)

for i in range(cells.shape[0]):
    m += cp.AllDifferent(cells[i,:])
    m += cp.AllDifferent(cells[:,i])
    m += cp.AllDifferent(blocks[i])

def print_grid(grid):
    for r in range(grid.shape[0]*2+1):
        row = ""
        if r == 0 or r == grid.shape[0]*2:
            for c in range(grid.shape[1]*2+1):
                if c % 2 == 0:
                    row += "+"
                else:
                    row += "---"
        elif r % 2 == 0:
            for c in range(grid.shape[1]*2+1):
                if c == 0 or c == grid.shape[1]*2:
                    row += "+"
                elif c % 2 == 0:
                    if C_WALLS[r//2 - 1, c//2 - 1] == 1:
                        row += "+"
                    else:
                        row += " "
                else:
                    if H_WALLS[r//2 - 1, c//2] == 1:
                        row += "---"
                    else:
                        row += "   "
        else:
            for c in range(grid.shape[1]*2+1):
                if c == 0 or c == grid.shape[1]*2:
                    row += "|"
                elif c % 2 == 0:
                    if V_WALLS[r//2, c//2 - 1] == 1:
                        row += "|"
                    else:
                        row += " "
                else:
                    if (r//2, c//2) not in EMITTERS:
                        row += "   "
                    else:
                        row += " " + str(EMITTERS[(r//2, c//2)]) + " "
        print(row)
    print("")

print("The puzzle is:")
print_grid(cells)

# print(m)

# write constraints to a file
with open("sudoku_schrodingers_rat.txt", "w") as f:
    for c in m.constraints:
        f.write(str(c) + "\n")

print("Number of constraints:", len(m.constraints))

start = time.time()
sol = m.solve(solver=solver)
end = time.time()

print(f"Solved in {end - start} seconds")
if sol:
    print("The solution is:")
    print(cells.value())
    print("The path is (0 if not on the path):")
    print(path.value())
    print("With the following sequence values (Before: value, row, column | After: value, row, column):")
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            if (r,c) in EMITTERS:
                print(f"Cell ({r},{c}): {sequence[r,c].value()}")
else:
    print("Model UNSAT, finding MUS...")

    res = mus(m.constraints, solver=solver)

    print(f"{len(res)} of {len(m.constraints)} constraints in the MUS:")

    print("MUS constraints:")
    for c in res:
        input("Press Enter to continue...")
        print(c)