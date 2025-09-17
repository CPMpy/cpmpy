import cpmpy as cp
import numpy as np

# This cpmpy example solves a sudoku by Zanno, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N39

# sudoku cells
cells = cp.intvar(1,9, shape=(9,9))
# path indices 0 if not on path else the var indicates at what point the rat passes this cell
path = cp.intvar(0, 81, shape=(9,9))
# line modifiers
line = cp.intvar(0,6, shape=(2,9,9))
# inducers for induced C_WALLS
inducers = cp.intvar(0, 80, shape=(8,8))

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
                    [0,0,0,0,0,0,0,1,0],
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
    return cp.all([cells[r1,c1] > cells[r2,c2], path[r1,c1] + 1 != path[r2,c2]])

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
            if (r,c) == (2,8):
                # path starts on emitter, the next pathcell must have its first line value equal to this emitters value
                constraints.append(cp.any([cp.all([path[neighbour[0], neighbour[1]] == 2, line[0, neighbour[0], neighbour[1]] == 1]) for neighbour in neighbours]))
            elif (r,c) == (6,6):
                # path ends on emitter, the previous pathcell must have its second line value equal to this emitters value
                constraints.append(cp.any([cp.all([path[neighbour[0], neighbour[1]] == cp.max(path)-1, line[1, neighbour[0], neighbour[1]] == 4]) for neighbour in neighbours]))
            else:
                
                # for any pathcell, the next pathcell must always be reachable
                constraints.append((path[r,c] != 0).implies(cp.any([path[neighbour[0], neighbour[1]] == path[r,c] + 1 for neighbour in neighbours])))
                if (r,c) not in EMITTERS:
                    # for any non-emitter pathcell, the next non-emitter pathcell must have the same line values
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[neighbour[0], neighbour[1]])).implies((line[0, r, c] == line[0, neighbour[0], neighbour[1]]) & (line[1, r, c] == line[1, neighbour[0], neighbour[1]])) for neighbour in non_emitter_neighbours]))
                else:
                    # for any emitter pathcell, the next pathcell must have its first line value equal to this emitters value
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[neighbour[0], neighbour[1]])).implies(line[0, neighbour[0], neighbour[1]] == EMITTERS[(r,c)]) for neighbour in neighbours]))
                    # for any emitter pathcell, the previous pathcell must have its second line value equal to this emitters value
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] == path[neighbour[0], neighbour[1]] + 1)).implies(line[1, neighbour[0], neighbour[1]] == EMITTERS[(r,c)]) for neighbour in neighbours]))
                # if any cell is not on the path, its line values must be 0
                constraints.append((path[r,c] == 0).implies((line[0, r, c] == 0) & (line[1, r, c] == 0)))
                # for any pathcell its two line values must be different
                constraints.append((path[r,c] != 0).implies(line[0, r, c] != line[1, r, c]))
    return constraints

def renban(array):
    # the line must form a set of consecutive non repeating digits
    return cp.all([cp.AllDifferent(array), cp.max(array) - cp.min(array) == len(array) - 1])

def nabner(array):
    # no two adjacent digits can be consecutive and no digit repeats
    return cp.all([cp.AllDifferent(array), cp.all([cp.abs(a - b) > 1 for a,b in zip(array, array[1:])])])

def modular(array):
    # every set of 3 consecutive digits must have a different value mod 3
    return cp.all([cp.AllDifferent(array[i:i+3] % 3) for i in range(len(array) - 2)])

def entropy(array):
    # every set of 3 consecutive digit must have 1 low digit (1-3) and 1 middle digit (4-6) and 1 high digit (7-9)
    return cp.all([((cp.sum([cp.Count(array[i:i+3], j) for j in [1,2,3]]) == 1) & (cp.sum([cp.Count(array[i:i+3], j) for j in [4,5,6]]) == 1) & (cp.sum([cp.Count(array[i:i+3], j) for j in [7,8,9]]) == 1)) for i in range(len(array) - 2)])

def region_sum(array):
    # box borders divide the line into segments of the same sum
    return cp.BoolVal(True) # TODO

def ten_sum(array):
    # the line can be divided into one or more non-overlapping segments that each sum to 10
    return cp.BoolVal(True) # TODO

def activate_lines(working_cells):
    # once the path has been fully traversed, apply the line constraints
    constraints = []
    # print(working_cells)
    for line_cells in working_cells:
        c1 = EMITTERS[line_cells[0]] # first cell is always an emitter
        c2 = EMITTERS[line_cells[-1]] # last cell is always an emitter
        line_cells_vars = np.array([cells[cell[0], cell[1]] for cell in line_cells])
        if c1 == 1 or c2 == 1:
            constraints.append(renban(line_cells_vars))
        elif c1 == 2 or c2 == 2:
            constraints.append(nabner(line_cells_vars))
        elif c1 == 3 or c2 == 3:
            constraints.append(modular(line_cells_vars))
        elif c1 == 4 or c2 == 4:
            constraints.append(entropy(line_cells_vars))
        elif c1 == 5 or c2 == 5:
            constraints.append(region_sum(line_cells_vars))
        elif c1 == 6 or c2 == 6:
            constraints.append(ten_sum(line_cells_vars))
    return cp.all(constraints)

def split_lines(working_cells, cell_to_add, checked):
    # Follow the path to find the constraint of the puzzle. Lines are split by emitters.

    working_cells[-1] = working_cells[-1] + [cell_to_add]
    # print(working_cells, cell_to_add)
    if cell_to_add == (6,6):
        return activate_lines(working_cells)
    # check if next cell is an emitter, if so start a new line
    new_line = cell_to_add in EMITTERS
    if new_line:
        working_cells.append([cell_to_add])
    
    # find next cell in path
    r, c = cell_to_add
    neighbours = get_reachable_neighbours(r, c)
    return cp.all([((path[r,c] + 1 == path[neighbour[0], neighbour[1]])).implies(split_lines(working_cells.copy(), tuple(neighbour), checked + [tuple(neighbour)])) for neighbour in neighbours if tuple(neighbour) not in checked])

m = cp.Model(

    # path givens
    path[2,8] == 1,
    cp.max(path) == path[6,6],

    # all totals different
    cp.AllDifferentExcept0(path), 

    # valid path traversal
    path_valid(path),
    
    # gates
    gate((0,4), (1,4)),
    gate((2,3), (2,2)),
    gate((6,6), (7,6)),
    
    # line constraints on path
    # split_lines([[]], (2,8), [(2,8)])
)

for idx in EMITTERS:
    r, c = idx
    m += (path[r,c]!=0).implies((line[0, r, c] == EMITTERS[idx]) & (line[1, r, c] == 0))

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
                    row += "-"
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
                        row += "-"
                    else:
                        row += " "
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
                    if grid[r//2, c//2].value() == None:
                        row += " "
                    else:
                        row += str(grid[r//2, c//2].value())
        print(row)
    print("")
    
print("The puzzle is:")
print_grid(cells)

# print(m)

print("Number of constraints:", len(m.constraints))

sol = m.solve()
print("The solution is:")
print(cells.value())
print("The path is (0 if not on the path):")
print(path.value())
print("The line values are (0-5):")
print(line.value())