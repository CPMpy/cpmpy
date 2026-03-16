import cpmpy as cp
import numpy as np

""" 
Puzzle source: https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000N39

Rules: 

- Normal Sudoku rules apply.

- Standart RAT RUN RULES apply: 
- The rat must reach the hole by finding a path through the maze. 
- The path must not visit any cell more than once, cross itself, or pass through any thick maze walls.
  As well as moving orthogonally, the rat may move diagonally if there's a 2x2 space in which to do so, 
  but may never pass diagonally through a round wall-spot on the corner of a cell.
- The rat may only pass directly through a purple arrow if moving in the direction the arrow is pointing.
- An arrow always points to the smaller of the two digits it sits between.

SCHRÖDINGER LINE EMITTERS: 
- Scattered around the lab are colored line emitters. 
  The path segment connecting two Emitters (including the emitter cells themselves) 
  must follow the rules of both emitters (see below). The emitters at both ends 
  of a segment must have different colors.
Clarification: The two segments extending from a single emitter are considered independent lines 
(For example, a digit may appear on both sides of a renban emitter without violating its rule). 
The segments extending from a region sum emitter may have different sums.

EMITTER RULES:
- RENBAN (purple): The line contains a set of consecutive digits (not necessarily in order).
- NABNER (yellow): No two digits are consecutive, and no digit repeats.
- MODULAR (teal): Every set of three consecutive digits must include one digit from {1,4,7}, one from {2,5,8}, and one from {3,6,9}.
- ENTROPY (peach): Every set of three consecutive digits must include one digit from {1,2,3}, one from {4,5,6}, and one from {7,8,9}.
- REGION SUM (blue): The sum of the digits on the line is the same in every 3×3 box it passes through. The line has to cross at least one box-border.
- TEN SUM (gray): The line can be divided into one or more non-overlapping segments that each sum to 10.
"""

# sudoku cells
cells = cp.intvar(1,9, shape=(9,9), name="values")
# path indices 0 if not on path else the var indicates at what point the rat passes this cell
path = cp.intvar(0, 81, shape=(9,9), name="path")
# list of cells (before, after) on path with indices and value, used for emitter constraint. (Due to the structure of the puzzle, we know there will never be more than 20 cells between two emitters on the path. This in itself is hard to know, but for efficiency reasons we reduce this amount using 'expert knowledge'.)
sequence = cp.intvar(-1, 9, shape=(9,9,6,20), name="sequence")  # for each cell: before value, before row, before column, after value, after row, after column | -1 if not applicable

# inducers for diagonal walls (prevent diagonal self-crossings)
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
    """
    the path can not pass from idx2 to idx1, and the value in idx1 must be greater than the value in idx2
    
    Args:
        idx1 (tuple): tuple representing the indices of cell 1
        idx2 (tuple): tuple representing the indices of cell 2
        
    Returns:
        cpmpy constraint enforcing the gate rule
    """
    r1, c1 = idx1
    r2, c2 = idx2
    return cp.all([cells[r1,c1] > cells[r2,c2], path[r1,c1] - 1 != path[r2,c2]])

def get_reachable_neighbours(row, column):
    """
    from a cell get the indices of its reachable neighbours
    
    Args:
        row (int): row index
        column (int): column index
        
    Returns:
        list: list of tuples representing the indices of reachable neighbours
    """
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
    """
    Add constraints to ensure the path is valid according to the walls and emitters.
    Enforce Sequence values.
    
    Args:
        path (cp.intvar): cpmpy variable representing the path indices
        
    Returns:
        list: list of cpmpy constraints enforcing the path validity rules
    """
    constraints = []
    for r in range(path.shape[0]):
        for c in range(path.shape[0]):
            neighbours = get_reachable_neighbours(r, c)
            non_emitter_neighbours = [n for n in neighbours if tuple(n) not in EMITTERS]
            emitter_neighbours = [n for n in neighbours if tuple(n) in EMITTERS]
            if (r,c) == (2,8):
                # The path starts on emitter. It doesn't have any previous cells so the first 3 vectors in the sequence must be fully -1. As for the the last 3, they are taken from the neighbour. Vice versa also applies.
                constraints.append(cp.all([cp.all(sequence[r,c,:3].flatten() == -1)])) # no previous cells in sequence
                constraints.append(cp.any([cp.all([path[nr,nc] == 2, # next cell must be the second on the path
                                                   sequence[r,c,3,0] == cells[nr,nc], # the immediate next cell value must be that of the neighbour
                                                   sequence[r,c,4,0] == nr, # the immediate next cell row must be that of the neighbour
                                                   sequence[r,c,5,0] == nc, # the immediate next cell column must be that of the neighbour
                                                   cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                                   cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                                   cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                                   sequence[nr,nc,0,0] == cells[r,c], # the immediate previous cell value of the neighbour must be that of the current cell
                                                   sequence[nr,nc,1,0] == r, # the immediate previous cell row of the neighbour must be that of the current cell
                                                   sequence[nr,nc,2,0] == c, # the immediate previous cell column of the neighbour must be that of the current cell
                                                   cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                                   cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                                   cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19]) # the rest of the sequence vectors must be carried over from the current cell
                                                   ]) for nr, nc in neighbours]))
            elif (r,c) == (6,6):
                # nothing comes after this emitter on the path
                constraints.append(cp.all([cp.all(sequence[r,c,3:].flatten() == -1)]))
                constraints.append(path[r,c] == cp.max(path))
            else:
                # for any pathcell, the next pathcell must always be reachable
                constraints.append((path[r,c] != 0).implies(cp.any([path[neighbour[0], neighbour[1]] == path[r,c] + 1 for neighbour in neighbours])))
                if (r,c) not in EMITTERS:
                    # carry over sequence values from non-emitter to non-emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies( # non-emitter neighbour that is next on path
                        cp.all([sequence[r,c,3,0] == cells[nr,nc], # the immediate next cell value must be that of the neighbour
                                sequence[r,c,4,0] == nr, # the immediate next cell row must be that of the neighbour
                                sequence[r,c,5,0] == nc, # the immediate next cell column must be that of the neighbour
                                cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                sequence[nr,nc,0,0] == cells[r,c], # the immediate previous cell value of the neighbour must be that of the current cell
                                sequence[nr,nc,1,0] == r, # the immediate previous cell row of the neighbour must be that of the current cell
                                sequence[nr,nc,2,0] == c, # the immediate previous cell column of the neighbour must be that of the current cell
                                cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19]) # the rest of the sequence vectors must be carried over from the current cell
                                ])) for nr,nc in non_emitter_neighbours]))
                    # carry over sequence values from non-emitter to emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies( # emitter neighbour that is next on path
                        cp.all([sequence[r,c,3,0] == cells[nr,nc], # the immediate next cell value must be that of the neighbour
                                sequence[r,c,4,0] == nr, # the immediate next cell row must be that of the neighbour
                                sequence[r,c,5,0] == nc, # the immediate next cell column must be that of the neighbour
                                cp.all(sequence[r,c,3,1:] == -1), # no continuation after emitter
                                cp.all(sequence[r,c,4,1:] == -1), # no continuation after emitter
                                cp.all(sequence[r,c,5,1:] == -1), # no continuation after emitter
                                sequence[nr,nc,0,0] == cells[r,c], # the immediate previous cell value of the neighbour must be that of the current cell
                                sequence[nr,nc,1,0] == r, # the immediate previous cell row of the neighbour must be that of the current cell
                                sequence[nr,nc,2,0] == c, # the immediate previous cell column of the neighbour must be that of the current cell
                                cp.all(sequence[nr,nc,0,1:] == sequence[r,c,0,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                cp.all(sequence[nr,nc,1,1:] == sequence[r,c,1,:19]), # the rest of the sequence vectors must be carried over from the current cell
                                cp.all(sequence[nr,nc,2,1:] == sequence[r,c,2,:19]) # the rest of the sequence vectors must be carried over from the current cell
                                ])) for nr,nc in emitter_neighbours]))
                else:
                    # carry over sequence values from emitter to non-emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies( # non-emitter neighbour that is next on path
                        cp.all([sequence[r,c,3,0] == cells[nr,nc], # the immediate next cell value must be that of the neighbour
                                sequence[r,c,4,0] == nr, # the immediate next cell row must be that of the neighbour
                                sequence[r,c,5,0] == nc, # the immediate next cell column must be that of the neighbour
                                cp.all(sequence[r,c,3,1:] == sequence[nr,nc,3,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                cp.all(sequence[r,c,4,1:] == sequence[nr,nc,4,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                cp.all(sequence[r,c,5,1:] == sequence[nr,nc,5,:19]), # the rest of the sequence vectors must be carried over from the neighbour
                                sequence[nr,nc,0,0] == cells[r,c], # the immediate previous cell value of the neighbour must be that of the current cell
                                sequence[nr,nc,1,0] == r, # the immediate previous cell row of the neighbour must be that of the current cell
                                sequence[nr,nc,2,0] == c, # the immediate previous cell column of the neighbour must be that of the current cell
                                cp.all(sequence[nr,nc,0,1:] == -1), # no continuation before emitter
                                cp.all(sequence[nr,nc,1,1:] == -1), # no continuation before emitter
                                cp.all(sequence[nr,nc,2,1:] == -1) # no continuation before emitter
                                ])) for nr,nc in non_emitter_neighbours]))
                    # carry over sequence values from emitter to emitter
                    constraints.append(cp.all([((path[r,c] != 0) & (path[r,c] + 1 == path[nr,nc])).implies( # emitter neighbour that is next on path
                        cp.all([sequence[r,c,3,0] == cells[nr,nc], # the immediate next cell value must be that of the neighbour
                                sequence[r,c,4,0] == nr, # the immediate next cell row must be that of the neighbour
                                sequence[r,c,5,0] == nc, # the immediate next cell column must be that of the neighbour
                                cp.all(sequence[r,c,3,1:] == -1), # no continuation after emitter
                                cp.all(sequence[r,c,4,1:] == -1), # no continuation after emitter
                                cp.all(sequence[r,c,5,1:] == -1), # no continuation after emitter
                                sequence[nr,nc,0,0] == cells[r,c], # the immediate previous cell value of the neighbour must be that of the current cell
                                sequence[nr,nc,1,0] == r, # the immediate previous cell row of the neighbour must be that of the current cell
                                sequence[nr,nc,2,0] == c, # the immediate previous cell column of the neighbour must be that of the current cell
                                cp.all(sequence[nr,nc,0,1:] == -1), # no continuation before emitter
                                cp.all(sequence[nr,nc,1,1:] == -1), # no continuation before emitter
                                cp.all(sequence[nr,nc,2,1:] == -1) # no continuation before emitter
                                ])) for nr,nc in emitter_neighbours]))

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
    """
    check if two cells are in the same 3x3 region
    
    Args:
        r1 (int): row index of cell 1
        c1 (int): column index of cell 1
        r2 (int): row index of cell 2
        c2 (int): column index of cell 2
        
    Returns:
        cpmpy constraint enforcing the same region rule; will evaluate to BoolVal(True) if both cells are in the same 3x3 region, else BoolVal(False)
    """
    return cp.all([(r1 // 3 == r2 // 3), (c1 // 3 == c2 // 3)])

def renban(array, rs, cs):
    """
    the line must form a set of consecutive non repeating digits
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        
    Returns:
        list: list of cpmpy constraints enforcing the renban rule
    """
    cons = []
    renban_emitters = [k for k, v in EMITTERS.items() if v == 1]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in renban_emitters])))  # if the next cell is -1, the current cell can't be a renban emitter
    cons.append(cp.all([cp.AllDifferentExceptN(array, -1), cp.max(array) - cp.min([a + 10*(a == -1) for a in array]) + 1 == len(array) - cp.Count(array, -1)]))
    return cons

def nabner(array, rs, cs):
    """
    no two digits are consecutive and no digit repeats
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        
    Returns:
        list: list of cpmpy constraints enforcing the nabner rule
    """
    cons = []
    nabner_emitters = [k for k, v in EMITTERS.items() if v == 2]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in nabner_emitters])))  # if the next cell is -1, the current cell can't be a nabner emitter

    for i in range(len(array) - 1):
        cons.append((array[i] != -1).implies(cp.all([cp.abs(array[i] - array[j]) > 1 for j in range(i+1,len(array))])))
    cons.append(cp.AllDifferentExceptN(array, -1))
    return cons

def modular(array, rs, cs):
    """
    every set of 3 consecutive digits must have a different value mod 3
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        
    Returns:
        list: list of cpmpy constraints enforcing the modular rule
    """
    cons = []
    modular_emitters = [k for k, v in EMITTERS.items() if v == 3]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in modular_emitters])))  # if the next cell is -1, the current cell can't be a modular emitter
    arr = cp.intvar(-1, 2, shape=len(array))
    cons.append(cp.all([cp.all(arr == (array % 3)), cp.all([cp.AllDifferentExceptN(arr[i:i+3], -1) for i in range(len(arr) - 2)])]))
    return cons

def entropy(array, rs, cs):
    """
    every set of 3 consecutive digit must have 1 low digit (1-3) and 1 middle digit (4-6) and 1 high digit (7-9)
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        
    Returns:
        list: list of cpmpy constraints enforcing the entropy rule
    """
    cons = []
    entropy_emitters = [k for k, v in EMITTERS.items() if v == 4]

    for i in range(len(array) - 1):
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in entropy_emitters])))  # if the next cell is -1, the current cell can't be an entropy emitter
    cons.append(cp.all([cp.AllDifferentExceptN([(a+2) // 3 for a in array[i:i+3]], 0) for i in range(len(array) - 2)]))
    return cons

def region_sum(array, rs, cs, order):
    """
    box borders divide the line into segments of the same sum
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        order (string): order of the line segment (before or after emitter)
        
    Returns:
        list: list of cpmpy constraints enforcing the region sum rule
    """
    running_sums = cp.intvar(-1, 45, shape=len(array), name=f"running_sums_{rs[0]}_{cs[0]}_{order}")
    region_sum = cp.intvar(1, 45, name=f"region_sum_{rs[0]}_{cs[0]}_{order}")
    cons = []
    region_sum_emitters = [k for k, v in EMITTERS.items() if v == 5]

    cons.append(region_sum == running_sums[0])
    cons.append(running_sums[-1] == array[-1])
    for i in range(len(array)-1):
        cons.append((cp.all([array[i+1] != -1, ~same_region(rs[i], cs[i], rs[i+1], cs[i+1])])).implies(cp.all([running_sums[i] == array[i], region_sum == running_sums[i+1]])))
        cons.append((cp.all([(array[i+1] == -1), (array[i] != -1)])).implies(running_sums[i] == array[i]))
        cons.append((cp.all([array[i+1] != -1, same_region(rs[i], cs[i], rs[i+1], cs[i+1])])).implies(cp.all([running_sums[i] == running_sums[i+1] + array[i]])))
        cons.append((array[i] == -1).implies(running_sums[i] == array[i]))
        # the last cell can not be another region_sum emitter
        cons.append((cp.all([array[i] != -1, array[i+1] == -1])).implies(cp.all([cp.any([rs[i] != er, cs[i] != ec]) for (er, ec) in region_sum_emitters])))  # if the next cell is -1, the current cell can't be a region_sum emitter
    cons.append(cp.Count(running_sums, region_sum) >= 2)  # at least two segments
    return cons

def ten_sum(array, rs, cs, order):
    """
    the line can be divided into one or more non-overlapping segments that each sum to 10
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells in the line segment
        rs (list): list of row indices for the cells in the line segment
        cs (list): list of column indices for the cells in the line segment
        order (string): order of the line segment (before or after emitter)
        
    Returns:
        list: list of cpmpy constraints enforcing the ten sum rule
    """
    running_sums = cp.intvar(-1, 45, shape=len(array), name=f"ten_running_sums_{rs[0]}_{cs[0]}_{order}")
    splits = cp.boolvar(shape=len(array)-1, name=f"splits_{rs[0]}_{cs[0]}_{order}")
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
    return cons

def activate_lines(sequence):
    """
    Iterate over all emitters and activate their respective constraints based on if they are on the path
    
    Args:
        sequence (cp.intvar): cpmpy variable representing the sequence of cells (values, rows, columns) before and after each cell
    
    Returns:
        list: list of cpmpy constraints enforcing the emitter rules
    """
    constraints = []

    for er,ec in EMITTERS:
        line = EMITTERS[(er,ec)]
        before = np.concatenate(([cells[er,ec]], sequence[er,ec,0,:]))  # before emitter
        after = np.concatenate(([cells[er,ec]], sequence[er,ec,3,:]))   # after emitter
        before_rs = np.concatenate(([er], sequence[er,ec,1,:]))  # before emitter rows
        before_cs = np.concatenate(([ec], sequence[er,ec,2,:]))  # before emitter columns
        after_rs = np.concatenate(([er], sequence[er,ec,4,:]))   # after emitter rows
        after_cs = np.concatenate(([ec], sequence[er,ec,5,:]))   # after emitter columns
        
        cons_before = []
        cons_after = []

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
            cons_before = region_sum(before, before_rs, before_cs, "before")
            cons_after = region_sum(after, after_rs, after_cs, "after")
        elif line == 6:
            cons_before = ten_sum(before, before_rs, before_cs, "before")
            cons_after = ten_sum(after, after_rs, after_cs, "after")
        for c in cons_before:
            constraints.append((path[er,ec] > 1).implies(c))
        for c in cons_after:
            constraints.append((cp.all([0 < path[er,ec],path[er,ec] < cp.max(path)])).implies(c))
    return constraints

m = cp.Model(

    # path givens
    path[2,8] == 1,
    cp.max(path) == path[6,6],

    # no duplicate path indices
    cp.AllDifferentExcept0(path),

    # gates
    gate((0,4), (1,4)),
    gate((2,3), (2,2)),
    gate((6,6), (7,6)),
    
    # emitter constraints
    activate_lines(sequence),
    # general path constraints
    path_valid(path)
)

def regroup_to_blocks(grid):
    """
    Regroup the 9x9 grid into its 3x3 blocks.
    
    Args:
        grid (cp.intvar): A 9x9 grid of integer variables.
        
    Returns:
        list: A list of 9 lists, each containing the elements of a 3x
    """
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
    """Print the start grid with walls and emitters."""
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

print("Number of constraints:", len(m.constraints))

sol = m.solve()

print("The solution is:")
print(cells.value())
print("The path is (0 if not on the path):")
print(path.value())

assert (cells.value() == [[9, 7, 3, 5, 6, 4, 2, 8, 1],
                          [1, 6, 5, 8, 3, 2, 9, 7, 4],
                          [8, 4, 2, 9, 1, 7, 5, 6, 3],
                          [3, 8, 4, 2, 9, 6, 7, 1, 5],
                          [7, 1, 9, 3, 5, 8, 6, 4, 2],
                          [2, 5, 6, 4, 7, 1, 3, 9, 8],
                          [6, 2, 1, 7, 4, 3, 8, 5, 9],
                          [4, 9, 8, 6, 2, 5, 1, 3, 7],
                          [5, 3, 7, 1, 8, 9, 4, 2, 6]]).all()

assert (path.value() == [[ 0,  0,  0, 16, 15, 14,  0,  0,  0],
                         [ 0, 18, 17,  0, 12, 13,  0,  0,  0],
                         [20, 19,  0,  0, 11,  0,  0,  0,  1],
                         [21,  0,  0,  0, 10,  0,  0,  0,  2],
                         [22,  0,  0,  0,  9,  0,  5,  4,  3],
                         [23,  0,  0,  0,  8,  6,  0,  0,  0],
                         [24,  0,  0,  0, 31,  7, 36, 35,  0],
                         [25,  0, 29, 30,  0, 32, 34,  0,  0],
                         [26, 27, 28,  0,  0, 33,  0,  0,  0]]).all()