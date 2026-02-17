import cpmpy as cp
import numpy as np


""" 
Puzzle source: https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000KBK

Rules:

- Schrödinger Sudoku: Place one or two digits from 0-6 in every empty cell. 
  Every row, column and box must contain each digit 0-6 exactly once.
- Values: The value of a cell is the sum of its digit(s).
- Line: Values can't repeat on the line.
- Schrödingers Flat Mates: Every cell with value 5 must have a cell with value 1 directly above it 
  and/or a cell with value 9 directly below it.
"""
# This cpmpy example solves a sudoku by gdc, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000KBK

# sudoku cells
cells = cp.intvar(0,6, shape=(6,6))
# schrodinger cells, -1 for no schrodinger
schrodinger = cp.intvar(-1,6, shape=(6,6))
# true values
values = cp.intvar(0,11, shape=(6,6))

def schrodinger_flats(cells, schrodinger, values):
    """
    schrodinger flatmate rules
    
    Args:
        cells (cp.intvar): cpmpy variable representing the sudoku cells
        schrodinger (cp.intvar): cpmpy variable representing the schrodinger cells
        values (cp.intvar): cpmpy variable representing the true values

    Returns:
        list: list of cpmpy constraints enforcing the schrodinger flatmate rules
    """
    
    # go over all cells
    constraints = []
    for r in range(6):
        for c in range(6):
            # get value
            # for normal
            constraints.append((schrodinger[r,c] != -1).implies(values[r,c] == cells[r,c]+schrodinger[r,c]))
            # for schrodinger
            constraints.append((schrodinger[r,c] == -1).implies(values[r,c] == cells[r,c]))
            # symmetry breaking
            constraints.append(schrodinger[r,c] < cells[r,c])
            # flatmate rules
            if r == 0:
                constraints.append((values[r, c] == 5).implies(values[r + 1, c] == 9))
            elif r == 5:
                constraints.append((values[r, c] == 5).implies(values[r - 1, c] == 1))
            else:
                constraints.append((values[r,c] == 5).implies((values[r-1, c] == 1) | (values[r+1,c] == 9)))

    return constraints


m = cp.Model(

    # givens
    cells[0,3]==5,
    schrodinger[0,3]==-1,
    values[0,3]==5,
    cells[5,1]==1,
    schrodinger[5,1]==-1,
    values[5,1]==1,

    # unique line
    cp.AllDifferent(np.array((values[1,2],
                              values[1,3],
                              values[1,4],
                              values[2,1],
                              values[2,2],
                              values[2,4],
                              values[3,1],
                              values[3,5],
                              values[4,2],
                              values[4,5],
                              values[5,3],
                              values[5,4]))),

    schrodinger_flats(cells, schrodinger, values)

)

def regroup_to_blocks(grid):
    """
    Regroup the 6x6 grid into its 2x3 blocks.
    
    Args:
        grid (cp.intvar): cpmpy variable representing the 6x6 sudoku grid
        
    Returns:
        list: list of lists representing the 2x3 blocks of the sudoku grid
    """
    # Create an empty list to store the blocks
    blocks = [[] for _ in range(6)]

    for row_index in range(6):
        for col_index in range(6):
            # Determine which block the current element belongs to
            block_index = (row_index // 2) * 2 + (col_index // 3)
            # Add the element to the appropriate block
            blocks[block_index].append(grid[row_index][col_index])

    return blocks


blocks = regroup_to_blocks(cells)

schrodinger_blocks = regroup_to_blocks(schrodinger)

for i in range(cells.shape[0]):
    m += cp.AllDifferentExceptN(np.concatenate((cells[i,:], schrodinger[i,:])), -1)
    m += cp.AllDifferentExceptN(np.concatenate((cells[:,i], schrodinger[:,i])), -1)
    m += cp.AllDifferentExceptN(np.concatenate((blocks[i], schrodinger_blocks[i])), -1)
    # at least one schrodinger cell per unit (at most covered by all diffs above)
    m += cp.sum(schrodinger[i,:]) != -6
    m += cp.sum(schrodinger[:,i]) != -6
    m += cp.sum(schrodinger_blocks[i]) != -6

sol = m.solve()
print("The solution is:")
print(cells.value())
print("With these schrödinger cells (-1 if not schrödinger):")
print(schrodinger.value())
print("Resulting in these true values:")
print(values.value())

assert (cells.value() == [[6, 3, 1, 5, 2, 4],
                          [2, 4, 5, 6, 0, 1],
                          [4, 5, 3, 0, 1, 6],
                          [1, 6, 0, 2, 4, 5],
                          [5, 0, 6, 1, 3, 2],
                          [3, 1, 2, 4, 6, 0]]).all()
                          
assert (schrodinger.value() == [[ 0, -1, -1, -1, -1, -1],
                                [-1, -1, -1,  3, -1, -1],
                                [-1,  2, -1, -1, -1, -1],
                                [-1, -1, -1, -1, -1,  3],
                                [-1, -1,  4, -1, -1, -1],
                                [-1, -1, -1, -1,  5, -1]]).all()