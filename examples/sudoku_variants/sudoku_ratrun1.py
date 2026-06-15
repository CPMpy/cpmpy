import cpmpy as cp
import numpy as np

"""
Puzzle source: https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000IFI

Rules:

- Normal 6x6 sudoku rules apply; 
  fill the grid with the digits 1-6 so that digits don't repeat in a row, column or 3x2 box (marked with dotted lines.)
- AIM OF EXPERIMENT: Finkz the rat must reach the cupcake by finding a path through the maze. 
  The path will be a snaking line that passes through the centres of cells, without visiting any cell more than once, 
  crossing itself or passing through any thick maze walls.
- As well as moving orthogonally, Finkz may move diagonally if there's a 2x2 space in which to do so, 
  but may never pass diagonally through the rounded end / corner of a wall.
- TEST CONSTRAINT: In this experiment, any two cells that are adjacent along the correct path must sum to a prime number. 
  Also, all the digits that lie anywhere on the correct path within the same 3x2 sudoku box must sum to a prime number too.
"""

# sudoku cells
cells = cp.intvar(1,6, shape=(6,6))
# path indices 0 if not on path else the var indicates at what point the rat passes this cell
path = cp.intvar(0,36, shape=(6,6))

# givens
# vertical walls
V_WALLS = np.array([[0,1,1,0,0],
                    [1,0,0,0,0],
                    [0,1,0,0,1],
                    [1,0,0,1,0],
                    [0,0,0,0,1],
                    [0,0,0,0,0]])
# horizontal walls
H_WALLS = np.array([[0,0,0,0,0,0],
                    [0,1,1,1,1,0],
                    [0,0,0,0,0,0],
                    [0,1,1,0,1,0],
                    [0,0,0,0,1,0]])
# corners walls (to block diagonal movement and future proofing for puzzles that only include walls just made out of a corner)
C_WALLS = np.array([[1,1,1,1,1],
                    [1,1,1,1,1],
                    [1,1,0,1,1],
                    [1,1,1,1,1],
                    [0,0,0,1,1]])

# all reachable primes when summing 2 digits in range [1,6] or when summing 6 different digits in range [1,6]
PRIMES = [2,3,5,7,11,13,17,19]

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
        if column != 5:
            if C_WALLS[row-1, column] == 0:
                reachable_neighbours.append([row - 1, column + 1])
    if row != 5:
        if H_WALLS[row, column] == 0:
            reachable_neighbours.append([row + 1, column])
        if column != 0:
            if C_WALLS[row, column-1] == 0:
                reachable_neighbours.append([row + 1, column - 1])
        if column != 5:
            if C_WALLS[row, column] == 0:
                reachable_neighbours.append([row + 1, column + 1])
    if column != 0:
        if V_WALLS[row, column-1] == 0:
            reachable_neighbours.append([row, column - 1])
    if column != 5:
        if V_WALLS[row, column] == 0:
            reachable_neighbours.append([row, column + 1])
    return reachable_neighbours

def path_valid(path):
    """Check if the path is valid according to the maze rules."""
    constraints = []
    for r in range(path.shape[0]):
        for c in range(path.shape[0]):
            if (r,c) != (4,4): # cupcake is the end of path
                neighbours = get_reachable_neighbours(r, c)

                # for any pathcell, the next pathcell must always be reachable
                constraints.append((path[r,c] != 0).implies(cp.any([path[neighbour[0], neighbour[1]] == path[r,c] + 1 for neighbour in neighbours])))
                # the two must also have a prime sum
                constraints.append(cp.all([cp.any([((path[r,c] != 0) & (path[r,c] + 1 == path[neighbour[0], neighbour[1]])).implies(cells[r,c] + cells[neighbour[0], neighbour[1]] == p) for p in PRIMES]) for neighbour in neighbours]))

    return constraints


def prime_block(block, path):
    """
    sum of pathcells in block must be prime
    
    Args:
        block (list): list of cpmpy variables representing the cells in the block
        path (list): list of cpmpy variables representing the path cells in the block
        
    Returns:
        cpmpy constraint enforcing the prime sum rule for the block
    """
    return cp.any([cp.sum(block[i]*(path[i]!=0) for i in range(len(block))) == p for p in PRIMES])


m = cp.Model(

    # path givens
    path[0,2] == 1,
    cp.max(path) == path[4,4],

    # all totals different
    cp.AllDifferentExcept0(path), # might already be implied by other constraints so check if can be removed

    # valid path traversal
    path_valid(path)

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

path_blocks = regroup_to_blocks(path)

for i in range(cells.shape[0]):
    m += cp.AllDifferent(cells[i,:])
    m += cp.AllDifferent(cells[:,i])
    m += cp.AllDifferent(blocks[i])
    m += prime_block(blocks[i], path_blocks[i])

sol = m.solve()
print("The solution is:")
print(cells.value())
print("The path is (0 if not on the path):")
print(path.value())

assert (cells.value() == [[1, 4, 2, 6, 3, 5],
                          [6, 3, 5, 2, 1, 4],
                          [2, 5, 4, 1, 6, 3],
                          [3, 6, 1, 4, 5, 2],
                          [4, 1, 3, 5, 2, 6],
                          [5, 2, 6, 3, 4, 1]]).all()

assert (path.value() == [[ 0,  0,  1,  0,  0,  0],
                         [ 0,  0,  2,  3,  4,  5],
                         [14, 13,  0, 10,  9,  6],
                         [15, 12, 11,  0,  8,  7],
                         [16, 17,  0, 19, 20,  0],
                         [ 0,  0, 18,  0,  0,  0]]).all()