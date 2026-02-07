import cpmpy as cp
import numpy as np

""" 
Puzzle source: https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000ONO

Rules:

- Place six 3x2 boxes into the grid. They can be placed either horizontally or vertically, but cannot overlap.

- Place the digits 1-6 in each box so that these digits do not repeat in a row, column or box. 
  Any cell outside a box contains a zero.

The following line rules apply to all digits along a line, (even the zeroes!)

      - (RED) ALTERNATING PARITY: Adjacent digits on a red line contain one even digit and one odd digit.
      - (TURQUOISE) SAME DIFFERENCE: Adjacent digits on a turquoise line have the same difference, to be determined.
      - (YELLOW) PALINDROME: A yellow line reads the same in either direction.
      - (GREEN) GERMAN WHISPER: Adjacent digits on a green line have a difference of at least FIVE.
      - (DARK BLUE) REGION SUM: Box borders divide the darker blue line into segments. 
                                The digits on each segment have the same sum.
      - (LAVENDER) ZIPPER: Two digits that are an equal distance from the central spot on a lavender line
                           sum to the digit on that central spot.
      - (GREY) SLOW THERMO: Starting at the end with the bulb and moving along it, 
                            digits on a slow thermo always increase or stay the same (but never decrease.)
"""

# w, h of block
w, h = 3, 2

# sudoku cells
cells = cp.intvar(0,6, shape=(9,9))

# orientations of blocks
orientations = cp.boolvar(shape=6)

# starts and ends
starts = cp.intvar(0, 7, shape=(6, 2))
ends = cp.intvar(2, 9, shape=(6, 2))
# block shapes
blocks = cp.intvar(2, 3, shape=(6, 2))

# intvars for cell to block mapping
cell_blocks = cp.intvar(0, 6, shape=(9, 9))

def same_difference(array):
    """
    adjacent cells on the line must all have the same difference
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the same difference rule
    """    
    diff = cp.intvar(0,6, shape=1)
    return cp.all([abs(a-b) == diff for a,b in zip(array, array[1:])])

def zipper(array):
    """
    equidistant cells from the middle, sum to the value in the value in the middle
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the zipper rule
    """
    assert len(array) % 2 == 1
    mid = len(array) // 2
    return cp.all([array[i] + array[len(array)-1-i] == array[mid] for i in range(mid)])

def parity(array):
    """
    adjacent cells on the line must have different parity
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the parity rule
    """
    return cp.all([a % 2 != b % 2 for a,b in zip(array, array[1:])])

def whisper(array):
    """
    adjacent cells on the line must have a difference of at least five (5)
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the whisper rule
    """
    return cp.all([abs(a-b) >= 5 for a,b in zip(array, array[1:])])

def palindrome(array):
    """
    the line must read the same forwards and backwards
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the palindrome rule
    """
    return cp.all([array[i] == array[len(array)-1-i] for i in range(len(array)//2)])

def slow_thermo(array):
    """
    digits on a slow thermo must increase or stay equal
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        
    Returns:
        cpmpy constraint enforcing the slow thermo rule
    """
    return cp.all([a <= b for a,b in zip(array, array[1:])])

def region_sum(args):
    """
    box borders divide the line into segments of the same sum
    use cell_blocks vars, track sum of segment until adjacent digits belong to different blocks
    
    Args:
        args (list): list of tuples representing the (row, col) indices of the cells on the line
        
    Returns:
        cpmpy constraint enforcing the region sum rule
    """
    # the input in this case is actually the indices which can be used both for the cells vars and the cell_blocks vars

    running_sums = cp.intvar(0, 21, shape=len(args))
    region_sum = cp.intvar(0, 21)
    cons = []
    cons.append(region_sum == running_sums[0])
    cons.append(running_sums[-1] == cells[args[-1,0], args[-1,1]])
    for i in range(len(args)-1):
        cons.append((cell_blocks[args[i, 0], args[i, 1]] != cell_blocks[args[i+1, 0], args[i+1, 1]]).implies(cp.all([running_sums[i] == cells[args[i, 0], args[i, 1]], region_sum == running_sums[i+1]])))
        cons.append((cell_blocks[args[i, 0], args[i, 1]] == cell_blocks[args[i+1, 0], args[i+1, 1]]).implies(cp.all([running_sums[i] == running_sums[i+1] + cells[args[i,0], args[i,1]]])))

    return cp.all(cons)


m = cp.Model(
    # add line constraints
    parity(cells[0,:3]),
    same_difference(np.concatenate((cells[0,3:5], [cells[1,5]], cells[0,6:8]))),
    zipper(np.concatenate((cells[3,5:], cells[2::-1,8]))),
    parity(np.array([cells[1,6], cells[2,6], cells[2,7]])),
    whisper(np.array([cells[1,3], cells[2,2], cells[3,3], cells[2,3]])),
    palindrome(np.array([cells[5,3], cells[6,3], cells[6,4]])),
    palindrome(np.array([cells[5,4], cells[4,4], cells[4,5]])),
    slow_thermo(np.concatenate((cells[8,2::-1], cells[7:4:-1,0], [cells[5,1]], [cells[4,2]], [cells[3,2]]))),
    slow_thermo(np.array([cells[6,2], cells[7,3], cells[8,4]])),
    zipper(np.array([cells[5,6], cells[6,7], cells[7,6]])),
    same_difference(np.array([cells[4,8], cells[5,7], cells[6,8]])),
    same_difference(cells[8,6:]),
    region_sum(np.array([(1,0), (2,0), (3,0), (3,1), (2,1), (1,1)]))
)

for i in range(cells.shape[0]):
    m += cp.AllDifferentExcept0(cells[i,:])
    m += cp.AllDifferentExcept0(cells[:,i])
    
for i in range(6):
    # block orientation
    m += (orientations[i] == 0).implies(blocks[i,0] == w)
    m += (orientations[i] == 1).implies(blocks[i,0] == h)
    m += (orientations[i] == 0).implies(blocks[i,1] == h)
    m += (orientations[i] == 1).implies(blocks[i,1] == w)
    # block starts and ends
    m += starts[i,0] + blocks[i,0] == ends[i,0]
    m += starts[i,1] + blocks[i,1] == ends[i,1]

# block constraints
# Create mapping of value, block to unique ID, then all different. This also ensures there is just 6 cells for each block
m += cp.AllDifferentExceptN([cell_blocks[i,j]*6 + cells[i,j]-1 for i in range(9) for j in range(9)], -1)

# Use start and end to restrict cell_blocks values
for i in range(9):
    for j in range(9):
        for b in range(6):
            in_block = cp.boolvar()
            in_block = cp.all([
                cp.all([i >= starts[b,0], i < ends[b,0]]),
                cp.all([j >= starts[b,1], j < ends[b,1]])]
            )
            
            # this in also enforces the no_overlap2d constraint in a non-global way, but it is useful to actually have the mapping
            m += (in_block).implies(cp.all([cell_blocks[i,j] == b+1, cells[i,j] != 0]))
            m += (~in_block).implies(cell_blocks[i,j] != b+1)
        m += (cell_blocks[i,j] == 0).implies(cells[i,j] == 0)
        

for i in range(len(starts)-1):
    r1, c1 = starts[i]
    for j in range(i+1, len(starts)):
        r2, c2 = starts[j]
        m += (r1 < r2) | (cp.all([r1 == r2, c1 < c2])) # symmetry breaking, slow

sol = m.solve()
print("The solution is:")
print(cells.value())
print("The blocks are mapped like this:")
print(cell_blocks.value())

assert (cells.value() == [[2, 5, 0, 4, 3, 6, 1, 0, 0],
                          [3, 1, 0, 5, 6, 2, 4, 0, 0],
                          [4, 6, 0, 1, 2, 3, 5, 0, 0],
                          [5, 2, 4, 6, 1, 0, 0, 0, 0],
                          [6, 4, 3, 2, 5, 0, 0, 0, 0],
                          [1, 3, 0, 0, 0, 0, 2, 4, 0],
                          [0, 0, 0, 0, 0, 0, 6, 5, 0],
                          [0, 0, 0, 0, 0, 0, 3, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]]).all()

assert (cell_blocks.value() == [[1, 1, 0, 2, 2, 3, 3, 0, 0],
                                [1, 1, 0, 2, 2, 3, 3, 0, 0],
                                [1, 1, 0, 2, 2, 3, 3, 0, 0],
                                [4, 4, 5, 5, 5, 0, 0, 0, 0],
                                [4, 4, 5, 5, 5, 0, 0, 0, 0],
                                [4, 4, 0, 0, 0, 0, 6, 6, 0],
                                [0, 0, 0, 0, 0, 0, 6, 6, 0],
                                [0, 0, 0, 0, 0, 0, 6, 6, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0]]).all()