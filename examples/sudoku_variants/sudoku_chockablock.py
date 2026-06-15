import cpmpy as cp
import numpy as np


"""
Puzzle source: https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=000I2P

Rules:

- Normal sudoku rules apply.
- Unique N-Lines: ALL lines in the grid are N-Lines. Each N-line is composed of one or more non-overlapping sets of adjacent digits, each of which sum to N. Every line has a different N value.
- Coloured lines (not including the grey ones) each have an additional rule:
- Renban (pink): Digits on a pink line form a non-repeating consecutive set, which can be arranged in any order.
- Even Sum Lines (darker blue): All the digits on a darker blue line sum to an even number.
- Prime Lines (purple): Adjacent digits on a purple line sum to a prime number.
- Anti-Kropki Lines (red): No two digits anywhere on the same red line are consecutive, or in a 1:2 ratio (but they may repeat).
- Same Difference Lines (turquoise): Each pair of adjacent digits on a turquoise line have the same difference. This difference must be determined for each turquoise line.
- The grey perimeter can be used for making notes.
"""

# sudoku cells
cells = cp.intvar(1,9, shape=(9,9))
# N-values
totals = cp.intvar(1,27, shape=27)


def n_lines(array, total):
    """"
    N-lines are composed of one or more non-overlapping sets of adjacent cells
    The number of such partitions is exponential, luckily all the lines are of length 3 so we can easily hardcode them
    partitions = [(3), (2,1), (1,2), (1,1,1)]
    For every partition we can check whether each subpartition sums to the same number
    the N variable should be equal to a partition where it holds that each subpartition sums to the same number
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the N-line rule
    """
    all_partitions = [[array], [array[:2], array[2:]], [array[:1], array[1:]], [array[:1], array[1:2], array[2:]]]
    sums = cp.intvar(0,27, shape=4)
    partial_sums = []
    for partition in all_partitions:
        partial_sums.append([cp.sum(subset) for subset in partition])

    constraints = []
    for i, partial_sum in enumerate(partial_sums):
        # keep track of possible N-values for the line
        # note that the trivial partition of just taking all the cells on a line as one set will always be valid
        constraints.append(cp.AllEqual(partial_sum).implies(sums[i] == partial_sum[0]))
        # deactivate partitions not summing to a constant value by setting their sum to 0
        constraints.append(cp.AllEqual(partial_sum) | (sums[i] == 0))

    # the N-value must be equal to one of the values
    constraints.append(cp.any([total == s for s in sums]))
    return cp.all(constraints)

def even_sum(array, total):
    """
    line must sum to an even total
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the even sum rule and the n-line rule
    """
    return cp.all([n_lines(array, total), total % 2 == 0])

def prime_sum(array, total):
    """
    line must sum to a prime total
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the prime sum rule and the n-line rule
    """
    primes = [2, 3, 5, 7, 11, 13, 17]
    pairs = [(a,b) for a,b in zip(array, array[1:])]
    return cp.all([n_lines(array, total), cp.all([cp.any([cp.sum(pair) == p for p in primes]) for pair in pairs])])

def renban(array, total):
    """
    digits on a pink renban form a set of consecutive non repeating digits
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the renban rule and the n-line rule
    """
    return cp.all([n_lines(array, total), cp.AllDifferent(array), cp.max(array) - cp.min(array) == len(array) - 1])

# there are no kropki dots in this sudoku the two following functions are used for the anti-kropki line
def white_kropki(a, b):
    """
    digits separated by a white dot differ by 1
    
    Args:
        a (cp.intvar): cpmpy variable representing cell a
        b (cp.intvar): cpmpy variable representing cell b
        
    Returns:
        cpmpy constraint enforcing the white kropki rule
    """
    return abs(a-b) == 1

def black_kropki(a, b):
    """
    digits separated by a black dot are in a 1:2 ratio
    
    Args:
        a (cp.intvar): cpmpy variable representing cell a
        b (cp.intvar): cpmpy variable representing cell b
        
    Returns:
        cpmpy constraint enforcing the black kropki rule
    """
    return  cp.any([a * 2 == b, a == 2 * b])

def anti_kropki(array, total):
    """
    no pair anywhere on the line may be in a kropki relationship
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the anti-kropki rule and the n-line rule
    """
    all_pairs = [(a, b) for idx, a in enumerate(array) for b in array[idx+1:]]
    constraints = []
    for pair in all_pairs:
        constraints.append(cp.all([~white_kropki(pair[0], pair[1]), ~black_kropki(pair[0], pair[1])]))
    return cp.all([cp.all(constraints), n_lines(array, total)])

def same_difference(array, total):
    """
    adjacent cells on the line must all have the same difference
    
    Args:
        array (cp.intvar): cpmpy variable representing the cells on the line
        total (cp.intvar): cpmpy variable representing the total sum for the line
        
    Returns:
        cpmpy constraint enforcing the same difference rule and the n-line rule
    """
    diff = cp.intvar(0,8, shape=1)
    return cp.all([cp.all([abs(a-b) == diff for a,b in zip(array, array[1:])]), n_lines(array, total)])


def regroup_to_blocks(grid):
    """
    Regroup the 9x9 grid into its 3x3 blocks.
    
    Args:
        grid (cp.intvar): cpmpy variable representing the 9x9 sudoku grid
        
    Returns:
        list: list of lists representing the 3x3 blocks of the sudoku grid
    """
    blocks = [[] for _ in range(9)]

    for row_index in range(9):
        for col_index in range(9):
            # Determine which block the current element belongs to
            block_index = (row_index // 3) * 3 + (col_index // 3)
            # Add the element to the appropriate block
            blocks[block_index].append(grid[row_index][col_index])

    return blocks

m = cp.Model(

    # all totals different
    cp.AllDifferent(totals),

    # nlines
    n_lines(cells[0,3:6], totals[0]),
    n_lines(np.concatenate(([cells[1,3]], cells[:2,2])), totals[1]),
    n_lines(np.concatenate(([cells[0,8]], cells[1,7:])), totals[2]),
    n_lines(np.array((cells[2,7], cells[3,8], cells[2,8])), totals[3]),
    n_lines(cells[8, 2:5], totals[4]),
    n_lines(np.array((cells[6,5],cells[5,6],cells[4,7])), totals[5]),

    # renbans
    renban(np.array((cells[1,1], cells[0,0], cells[0,1])), totals[6]),
    renban(np.array((cells[8,0],cells[8,1], cells[7,2])), totals[7]),
    renban(np.array((cells[7,7], cells[7,6], cells[8,7])), totals[8]),
    renban(cells[6:,8], totals[9]),

    # evensum
    even_sum(np.array((cells[1,0], cells[2,1], cells[2,2])), totals[10]),
    even_sum(np.array((cells[4,0], cells[5,0], cells[6,1])), totals[11]),
    even_sum(np.array((cells[5,1],cells[6,2], cells[7,3])), totals[12]),

    # primes
    prime_sum(np.array((cells[1,5],cells[2,6],cells[3,7])), totals[13]),
    prime_sum(np.array((cells[3,2],cells[4,1],cells[5,2])), totals[14]),
    prime_sum(np.array((cells[3,3],cells[4,2],cells[5,3])), totals[15]),

    # anti-kropki
    anti_kropki(np.array((cells[3,0], cells[2,0], cells[3,1])), totals[16]),
    anti_kropki(np.array((cells[0,6],cells[0,7],cells[1,6])), totals[17]),
    anti_kropki(np.array((cells[2,3],cells[3,4],cells[4,5])), totals[18]),
    anti_kropki(np.array((cells[4,3],cells[5,4],cells[6,3])), totals[19]),
    anti_kropki(np.array((cells[4,4],cells[5,5],cells[6,4])), totals[20]),
    anti_kropki(np.array((cells[4,8],cells[5,8],cells[6,7])), totals[21]),
    anti_kropki(np.array((cells[7,0],cells[6,0],cells[7,1])), totals[22]),
    anti_kropki(np.array((cells[7,4],cells[8,5],cells[8,6])), totals[23]),

    # same difference
    same_difference(np.array((cells[1,4],cells[2,4],cells[3,5])), totals[24]),
    same_difference(np.array((cells[2,5],cells[3,6],cells[4,6])), totals[25]),
    same_difference(np.array((cells[7,5],cells[6,6],cells[5,7])), totals[26]),

)
blocks = regroup_to_blocks(cells)

for i in range(cells.shape[0]):
    m += cp.AllDifferent(cells[i,:])
    m += cp.AllDifferent(cells[:,i])
    m += cp.AllDifferent(blocks[i])

sol = m.solve()
print("The solution is:")
print(cells.value())

assert (cells.value() == [[3, 1, 9, 5, 6, 8, 7, 2, 4],
                         [7, 2, 8, 9, 4, 1, 5, 3, 6],
                         [4, 5, 6, 7, 2, 3, 1, 8, 9],
                         [9, 7, 5, 2, 3, 4, 6, 1, 8],
                         [2, 8, 1, 6, 9, 5, 3, 4, 7],
                         [6, 4, 3, 1, 8, 7, 2, 9, 5],
                         [1, 6, 4, 8, 7, 2, 9, 5, 3],
                         [5, 3, 7, 4, 1, 9, 8, 6, 2],
                         [8, 9, 2, 3, 5, 6, 4, 7, 1]]).all()