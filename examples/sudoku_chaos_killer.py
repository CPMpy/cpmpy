import cpmpy as cp
import numpy as np
import time

# This CPMpy example solves a sudoku by KNT, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=0009KE

SIZE = 9

# sudoku cells
cell_values = cp.intvar(1,SIZE, shape=(SIZE,SIZE))
# decision variables for the regions
cell_regions = cp.intvar(1,SIZE, shape=(SIZE,SIZE))


def killer_cage(idxs, values, regions, total):
    # the sum of the cells in the cage must equal the total
    # all cells in the cage must be in the same region
    constraints = []
    constraints.append(cp.sum([values[r, c] for r,c in idxs]) == total)
    constraints.append(cp.AllEqual([regions[r, c] for r,c in idxs]))
    return constraints

def get_neighbours(r, c):
    # a cell must be orthogonally adjacent to a cell in the same region

    # check if on top row
    if r == 0:
        if c == 0:
            return [(r, c+1), (r+1, c)]
        elif c == SIZE-1:
            return [(r, c-1), (r+1, c)]
        else:
            return [(r, c-1), (r, c+1), (r+1, c)]
    # check if on bottom row
    elif r == SIZE-1:
        if c == 0:
            return [(r, c+1), (r-1, c)]
        elif c == SIZE-1:
            return [(r, c-1), (r-1, c)]
        else:
            return [(r, c-1), (r, c+1), (r-1, c)]
    # check if on left column
    elif c == 0:
        return [(r-1, c), (r+1, c), (r, c+1)]
    # check if on right column
    elif c == SIZE-1:
        return [(r-1, c), (r+1, c), (r, c-1)]
    # check if in the middle
    else:
        return [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    
def connected(idx1, idx2, regions, manhattan_allowance=SIZE-1, checked=None):
    if checked is None:
        checked = []
    # the two cells must be in the same region
    r1, c1 = idx1
    r2, c2 = idx2
    
    # print(f"connected({idx1}, {idx2}, {manhattan_allowance})")
    
    manhattan_distance = abs(r1 - r2) + abs(c1 - c2)
    # print(f"manhattan_distance: {manhattan_distance}")
    
    if (manhattan_distance > manhattan_allowance):
        # print(f"manhattan_distance > manhattan_allowance")
        return cp.BoolVal(False)
    else:
        if (manhattan_distance == 1):
            return cp.BoolVal(True)
        else:
            constraints = []
            neighbours = get_neighbours(r1, c1)
            checked.append((r1, c1))
            for n in neighbours:
                # print(f"n: {n}")
                r, c = n
                if n in checked:
                    continue
                new_constraints = connected(n, idx2, regions, manhattan_allowance - 1, list(checked))
                # print(f"new_constraints: {new_constraints}")
                # print(new_constraints)
                constraints.append(cp.all([cell_regions[r1, c1] == cell_regions[r, c], new_constraints]))
                # print(constraints)
            return cp.any(constraints)
    

m = cp.Model(
    killer_cage(np.array([[0,0],[0,1],[1,1]]), cell_values, cell_regions, 18),
    killer_cage(np.array([[1,0],[2,0],[2,1]]), cell_values, cell_regions, 8),
    killer_cage(np.array([[3,0],[3,1],[3,2],[2,2],[1,2]]), cell_values, cell_regions, 27),
    killer_cage(np.array([[4,0],[4,1]]), cell_values, cell_regions, 17),
    killer_cage(np.array([[5,0],[5,1]]), cell_values, cell_regions, 8),
    killer_cage(np.array([[1,3],[1,4]]), cell_values, cell_regions, 6),
    killer_cage(np.array([[0,7],[0,8]]), cell_values, cell_regions, 4),
    killer_cage(np.array([[2,3],[3,3],[3,4],[4,3]]), cell_values, cell_regions, 12),
    killer_cage(np.array([[1,5],[2,5],[3,5],[2,4]]), cell_values, cell_regions, 28),
    killer_cage(np.array([[4,4],[4,5],[5,5],[4,6]]), cell_values, cell_regions, 16),
    killer_cage(np.array([[2,8],[3,8]]), cell_values, cell_regions, 15),
    killer_cage(np.array([[3,6],[3,7],[4,7],[4,8]]), cell_values, cell_regions, 17),
    killer_cage(np.array([[6,2],[6,3],[7,3]]), cell_values, cell_regions, 21),
    killer_cage(np.array([[7,2],[8,2]]), cell_values, cell_regions, 5),
    killer_cage(np.array([[8,3],[8,4]]), cell_values, cell_regions, 15),
    killer_cage(np.array([[6,4],[7,4]]), cell_values, cell_regions, 8),
    killer_cage(np.array([[6,5],[7,5],[6,6]]), cell_values, cell_regions, 19),
    killer_cage(np.array([[8,5],[8,6]]), cell_values, cell_regions, 11),
    killer_cage(np.array([[7,6],[7,7],[8,7]]), cell_values, cell_regions, 10),
)

for i in range(cell_values.shape[0]):
    m += cp.AllDifferent(cell_values[i,:])
    m += cp.AllDifferent(cell_values[:,i])


total_cells = SIZE**2
for i in range(total_cells-1):
    r1 = i // SIZE
    c1 = i % SIZE
    
    for j in range(i+1, total_cells):
        r2 = j // SIZE
        c2 = j % SIZE
        if (abs(r1 - r2) + abs(c1 - c2) > SIZE-1): # can never be in the same region because too far apart (redundant constraint but should guide search?)
            m += cell_regions[r1, c1] != cell_regions[r2, c2]
        else:
            m += (cell_regions[r1, c1] == cell_regions[r2, c2]).implies(connected((r1, c1), (r2, c2), cell_regions)) # TODO: make global, i.e. not pairwise connected, but whole region should be connected
        

for r in range(1, SIZE+1):
    # Enforce size for each region
    m += cp.Count(cell_regions, r) == SIZE  # each region must be of size SIZE
    
    
# Create unique IDs for each (value, region) pair to enforce all different
m += cp.AllDifferent([(cell_values[i,j]-1)*SIZE+cell_regions[i,j]-1 for i in range(SIZE) for j in range(SIZE)])

    
# TODO: Symmetry breaking for the regions?
# fix top-left and bottom-right region to reduce symmetry
m += cell_regions[0,0] == 1
m += cell_regions[SIZE-1, SIZE-1] == SIZE
# TODO: symmetry breaking for all regions but slow
# Probably not needed, because we don't really look for multiple solutions anyway
# flat_regions = cell_regions.flatten()
# for i in range(1, SIZE*SIZE):
#     for j in range(i//SIZE+2, SIZE+1):
#         m += (flat_regions[i] == j).implies(cp.any([flat_regions[k] == j-1 for k in range(i)]))  # ensure that regions are assigned in order




# print("The model is:")
# print(m)
print(f"There are {len(m.constraints)} constraints")
start = time.time()
sol = m.solve()
end = time.time()
print(f"Solved: {sol} in {end-start} seconds")
print("The solution is:")
print(cell_values.value())
print("The regions are:")
print(cell_regions.value())