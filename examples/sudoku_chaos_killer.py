import cpmpy as cp
import numpy as np
import time

# This CPMpy example solves a sudoku by KNT, which can be found on https://logic-masters.de/Raetselportal/Raetsel/zeigen.php?id=0009KE

SIZE = 9

# sudoku cells
cell_values = cp.intvar(1,SIZE, shape=(SIZE,SIZE))
# decision variables for the regions
cell_regions = cp.intvar(1,SIZE, shape=(SIZE,SIZE))
# regions cardinals
region_cardinals = cp.intvar(1,SIZE, shape=(SIZE,SIZE))


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
    
    neighbours = get_neighbours(r1, c1)
    
    # at least one neighbour must be in the same region with a smaller cardinal number (or the cell itself must have cardinal number 1)
    m += cp.any([cp.all([cell_regions[r1,c1] == cell_regions[r2,c2], region_cardinals[r1, c1] > region_cardinals[r2, c2]]) for r2,c2 in neighbours]) | (region_cardinals[r1,c1] == 1)
        
        

for r in range(1, SIZE+1):
    # Enforce size for each region
    m += cp.Count(cell_regions, r) == SIZE  # each region must be of size SIZE
    
    
# Create unique IDs for each (value, region) pair to enforce all different
m += cp.AllDifferent([(cell_values[i,j]-1)*SIZE+cell_regions[i,j]-1 for i in range(SIZE) for j in range(SIZE)])
# Only 1 cell with cardinal number 1 per region
for r in range(1, SIZE+1):
    m += cp.Count([(region_cardinals[i,j])*(cell_regions[i,j] == r) for i in range(SIZE) for j in range(SIZE)], 1) == 1

    
# Symmetry breaking for the regions
# fix top-left and bottom-right region to reduce symmetry
m += cell_regions[0,0] == 1
m += cell_regions[SIZE-1, SIZE-1] == SIZE




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
print("The region cardinals are:")
print(region_cardinals.value())