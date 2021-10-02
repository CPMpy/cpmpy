# Implementation of LP/CP Contest 2021
# Problem 1, Crazy Frog Puzzle
# https://github.com/alviano/lpcp-contest-2021/tree/main/problem-1
#
# Tias Guns, 2021

import requests # to get the data
import numpy as np
from cpmpy import *

# Nice example of planning-as-sat (for fixed step length),
# 3D tensors, numpy indexing, and vectorized constraints
#
# S = Grid size (shape=(S,S))
# I = start row position of frog (offset 1)
# J = start column position of frog (offset 1)
# grid = S*S shaped; 0 if insect, 1 if obstacle
def model_frog(S,I,J,grid):
    m = Model()

    N = (grid == 0).sum() # nr of steps
    print("Steps: ",N)

    
    # State-spaces
    # indicator matrix of frog position (N*S*S bool)
    frogpos = boolvar(shape=(N,S,S), name="frog")
    # indicator matrix of 'free' places (N*S*S bool)
    freepos = boolvar(shape=(N,S,S), name="free")    

    # Start state: frog and free positions
    frogstart = np.zeros(shape=(S,S), dtype=int)
    frogstart[I-1,J-1] = 1 # omg, its offset 1...
    m += (frogpos[0] == frogstart)
    m += (freepos[0] == (1-grid)) 
    
    # Stop state, frog at last free item
    m += (freepos[-1] == frogpos[-1])
    
    # Transition: define next based on prev + invariants
    def transition(m, prev_frog, prev_free, next_frog, next_free):
        # next free is previous free except previous frog
        m += (next_free == (prev_free & ~prev_frog))
    
        # next frog: for each position, determine its reachability
        # only reachable if(implies) free and prev_frog on same row or column
        for i in range(S):
            for j in range(S):
                m += next_frog[i,j].implies(next_free[i,j] & \
                                            (any(prev_frog[i,:]) | any(prev_frog[:,j])))
            
        # invariant: exactly one frog
        m += (next_frog.sum() == 1)
    # apply transitions (0,1) (1,2) (2,3) ...
    for i in range(1, N):
        transition(m, frogpos[i-1], freepos[i-1], frogpos[i], freepos[i])
    
    return (m, frogpos, freepos)

# Let's get the data from github directly
# n=1..10; up to '4' its fast to solve, afterwards not at all
def get_instance(n):
    response = requests.get(f"https://github.com/alviano/lpcp-contest-2021/raw/main/problem-1/instance.{n}.in")
    response.raise_for_status()
    
    # split on newline (and remove last empty newline)
    response_rows = response.text.rstrip().split('\n')
    
    # first row: 4 3 3
    S,I,J = np.array(response_rows[0].split(' '), dtype=int) # first row
    # remaining rows: the grid
    grid = np.array([r.split(' ') for r in response_rows[1:]], dtype=int) # all but first
    
    return (S,I,J,grid)

if __name__ == "__main__":
    inst = 1
    print(f"Getting instance {inst}")
    (S,I,J,grid) = get_instance(inst)
    print(grid, S,I,J)

    (model, frogpos, freepos) = model_frog(S,I,J,grid)
    print("Model ready...")

    model.solve()
    print(model.status())

    # pretty print
    for i in range(frogpos.shape[0]):
        # frog will be nr '2'
        print(frogpos[i].value() + freepos[i].value())
