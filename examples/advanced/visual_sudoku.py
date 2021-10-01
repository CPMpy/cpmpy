"""
Visual Sudoku problem in CPMpy
"""

from cpmpy import * 
import numpy as np 
from scipy.stats import poisson
from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.solver_interface import ExitStatus

PRECISION = 1e-3

def sudoku_model(grid):
    n = len(grid)
    b = np.sqrt(n).astype(int)

    # decision vars
    puzvar = IntVar(1,n, shape=grid.shape, name='cell')
    
    # alldiff constraint
    constraints = []
    constraints += [alldifferent(row) for row in puzvar]
    constraints += [alldifferent(col) for col in puzvar.T]
    for i in range(0,n,b):
        for j in range(0,n,b):
            constraints += [alldifferent(puzvar[i: i +b, j:j+b])]
    
    return puzvar, constraints 


def solve_vizsudoku_baseline(puzvar, constraints, logprobs, is_given):
        #puzvar, constraints = sudoku_model(is_given)
        cons = [*constraints]
        # Baseline: take most likely digit as deterministic input
        givens = np.argmax(logprobs, axis=2)
        cons += [puzvar[is_given] == givens[is_given]]
        model = Model(cons)
        if model.solve():
            return puzvar.value()
        else:
            return np.zeros_like(puzvar)


def solve_vizsudoku_hybrid1(puzvar, constraints, logprobs, is_given):
    #puzvar, constraints = sudoku_model(is_given)
    # objective function: max log likelihood of all digits
    lprobs = np.array(-logprobs/PRECISION).astype(int)
    obj = sum(Element(lp, v) for lp,v in zip(lprobs[is_given], puzvar[is_given]))
    #obj = sum(Element(logprobs[i,j], puzvar[i,j]) for i in range(n) for j in range(n) if is_given[i,j])
    model = Model(constraints, minimize=obj)

    if model.solve():
            return puzvar.value()
    else:
        return np.zeros_like(puzvar)

def is_unique(solution, is_given):
    puzvar, constraints = sudoku_model(solution)
    constraints += [all(puzvar[is_given] == solution[is_given])]
    # forbid current solution 
    constraints += [any((puzvar != solution).flatten())] #FIXME auto-flatten 2d dvar arrays?
    model= CPM_ortools(Model(constraints))
    return model.solve(stop_after_first_solution=True) == ExitStatus.UNSATISFIABLE


def solve_vizsudoku_hybrid2(puzvar, constraints, logprobs, is_given, max_iter=10):
    #puzvar, constraints = sudoku_model(is_given)
    solution = solve_vizsudoku_hybrid1(puzvar, constraints, logprobs, is_given)
    i = 0
    while not is_unique(solution, is_given):
        if i == max_iter:
            break 
        # forbid current solution
        constraints += [any(puzvar[is_given] != solution[is_given])]
        solution = solve_vizsudoku_hybrid1(puzvar, constraints, logprobs, is_given)
        i += 1
    print(i)
    return solution


def cnn_output_simulation(puzzle):
    """
    Poisson distribution probability tensor containing a probability 
    score for each possible value in each cell.
    """
    probs = np.zeros(shape=(puzzle.shape + (len(puzzle)+ 1,) ))
    values = np.arange(len(puzzle)+ 1) 
    for r in range(len(puzzle)):
        for c in range(len(puzzle[0])):
            pss = poisson.pmf(values,puzzle[r,c]+1)
            pss[puzzle[r,c]] += 0.01 # break equal probs
            probs[r,c] = pss/np.sum(pss)
    return probs

if __name__ == '__main__':
    puzzle = np.array(
        [[0,0,0, 2,0,5, 0,0,0],
        [0,9,0, 0,0,0, 7,3,0],
        [0,0,2, 0,0,9, 0,6,0],
        [2,0,0, 0,0,0, 4,0,9],
        [0,0,0, 0,7,0, 0,0,0],
        [6,0,9, 0,0,0, 0,0,1],
        [0,8,0, 4,0,0, 1,0,0],
        [0,6,3, 0,0,0, 0,8,0],
        [0,0,0, 6,0,8, 0,0,0]]
    )
    is_given = puzzle > 0
    probs = cnn_output_simulation(puzzle) 
    # add some noise 
    print('truth',np.argmax(probs, axis=-1))
    probs[0,3], probs[8,5] = probs[8,5], probs[0,3]
    probs[puzzle==0][0] = 0.8
    logprobs = np.log(np.maximum(probs, PRECISION))
    print(np.argmax(logprobs, axis=-1))


    dvar, cons = sudoku_model(puzzle)

    sol1 = solve_vizsudoku_baseline(dvar, cons,logprobs, is_given)
    print('vizsudoku baseline solution', sol1,sep='\n')

    sol2 = solve_vizsudoku_hybrid1(dvar, cons,logprobs, is_given)
    print('vizsudoku hybrid1 solution', sol2,sep='\n')

    sol3 = solve_vizsudoku_hybrid2(dvar, cons,logprobs, is_given)
    print('vizsudoku hyrbid2 solution', sol3,sep='\n')