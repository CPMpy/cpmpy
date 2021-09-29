"""
Visual Sudoku problem in CPMpy
"""

from cpmpy import * 
import numpy as np 
from scipy.stats import poisson

def solve_vizsudoku_cppy(logprobs, is_given, baseline=False):
        n = len(is_given)
        b = np.sqrt(n).astype(int)

        scale = 1e-3

        # decision vars
        puzvar = IntVar(1,n, shape=(n,n), name='cell')
        
        # alldiff constraint
        constraint = []
        constraint += [alldifferent(row) for row in puzvar]
        constraint += [alldifferent(col) for col in puzvar.T]
        for i in range(0,n,b):
            for j in range(0,n,b):
                constraint += [alldifferent(puzvar[i: i +b, j:j+b])]

        # Baseline: take most likely digit as deterministic input
        if baseline:
            givens = np.argmax(logprobs, axis=2)
            constraint += [puzvar[givens>0] == givens[givens>0]]
            model = Model(constraint)
        else:
            # objective function: max log likelihood of all digits
            logprobs = np.array(-logprobs/scale).astype(int)
            #TODO need for broadcast? line would look like:
            #obj = sum(Element([logprobs[is_given], puzvar[is_given]]))
            #obj = sum(Element([logprobs[i,j], puzvar[i,j]]) for i in range(n) for j in range(n) if is_given[i,j])
            terms = []
            for i in range(n):
                for j in range(n): 
                    if is_given[i,j]:
                        terms.append(Element(logprobs[i,j], puzvar[i,j]))
            obj = sum(terms)
            model = Model(constraint, minimize=obj)
        
        #print(model)
        if model.solve():
            return puzvar.value()
        else:
            return np.zeros_like(puzvar)


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
    puzzle = np.array([[0,0,0, 2,0,5, 0,0,0],
                             [0,9,0, 0,0,0, 7,3,0],
                             [0,0,2, 0,0,9, 0,6,0],
                             [2,0,0, 0,0,0, 4,0,9],
                             [0,0,0, 0,7,0, 0,0,0],
                             [6,0,9, 0,0,0, 0,0,1],
                             [0,8,0, 4,0,0, 1,0,0],
                             [0,6,3, 0,0,0, 0,8,0],
                             [0,0,0, 6,0,8, 0,0,0]])
    is_given = puzzle > 0
    probs = cnn_output_simulation(puzzle) 
    probs[puzzle==0][0] = 0.8
    logprobs = np.log(probs)
    print(logprobs[0,3])

    sol4 = solve_vizsudoku_cppy(logprobs, is_given)
    print('cpmpy solution', sol4,sep='\n')