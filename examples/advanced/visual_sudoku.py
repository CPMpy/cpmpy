#%%
"""
Visual Sudoku problem in CPMpy
"""

from cpmpy import * 
import numpy as np 
from scipy.stats import poisson
import torch 
from torchvision import datasets, transforms

from cpmpy.solvers.ortools import CPM_ortools
from cpmpy.solvers.solver_interface import ExitStatus

import matplotlib.pyplot as plt

PRECISION = 1e-4
# TODO: Add verbosity to python script

def model_sudoku(grid):
    '''
        TODO: Basic doc.
    '''
    # TODO: Instead of returning constraints, you should return a model
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
        #TODO: either give model or ask to build new model
        # TODO: IS given may be recomputed everytime for clarity :)
        cons = [*constraints]
        # Baseline: take most likely digit as deterministic input
        givens = np.argmax(logprobs, axis=2)
        cons += [puzvar[is_given] == givens[is_given]]
        model = Model(cons)

        if model.solve():
            return puzvar.value()
        else:
            return np.zeros_like(puzvar)

# TODO: REname functions hybrid 1 / hybrid 2
def solve_vizsudoku_hybrid1(puzvar, constraints, logprobs, is_given):
    '''
        TODO: Please explain difference between hybrid 1 and hybrid 2
    '''
    # TODO: constraints => cpmpy model
    # TODO: IS given may be recomputed everytime for clarity :)
    #puzvar, constraints = sudoku_model(is_given)
    # objective function: max log likelihood of all digits

    # Emilio: Why do you need to devide by precision ?
    lprobs = np.array(-logprobs/PRECISION).astype(int)
    obj = sum(Element(lp, v) for lp,v in zip(lprobs[is_given], puzvar[is_given]))
    model = Model(constraints, minimize=obj)

    if model.solve():
        return puzvar.value()
    else:
        return np.zeros_like(puzvar)

def is_unique(solution, is_given):
    # TODO: constraints => cpmpy model
    # TODO: IS given may be recomputed everytime for clarity :)
    puzvar, constraints = model_sudoku(solution)
    constraints += [puzvar[is_given] == solution[is_given]]
    # forbid current solution 
    constraints += [any((puzvar != solution).flatten())] #FIXME auto-flatten 2d dvar arrays?
    model= CPM_ortools(Model(constraints))
    return not model.solve(stop_after_first_solution=True)

# TODO: REname functions hybrid 1 / hybrid 2
def solve_vizsudoku_hybrid2(puzvar, constraints, logprobs, is_given, max_iter=10):
    # TODO: create a new sudoku model/give one as argument here, instead of reusing the constraints for understandibility
    # TODO: IS given may be recomputed everytime for clarity :)
    #puzvar, constraints = sudoku_model(is_given)
    cons = [*constraints]
    solution = solve_vizsudoku_hybrid1(puzvar, cons, logprobs, is_given)
    i = 0
    while not is_unique(solution, is_given):
        if i == max_iter:
            break 
        # forbid current solution
        # TODO: see enumerating solutions for cpmpy:
        # https://cpmpy.readthedocs.io/en/latest/multiple_solutions.html
        cons += [any(puzvar[is_given] != solution[is_given])]
        solution = solve_vizsudoku_hybrid1(puzvar, cons, logprobs, is_given)
        i += 1
    # TODO: remove print(i)?
    print(i)
    return solution


# sample a dataset index for each non-zero number
def sample_visual_sudoku(sudoku_puzzle):
    '''
        TODO: Missing Documentation
    '''

    N_COLORS = 1 # Only consider shades of grey
    IMAGE_WIDTH = 28 # MNIST image width
    IMAGE_HEIGHT = 28 # MNIST image height
    # TODO: Here you write for a 9x9 sudoku but you model it in model_sudoku for any kind of grid!
    # Visual Sudoku Tensor init: (9x9 grid) x 1 (black/white) x (image_height x image_width pixels)
    sudoku_torch_dimension = sudoku_puzzle.shape + (N_COLORS, IMAGE_WIDTH, IMAGE_HEIGHT,)

    vizsudoku = torch.zeros(sudoku_torch_dimension, dtype=torch.float32)

    # TODO: explain standard transformation?
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the MNIST data
    testset = datasets.MNIST('.', download=True, train=False, transform=transform)

    # Dictionary (number, corresponding sample images)
    digit_indices = {number: torch.LongTensor(*np.where(testset.targets == number)) for number in range(1,10)}

    # replace all numeric values in sudoku torch tensor grid by an mnist digit
    
    # TODO: Why do you not use np.ndenumerate ? and then for each number in the 
    #       grid you loop select a random image?
    # IS it for having differnet images ? if yes, motivate or comment :)
    # TODO: PROPOSED:
    for idx, val in np.ndenumerate(sudoku_puzzle[sudoku_puzzle > 0]):
        image_tensor = np.random.choice(digit_indices[val])
        vizsudoku[idx] = image_tensor

    ## OLD:
    for val in np.unique(sudoku_puzzle[sudoku_puzzle > 0]):

        val_idx = np.where(sudoku_puzzle == val)
        idx = torch.LongTensor(np.random.choice(digit_indices[val], len(sudoku_puzzle[val_idx])))
        vizsudoku[val_idx] = torch.stack([testset[i][0] for i in idx])

    return vizsudoku



from torch import nn 
import torch.nn.functional as F

class LeNet(nn.Module):
    '''
        TODO: Write doc+ reference to paper
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*16) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_clf(clf_classname, path):
    '''
        TODO: Write doc.
    '''
    net = clf_classname()
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    return net


@torch.no_grad() #TODO: What does this do ?
def predict_proba_sudoku(model, vizsudoku):
    #TODO: Change shape of output to work for differnet kinds of sudoku shapes (4x4, 9x9, ...)
    # reshape from 9x9x1x28x28 to 81x1x28x28
    pred = model(vizsudoku.flatten(0,1))
    # our NN returns 81 probabilistic vector: an 81x10 matrix
    return pred.reshape(9,9,10).detach() # reshape as 9x9x10 tensor for easier visualisation

if __name__ == '__main__':
    '''
        Missing a bit of explanations even small bit
    '''
    e = 0 # empty cell
    sample_sudoku_grid = np.array([
        [e, e, e,  2, e, 5,  e, e, e],
        [e, 9, e,  e, e, e,  7, 3, e],
        [e, e, 2,  e, e, 9,  e, 6, e],

        [2, e, e,  e, e, e,  4, e, 9],
        [e, e, e,  e, 7, e,  e, e, e],
        [6, e, 9,  e, e, e,  e, e, 1],

        [e, 8, e,  4, e, e,  1, e, e],
        [e, 6, 3,  e, e, e,  e, 8, e],
        [e, e, e,  6, e, 8,  e, e, e]
    ])

    # Emilio: Missing a bit of context here ?
    vs = sample_visual_sudoku(sample_sudoku_grid)

    # TODO: (Minor) Does not work from outside of folder + what is clf ? (classifier?)
    model = load_clf(LeNet,'lenet_mnist_e15.pt')

    # (log)probabilities for each cell
    logprobs = predict_proba_sudoku(model, vs)

    # maximum likelihood class 
    ml_digits = np.argmax(logprobs, axis=-1)

    # Emilio:  dvar ? decision vars (puzzle_vars ?)
    dvar, sudoku_model = model_sudoku(sample_sudoku_grid)

    # Emilio: sol3 ? where is sol 1 and 2?
    sol3 = solve_vizsudoku_hybrid2(dvar, sudoku_model,logprobs, sample_sudoku_grid)

    sudoku_model += [dvar[sample_sudoku_grid > 0] == sample_sudoku_grid[sample_sudoku_grid > 0]]

    # TODO: why do you need to create a new model ?
    # old: model = Model(cons)

    model.solve()
    solution = dvar.value()

    # TODO: pretty printing ?
    print('True solution', *solution)
    print('hybrid model output', *sol3)

# %%

## OLD!
# puzzle = np.array(
#         [[0,0,0, 2,0,5, 0,0,0],
#         [0,9,0, 0,0,0, 7,3,0],
#         [0,0,2, 0,0,9, 0,6,0],
#         [2,0,0, 0,0,0, 4,0,9],
#         [0,0,0, 0,7,0, 0,0,0],
#         [6,0,9, 0,0,0, 0,0,1],
#         [0,8,0, 4,0,0, 1,0,0],
#         [0,6,3, 0,0,0, 0,8,0],
#         [0,0,0, 6,0,8, 0,0,0]]
#     )


# vs = sample_visual_sudoku(puzzle)

# model = load_clf(LeNet,'lenet_mnist_e15.pt' )
# # (log)probabilities for each cell
# logprobs = predict_proba_sudoku(model, vs)
# is_given = puzzle > 0
# # maximum likelihood class 
# ml_digits = np.argmax(logprobs, axis=-1)
# dvar, cons = sudoku_model(puzzle)
# sol3 = solve_vizsudoku_hybrid2(dvar, cons,logprobs, is_given)
# cons += [dvar[is_given] == puzzle[is_given]]
# model = Model(cons) 
# model.solve()
# solution = dvar.value()

# print('True solution', *solution)
# print('hybrid model output', *sol3)