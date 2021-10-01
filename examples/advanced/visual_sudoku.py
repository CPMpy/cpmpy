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
    return not model.solve(stop_after_first_solution=True)


def solve_vizsudoku_hybrid2(puzvar, constraints, logprobs, is_given, max_iter=10):
    #puzvar, constraints = sudoku_model(is_given)
    cons = [*constraints]
    solution = solve_vizsudoku_hybrid1(puzvar, cons, logprobs, is_given)
    i = 0
    while not is_unique(solution, is_given):
        if i == max_iter:
            break 
        # forbid current solution
        cons += [any(puzvar[is_given] != solution[is_given])]
        solution = solve_vizsudoku_hybrid1(puzvar, cons, logprobs, is_given)
        i += 1
    print(i)
    return solution


# %%
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the MNIST data
testset = datasets.MNIST('.', download=True, train=False, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
digit_indices = {k:torch.LongTensor(*np.where(testset.targets == k)) for k in range(1,10)}

# sample a dataset index for each non-zero number
def sample_visual_sudoku(sudoku_p):
    nonzero = sudoku_p > 0
    vizsudoku = torch.zeros((9,9,1,28,28), dtype=torch.float32)
    for val in np.unique(sudoku_p[nonzero]):
        val_idx = np.where(sudoku_p == val)
        idx = torch.LongTensor(np.random.choice(digit_indices[val], len(sudoku_p[val_idx])))
        vizsudoku[val_idx] = torch.stack([testset[i][0] for i in idx])
    return vizsudoku



from torch import nn 
import torch.nn.functional as F
class LeNet(nn.Module):
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
    net = clf_classname()
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    return net

@torch.no_grad()
def predict_proba_sudoku(model, vizsudoku):
    # reshape from 9x9x1x28x28 to 81x1x28x28
    pred = model(vizsudoku.flatten(0,1))
    # our NN return 81 probabilistic vector: an 81x10 matrix
    return pred.reshape(9,9,10).detach() # reshape as 9x9x10 tensor for easier visualisation

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


vs = sample_visual_sudoku(puzzle)

model = load_clf(LeNet,'lenet_mnist_e15.pt' )
# (log)probabilities for each cell
logprobs = predict_proba_sudoku(model, vs)
is_given = puzzle > 0
# maximum likelihood class 
ml_digits = np.argmax(logprobs, axis=-1)
dvar, cons = sudoku_model(puzzle)
sol3 = solve_vizsudoku_hybrid2(dvar, cons,logprobs, is_given)
cons += [dvar[is_given] == puzzle[is_given]]
model = Model(cons) 
model.solve()
solution = dvar.value()

print('True solution', *solution)
print('hybrid model output', *sol3)

# %%

