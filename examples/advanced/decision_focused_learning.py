import numpy as np
import pyepo
from cpmpy import *
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# CPMPy implementation of the knapsack problem
class KnapsackProblem():
    def __init__(self, weights, capacity):
        self.weights = np.array(weights)
        self.capacity = capacity
        self.num_item = len(weights)
        self.x = boolvar(shape=self.num_item, name="x")
        self.model = Model(sum(self.x * self.weights) <= self.capacity)

    def solve(self, y):
        self.model.maximize(sum(self.x * y))
        self.model.solve(solver="gurobi")
        return self.x.value().astype(int)


# Predictive model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        out = self.linear(x)
        return out


# SPO+ surrogate loss
class SPOPlus(torch.autograd.Function):
    """
    Implementation of the SPO+ surrogate loss from
    Elmachtoub, A. N., & Grigas, P. (2021). Smart "predict, then optimize". Management Science.
    """
    @staticmethod
    def forward(ctx, y_pred, y_true, sol_true, optimization_problem):
        """
        The forward pass computes and stores the solution for the SPO-perturbed cost
        vector (for the backward pass).

        :param ctx: the context object
        :param y_pred: the predicted cost vector
        :param y_true: The true cost vector
        :param sol_true: The true solution
        :param optimization_problem: The parametric optimization problem
        """
        cost_vector = 2 * y_pred - y_true
        sol_spo = torch.tensor(optimization_problem.solve(cost_vector.detach().cpu().numpy()))
        ctx.save_for_backward(sol_spo, sol_true)

        # We can just return a dummy value, rather than the actual training SPO+ loss, without affecting backprop
        return torch.tensor(1.0)

    @staticmethod
    def backward(ctx, grad_output):
        sol_spo, sol_true = ctx.saved_tensors
        return -2 * (sol_true - sol_spo), None, None, None, None, None


# Custom datawrapper
class DataWrapper(Dataset):
    def __init__(self, x, y, sol):
        self.x = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).float()
        self.y = y if isinstance(x, torch.Tensor) else torch.from_numpy(y).float()
        self.sol = sol if isinstance(x, torch.Tensor) else torch.from_numpy(sol).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.sol[index]


if __name__ == "__main__":
    # Data configuration
    num_data = 1000
    num_feat = 5
    num_item = 10
    deg = 6
    noise_width = 0
    seed = 42

    # Training configuration
    lr = 0.01
    num_epochs = 15

    # Generate data
    weights, x, y = pyepo.data.knapsack.genData(num_data, num_feat, num_item, 1, deg, noise_width, seed)
    weights = weights[0]
    capacity = 0.5 * sum(weights)

    # Initialize knapsack problem
    knapsack_problem = KnapsackProblem(weights, capacity)

    # Initialize linear regressor
    pred_model = LinearRegression()

    # Use ADAM optimizer
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr)

    # Compute ground-truth solutions
    sols = np.array(
        [knapsack_problem.solve(y[i]) for i in range(len(x))]
    )

    # Split dataset into training and test sets
    n_training = int(num_data * 0.8)
    x_train, y_train, sol_train = x[:n_training], y[:n_training], sols[:n_training]
    x_test, y_test, sol_test = x[n_training:], y[n_training:], sols[n_training:]

    # Data loaders
    train_dl = DataLoader(DataWrapper(x_train, y_train, sol_train), batch_size=32, shuffle=True)
    test_dl = DataLoader(DataWrapper(x_test, y_test, sol_test), batch_size=len(x_test))

    # Training
    pred_model.train()
    training_regrets = []
    for epoch in range(num_epochs):
        for data in train_dl:
            x, y, sol = data

            # Forward pass
            y_pred = pred_model(x)
            loss = 0
            for i in range(len(x)):
                loss += SPOPlus.apply(y_pred[i], y[i], sol[i], knapsack_problem)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Calculate training regret after each epoch
        total_regret = 0
        pred_model.eval()  # Switch to eval mode for regret calculation
        with torch.no_grad():
            for data in train_dl:
                x, y, sol = data
                for i in range(len(x)):
                    y_pred = pred_model(x[i])
                    sol_pred = knapsack_problem.solve(y_pred.numpy())

                    sol_i = sol[i].detach().cpu().numpy()
                    y_i = y[i].detach().cpu().numpy()
                    total_regret += (np.dot(sol_i, y_i) - np.dot(sol_pred, y_i)) / np.dot(sol_i, y_i)
        
        avg_train_regret = total_regret / len(x_train)
        training_regrets.append(avg_train_regret)
        print(f"Epoch {epoch + 1}, Training relative regret: {avg_train_regret:.4f}")
        pred_model.train()  # Switch back to training mode

    # Plot training regrets
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, num_epochs + 1),
        training_regrets,
        marker='o',
        linestyle='-',
        linewidth=2.5,
        markersize=8,
        color='#E24A33',
        alpha=0.9
    )
    plt.title('Learning Curves', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, labelpad=10)
    plt.ylabel('Relative regret', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.show()

    # Evaluate on the test set
    pred_model.eval()
    average_regret = 0
    for data in test_dl:
        x, y, sol = data
        for i in range(len(x)):
            y_pred = pred_model(x[i])
            sol_pred = knapsack_problem.solve(y_pred.detach().numpy())

            sol_i = sol[i].detach().cpu().numpy()
            y_i = y[i].detach().cpu().numpy()
            average_regret += (np.dot(sol_i, y_i) - np.dot(sol_pred, y_i)) / np.dot(sol_i, y_i)

        average_regret /= len(x)
    print(f"Average test regret: {average_regret}")