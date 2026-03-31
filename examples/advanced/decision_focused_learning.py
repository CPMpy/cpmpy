import pyepo
import cpmpy as cp
import torch
from torch import nn
import matplotlib.pyplot as plt


# Generic PyEPO wrapper for CPMpy models
class optCPMpyModel(pyepo.model.opt.optModel):
    def __init__(self, model, dvars, sense, solver=None, prec=None):
        super().__init__()
        self._model = model
        self.x = dvars
        self.s = cp.SolverLookup.get(solver, model)
        self.modelSense = sense
        self.prec = prec

    def _getModel(self):
        return None, None  # created by constructor

    def setObj(self, c):
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()

        if self.modelSense == pyepo.EPO.MAXIMIZE:
            self.s.maximize(cp.sum(self.x*c))
        else:
            self.s.minimize(cp.sum(self.x*c))

    def solve(self):
        self.s.solve()
        return self.x.value().astype(int), self.s.objective_value()


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
    capacity = 0.2 * weights.sum()

    
    # Initialize PyEPO-wrapped optimisation model (just the constraints, works for any CPMpy model)
    dv = cp.boolvar(shape=num_item, name="x")
    m = cp.Model(
        weights * dv <= capacity,  # capacity constraint
    )
    optmodel = optCPMpyModel(m, dv, sense=pyepo.EPO.MAXIMIZE, solver="gurobi")

    # Initialize machine learning model, optimizer and PyEPO-wrapped loss
    pred_model = nn.Linear(num_feat, num_item)  # linear regressor
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr)
    spo_plus = pyepo.func.SPOPlus(optmodel, processes=1)  # PyEPO's SPO+ loss over the PyEPO optimisation model
    

    # Split dataset into training and test sets
    n_training = int(num_data * 0.8)
    x_train, y_train = x[:n_training], y[:n_training]
    x_test, y_test = x[n_training:], y[n_training:]

    # Data loaders
    train_dataset = pyepo.data.dataset.optDataset(optmodel, x_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = pyepo.data.dataset.optDataset(optmodel, x_test, y_test)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    
    # Training
    pred_model.train()
    training_regrets = []
    for epoch in range(num_epochs):
        for data in train_dl:
            x, y, sol, obj = data

            # Forward pass
            y_pred = pred_model(x)
            loss = spo_plus(y_pred, y, sol, obj)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate training regret after each epoch
        normalized_regret = pyepo.metric.regret(pred_model, optmodel, train_dl)
        training_regrets.append(normalized_regret)
        print(f"Epoch {epoch + 1}, Training normalized regret: {normalized_regret:.4f}")
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
