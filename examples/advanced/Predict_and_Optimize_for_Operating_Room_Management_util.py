import numpy as np
from matplotlib import pyplot as plt
import cpmpy as cp


def generate_synthetic_data(num_features=5, num_samples=1000, deg=4, noise_width=0.5):
    B = np.random.uniform(-1, 1, num_features)
    features = np.random.rand(num_samples, num_features)

    labels = []
    for i in range(num_samples):
        label = (np.dot(B, features[i]) / np.sqrt(5) + 3) ** deg
        label /= 3.5 ** deg
        label += 1

        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width)
        label *= epsilon
        labels.append(label)

    max_val = np.max(labels)
    labels /= max_val
    labels *= 300

    return features, labels


def sample_one_instance(X, y, min_patients, max_patients):
    n_samples = np.random.randint(min_patients, max_patients + 1)
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    return X_sample, y_sample

def generate_instances(X, y, num_instances, min_patients, max_patients):
    batches = []
    labels = []
    for _ in range(num_instances):
        X_s, y_s = sample_one_instance(X, y, min_patients, max_patients)
        batches.append(X_s)
        labels.append(y_s)
    num_features = X.shape[1]

    X_tensor = np.zeros((num_instances, max_patients, num_features))
    y_tensor = np.zeros((num_instances, max_patients, 1))

    for i, (xb, yb) in enumerate(zip(batches, labels)):
        n_patients = xb.shape[0]
        X_tensor[i, :n_patients, :] = xb
        y_tensor[i, :n_patients, 0] = np.squeeze(yb)

    return X_tensor, y_tensor

# Helper to flatten and ignore padded values (assumes padding is zeros)
def flatten_ignore_padding(X_tensor, y_tensor):
    mask = ~(X_tensor == 0).all(axis=2)
    X_flat = X_tensor[mask]
    y_flat = y_tensor[mask[:, :]]
    return X_flat, y_flat


def plot(assignments, title, num_ops, operation_durations, capacity, save = False):
    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
    import matplotlib.cm as cm
    op_colors = cm.get_cmap('tab20', num_ops)

    # Plot each room dynamically based on the number of rooms (length of capacity)
    num_rooms = len(capacity)
    for room_idx in range(num_rooms):
        ax.barh(y=num_rooms - 1 - room_idx, width=capacity[room_idx], left=0, height=0.5, color='white', edgecolor='black', linewidth=2, alpha=1, zorder=0)
        start = 0
        for op in range(num_ops):
            if assignments[op, room_idx] > 0.5:
                ax.barh(y=num_rooms - 1 - room_idx, width=operation_durations[op], left=start, height=0.4, color=op_colors(op), edgecolor='black', zorder=1)
                start += operation_durations[op]

    # --- Add "Unused" bar ---
    unused_ops = [op for op in range(num_ops) if np.all(assignments[op, :] <= 0.5)]
    if unused_ops:
        start = 0
        y_unused = num_rooms
        #ax.barh(y=y_unused, width=max(capacity), left=0, height=0.5, color='white', edgecolor='black', linewidth=2, alpha=1, zorder=0)
        for op in unused_ops:
            ax.barh(y=y_unused, width=operation_durations[op], left=start, height=0.4, color=op_colors(op), edgecolor='black', zorder=1)
            start += operation_durations[op]

    yticks = list(range(num_rooms))
    yticklabels = [f'OR {room_idx}' for room_idx in range(num_rooms, 0, -1)]
    if unused_ops:
        yticks = yticks + [num_rooms]
        yticklabels = yticklabels + ["Not assigned"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Duration (minutes)')

    ax.set_title(title)

    # Create legend for operations
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=op_colors(op), edgecolor='black', label=f'Surgery {op+1}') for op in range(num_ops) if operation_durations[op] > 0]
    legend_elements += [Patch(facecolor='none', edgecolor='black', label='OR Capacities', linestyle='-')]

    # Put legend outside the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    plt.tight_layout()
    ax.set_xlim(0, 450)
    if save:
        plt.savefig(f"{title}.png", bbox_inches='tight', dpi=300)
    plt.show()

class DAP_Model():
    def __init__(self, n_patients, OR_capacity, underuse, overuse, symmetry_breaking=True):

        # Number of patients
        self.n_patients = n_patients

        # OR blocks defined as tuple
        self.B = len(OR_capacity)

        # Capacity (0 = no OR block)
        self.c = np.zeros(self.B, dtype=int)
        for i in range(self.B):
            self.c[i] = OR_capacity[i]

        # Create a dict to collect indices of OR with same capacity
        groups_dict = {}
        # Group indices by OR capacity
        for idx, value in enumerate(OR_capacity):
            if value not in groups_dict:
                groups_dict[value] = []
            groups_dict[value].append(idx)
        # Convert into a list
        self.groups = list(groups_dict.values())

        # Objective function coefficients (beta > alpha)
        self.ALPHA = underuse
        self.BETA = overuse
        self.symmetry_breaking = symmetry_breaking

    def get_objective_value(self, d, x):
        u = []
        for b in range(self.B):
            u.append(max(0, self.c[b] -
                         sum([d[i] * x[i, b] for i in range(self.n_patients)])))
        v = []
        for b in range(self.B):
            v.append(max(0, sum([d[i] * x[i, b] for i in range(self.n_patients)]) -
                         self.c[b]))

        obj_val = float(self.ALPHA * sum(u) + self.BETA * sum(v))
        return obj_val

    def solve(self, d):
        d = [int(round(dd[0])) if isinstance(dd, (list, np.ndarray)) else int(round(dd)) for dd in d]

        model = cp.Model()

        x = cp.boolvar(shape=(self.n_patients, self.B), name="x")

        total_assigned = [cp.sum([d[i] * x[i, b] for i in range(self.n_patients)]) for b in range(self.B)]

        underuse = [cp.Maximum([0, (self.c[b] - total_assigned[b])]) for b in range(self.B)]
        overuse = [cp.Maximum([0, (total_assigned[b] - self.c[b])]) for b in range(self.B)]

        obj = self.ALPHA * cp.sum(underuse) + self.BETA * cp.sum(overuse)

        model += [cp.sum(x[i, b] for b in range(self.B)) <= 1 for i in range(self.n_patients)]

        if self.symmetry_breaking:
            for group in self.groups:
                for (j, b) in enumerate(group):
                    for i in range(self.n_patients):
                        if j > i:
                            model += [x[i, b] == 0]

                    for i in range(1, self.n_patients):
                        if j > 0:
                            model += [x[i, b] <= cp.sum([x[ii, group[j - 1]] for ii in range(i)])]

        for i in range(self.n_patients):
            if d[i] <= 0.001:
                for b in range(self.B):
                    model += [x[i, b] == 0]
        # Solve

        model.minimize(obj)
        status = model.solve()  # solver="gurobi", if you wish so
        if status:
            x_sol = np.array([x[i, b].value() for i in range(self.n_patients) for b in range(self.B)]).reshape(
                self.n_patients, self.B)
            return float(obj.value()), {"x": x_sol}
        else:
            raise Exception("No feasible solution was found")
