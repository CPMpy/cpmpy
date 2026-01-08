"""
Collection of visualisation tools for processing the result of a `benchmark.py` run.

Best used though its CLI, a command-line tool to visualize and analyze solver performance 
based on CSV output files.

E.g. to compare the results of multiple solvers:

.. code-block:: console

    python analyze.py <results_dir>

Positional Arguments
--------------------
files : str
    One or more CSV files (or a single directory) containing performance data to analyze.

Optional Arguments
------------------
--time_limit : float, optional
    Maximum time limit (in seconds) to display on the x-axis of the plot.

--output, -o : str, optional
    Path to save the generated plot image (e.g., "output.png"). If not provided, the plot will be displayed interactively.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _extract_cost(solution_str):
    """
    Extract numeric cost from solution string like '<instantiation ... cost="69">'
    """
    if isinstance(solution_str, str):
        match = re.search(r'cost="(\d+)"', solution_str)
        if match:
            return float(match.group(1))
    return np.nan


OPT = 'OPTIMUM FOUND'
UNS = 'UNSATISFIABLE'
SAT = 'SATISFIABLE'
MEM = 'MEMORY'
ERR = 'ERROR'

def xcsp3_plot(df, time_limit=None, metric="time_solve", filter="solved"):
    # Get unique solvers
    solvers = df['solver'].unique()

    # Determine the status to plot (Opt if at least one opt, otherwise sat)
    if filter == "solved":
        status_filter = (OPT, UNS)
    elif filter == "feasible":
        status_filter = (OPT, UNS, SAT)
    else:
        raise Exception

    df = df[(df['status'].isin(status_filter))]  # only those that reached the desired status
    # print(df[["solver", "instance", "status", metric]])

    # Count how many instances each solver solved (with correct status)
    solver_counts = df['solver'].value_counts()

    # Sort solvers descending by number of instances solved
    solvers_sorted = solver_counts.sort_values(ascending=False).index.tolist()
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    for solver in sorted(solvers): # Sort solver names for consistent ordering
        # Get data for this solver
        solver_data = df[df['solver'] == solver]
        
        # Sort by time_total
        solver_data = solver_data.sort_values(metric)
        
        # If time_limit is set, truncate data
        if time_limit is not None:
            solver_data = solver_data[solver_data[metric] <= time_limit]
        
        # Build x and y values
        x = [0.0] + solver_data[metric].tolist()
        y = [0] + list(range(1, len(solver_data) + 1))
        
        # Plot the performance curve
        plt.plot(x, y, label=f"{solver} ({len(solver_data)})", linewidth=2.5)
    
    # Set plot properties
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Number of instances returning \'{','.join(status_filter)}\'')
    # Get unique year-track combinations
    year_track_pairs = df[['year', 'track']].drop_duplicates()
    datasets = ', '.join([f'{row.year}:{row.track}' for _, row in year_track_pairs.iterrows()])
    plt.title(f'Performance Plot ({datasets})')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis limit if specified
    if time_limit is not None:
        plt.xlim(0, time_limit)

    return fig

def get_cost(row):
    """
    Get the achieved objective value from the provided row.
    If intermediate solutions are available, get the best found (not neccesarily proven optimal).
    """
    intermediate = row['intermediate']

    # Try to parse string representations safely
    if isinstance(intermediate, str):
        try:
            intermediate = ast.literal_eval(intermediate)
        except (ValueError, SyntaxError):
            intermediate = None

    # If it's a valid list of tuples, return the last objective
    if isinstance(intermediate, list) and len(intermediate) > 0:
        try:
            return intermediate[-1][1]
        except (IndexError, TypeError):
            pass

    # Fallback to extracting from solution
    return _extract_cost(row['solution'])

def xcsp3_objective_performance_profile(df):
    # Parse cost from the solution string
    df = df.copy()
    # print(df["intermediate"])
    df['cost'] = df.apply(get_cost, axis=1)
    # df['cost'] = df['solution'].apply(extract_cost)

    # Pivot to get costs per instance per solver
    pivot = df.pivot_table(index='instance', columns='solver', values='cost')

    # Drop instances not solved by all solvers (for fair comparison)
    pivot = pivot.dropna(how='all')

    # Compute the best (minimum) cost per instance
    best_costs = pivot.min(axis=1)

    # Compute performance ratios: solver_cost / best_cost
    perf_ratios = pivot.divide(best_costs, axis=0)

    # Replace inf or NaN with a large number for safe plotting
    perf_ratios = perf_ratios.replace([np.inf, np.nan], np.max(perf_ratios.values) * 10)

    # Compute a score for sorting: fraction of instances with ratio ≤ 1.1 (or similar)
    score_threshold = 1.1
    solver_scores = (perf_ratios <= score_threshold).mean().sort_values(ascending=False)
    sorted_solvers = solver_scores.index.tolist()

    # τ range for plotting
    tau_vals = np.linspace(1, perf_ratios.max().max(), 500)

    # Plotting
    fig = plt.figure(figsize=(10, 6))

    for solver in sorted_solvers:
        y_vals = [(perf_ratios[solver] <= tau).mean() for tau in tau_vals]
        plt.plot(tau_vals, y_vals, label=solver, linewidth=2.5)

    plt.xlabel(r'Objective ratio $\tau$')
    plt.ylabel('Fraction of instances')
    year_track_pairs = df[['year', 'track']].drop_duplicates()
    datasets = ', '.join([f'{row.year}:{row.track}' for _, row in year_track_pairs.iterrows()])
    plt.title(f'Objective Performance Profile ({datasets})')
    plt.grid(True)
    plt.legend()
    plt.xlim(left=1)

    return fig

def xcsp3_stats(df):

    for phase in ['parse', 'model', 'post']:
        slowest_idx = df[f'time_{phase}'].idxmax()
        if slowest_idx is not None and not pd.isna(slowest_idx):
            print(f"Slowest {phase}: {df.loc[slowest_idx, f'time_{phase}']}s ({df.loc[slowest_idx, 'instance']}, {df.loc[slowest_idx, 'solver']})")

    df['problem'] = df['instance'].map(lambda x: x.split("-")[0])
    def get_metadata(x):
        with open(x) as f:
            metadata = json.load(f)
        return metadata["area"]
    df["file_name"] = df["year"].map(str) + "/" + df["track"] + "/" + df["instance"].map(lambda x: x[:-4] + ".json")
    df["area"] = df["file_name"].map(get_metadata)
    pd.set_option('display.float_format', '{:0.1f}'.format)

    df["error"] = df["status"] == ERR
    df["memory"] = df["status"] == MEM
    df["feasible"] = df["status"].isin((OPT, SAT, UNS))
    df["solved"] = df["status"].isin((OPT, UNS))
    df["posted"] = ~df["time_post"].isna()
    df["time_cb"] = df["cb_time"].fillna(value=0.0)
    df["cb_rel"] = 100 * (df["time_cb"] / df["time_solve"])

    df = df[df["solver"].isin(("gurobi", "lazy_gurobi"))]
    print(df[["instance", "solver", "time_solve", "area", "cb_rel"]])

    TO = 600
    TIMES = ("post", "solve")

    for t in TIMES:
        df[f"time_{t}_p2"] = df[f"time_{t}"].fillna(value=TO*2)

    for grouping in (['problem', 'solver'], ['solver']):
        groups = df.groupby(grouping).agg(
                area = ('area', 'mean'),
                t_totl_hr = ('time_total', 'sum'),
                t_post_p2 = ('time_post_p2', 'sum'),
                t_solv_p2 = ('time_solve_p2', 'sum'),
                insts = ('status', 'count'),
                error = ('error', 'sum'),
                memory = ('memory', 'sum'),
                posted = ('posted', 'sum'),
                feasib = ('feasible', 'sum'),
                solved = ('solved', 'sum'),
                cb_rel = ('cb_rel', 'mean'),
                )

        # groups = groups.sort_index(level=["problem"], by="area")
        # groups = groups.sort_values(by="area", ascending=False)
        groups["area"] = groups["area"].map(lambda x: f"{x:.1e}")
        groups["t_totl_hr"] = groups["t_totl_hr"].map(lambda x: x / 3600)
        # groups.loc[('Total')] = groups.sum(numeric_only=True)

        print(groups)

    
    
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze XCSP3 solver performance data')
    parser.add_argument('files', nargs='+', help='List of CSV files or directories to analyze')
    parser.add_argument('--time-limit', type=float, default=None, 
                        help='Maximum time limit in seconds to show on x-axis')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to save the plot image (e.g., output.png)')
    args = parser.parse_args()
    
    # Gather all CSV files
    csv_files = []
    for path_str in args.files:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.csv':
            csv_files.append(path)
        elif path.is_dir():
            csv_files.extend(path.rglob('*.csv'))
        else:
            print(f"Warning: {path} is not a valid CSV file or directory")

    if not csv_files:
        print("No CSV files found.")
        return

    # Read and merge all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Print some stats
    xcsp3_stats(df)
    
    # Create performance plot
    # df[cb_time] = df[f"time_{t}"].fillna(value=TO*2)
    # df["t_solve_wo_cb"] = df["time_solve"] - df["cb_time"].fillna(value=0)
    # fig = xcsp3_plot(df, args.time_limit, metric="t_solve_wo_cb")
    fig = xcsp3_plot(df, args.time_limit, filter="feasible")
    # fig = xcsp3_objective_performance_profile(merged_df)

    # Save or show plot
    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
