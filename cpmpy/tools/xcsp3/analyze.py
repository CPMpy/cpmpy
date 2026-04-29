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
from pathlib import Path
import re
import matplotlib
import pandas as pd  # type: ignore[import-untyped]
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


def xcsp3_plot(df, time_limit=None):
    # Get unique solvers
    solvers = df['solver'].unique()

    # Determine the status to plot (Opt if at least one opt, otherwise sat)
    statuses = df['status'].unique()
    if 'OPTIMUM FOUND' in statuses:
        status_filter = 'OPTIMUM FOUND'
    else:
        status_filter = 'SATISFIABLE'
    df = df[(df['status'] == status_filter) | (df['status'] == 'UNSATISFIABLE')]  # only those that reached the desired status

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
        # key = "time_total"
        key = "time_post"
        solver_data = solver_data.sort_values(key)
        
        # If time_limit is set, truncate data
        if time_limit is not None:
            solver_data = solver_data[solver_data['time_total'] <= time_limit]
        
        # Build x and y values
        x = [0.0] + solver_data['time_total'].tolist()
        y = [0] + list(range(1, len(solver_data) + 1))
        
        # Plot the performance curve
        plt.plot(x, y, label=f"{solver} ({len(solver_data)})", linewidth=2.5)
    
    # Set plot properties
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Number of instances returning \'{status_filter}\'')
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

    for solver in df['solver'].unique():
        sdf = df[df['solver'] == solver]
        print(f"\n=== {solver} ({len(sdf)} instances) ===")

        # Status counts
        status_counts = sdf['status'].value_counts()
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

        # Time stats for solved instances
        solved = sdf[sdf['status'].isin(['SATISFIABLE', 'UNSATISFIABLE', 'OPTIMUM FOUND'])]
        if not solved.empty:
            for col in ['time_total', 'time_post', 'time_solve']:
                if col in solved.columns and solved[col].notna().any():
                    vals = solved[col].dropna()
                    print(f"  {col}: mean={vals.mean():.2f}s, median={vals.median():.2f}s, max={vals.max():.2f}s")
            print(f"  Grand total: {sdf['time_total'].sum()/60:.2f} minutes")

        # Slowest phases
        for phase in ['parse', 'model', 'post']:
            col = f'time_{phase}'
            if col in sdf.columns and sdf[col].notna().any():
                idx = sdf[col].idxmax()
                print(f"  Slowest {phase}: {sdf.loc[idx, col]:.2f}s ({sdf.loc[idx, 'instance']})")
    
    
def xcsp3_time_comparison(df, time_limit=300, solver_order=None):
    """
    Compare time differentials between solvers on shared instances.
    Prints per-instance and aggregate timing differences.
    NaN time values are replaced with 2*time_limit as a penalty.
    """
    solvers = sorted(df['solver'].unique())
    if len(solvers) < 2:
        print(f"Time comparison requires at least 2 solvers, got {len(solvers)}: {list(solvers)}")
        return

    if solver_order and len(solver_order) >= 2:
        s1, s2 = solver_order[0], solver_order[1]
    else:
        # Compare the two most recent runs (sorted alphabetically, timestamps ensure order)
        s1, s2 = solvers[-2], solvers[-1]
    print(f"\nComparing: {s1} vs {s2}")
    df = df[(df['solver'] == s1) | (df['solver'] == s2)]

    # Only keep instances that both solvers attempted
    instances_s1 = set(df[df['solver'] == s1]['instance'])
    instances_s2 = set(df[df['solver'] == s2]['instance'])
    shared_instances = instances_s1 & instances_s2
    skipped = (instances_s1 | instances_s2) - shared_instances
    if skipped:
        print(f"  Skipping {len(skipped)} instances not present in both solvers")
    df = df[df['instance'].isin(shared_instances)]

    # UNKNOWN instances get 2*time_limit penalty for time_total
    unknown_mask = df['status'] == 'UNKNOWN'
    df.loc[unknown_mask, 'time_total'] = 2 * time_limit
    df.loc[unknown_mask, 'time_solve'] = 2 * time_limit

    # time_cols = ['time_total']
    # df["time"] = df['time_parse', 'time_model', 'time_post', 'time_solve'].sum()

    # # Replace NaN time values with 2*time_limit as penalty
    # time_cols = ['time_total', 'time_parse', 'time_model', 'time_post', 'time_solve']

    for col in ['time_parse', 'time_model', 'time_post']:
        if col in df.columns:
            df[col] = df[col].fillna(2 * time_limit)

    time_cols = ['time_total', 'time_post', 'time_solve']

    # Pivot per time column
    for col in time_cols:
        pivot = df.pivot_table(index='instance', columns='solver', values=col)
        # Skip if either solver has no data for this column
        if s1 not in pivot.columns or s2 not in pivot.columns:
            continue
        shared = pivot.dropna()
        if shared.empty:
            continue

        diff = shared[s1] - shared[s2]

        print(f"\n=== {col} ({s1} minus {s2}) ===")
        print(f"  Instances compared: {len(diff)}")
        print(f"  Mean diff:   {diff.mean():+.3f}s")
        print(f"  Median diff: {diff.median():+.3f}s")
        print(f"  Total diff:  {diff.sum():+.3f}s")
        print(f"  {s1} faster on {(diff < -1).sum()}/{len(diff)} instances")
        print(f"  {s2} faster on {(diff > 1).sum()}/{len(diff)} instances")

        # Table of instances with |diff| > 1s
        top = diff[diff.abs() > 1].sort_values()
        if not top.empty:
            table = pd.DataFrame({
                'instance': top.index,
                s1: [f"{shared.loc[i, s1]:.2f}" for i in top.index],
                s2: [f"{shared.loc[i, s2]:.2f}" for i in top.index],
                'diff': [f"{d:+.3f}" for d in top.values],
            })
            print(table.to_string(index=False))


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze XCSP3 solver performance data')
    parser.add_argument('files', nargs='+', help='List of CSV files or directories to analyze')
    parser.add_argument('--time_limit', type=float, default=None, 
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

    SOLVER_RENAMES = None

    # # Rename solvers: map auto-generated "solver_timestamp" names to readable labels
    # SOLVER_RENAMES = {
    #     "gurobi_expr": "expr",
    #     "gurobi_base": "base",
    # }

    # Read and merge all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # ts = "_".join(file.stem.split("_")[4:6])
        solver = file.stem
        df["solver"] = df["solver"] + f"_{solver}"
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Print some stats
    xcsp3_stats(merged_df)
    
    # Create performance plot
    fig = xcsp3_plot(merged_df, args.time_limit)
    # fig = xcsp3_objective_performance_profile(merged_df)

    # Save or show plot
    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
