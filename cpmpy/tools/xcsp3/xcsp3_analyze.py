import argparse
from pathlib import Path
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def xcsp3_plot(df, time_limit=None):
    # Get unique solvers
    solvers = df['solver'].unique()

    # Determine the status to plot (Opt if at least one opt, otherwise sat)
    statuses = df['status'].unique()
    if 'OPTIMUM FOUND' in statuses:
        status_filter = 'OPTIMUM FOUND'
    else:
        status_filter = 'SATISFIABLE'
    df = df[df['status'] == status_filter]  # only those that reached the desired status
    
    # Create figure
    fig = plt.figure(figsize=(10, 6))
    
    for solver in solvers:
        # Get data for this solver
        solver_data = df[df['solver'] == solver]
        
        # Sort by time_total
        solver_data = solver_data.sort_values('time_total')
        
        # If time_limit is set, truncate data
        if time_limit is not None:
            solver_data = solver_data[solver_data['time_total'] <= time_limit]
        
        # Build x and y values
        x = [0.0] + solver_data['time_total'].tolist()
        y = [0] + list(range(1, len(solver_data) + 1))
        
        # Plot the performance curve
        plt.plot(x, y, label=solver, linewidth=2.5)
    
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

def xcsp3_stats(df):

    for phase in ['parse', 'model', 'post']:
        slowest_idx = df[f'time_{phase}'].idxmax()
        print(f"Slowest {phase}: {df.loc[slowest_idx, f'time_{phase}']}s ({df.loc[slowest_idx, 'instance']}, {df.loc[slowest_idx, 'solver']})")

    for solver in df['solver'].unique():
        solver_total = df[df['solver'] == solver]['time_total'].sum()
        print(f"Grand total for {solver}: {solver_total/60:.2f} minutes")
    
    
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

    # Read and merge all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Print some stats
    xcsp3_stats(merged_df)
    
    # Create performance plot
    fig = xcsp3_plot(merged_df, args.time_limit)

    # Save or show plot
    if args.output:
        fig.savefig(args.output, bbox_inches='tight')
        print(f"Plot saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
