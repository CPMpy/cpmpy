import argparse
import pandas as pd
import matplotlib.pyplot as plt

def xcsp3_plot(df, time_limit=None):
    # Get unique solvers
    solvers = df['solver'].unique()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # For each solver, create a performance curve
    for solver in solvers:
        # Get data for this solver
        solver_data = df[df['solver'] == solver]
        
        # Sort by time_total
        solver_data = solver_data.sort_values('time_total')
        
        # If time_limit is set, truncate data
        if time_limit is not None:
            solver_data = solver_data[solver_data['time_total'] <= time_limit]
        
        # Calculate cumulative count
        cumulative_count = list(range(1, len(solver_data) + 1))
        
        # Plot the line with increased linewidth
        plt.plot(solver_data['time_total'], cumulative_count, label=solver, linewidth=2.5)
    
    # Set plot properties
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of instances solved')
    plt.title('Performance Plot')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis limit if specified
    if time_limit is not None:
        plt.xlim(0, time_limit)
    
    # Show plot
    plt.show()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze XCSP3 solver performance data')
    parser.add_argument('files', nargs='+', help='List of CSV files to analyze')
    parser.add_argument('--time_limit', type=float, default=None, 
                       help='Maximum time limit in seconds to show on x-axis')
    args = parser.parse_args()
    
    # Read and merge all CSV files
    dfs = []
    for file in args.files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Create performance plot
    xcsp3_plot(merged_df, args.time_limit)

if __name__ == '__main__':
    main()
