"""
Some utilities for analysing the results of running a mock of the XCSP3 competition.
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_cumulative_solve_times(*csv_file_paths):
    plt.figure(figsize=(10, 6))

    for csv_file_path in csv_file_paths:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        df['total'] = df['t_add'] + df["t_parse"] + df["t_solve"]
        # Sort the DataFrame by total solve time
        df = df.sort_values(by='total')

        # Create a cumulative count of models solved within each time
        df['cumulative_count'] = range(1, len(df) + 1)

        # Plot the cumulative solve times for each CSV
        plt.plot(df['total'], df['cumulative_count'], marker='', linestyle='-', label=csv_file_path)

    plt.xlabel('Solve Time (seconds)')
    plt.ylabel('Number of Models Solved')
    plt.title('Cumulative Number of Models Solved Over Time')
    plt.grid(True)
    plt.legend(title='CSV Files')
    plt.show()

def compare_solve_times(csv_file_path1, csv_file_path2):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)

    #df1 = df1[df1['t_solve'] <= 1799]
    #df2 = df2[df2['t_solve'] <= 1799]
    df1['t_post'] = df1['t_add'] - df1['t_transform']
    df2['t_post'] = df2['t_add'] - df2['t_transform']
    # Merge the DataFrames on the common columns (assuming 'solver' and 'model' are common)
    merged_df = pd.merge(df1, df2, on=['solver', 'model_name'], suffixes=('_1', '_2'))  # inner join, only keep shared instances
    total_times = merged_df.groupby(['solver']).sum()
    # Calculate the difference in solve time between the two DataFrames
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(total_times)
    return total_times[['solver', 'model_name', 't_solve_1', 't_solve_2', 'solve_time_diff']]

if __name__ == "__main__":
    csv_file_path = 'output.csv'
    plot_cumulative_solve_times(csv_file_path)
