"""
Very basic script to create ECDF plots comparing multiple benchmark runs.
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# === USER-DEFINED INPUT FILES ===
input_files = {
    "results/xcsp3_2024_CSP_ortools_20250522_100934.csv": "old",
    "results/xcsp3_2024_CSP_ortools_20250522_101358.csv": "new",
}

# Time columns to compare
time_columns = ['time_total', 'time_parse', 'time_model', 'time_post', 'time_solve']
output_folder = "ecdf_plots"  # Folder to save images


# Set style for seaborn
sns.set_theme(style="whitegrid")

os.makedirs(output_folder, exist_ok=True)

# Read CSVs and extract metadata
dataframes = {}
file_metadata = {}  # label -> (year, track)
for path, label in input_files.items():
    df = pd.read_csv(path)
    dataframes[label] = df

    # Extract year and track from filename
    filename = os.path.basename(path)
    match = re.match(r'xcsp3_(\d{4})_(\w+)_', filename)
    if match:
        year, track = match.groups()
    else:
        year, track = "UnknownYear", "UnknownTrack"
    file_metadata[label] = (year, track)

# Generate ECDF plots for each time metric
# Generate ECDF plots for each time metric
for time_col in time_columns:
    plt.figure(figsize=(10, 6))

    max_instances = 0  # Track overall max for the dotted line

    for label, df in dataframes.items():
        if time_col in df.columns:
            df_clean = df[df[time_col].notna()]
            n = len(df_clean)
            if n == 0:
                continue
            max_instances = max(max_instances, n)

            sorted_values = df_clean[time_col].sort_values()
            y_values = range(1, n + 1)
            plt.step(sorted_values, y_values, where="post", label=label)

    # Get year/track from one file (assume same for all)
    example_label = next(iter(file_metadata))
    year, track = file_metadata[example_label]

    # Plot the max line
    plt.axhline(y=max_instances, color="gray", linestyle="dotted", linewidth=1)
    
    # Labels and layout
    plt.title(f"ECDF of {time_col} - Track: {track}, Year: {year}")
    plt.xlabel(f"{time_col} (seconds)")
    plt.ylabel("Number of Instances")
    plt.legend(title="Version")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    file_path = os.path.join(output_folder, f"ecdf_{time_col}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

print(f"Saved ECDF plots to: {output_folder}")