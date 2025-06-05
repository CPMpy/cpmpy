"""
CLI tool to extract the latest run from each solver.

E.g.

    python cpmpy/tools/xcsp3/utils/result_copier.py results/2024/COP/

Will place the latest runs in results/2024/COP/best

Usefull when analysing runs with xcsp3_analyzer.py
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

def copy_latest_csvs(source_dir, destination_dir):
    source = Path(source_dir)
    destination = Path(destination_dir)

    # If destination exists, remove all files inside it
    if destination.exists():
        for file in destination.glob('*'):
            if file.is_file():
                file.unlink()
    else:
        destination.mkdir(parents=True, exist_ok=True)

    # Regex pattern to match: prefix_timestamp (e.g. xcsp3_2023_CSP23_choco_20250515_134044.csv)
    pattern = re.compile(r'(.+?)_(\d{8}_\d{6})\.csv$')

    latest_files = {}

    for file in source.glob('*.csv'):
        match = pattern.match(file.name)
        if not match:
            continue
        prefix, timestamp = match.groups()
        if prefix not in latest_files or timestamp > latest_files[prefix][0]:
            latest_files[prefix] = (timestamp, file)

    # Copy files
    for prefix, (timestamp, file) in latest_files.items():
        target = destination / file.name
        shutil.copy2(file, target)
        print(f"Copied: {file.name} -> {target}")

def main():
    parser = argparse.ArgumentParser(description="Copy the latest CSVs from source to destination.")
    parser.add_argument(
        "source",
        type=str,
        help="Path to the source directory containing CSV files"
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: {source_dir} is not a valid directory.")
        return

    destination_dir = source_dir / "best"
    copy_latest_csvs(source_dir, destination_dir)

if __name__ == "__main__":
    main()
