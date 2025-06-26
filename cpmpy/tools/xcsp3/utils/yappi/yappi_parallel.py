# run_all_profiles.py

import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from tqdm import tqdm

PROFILE_DIR = Path("profiles")
PROFILE_DIR.mkdir(exist_ok=True)

MAX_PARALLEL_JOBS = 4  # Adjust to your machine

def profile_instance(idx, filename):
    profile_file = PROFILE_DIR / f"profile_{idx}_{Path(filename).stem}.pstat"
    return subprocess.run(
        ["python", "yappi/profile_worker.py", filename, str(profile_file)],
        capture_output=True
    )

def main():
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)
    tasks = [(i, str(file)) for i, (file, _) in enumerate(dataset)]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_idx = {
            executor.submit(profile_instance, i, fname): (i, fname) for i, fname in tasks
        }

        with tqdm(total=len(future_to_idx), desc="Profiling Benchmarks") as pbar:
            for future in as_completed(future_to_idx):
                i, fname = future_to_idx[future]
                result = future.result()
                if result.returncode != 0:
                    print(f"\n[ERROR] {fname}:\n{result.stderr.decode()}")
                else:
                    pass  # silently record success
                pbar.update(1)

if __name__ == "__main__":
    main()
