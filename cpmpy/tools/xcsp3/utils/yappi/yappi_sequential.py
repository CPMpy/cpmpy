import cpmpy
import lzma, tqdm
import io
import os
import yappi
import logging
import timeit
from pathlib import Path

from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from cpmpy.tools.xcsp3 import _load_xcsp3, _parse_xcsp3

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('benchmark_log.txt')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Output directory for profiling results
profile_output_dir = Path("profiles")
profile_output_dir.mkdir(exist_ok=True)

def run_with_solver(solver: str):
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)

    for i, (filename, metadata) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        profile_name = profile_output_dir / f"profile_{i}_{Path(filename).stem}.pstat"

        try:
            f = lzma.open(filename, 'rt', encoding='utf-8')
            try:
                xml_file = io.StringIO(f.read())

                # Start profiler
                yappi.set_clock_type("wall")
                yappi.clear_stats()
                yappi.start()

                start = timeit.default_timer()
                parser = _parse_xcsp3(xml_file)
                model = _load_xcsp3(parser)
                # model.solve(solver, time_limit=10)
                end = timeit.default_timer()

                # Stop profiler
                yappi.stop()

                # Save profiling stats
                func_stats = yappi.get_func_stats()
                func_stats.save(str(profile_name), type='pstat')

                logger.info(f"Benchmark completed for {filename} in {end - start:.2f}s")

            finally:
                f.close()
        except Exception as e:
            logger.error(f"Failed benchmark for {filename}: {e}")
            raise

        # if i >= 20:
        #     break

if __name__ == "__main__":
    solvers = ["choco"]
    for solver in solvers:
        run_with_solver(solver)
