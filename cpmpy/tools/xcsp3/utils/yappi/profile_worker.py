# profile_worker.py

import sys
import yappi
import io
import lzma
from cpmpy.tools.xcsp3 import _parse_xcsp3, _load_xcsp3
from pathlib import Path
import timeit
import logging

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_profile(xcsp3_file_path: str, output_profile_path: str):
    try:
        with lzma.open(xcsp3_file_path, 'rt', encoding='utf-8') as f:
            xml_file = io.StringIO(f.read())

        start = timeit.default_timer()
        parser = _parse_xcsp3(xml_file)

        yappi.set_clock_type("wall")
        yappi.clear_stats()
        yappi.start()
        model = _load_xcsp3(parser)
        # Optional: model.solve("choco", time_limit=10)
        end = timeit.default_timer()

        yappi.stop()
        stats = yappi.get_func_stats()
        stats.save(output_profile_path, type="pstat")

        logger.info(f"Profiling completed: {xcsp3_file_path} in {end - start:.2f}s")

    except Exception as e:
        logger.error(f"Error profiling {xcsp3_file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python profile_worker.py <input_file.xcsp3.xz> <output_profile.pstat>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_profile = sys.argv[2]
    run_profile(input_file, output_profile)
