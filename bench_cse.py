import cpmpy as cp

from cpmpy.tools.xcsp3.xcsp3_dataset import XCSP3Dataset
from cpmpy.tools.xcsp3 import read_xcsp3

import lzma
from io import StringIO

from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

import os
import resource
import sys
import multiprocessing as mp
import signal
import time

def do_transform(fname, solvername):
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(60)

    with open(os.path.join(dirname, fname), "rb") as f:
        
        xml_str = lzma.decompress(f.read()).decode("utf-8")
        model = read_xcsp3(StringIO(xml_str))

        solver = cp.SolverLookup.get(solvername)
        start = time.time()
        transformed = solver.transform(model.constraints)
        transform_time = time.time() - start

        return  {
            "nb_vars_orig": len(get_variables(model.constraints)),
            "nb_vars_trans": len(get_variables(transformed)),
            
            "nb_cons_orig": len(toplevel_list(model.constraints)),
            "nb_cons_trans": len(transformed),

            "transform_time": transform_time,
            "solver": solvername
        }
    
    
def signal_handler(signum, frame):
    raise TimeoutError

def read_file_and_transform(fname):
    print(f"Processing {fname}...")
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))
    pretty_instance = fname.replace(".xml.lzma", "")
    if pretty_instance.startswith("BinaryPuzzle"):
        return dict(instance=pretty_instance, error="skipped")
    
    with open(os.path.join("2023/CSP23", fname), "rb") as f:
        try:
            stats = do_transform(fname, "exact")
            stats['instance'] = pretty_instance
        except TimeoutError:
            stats = dict(instance=pretty_instance, error="timeout")
        except MemoryError:
            stats = dict(instance=pretty_instance, error="memory")
        
        return stats


if __name__ == "__main__":

    year = 2023
    track = "CSP"

    dirname = f"{year}/{track}{str(year)[2:]}"
    fnames = sorted(os.listdir(dirname))

    import pandas as pd
    all_stats = []

    with mp.Pool(processes=3) as pool:
        all_stats = pool.map(read_file_and_transform, fnames)

    df = pd.DataFrame(all_stats)
    df.set_index("instance", inplace=True)
    df.to_csv(f"xcsp3_stats_{str(year)[2:]}_{track}_cse_exact.csv")

    print(df)

        
        






