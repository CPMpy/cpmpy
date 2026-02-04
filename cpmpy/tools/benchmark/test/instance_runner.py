import argparse
import inspect
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional

from cpmpy.tools.benchmark.test.runner import Runner 


def create_output_file(output_file: Optional[str], base_dir: Optional[str] = None, *args) -> str:
    """
    Create an output file path.
    
    Args:
        output_file: The output file path (can be relative or absolute)
        base_dir: Base directory for output files (default: "results/")
        *args: Additional arguments used to construct default filename if output_file is None
    
    Returns:
        The full output file path
    """
    if base_dir is None:
        base_dir = "results"
    
    if output_file is None:
        output_file = f"{'_'.join(args)}.txt"
    
    # If output_file is already absolute, use it as-is
    if os.path.isabs(output_file):
        full_path = output_file
    else:
        # Otherwise, join with base_dir
        full_path = os.path.join(base_dir, output_file)
    
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    
    return full_path
class InstanceRunner:
    
    def __init__(self):
        self.additional_observers = []
        # Get the file path of the concrete class, not the base class
        # This allows subclasses to reference their own file path
        self.this_file_path = os.path.abspath(inspect.getfile(type(self)))
        self.this_python = sys.executable

    def get_runner(self, instance: str, solver: str = "ortools", output_file: str = None, overwrite: bool = True, **kwargs):

        runner = Runner(reader=self.reader)
        # Store reference to instance_runner so observers can access it for formatting
        runner.instance_runner = self

        # Register default observers
        import inspect as inspect_module
        for observer in self.default_observers:
            # Check if observer accepts output_file and overwrite parameters
            sig = inspect_module.signature(observer.__init__)
            if 'output_file' in sig.parameters or 'overwrite' in sig.parameters:
                runner.register_observer(observer(output_file=output_file, overwrite=overwrite))
            else:
                runner.register_observer(observer())
        
        # Register any additional observers that were added programmatically
        # Track file paths to avoid duplicate WriteToFileObserver registrations
        registered_file_paths = set()
        for observer in self.get_additional_observers():
            # If observer is a partial function, call it to get the instance
            if isinstance(observer, partial):
                obs_instance = observer()
                # Check if it's a WriteToFileObserver and if we've already registered one for this file
                if hasattr(obs_instance, 'file_path'):
                    if obs_instance.file_path in registered_file_paths:
                        continue  # Skip duplicate WriteToFileObserver for the same file
                    registered_file_paths.add(obs_instance.file_path)
                runner.register_observer(obs_instance)
            # If observer is already an instance, use it directly
            elif hasattr(observer, '__class__') and not inspect.isclass(observer):
                # Check if it's a WriteToFileObserver and if we've already registered one for this file
                if hasattr(observer, 'file_path'):
                    if observer.file_path in registered_file_paths:
                        continue  # Skip duplicate WriteToFileObserver for the same file
                    registered_file_paths.add(observer.file_path)
                runner.register_observer(observer)
            # If observer is a class, instantiate it
            else:
                sig = inspect_module.signature(observer.__init__)
                if 'output_file' in sig.parameters or 'overwrite' in sig.parameters:
                    obs_instance = observer(output_file=output_file, overwrite=overwrite)
                    # Check if it's a WriteToFileObserver and if we've already registered one for this file
                    if hasattr(obs_instance, 'file_path'):
                        if obs_instance.file_path in registered_file_paths:
                            continue  # Skip duplicate WriteToFileObserver for the same file
                        registered_file_paths.add(obs_instance.file_path)
                    runner.register_observer(obs_instance)
                else:
                    runner.register_observer(observer())

        # Create output file path
        output_file = create_output_file(output_file, None, solver, instance)
        
        return runner
    
    def cmd(self, instance: str):
        pass

    def base_cmd(self, instance: str):
        return [
            self.this_python,
            self.this_file_path,
            instance,
        ]

    def argparser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("instance", type=str)
        parser.add_argument("--solver", type=str, default="ortools")
        parser.add_argument("--output_file", type=str, default=None)
        parser.add_argument("--verbose", action="store_true", default=False)
        parser.add_argument("--time_limit", type=float, default=None)
        parser.add_argument("--mem_limit", type=int, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--intermediate", action="store_true", default=False)
        parser.add_argument("--cores", type=int, default=None)
        parser.add_argument("--observers", type=list[str], default=None)
        return parser

    def print_comment(self, comment: str):
        """Print a comment. Subclasses can override to add formatting (e.g., 'c ' prefix)."""
        # Default implementation: just print (subclasses can override to add formatting)
        print(comment)
    
    def register_observer(self, observer):
        """Register an observer to be added when run() is called."""
        self.additional_observers.append(observer)
    
    def get_additional_observers(self):
        """Get the list of additional observers that should be registered."""
        return self.additional_observers

    def run(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):

        

        
        self.runner = self.get_runner(instance, solver, output_file, **kwargs)
        self.runner.run(instance=instance, solver=solver, output_file=output_file, **kwargs)
