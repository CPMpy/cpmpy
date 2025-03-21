# XCSP3 CPMpy tools

For now, the executable from the 2024 competition entry has been copied as-is.

Run an example using the following command:
```python
python cpmpy/tools/xcsp3/xcsp3_cpmpy.py cpmpy/tools/xcsp3/models/Fillomino-mini-5-0_c24.xml
```

## Ideas

Some of the utilities used within the executable should probably be extracted into their own re-usable functions. Functions which are part of CPMpy's suite of tools, so that a user can call them from their own code allowing them to make their own runners and not necesarily needing to use our XCSP3 competition runner (e.g. if they just want to load a XCSP3 instance into a CPMpy model, without all the other competition related stuff).

**Read and write** functions for the XCSP3 format, similarly as to how the DIMACS tool works:
```python
def write_xcsp3(model: cp.Model, ...) -> str:
    """
    Takes a CPMpy model and formats its solution according to the XCSP3 specification (in a XML format).

    Like what currently happens in `xcsp3_solution.py`.
    """
    pass

def read_xcsp3(fname:os.PathLike) -> cp.Model:
    """
    Take a path to a XCSP3 instance (`.xml` file), return the matching CPMpy model.
    """
    parser = ParserXCSP3(fname)
    callbacks = CallbacksCPMPy()
    callbacks.force_exit = True
    callbacker = CallbackerXCSP3(parser, callbacks)
    callbacker.load_instance()
    model = callbacks.cpm_model
    return model
```

Utils for proper **formatting of print statements**:
```python

def status_line_start() -> str:
    return 's' + chr(32)

def value_line_start() -> str:
    return 'v' + chr(32)

def objective_line_start() -> str:
    return 'o' + chr(32)

def comment_line_start() -> str:
    return 'c' + chr(32)

class ExitStatus(Enum):
    unsupported:str = "UNSUPPORTED" # instance contains a unsupported feature (e.g. a unsupported global constraint)
    sat:str = "SATISFIABLE" # CSP : found a solution | COP : found a solution but couldn't prove optimality
    optimal:str = "OPTIMUM" + chr(32) + "FOUND" # optimal COP solution found
    unsat:str = "UNSATISFIABLE" # instance is unsatisfiable
    unknown:str = "UNKNOWN" # any other case

def print_status(status: ExitStatus) -> None:
    print(status_line_start() + status.value, end="\n", flush=True)

def print_value(value: str) -> None:
    value = value[:-2].replace("\n", "\nv" + chr(32)) + value[-2:]
    print(value_line_start() + value, end="\n", flush=True)

def print_objective(objective: int) -> None:
    print(objective_line_start() + str(objective), end="\n", flush=True)

def print_comment(comment: str) -> None:
    print(comment_line_start() + comment.rstrip('\n'), end="\r\n", flush=True)

@contextmanager
def prepend_print():
    """
    A context manager to capture stdout (e.g. to capture prints coming directly from the solvers themselves)
    """
    # Save the original stdout
    original_stdout = sys.stdout
    
    class PrependStream:
        def __init__(self, stream):
            self.stream = stream
        
        def write(self, message):
            # Prepend 'c' to each message before writing it
            if message.strip():  # Avoid prepending 'c' to empty messages (like newlines)
                self.stream.write('c ' + message)
            else:
                self.stream.write(message)
        
        def flush(self):
            self.stream.flush()
    
    # Override stdout with our custom stream
    sys.stdout = PrependStream(original_stdout)
    
    try:
        yield
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout
```

Next we have a collection of **callback handlers** for solvers that support intermediate solutions: OR-Tools and Gurobi.


Then we have a collection of utilities related to **resource management**, like setting memory limits and setting timeouts (including the pycsp3 parser).

```python
# Set memory limits
soft = max(args.mem_limit - mb_as_bytes(MEMORY_BUFFER_SOFT), mb_as_bytes(MEMORY_BUFFER_SOFT))
hard = max(args.mem_limit - mb_as_bytes(MEMORY_BUFFER_HARD), mb_as_bytes(MEMORY_BUFFER_HARD))
print_comment(f"Setting memory limit: {soft}-{hard}")
resource.setrlimit(resource.RLIMIT_AS, (soft, hard)) # limit memory in number of bytes
```

Setting a time limit on a function call in Python is tricky, expecially making it cross-platform. Many solutions rely on `signal` timers, which are not available on `Windows`. Recently I found this nice library that does all the work for you: https://pypi.org/project/stopit/. I've succesfully been able to use it in combination with XCSP3 in some previous profiling experiments.

Next is a collection of functions for setting the **solver args** with options specifically for the competition.

I believe all of the above should be separated from the **CLI runner**. They should all be accessible from cpmpy.tools.xcsp3 outside of the xcsp3-competition-specific executable that we provide. Maybe some parts (like limiting resources) should be made as part of a more general tool, as they might be usefull outside of a xcsp3 context. E.g. the ability to set a time limit which includes the transformation time. 

Some work on solver arguments might also be generalisable, although this might prove to be more difficult. The "seed" option for example. Not all solvers support it, OR-Tools is very tricky to make deterministics, and Exact expects kwargs at construction time instead of the default as part of a `.solve()` call.

Besides having a CLI tool to run a single instance, some tooling for running a large batch might also be nice. For the competition we used PyTest (to easily re-run instances which failed), which turned out to be more finicky then expected.

## TODOs / issues

The runner currently makes use of the `resource` module of python (for setting a limit on memory consumption), which is UNIX exclusive. Windows seems to require more manual fiddeling with the win32 API: https://stackoverflow.com/questions/54949110/limit-python-script-ram-usage-in-windows