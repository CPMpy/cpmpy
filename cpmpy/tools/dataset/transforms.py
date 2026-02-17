"""
Composable Transforms for CPMpy Datasets

Provides composable transform classes inspired by torchvision.transforms.
Transforms can be chained using :class:`Compose` and passed as the
``transform`` or ``target_transform`` argument to any Dataset subclass.

=================
List of classes
=================

.. autosummary::
    :nosignatures:

    Compose
    Open
    Load
    Serialize
    Translate
    SaveToFile
    Lambda

Example usage::

    from cpmpy.tools.dataset import MSEDataset, Compose, Load, Serialize
    from cpmpy.tools.io.wcnf import load_wcnf

    dataset = MSEDataset(root=".", year=2024, track="exact-weighted")

    # Chain: load WCNF files, then serialize to DIMACS
    transform = Compose([
        Load(load_wcnf, open=dataset.open),
        Serialize("dimacs"),
    ])
    dataset.transform = transform

    for dimacs_string, metadata in dataset:
        print(dimacs_string[:100])
"""

import json
import os
import re

_builtins_open = open  # capture before any parameter shadowing


def extract_format_metadata(content, format_name):
    """Extract format-specific metadata from a translated file content string.

    Parses format headers to extract statistics like variable/constraint counts.

    Arguments:
        content (str): The file content string.
        format_name (str): The format name (e.g., ``"opb"``, ``"dimacs"``, ``"mps"``).

    Returns:
        dict with format-prefixed metadata fields.
    """
    result = {}

    if format_name == "opb":
        for line in content.split('\n'):
            if not line.startswith('*'):
                break
            match = re.search(r'#variable=\s*(\d+)', line)
            if match:
                result["opb_num_variables"] = int(match.group(1))
            match = re.search(r'#constraint=\s*(\d+)', line)
            if match:
                result["opb_num_constraints"] = int(match.group(1))
            match = re.search(r'#product=\s*(\d+)', line)
            if match:
                result["opb_num_products"] = int(match.group(1))

    elif format_name == "dimacs":
        match = re.search(r'^p\s+(?:w?cnf)\s+(\d+)\s+(\d+)', content, re.MULTILINE)
        if match:
            result["dimacs_num_variables"] = int(match.group(1))
            result["dimacs_num_clauses"] = int(match.group(2))

    elif format_name == "mps":
        section = None
        num_rows = 0
        columns = set()
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith("NAME"):
                section = "NAME"
            elif stripped == "ROWS":
                section = "ROWS"
            elif stripped == "COLUMNS":
                section = "COLUMNS"
            elif stripped in ("RHS", "RANGES", "BOUNDS", "ENDATA"):
                section = stripped
            elif section == "ROWS" and stripped:
                parts = stripped.split()
                if parts[0] != "N":
                    num_rows += 1
            elif section == "COLUMNS" and stripped:
                parts = stripped.split()
                if parts:
                    columns.add(parts[0])
            if section == "ENDATA":
                break
        if num_rows or columns:
            result["mps_num_rows"] = num_rows
            result["mps_num_columns"] = len(columns)

    elif format_name == "lp":
        # Count constraints in the "Subject To" section
        in_subject_to = False
        num_constraints = 0
        for line in content.split('\n'):
            stripped = line.strip().lower()
            if stripped in ("subject to", "st", "s.t."):
                in_subject_to = True
            elif stripped in ("bounds", "binary", "generals", "end"):
                in_subject_to = False
            elif in_subject_to and stripped and ":" in stripped:
                num_constraints += 1
        if num_constraints:
            result["lp_num_constraints"] = num_constraints

    return result


def _enrich_from_model(model, metadata):
    """Add decision variable and objective info from a CPMpy Model to metadata.

    This is called by transforms that produce CPMpy models (Load, Translate)
    via their ``enrich_metadata`` method. It adds:

    - ``decision_variables``: list of dicts with name, type, lb, ub for each variable
    - ``objective``: string representation of the objective expression (if any)
    - ``objective_is_min``: True if minimizing, False if maximizing (if any)
    """
    if not hasattr(model, 'constraints'):
        return metadata  # not a CPMpy Model

    from cpmpy.transformations.get_variables import get_variables_model
    from cpmpy.expressions.variables import _BoolVarImpl

    variables = get_variables_model(model)
    metadata['decision_variables'] = [
        {
            "name": v.name,
            "type": "bool" if isinstance(v, _BoolVarImpl) else "int",
            "lb": int(v.lb),
            "ub": int(v.ub),
        }
        for v in variables
    ]

    if model.objective_ is not None:
        metadata['objective'] = str(model.objective_)
        metadata['objective_is_min'] = bool(model.objective_is_min)

    return metadata


class Compose:
    """
    Composes several transforms together, applying them sequentially.

    Each transform in the sequence receives the output of the previous one.
    Transforms that define ``enrich_metadata(data, metadata)`` can contribute
    additional fields to the metadata dictionary. Each sub-transform's
    ``enrich_metadata`` receives the intermediate result *it* produced, so a
    :class:`Load` inside ``Compose([Load(...), Serialize(...)])`` sees the
    CPMpy model, not the final serialized string.

    Arguments:
        transforms (list[callable]): List of transforms to compose.

    Example::

        >>> transform = Compose([
        ...     Load(load_wcnf, open=dataset.open),
        ...     Serialize("dimacs"),
        ... ])
        >>> dataset = MSEDataset(transform=transform)
        >>> dimacs_string, metadata = dataset[0]
    """

    def __init__(self, transforms):
        if not isinstance(transforms, (list, tuple)):
            raise TypeError("transforms must be a list or tuple of callables")
        self.transforms = list(transforms)
        self._steps = []  # (transform, its_output) pairs from last __call__

    def __call__(self, x):
        self._steps = []
        for t in self.transforms:
            x = t(x)
            self._steps.append((t, x))
        return x

    def enrich_metadata(self, data, metadata):
        """Delegate to each sub-transform's enrich_metadata with its own output."""
        for t, result in self._steps:
            if hasattr(t, 'enrich_metadata'):
                metadata = t.enrich_metadata(result, metadata)
        return metadata

    def __repr__(self):
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"    {t},")
        lines.append("])")
        return "\n".join(lines)


class Open:
    """
    Transform that opens a file path and returns its text contents.
    Handles decompression via the provided ``open`` callable.

    Arguments:
        open (callable): A callable that opens a file path and returns a
            file-like object. Typically ``dataset.open``. Defaults to
            Python's built-in ``open``.

    Example::

        >>> dataset = MSEDataset(transform=Open(open=dataset.open))
        >>> raw_string, metadata = dataset[0]
    """

    def __init__(self, open=_builtins_open):
        self._open = open

    def __call__(self, file_path):
        with self._open(file_path) as f:
            return f.read()

    def __repr__(self):
        if self._open is _builtins_open:
            return f"{self.__class__.__name__}()"
        return f"{self.__class__.__name__}(open={self._open})"


class Load:
    """
    Transform that loads a file path into a CPMpy model.
    
    Loading always handles reading internally. This transform combines reading
    (decompressing + reading raw contents) and loading (turning raw contents
    into a CPMpy model) into a single step.

    Implements ``enrich_metadata`` to add model verification information
    (decision variables, objective) to the metadata dictionary. This is
    called automatically by the dataset's ``__getitem__``.

    Arguments:
        loader (callable): A loader function that takes raw content string and
            returns a CPMpy model. Can be a dataset's ``loader`` method or a
            loader function that supports raw strings (e.g., ``load_wcnf``,
            ``load_opb``, ``load_xcsp3``, etc.).
        open (callable, optional): A callable to open files for reading.
            Typically ``dataset.open``. Defaults to Python's built-in ``open``.
        **kwargs: Additional keyword arguments passed to the loader (if supported).

    Example::

        >>> # Using dataset's loader method
        >>> dataset = MSEDataset(transform=Load(dataset.loader, open=dataset.open))
        >>> model, metadata = dataset[0]
        
        >>> # Using a loader function that supports raw strings
        >>> from cpmpy.tools.io.wcnf import load_wcnf
        >>> dataset = MSEDataset(transform=Load(load_wcnf, open=dataset.open))
        >>> model, metadata = dataset[0]
        >>> metadata['decision_variables']  # list of variable descriptors
        >>> metadata['objective']           # objective expression string (if any)
    """

    def __init__(self, loader, open=None, **kwargs):
        self.loader = loader
        self._open = open if open is not None else _builtins_open
        self.kwargs = kwargs

    def __call__(self, file_path):
        # Step 1: Reading - decompress and read raw file contents
        with self._open(file_path) as f:
            content = f.read()
        
        # Step 2: Loading - turn raw contents into CPMpy model
        # Prepare kwargs, ensuring 'open' doesn't conflict
        kwargs = {k: v for k, v in self.kwargs.items() if k != 'open'}
        
        # Handle both regular functions and classmethods/staticmethods
        if hasattr(self.loader, '__self__') or isinstance(self.loader, classmethod):
            # It's a bound method or classmethod, call it directly
            return self.loader(content, **kwargs)
        else:
            # It's a regular function, call it normally
            return self.loader(content, **kwargs)

    def enrich_metadata(self, data, metadata):
        """Add model verification info if data is a CPMpy Model."""
        return _enrich_from_model(data, metadata)

    def __repr__(self):
        loader_name = getattr(self.loader, '__name__', repr(self.loader))
        return f"{self.__class__.__name__}(loader={loader_name})"


class Serialize:
    """
    Transform that serializes a CPMpy model to a string in a given format.

    Arguments:
        writer (callable or str): Either a writer function (e.g., ``write_dimacs``, ``write_opb``)
            or a format name string (e.g., ``"dimacs"``, ``"mps"``, ``"opb"``) that will be resolved
            to the appropriate writer function. If a string, must be a format supported by
            :func:`cpmpy.tools.io.writer.write`.
        **kwargs: Additional keyword arguments passed to the writer
            (e.g., ``header``, ``verbose``).

    Example::

        >>> # Using format name string
        >>> transform = Compose([
        ...     Load(load_wcnf, open=dataset.open),
        ...     Serialize("dimacs"),
        ... ])
        
        >>> # Using writer function directly
        >>> from cpmpy.tools.dimacs import write_dimacs
        >>> transform = Compose([
        ...     Load(load_wcnf, open=dataset.open),
        ...     Serialize(write_dimacs),
        ... ])
    """

    def __init__(self, writer, **kwargs):
        self.writer = writer
        self.kwargs = kwargs

    def __call__(self, model):
        # Determine writer function
        if callable(self.writer):
            # writer is a callable function
            return self.writer(model, fname=None, **self.kwargs)
        else:
            # writer is a format name string, use unified write function
            from cpmpy.tools.io.writer import write
            return write(model, format=self.writer, file_path=None, **self.kwargs)

    def __repr__(self):
        if callable(self.writer):
            writer_name = getattr(self.writer, '__name__', repr(self.writer))
            return f"{self.__class__.__name__}(writer={writer_name})"
        else:
            return f"{self.__class__.__name__}(writer='{self.writer}')"


class Translate:
    """
    Transform that translates a file from one format to another.
    Combines reading (decompressing + reading raw contents), loading (turning raw
    contents into a CPMpy model), and writing (serializing the model) in one step.

    Implements ``enrich_metadata`` to add model verification information
    from the intermediate CPMpy model to the metadata dictionary.

    Arguments:
        loader (callable): A loader function that takes raw content string and
            returns a CPMpy model. Can be a dataset's ``loader`` method or a
            loader function that supports raw strings (e.g., ``load_wcnf``,
            ``read_opb``, ``read_xcsp3``, etc.).
        writer (callable or str): Either a writer function (e.g., ``write_dimacs``, ``write_opb``)
            or a format name string (e.g., ``"dimacs"``, ``"mps"``) that will be resolved
            to the appropriate writer function.
        open (callable, optional): A callable to open compressed files for reading.
            Typically ``dataset.open``. Defaults to Python's built-in ``open``.
        **kwargs: Additional keyword arguments passed to the writer.

    Example::

        >>> # Using format name string
        >>> transform = Translate(dataset.loader, "dimacs", open=dataset.open)
        >>> dataset = MSEDataset(transform=transform)
        >>> dimacs_string, metadata = dataset[0]
        
        >>> # Using writer function directly
        >>> from cpmpy.tools.dimacs import write_dimacs
        >>> transform = Translate(dataset.loader, write_dimacs, open=dataset.open)
        >>> dataset = MSEDataset(transform=transform)
        >>> dimacs_string, metadata = dataset[0]
        >>> metadata['decision_variables']  # from the intermediate model
    """

    def __init__(self, loader, writer, open=None, **kwargs):
        self.loader = loader
        self.writer = writer
        self._open = open if open is not None else _builtins_open
        self.kwargs = kwargs
        self._last_model = None

    def __call__(self, file_path):
        # Step 1: Reading - decompress and read raw file contents
        with self._open(file_path) as f:
            content = f.read()
        
        # Step 2: Loading - turn raw contents into CPMpy model
        loader_kwargs = {k: v for k, v in self.kwargs.items() if k != 'open'}
        
        # Handle both regular functions and classmethods/staticmethods
        if hasattr(self.loader, '__self__') or isinstance(self.loader, classmethod):
            model = self.loader(content, **loader_kwargs)
        else:
            model = self.loader(content, **loader_kwargs)

        self._last_model = model
        
        # Step 3: Writing - serialize model to string
        writer_kwargs = {k: v for k, v in self.kwargs.items() if k != 'open'}
        if callable(self.writer):
            # writer is a callable function
            return self.writer(model, fname=None, **writer_kwargs)
        else:
            # writer is a format name string, use unified write function
            from cpmpy.tools.io.writer import write
            return write(model, format=self.writer, file_path=None, **writer_kwargs)

    def enrich_metadata(self, data, metadata):
        """Add model verification info from the intermediate model."""
        if self._last_model is not None:
            metadata = _enrich_from_model(self._last_model, metadata)
        return metadata

    def __repr__(self):
        loader_name = getattr(self.loader, '__name__', repr(self.loader))
        if callable(self.writer):
            writer_name = getattr(self.writer, '__name__', repr(self.writer))
            return f"{self.__class__.__name__}(loader={loader_name}, writer={writer_name})"
        else:
            return f"{self.__class__.__name__}(loader={loader_name}, writer='{self.writer}')"


class SaveToFile:
    """
    Transform that writes its input string to a file and returns the file path.

    When ``write_metadata=True``, also writes a ``.meta.json`` sidecar file
    alongside each output file. The sidecar contains portable instance
    metadata from the dataset (filtered by
    :func:`~cpmpy.tools.dataset._base.portable_instance_metadata`) and
    format-specific metadata extracted from the written content.

    Arguments:
        output_dir (str): Directory to write files to (created if needed).
        extension (str): File extension for output files (e.g., ``".cnf"``, ``".mps"``).
        naming (callable, optional): Function that receives the current data
            and returns a filename stem. If None, uses a counter.
        write_metadata (bool): If True, writes a ``.meta.json`` sidecar file
            next to each saved file. Requires being used inside a
            :class:`Compose` with the dataset's ``__getitem__``.
        target_format (str, optional): Target format name for format-specific
            metadata extraction. If None, inferred from extension.

    Example::

        >>> transform = Compose([
        ...     Translate(load_wcnf, "dimacs", open=dataset.open),
        ...     SaveToFile("output/", extension=".cnf", write_metadata=True),
        ... ])
    """

    def __init__(self, output_dir, extension="", naming=None,
                 write_metadata=False, target_format=None):
        self.output_dir = output_dir
        self.extension = extension
        self.naming = naming
        self.write_metadata = write_metadata
        self.target_format = target_format
        self._counter = 0
        self._last_path = None
        self._last_content = None

    def __call__(self, content):
        os.makedirs(self.output_dir, exist_ok=True)

        if self.naming is not None:
            name = self.naming(content)
        else:
            name = f"instance_{self._counter}"
            self._counter += 1

        file_path = os.path.join(self.output_dir, name + self.extension)
        with _builtins_open(file_path, "w") as f:
            f.write(content)
        self._last_path = file_path
        self._last_content = content
        return file_path

    def enrich_metadata(self, data, metadata):
        """Write a metadata sidecar alongside the saved file if enabled.

        The sidecar mirrors the structure used by ``translate_datasets.py``:
        ``dataset``, ``instance_name``, ``category``, ``instance_metadata``,
        ``translation``, and ``format_metadata`` sections.
        """
        if not self.write_metadata or self._last_path is None:
            return metadata

        from cpmpy.tools.dataset._base import portable_instance_metadata

        sidecar = {}

        # Dataset-level metadata (if present in the metadata dict)
        if "dataset" in metadata:
            # When called from __getitem__, metadata has 'dataset' as a string name.
            # Try to reconstruct richer dataset info from what's available.
            sidecar["dataset"] = {"name": metadata.get("dataset", "")}

        # Instance identification
        sidecar["instance_name"] = metadata.get("name", "")
        if "path" in metadata:
            sidecar["source_file"] = metadata["path"]

        # Category (year, track, variant, etc. — whatever the dataset provides)
        # These are the non-standard keys that category() returns
        _known_base = {"dataset", "name", "path"}
        category_keys = {
            k: v for k, v in metadata.items()
            if k in ("year", "track", "variant", "family")
        }
        if category_keys:
            sidecar["category"] = category_keys

        # Portable instance metadata
        sidecar["instance_metadata"] = portable_instance_metadata(metadata)

        # Translation info
        fmt = self.target_format or self._infer_format()
        import cpmpy
        from cpmpy.tools.io.writer import writer_dependencies
        translation = {
            "target_format": fmt or "",
            "cpmpy_version": cpmpy.__version__,
        }
        if fmt:
            deps = writer_dependencies(fmt)
            if deps:
                translation["writer_dependencies"] = deps
        sidecar["translation"] = translation

        # Format-specific metadata from the written content
        if fmt and self._last_content:
            sidecar["format_metadata"] = extract_format_metadata(
                self._last_content, fmt
            )

        sidecar_path = self._last_path + ".meta.json"
        with _builtins_open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        return metadata

    def _infer_format(self):
        """Infer format name from the file extension."""
        ext_to_format = {
            ".cnf": "dimacs", ".opb": "opb", ".mps": "mps",
            ".lp": "lp", ".fzn": "fzn", ".gms": "gms", ".pip": "pip",
        }
        return ext_to_format.get(self.extension)

    def __repr__(self):
        return f"{self.__class__.__name__}(output_dir='{self.output_dir}', extension='{self.extension}')"


class Lambda:
    """
    Wraps an arbitrary callable with a descriptive name for better repr.

    Arguments:
        fn (callable): The function to wrap.
        name (str, optional): Display name for repr. Defaults to the
            function's ``__name__`` attribute.

    Example::

        >>> transform = Compose([
        ...     Load(load_wcnf, open=dataset.open),
        ...     Lambda(lambda m: len(m.constraints), name="count_constraints"),
        ... ])
    """

    def __init__(self, fn, name=None):
        self.fn = fn
        self.name = name or getattr(fn, '__name__', 'lambda')

    def __call__(self, x):
        return self.fn(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
