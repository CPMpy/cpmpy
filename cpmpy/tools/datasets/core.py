"""
This module provides an abstract, PyTorch-style dataset interface for Constraint Optimisation (CO) benchmarks.
With a single line of code, classical benchmarks such as XCSP3, PSPLib, JSPLib, etc. can be downloaded and iterated over.

**Available datasets:**

- :doc:`XCSP3Dataset </api/tools/datasets/xcsp3>`: XCSP3 competition benchmark instances for constraint satisfaction and optimization.
- :doc:`JSPLibDataset </api/tools/datasets/jsplib>`: Job Shop Scheduling Problem benchmark library.
- :doc:`PSPLibDataset </api/tools/datasets/psplib>`: Project Scheduling Problem Library (RCPSP) benchmark instances.
- :doc:`MIPLibDataset </api/tools/datasets/miplib>`: Mixed Integer Programming Library benchmark instances.
- :doc:`MaxSATEvalDataset </api/tools/datasets/mse>`: MaxSAT Evaluation competition benchmark instances.
- :doc:`OPBDataset </api/tools/datasets/opb>`: Pseudo-Boolean Competition benchmark instances.
- :doc:`SATDataset </api/tools/datasets/sat>`: SAT competition benchmark instances (DIMACS CNF).
- :doc:`NurseRosteringDataset </api/tools/datasets/nurserostering>`: Nurse rostering benchmark instances.


.. note::

    Whilst the dataset class provides a PyTorch compatible access pattern, it has no actual dependency on 
    PyTorch and can be used without installing this library.

**Class hierarchy**::

    Dataset (ABC)
    └── FileDataset (ABC)
        └── XCSP3Dataset
        └── JSPLibDataset
        └── PSPLibDataset
        └── MIPLibDataset
        └── MaxSATEvalDataset
        └── OPBDataset
        └── SATDataset
        └── NurseRosteringDataset
        └── (your dataset here)

Whilst the class hierarchy will support more exotic dataset types in the future, with a structure put in place 
that takes inspiration from conventions within the ML community, currently only file-based datasets are supported, 
i.e. datasets where the instances are stored as files on disk. 

The base classes standardize:

- download and local storage of benchmark instances (file-based datasets)
- instance access via ``__len__`` / ``__getitem__`` (PyTorch compatibility)
- optional ``parse``/``transform``/``target_transform`` arguments
- dataset metadata (with sidecar collection)
    
To implement a new dataset, one needs to subclass one of the abstract dataset classes,
and provide implementation for the following methods:

- ``category``: return a dictionary of category labels, describing to which subset the dataset has been restricted (year, track, ...)
- ``download``: download the dataset (helper function :func:`_download_file` is provided)

Some optional methods to overwrite are:

- ``collect_instance_metadata``: collect metadata about individual instances (e.g. number of variables, constraints, ...), 
  potentially domain specific 
- ``open``: how to open the instance file (e.g. for compressed files using .xz, .lzma, .gz, ...)

Datasets must also implement the following dataset metadata attributes:

- ``name``: the name of the dataset
- ``description``: a short description of the dataset
- ``homepage``: a URL to the homepage of the dataset
- ``citation``: optionally, a list of citations for the dataset

All parts for which an implementation must be provided are marked with an @abstractmethod decorator, 
raising a NotImplementedError if not overwritten.

Dataset files are preferably downloaded as-is, without any preprocessing or decompression. Upon initial download,
instance-level metadata gets automatically collected and stored in a JSON sidecar file. All subsequent accesses to the dataset
will use the sidecar file to avoid re-collecting the metadata.

Iterating over the dataset is done in the same way as a PyTorch dataset. It returns 2-tuples (x,y) of:

- x: instance reference (a file path is the only supported instance reference type at the moment)
- y: instance metadata  (solution, features, origin, etc.)

Example:

.. code-block:: python

    dataset = MyDataset(download=True)
    for instance, info in dataset:
        print(instance, info)

The dataset also supports PyTorch-style transforms and target transforms.

.. code-block:: python

    dataset = MyDataset(download=True, transform=my_model_loader)
    for model, info in dataset:
        ...

===============
List of classes
===============

.. autosummary::
    :nosignatures:

    Dataset
    FileDataset

==================
List of functions
==================

.. autosummary::
    :nosignatures:

    from_files
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import pathlib
import io
import sys
import tempfile
from typing import Any, Optional, Tuple, List, Dict, Iterator, Callable, ClassVar
from urllib.error import URLError
from urllib.request import HTTPError, Request, urlopen

# tqdm as an optional dependency, provides prettier progress bars
tqdm: Any = None
try:
    from tqdm import tqdm as _tqdm
    tqdm = _tqdm
except ImportError:
    pass

import cpmpy as cp


class Dataset(ABC):
    """
    Abstract base class for CO datasets.

    The `Dataset` class is an abstract base class for all datasets. It provides a standardized interface for 
    the PyTorch-compatible access pattern for CO benchmark datasets. It is not meant to be instantiated directly, 
    but rather subclassed. Have a look at :class:`FileDataset` for a concrete implementation.

    Each instance in a dataset is characterised by a (x, y) pair of:

        - x: instance reference (e.g., file path, database key, generated seed, ...)
        - y: instance metadata  (solution, features, origin, etc.)

    Instances are indexed by a unique identifier can be accessed by that identifier. 
    For example its positional index within the dataset.

    Implementing this class requires implementing the following methods:

        - ``__len__``: return the total number of instances
        - ``__getitem__``: return the instance and metadata at the given index / identifier

    And providing the following class attributes:

        - ``name``: the name of the dataset
        - ``description``: a short description of the dataset
        - ``homepage``: a URL to the homepage of the dataset
        - ``citation``: optionally, a list of citations for the dataset

    Optional methods to overwrite:
    
        - ``instance_metadata``: return the metadata for a given instance
    """
    

    # -------------- Dataset-level metadata (override in subclasses) ------------- #

    name: ClassVar[str]
    description: ClassVar[str]
    homepage: ClassVar[str]
    citation: ClassVar[List[str]] = []
    
    # ---------------------------------------------------------------------------- #

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Arguments:
            transform (callable, optional):            Optional transform applied to the instance reference.
            target_transform (callable, optional):     Optional transform applied to the instance metadata.
        """
        self.transform = transform
        self.target_transform = target_transform

   
    # ---------------------------------------------------------------------------- #
    #                     Methods to implement in subclasses:                      #
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of instances.
        """
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return the instance and metadata at the given index / identifier.

        Returns:
            x: instance reference (e.g., file path, database key, generated seed, ...)
            y: instance metadata  (solution, features, origin, etc.)
        """
        pass

    @abstractmethod
    def instance_metadata(self, instance: Any) -> Dict[str, Any]:
        """
        Return the metadata for a given instance.

        Arguments:
            instance: the instance identifier for which to return the metadata

        Returns:
            dict: The metadata for the instance.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #

    @classmethod
    def dataset_metadata(cls) -> Dict[str, Any]:
        """
        Return dataset-level metadata as a dictionary.

        Returns:
            dict: The dataset-level metadata.
        """
        # Handle both single string and list of strings for citations
        if isinstance(cls.citation, str):
            citations = [cls.citation] if cls.citation else []
        else:
            citations = list(cls.citation)

        return {
            "name": cls.name,
            "description": cls.description,
            "homepage": cls.homepage,
            "citation": citations,
        }


    # ---------------------------------------------------------------------------- #
    #                                   Internals                                  #
    # ---------------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """
        Iterate over the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def __init_subclass__(cls, **kwargs: Any):
        """
        A collection of checks to ensure that the subclass is a valid Dataset subclass.
        """
        super().__init_subclass__(**kwargs)

        # Check that the subclass is not the Dataset class itself
        if cls is Dataset:
            raise TypeError("Dataset is an abstract base class and cannot be instantiated directly")

        # Abstract intermediate classes (e.g. FileDataset) still have
        # unimplemented abstractmethods; only concrete subclasses must define the
        # dataset-level attributes -> test to skip the next check
        is_abstract = any(
            getattr(getattr(cls, name, None), "__isabstractmethod__", False)
            for name in dir(cls)
        )
        if is_abstract:
            return

        # Check that the subclass defines the required class attributes
        for attr in ("name", "description", "homepage"):
            if attr not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define class attribute {attr!r}")


class FileDataset(Dataset):
    """
    Abstract base class for PyTorch-style datasets of file-based CO benchmarking sets.

    The `FileDataset` class provides a standardized interface for downloading and
    accessing file-backed benchmark instances. This class should not be used on its own.
    Either have a look at one of the concrete subclasses, providing access to 
    well-known datasets from the community, or use this class as the base for your own dataset.

    Two dataset styles are supported:

    - Model-defined instances: files directly encode variables/constraints/objective
      (for example XCSP3, OPB, DIMACS, FlatZinc). In this case, users typically
      pass a model-builder as ``transform``, converting the raw file instance into a model.
    - Data-only instances: files encode problem data for a fixed family, but no
      model. In this case, subclasses should override ``parse()`` and users can
      enable ``parse=True`` to obtain parsed intermediate data structures
      (for example table/dict structures for RCPSP-style scheduling data), then
      build a model separately or via a transform.
    """ 

    # Extension for metadata sidecar files
    METADATA_EXTENSION: ClassVar[str] = ".meta.json"

    def __init__(
            self,
            dataset_dir: str | os.PathLike[str] = ".",
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
            download: bool = False,
            parse: bool = False,
            extension: str = ".txt",
            **kwargs: Any
        ):
        """
        Constructor for the FileDataset base class.

        Arguments:
            dataset_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).
            parse (bool): If True, run ``self.parse(instance_path)`` before
                applying ``transform``. Intended for data-only datasets that do
                not directly encode a model in the source file.
            extension (str): Extension of the instance files. Used to filter instance files within the dataset directory.
            **kwargs: Advanced options. Currently supports:
                - ignore_sidecar (bool): If True, do not read/write metadata
                  sidecars and collect metadata on demand at iteration time
                  using ``collect_instance_metadata()`` (default=False).

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested dataset variant (e.g. year/track) is not available.
            ValueError: If the dataset directory does not contain any instance files.
        """

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.extension = extension
        self._parse = parse

        # Advanced options
        self._ignore_sidecar = kwargs.pop("ignore_sidecar", False)

        if not self._check_exists():
            if not download:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")
            else:
                self.download()
                if not self._ignore_sidecar:
                    self._collect_all_metadata()
                files = self._list_instances()
                print(f"Finished downloading {len(files)} instances")

        files = self._list_instances()
        if len(files) == 0:
            raise ValueError(f"Cannot find any instances inside dataset {self.dataset_dir}. Is it a valid dataset? If so, please report on GitHub.")

        super().__init__(transform=transform, target_transform=target_transform)

    def _check_exists(self) -> bool:
        """
        Check if the dataset exists (has been downloaded).
        """
        return self.dataset_dir.exists()

    # ---------------------------------------------------------------------------- #
    #                     Methods to implement in subclasses:                      #
    # ---------------------------------------------------------------------------- #


    @abstractmethod
    def categories(self) -> Dict[str, Any]:
        """
        Labels to distinguish instances into categories matching to those of the dataset,
        e.g. ``year`` or ``track``.
        """
        pass

    @abstractmethod
    def download(self, *args: Any, **kwargs: Any):
        """
        Download the dataset.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                        Methods to optionally overwrite                       #
    # ---------------------------------------------------------------------------- #

    def collect_instance_metadata(self, file: pathlib.Path) -> Dict[str, Any]:
        """
        Provide domain-specific instance metadata.
        Called once after download for each instance.

        Arguments:
            file: path to the instance file

        Returns:
            dict with instance-specific metadata fields
        """
        return {}

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        """
        How an instance file from the dataset should be opened.
        Especially usefull when files come compressed and won't work with
        Python standard library's 'open', e.g. '.xz', '.lzma'.

        Arguments:
            instance (os.PathLike): File path to the instance file.

        Returns:
            io.TextIOBase: The opened file handle.
        """
        return open(instance, "r")

    def read(self, instance: os.PathLike) -> str:    
        """
        Read raw file contents from an instance file.
        Handles optional decompression automatically via dataset.open().
        
        Arguments:
            instance (os.PathLike): File path to the instance file.
        Returns:
            str: The raw file contents.
        """
        with self.open(instance) as f:
            return f.read()

    def parse(self, instance: os.PathLike) -> Any:
        """
        Parse an instance file into intermediate data structures.

        Override this for datasets whose files contain problem data but not an
        explicit model. Typical outputs are structures like tables, arrays, and
        dictionaries that can then be passed to a separate model-construction
        function.

        Default behavior is ``read(instance)``, i.e. return raw text content.

        Arguments:
            instance (os.PathLike): File path to the instance file.

        Returns:
            The parsed intermediate data structure(s).
        """
        return self.read(instance)


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #


    def instance_metadata(self, instance: os.PathLike) -> Dict[str, Any]:
        """
        Return the metadata for a given instance file.

        Arguments:
            file (os.PathLike): Path to the instance file.

        Returns:
            dict: The metadata for the instance.
        """
        metadata = {
            'id': str(instance),
            'dataset': self.name,
            'categories': self.categories(),
            'name': self._instance_name(instance),
            'path': instance,
        }

        # Advanced mode: bypass sidecars and collect metadata on demand.
        if self._ignore_sidecar:
            metadata.update(self.collect_instance_metadata(pathlib.Path(instance)))
            return metadata
        else:
            # Load sidecar metadata if it exists
            meta_path = self._metadata_path(instance)
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    sidecar = json.load(f)
                # Structured: flatten instance_metadata
                metadata.update(sidecar.get("instance_metadata", {}))
            return metadata


    # ---------------------------------------------------------------------------- #
    #                                   Internals                                  #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ Instance access ----------------------------- #

    def _list_instances(self) -> List[pathlib.Path]:
        """
        List all instance files, excluding metadata sidecar files.

        Returns a sorted list of `pathlib.Path` objects for all instance files
        matching the dataset's extension pattern.
        """
        return sorted([
            f for f in self.dataset_dir.rglob(f"*{self.extension}")
            if f.is_file() and not str(f).endswith(self.METADATA_EXTENSION)
        ])

    def __len__(self) -> int:
        """
        Return the total number of instances.
        """
        return len(self._list_instances())

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return the instance and metadata at the given index.
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        files = self._list_instances()
        file_path = files[index]

        metadata = self.instance_metadata(file_path)
        if self.target_transform:
            metadata = self.target_transform(metadata)

        # Instance reference is a string path (PyTorch-style); parse/transform may replace it.
        data: Any = str(file_path)

        # Built-in parse stage: parse the instance file into intermediate data structures.
        # Mostly meant for datasets where files represent data and modeling is separate.
        if self._parse:
            data = self.parse(file_path)

        if self.transform:
            if isinstance(data, (str, os.PathLike)):
                data = self.transform(data, open=self.open)
            else:
                # Convenience for parse-first datasets where parse() returns
                # tuples and model builders take positional args.
                if isinstance(data, tuple):
                    data = self.transform(*data)
                else:
                    data = self.transform(data)
            # Let transforms contribute to metadata (e.g. model verification info)
            if hasattr(self.transform, 'enrich_metadata'):
                metadata = self.transform.enrich_metadata(data, metadata)

        return data, metadata


    # ---------------------------- Metadata collection --------------------------- #

    def _metadata_path(self, instance_path: os.PathLike) -> pathlib.Path:
        """
        Return the path to the `.meta.json` sidecar file for a given instance.

        Arguments:
            instance_path (os.PathLike): Path to the instance file.

        Returns:
            pathlib.Path: Path to the `.meta.json` sidecar file.
        """
        return pathlib.Path(str(instance_path) + self.METADATA_EXTENSION)

    def _instance_name(self, instance: os.PathLike) -> str:
        """
        Logical instance name: the filename with the dataset extension stripped.
        """
        name = pathlib.Path(instance).name
        ext = self.extension
        return name[:-len(ext)] if ext and name.endswith(ext) else name

    def _collect_all_metadata(self, force: bool = False):
        """
        Collect and store structured metadata sidecar files for all instances.

        Writes a structured `.meta.json` sidecar alongside each instance with:

        - `dataset`: dataset-level metadata (name, description, url, ...)
        - `instance_name`: logical instance name (filename stem)
        - `source_file`: path to the instance file
        - `categories`: dataset category labels (year, track, variant, family)
        - `instance_metadata`: portable domain-specific metadata

        Arguments:
            force (bool): If True, re-collect instance metadata even if sidecar
                files already exist.
        """
        files = self._list_instances()

        # Filter files that need processing
        if force:
            files_to_process = files
        else:
            files_to_process = [f for f in files if not self._metadata_path(f).exists()]

        if not files_to_process:
            return

        if tqdm is not None:
            file_iter = tqdm(files_to_process, desc="Collecting metadata", unit="file")
        else:
            file_iter = files_to_process
            print(f"Collecting metadata for {len(files_to_process)} instances...")

        for file_path in file_iter:
            # Build structured, self-contained sidecar. Let exceptions from
            # collect_instance_metadata() propagate: a failure signals a corrupt
            # instance or a bug, which should surface rather than be buried.
            sidecar: Dict[str, Any] = {
                "dataset": self.dataset_metadata(),
                "instance_name": self._instance_name(file_path),
                "source_file": str(file_path.relative_to(self.dataset_dir)),
                "categories": self.categories(),
                "instance_metadata": self.collect_instance_metadata(file_path),
            }

            with open(self._metadata_path(file_path), "w") as f:
                json.dump(sidecar, f, indent=2)


    # ----------------------------- Download methods ----------------------------- #

    @staticmethod
    def _download_file(url: str, target: str, destination: Optional[str] = None,
                        desc: Optional[str] = None,
                        chunk_size: int = 1024 * 1024) -> pathlib.Path:
        """
        Download a file from a URL with progress bar and speed information.

        This method provides a reusable download function with progress updates
        similar to pip and uv, showing download progress, speed, and ETA.

        Arguments:
            url (str): The original URL to download from (used as fallback).
            target (str): The target filename to download.
            destination (str, optional): The destination path to save the file.
            desc (str, optional): Description to show in the progress bar.
                                  If None, uses the filename.
            chunk_size (int): Size of each chunk for download in bytes (default=1MB).

        Returns:
            pathlib.Path: The destination path where the downloaded file is saved.
        """

        if desc is None:
            desc = target

        temp_destination = None
        if destination is None:
            temp_destination = tempfile.NamedTemporaryFile(delete=False)
            destination = temp_destination.name
        else:
            # Create parent directory if it doesn't exist and destination has a directory component
            dest_dir = os.path.dirname(destination)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)

        try:
            req = Request(url + target)
            with urlopen(req) as response:
                total_size = int(response.headers.get('Content-Length', 0))
            
            # Convert destination to Path for _download_sequential
            download_path = pathlib.Path(destination) if destination is not None else pathlib.Path(temp_destination.name)
            FileDataset._download_sequential(url + target, download_path, total_size, desc, chunk_size)
    
            if destination is None:
                temp_destination.close()

            return pathlib.Path(destination)

        except (HTTPError, URLError) as e:
            raise ValueError(f"Failed to download file from {url + target}. Error: {str(e)}")

    @staticmethod
    def _download_sequential(url: str, filepath: os.PathLike, total_size: int, desc: str,
                             chunk_size: int = 1024 * 1024):
        """
        Download file sequentially with progress bar.
        """
        filepath = pathlib.Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        req = Request(url)
        with urlopen(req) as response:
            if tqdm is not None:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True,
                             unit_divisor=1024, desc=f"Downloading {desc}", file=sys.stdout,
                             miniters=1, dynamic_ncols=True, ascii=False) as pbar:
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # Unknown size
                    with tqdm(unit='B', unit_scale=True, unit_divisor=1024,
                             desc=f"Downloading {desc}", file=sys.stdout, miniters=1,
                             dynamic_ncols=True, ascii=False) as pbar:
                        with open(filepath, 'wb') as f:
                            while True:
                                chunk = response.read(chunk_size)
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Fallback to simple download if tqdm is not available
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)


def from_files(dataset_dir: os.PathLike, extension: str = ".txt") -> FileDataset:
    """
    Create a FileDataset from a list of files.

    Example:

    .. code-block:: python

        dataset = from_files("path/to/dataset_files", extension=".txt")
        for x, y in dataset:
            print(x, y)
    """
    class FromFilesDataset(FileDataset):
        name = pathlib.Path(dataset_dir).name
        description = ""
        homepage = ""

        def __init__(self, dataset_dir: os.PathLike, extension: str = ".txt"):
            super().__init__(dataset_dir=dataset_dir, extension=extension)

        def categories(self) -> Dict[str, Any]:
            return {}

        def download(self):
            pass  # already in local files

    return FromFilesDataset(dataset_dir, extension)
