"""
Dataset Base Classes

This module defines multiple abstract datasets, a hierarchy of classes which together
serve as the foundation for competition and application oriented benchmarking datasets.

They enable the loading and managing of well-known benchmark instance collections 
from the Constraint Optimisation (CO) community.

It standardizes how datasets are downloaded, stored, accessed, and optionally transformed.

It provides a Pytorch compatible interface (constructor arguments like "transform" and the
methods __len__ and __getitem__ for iterating over the dataset).

Additionaly, it provides a collection of methods and helper functions to adapt the dataset
to the specific usecase requirements of constraint optimisation benchmarks.

To implement a new dataset, one needs to subclass one of the abstract dataset classes,
and provide implementation for the following methods:
- _loader: loads a CPMpy model from a string representation of the instance (file)
- category: return a dictionary of category labels, describing to which subset the dataset has been restricted (year, track, ...)
- download: download the dataset (helper function :func:`_download_file` is provided)

Some optional methods to overwrite are:
- collect_instance_metadata: collect metadata about individual instances (e.g. number of variables, constraints, ...), potentially domain specific 
- open: how to open the instance file (e.g. for compressed files, use .xz, .lzma, .gz, ...)

Datasets must also implement the following dataset metadata attributes:
- name: the name of the dataset
- description: a short description of the dataset
- homepage: a URL to the homepage of the dataset
- citation: a list of citations for the dataset

Optional dataset schema metadata:
- features: a :class:`FeaturesInfo` schema describing domain-level instance fields
  (for example ``jobs``, ``machines``, ``optimum``, ``horizon``).
  This schema is exposed in dataset-level metadata and used by dataset cards and
  export formats (e.g. Croissant) to document the meaning and types of fields in
  instance metadata.

``features`` is optional. Default behavior is ``features = None``:
- dataset cards are still generated, but the "Instance Features (Domain Metadata)"
  section is omitted.
- Croissant export is still generated with core fields (``id``, ``name``, ``path``)
  and standard CP model feature fields; only domain-specific schema fields from
  ``features`` are omitted.
- instance metadata collection and loading behavior are unchanged; ``features``
  only documents schema and export metadata.

Feature inheritance and extension:
- child dataset classes may declare only new fields in ``features``; these are
  merged with inherited fields from the nearest ancestor defining ``features``.
- child fields override inherited fields with the same name.
- to use a completely custom schema, define the full ``features`` object in the
  child class.

All parts for which an implementation must be provided are marked with an @abstractmethod decorator, 
raising a NotImplementedError if not overwritten.

Datasets files should be downloaded as-is, without any preprocessing or decompression. Upon initial download,
instance-level metadata gets auto collected and stored in a JSON sidecar file. All subsequent accesses to the dataset
will use the sidecar file to avoid re-collecting the metadata.

Iterating over the dataset is done in the same way as a PyTorch dataset. It returns 2-tuples (x,y) of:
- x: instance reference (a file path is the only supported type at the moment)
- y: metadata (solution, features, origin, etc.)

Example:

.. code-block:: python

    dataset = MyDataset(download=True)
    for x, y in dataset:
        print(x, y)

The dataset also supports PyTorch-style transforms and target transforms.

.. code-block:: python
    from cpmpy.tools.io import load_wcnf
    from cpmpy.tools.datasets.metadata import to_croissant

    dataset = MyDataset(download=True, transform=load_wcnf(x), target_transform=to_croissant)
    for model, croissant_record in dataset:
        ...

For advanced operations on the datasets, like filtering, mapping, splitting, shuffling, sorting, etc., 
make use of the PyTorch tooling ecosystem (thanks to our compatible interface).

Example:
.. code-block:: python
    dataset = MyDataset(download=True, transform=load_wcnf(x), target_transform=to_croissant)
    
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
import pathlib
import io
import tempfile
import warnings
from itertools import product
from typing import Any, Iterator, Optional, Tuple, List, Union, Callable
from urllib.error import URLError
from urllib.request import HTTPError, Request, urlopen
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

from .metadata import FeaturesInfo, DatasetInfo, InstanceInfo
from .utils import extract_model_features, portable_instance_metadata

# tqdm as an optional dependency, provides prettier progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import cpmpy as cp


def _format_bytes(bytes_num):
    """
    Format bytes into human-readable string (e.g., KB, MB, GB).

    Used to display download progress.
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0


class classproperty:
    """
    Descriptor that makes a method work as a class-level property (no () needed).
    Similar to @property, but for class methods.
    """

    def __init__(self, func):
        self.func = func
        self.__isabstractmethod__ = getattr(func, '__isabstractmethod__', False)

    def __get__(self, instance, owner):
        return self.func(owner)

class Dataset(ABC):
    """
    Abstract base class for CO datasets.

    Each instance in a dataset is characterised by a (x, y) pair of:
        x: instance reference (e.g., file path, database key, generated seed, ...)
        y: metadata (solution, features, origin, etc.)
    """
    

    # -------------- Dataset-level metadata (override in subclasses) ------------- #

    @classproperty
    @abstractmethod
    def name(self) -> str: pass

    @classproperty
    @abstractmethod
    def description(self) -> str: pass

    @classproperty
    @abstractmethod
    def homepage(self) -> str: pass

    @classproperty
    def citation(self) -> List[str]:
        return []

    # OPTIONAL
    features: Optional[FeaturesInfo] = None                # domain_metadata field schema
    
    # ---------------------------------------------------------------------------- #

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        """
        Arguments:
            transform (callable, optional): Optional transform applied to the instance reference.
            target_transform (callable, optional): Optional transform applied to the metadata.
        """
        self.transform = transform
        self.target_transform = target_transform

    def __init_subclass__(cls, **kwargs):
        """
        Auto-merge ``features`` when a subclass declares only its *new* fields.

        If a subclass explicitly defines ``features``, it is merged with the
        nearest ancestor's ``features`` so the subclass only needs to list
        what is new.  The subclass fields take precedence over inherited ones.

        .. code-block:: python

            class MyJSPDataset(JSPLibDataset):
                # No need to repeat {jobs, machines, optimum, ...} — they are
                # inherited and merged in automatically.
                features = FeaturesInfo({"difficulty": ("float", "Computed difficulty score")})

                def collect_instance_metadata(self, file):
                    meta = super().collect_instance_metadata(file)
                    meta["difficulty"] = ...
                    return meta

        To *replace* rather than extend the parent schema, explicitly set
        ``features`` to the complete schema you want (the auto-merge still
        runs, but if you start from scratch the parent's fields will be
        absent from the parent's FeaturesInfo and won't be merged).
        Alternatively, set ``features = None`` to clear the schema entirely.
        """
        super().__init_subclass__(**kwargs)
        subclass_features = cls.__dict__.get("features")
        if subclass_features is None:
            return
        # Walk the MRO to find the nearest ancestor that has features defined
        for base in cls.__mro__[1:]:
            parent_features = base.__dict__.get("features")
            if parent_features is not None:
                cls.features = parent_features | subclass_features
                return


    # ---------------------------------------------------------------------------- #
    #                     Methods to implement in subclasses:                      #
    # ---------------------------------------------------------------------------- #

    @abstractmethod
    def instance_metadata(self, instance) -> InstanceInfo:
        """
        Return the metadata for a given instance file.

        Returns an :class:`~metadata.InstanceInfo`, which is a ``dict`` subclass
        so all existing ``meta['year']``, ``meta.get('jobs')`` access is unchanged.
        Structured access via ``info.domain_metadata``, ``info.model_features``,
        ``info.id``, etc. is additive.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #

    @classmethod
    def dataset_metadata(cls) -> DatasetInfo:
        """
        Return dataset-level metadata as a :class:`~metadata.DatasetInfo`.

        :class:`~metadata.DatasetInfo` is the dataset metadata object.
        It offers dict-compatible access for straightforward key-based usage
        (for example ``dataset_metadata()['name']``), and also provides richer
        helper methods such as ``dataset_metadata().card()`` and
        ``dataset_metadata().to_croissant()``.

        Returns:
            DatasetInfo: The dataset-level metadata.
        """
        if isinstance(cls.citation, str):
            citations = [cls.citation] if cls.citation else []
        else:
            citations = list(cls.citation)

        return DatasetInfo({
            "name": cls.name,
            "description": cls.description,
            "homepage": cls.homepage,
            "citation": citations,
            "features": cls.features,
        })

    @classmethod
    def card(cls, format: str = "markdown") -> str:
        """
        Generate a dataset card for this dataset.

        Shorthand for ``cls.dataset_metadata().card(format=format)``.

        Follows HuggingFace Hub convention: YAML frontmatter (machine-readable)
        followed by a markdown body (human-readable).

        Arguments:
            format (str): Only ``"markdown"`` is currently supported.

        Returns:
            str: The dataset card as a string.
        """
        return cls.dataset_metadata().card(format=format)


class IndexedDataset(Dataset):
    """
    Abstract base class for indexed datasets.

    Indexed datasets are datasets where the instances are indexed by a unique identifier and 
    can be accessed by that identifier. For example its positional index within the dataset.

    Implementing this class requires implementing the following methods:
    - __len__: return the total number of instances
    - __getitem__: return the instance and metadata at the given index / identifier
    """

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
            y: metadata (solution, features, origin, etc.)
        """
        pass

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for i in range(len(self)):
            yield self[i]


def expand_varying_kwargs(
    vary: Union[str, List[str]],
    gen_kwargs: dict,
    mode: str = "zip",
) -> Iterator[dict]:
    """
    Expand gen_kwargs into a sequence of kwargs dicts for varying parameters.

    When ``vary`` is a single string, yields one kwargs dict per value in
    ``gen_kwargs[vary]``.

    When ``vary`` is a list of strings, each corresponding value in gen_kwargs
    must be an iterable. Yields one kwargs dict per tuple:
    - ``mode='zip'``: parallel iteration (zip), all iterables must have same length
    - ``mode='product'``: Cartesian product over the varying dimensions

    Arguments:
        vary: Name(s) of keys in gen_kwargs whose values are iterables to vary over.
        gen_kwargs: Base kwargs; keys in vary are replaced per iteration.
        mode: ``'zip'`` (default) or ``'product'``.

    Yields:
        dict: Full kwargs for each generator call.
    """
    varying_keys = [vary] if isinstance(vary, str) else list(vary)
    base_kwargs = {k: v for k, v in gen_kwargs.items() if k not in varying_keys}
    varying_iters = [gen_kwargs[k] for k in varying_keys]

    if mode == "zip":
        for values in zip(*varying_iters):
            yield {**base_kwargs, **dict(zip(varying_keys, values))}
    elif mode == "product":
        for values in product(*varying_iters):
            yield {**base_kwargs, **dict(zip(varying_keys, values))}
    else:
        raise ValueError(f"mode must be 'zip' or 'product', got {mode!r}")


class IterableDataset(Dataset):
    """
    Abstract base class for iterable datasets.

    Iterable datasets are datasets where the instances are iterable and can be accessed by an iterator.
    The dataset does not provide random access to the instances through an index or identifier.
    An example is a generator function that yields the instances based on a random seed.

    Implementing this class requires implementing the following method:
    - __iter__: return an iterator over the dataset
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        """
        Return an iterator over the dataset.

        Returns:
            Iterator[Tuple[Any, Any]]: Iterator over the dataset, yielding (x, y) pairs of:
                x: instance reference (e.g., file path, database key, generated seed, ...)
                y: metadata (solution, features, origin, etc.)
        """
        pass

    @staticmethod
    def from_generator(
        generator: Callable,
        gen_kwargs: Optional[dict] = None,
        vary: Optional[Union[str, List[str]]] = None,
        vary_mode: str = "zip",
    ) -> IterableDataset:
        """
        Create an IterableDataset from a generator.

        Wraps a Python generator function into an ``IterableDataset``.
        The method determines the number of ``generator(...)`` calls and their
        keyword arguments from ``gen_kwargs`` and ``vary``.

        ``gen_kwargs`` is the source of truth:
        keys are parameter names of ``generator``, values are argument values.
        ``vary`` selects which of these keys should be expanded.

        Behavior summary:
        - ``vary is None``:
          one call -> ``generator(**gen_kwargs)``.
        - ``vary`` is one key (e.g. ``"n"``):
          one call per value in ``gen_kwargs["n"]``, while all other
          keyword arguments from ``gen_kwargs`` are passed unchanged.
        - ``vary`` is multiple keys (e.g. ``["n", "seed"]``):
          one call per tuple of values for those keys, while all non-varied
          keyword arguments from ``gen_kwargs`` are passed unchanged. 
          Two options for the varying:
            - ``vary_mode="zip"``: parallel iteration
            - ``vary_mode="product"``: Cartesian product

        Important:
        - Every key mentioned in ``vary`` must already exist in ``gen_kwargs``.
        - If a key is varied, its value in ``gen_kwargs`` must be iterable.
        - Non-varied keys are reused unchanged for every generator call.

        Arguments:
            generator: Callable that returns an iterator yielding (x, y) pairs.
                When ``vary`` is None, called as ``generator()`` or
                ``generator(**gen_kwargs)``. When ``vary`` is set, called once
                per value (or tuple of values) of the varying kwarg(s).
            gen_kwargs: Optional dict of keyword arguments to pass to the generator.
            vary: Optional name or list of names of keys in gen_kwargs whose values
                are iterables. If a single string, the generator is called once per
                value. If a list of strings, the generator is called once per tuple
                from zip (default) or product of the iterables.
            vary_mode: When ``vary`` is a list, ``'zip'`` (parallel iteration,
                same-length iterables) or ``'product'`` (Cartesian product).

        Examples:

            .. code-block:: python

                def gen_graph_coloring(num_instances, n_vertices, edge_prob, seed):
                    import random
                    rng = random.Random(seed)
                    for i in range(num_instances):
                        x = {
                            "problem": "graph_coloring",
                            "n_vertices": n_vertices,
                            "edge_prob": edge_prob,
                            "instance_seed": rng.randint(0, 10**9),
                        }
                        y = {"family": "gc", "name": f"gc_{n_vertices}_{i}"}
                        yield x, y

            Fixed kwargs (single call):

                .. code-block:: python

                    ds = IterableDataset.from_generator(
                        gen_graph_coloring,
                        gen_kwargs={
                            "num_instances": 3,
                            "n_vertices": 40,
                            "edge_prob": 0.2,
                            "seed": 7,
                        },
                    )
                    # Calls gen_graph_coloring(...) once with fixed kwargs

            Vary one kwarg:

                .. code-block:: python

                    ds = IterableDataset.from_generator(
                        gen_graph_coloring,
                        gen_kwargs={
                            "num_instances": 3,
                            "n_vertices": 40,
                            "edge_prob": [0.1, 0.2, 0.3],
                            "seed": 7,
                        },
                        vary="edge_prob",
                    )
                    # Calls:
                    #   gen_graph_coloring(..., edge_prob=0.1, ...)
                    #   gen_graph_coloring(..., edge_prob=0.2, ...)
                    #   gen_graph_coloring(..., edge_prob=0.3, ...)
                    # Other kwargs (num_instances, n_vertices, seed) stay fixed.

            Vary multiple kwargs with zip (default):

                .. code-block:: python

                    def gen_rcpsp_like(num_instances, n_jobs, n_resources, tightness, seed):
                        import random
                        rng = random.Random(seed)
                        for i in range(num_instances):
                            x = {
                                "problem": "rcpsp",
                                "n_jobs": n_jobs,
                                "n_resources": n_resources,
                                "tightness": tightness,
                                "instance_seed": rng.randint(0, 10**9),
                            }
                            y = {"family": "rcpsp", "name": f"j{n_jobs}_r{n_resources}_{i}"}
                            yield x, y

                    ds = IterableDataset.from_generator(
                        gen_rcpsp_like,
                        gen_kwargs={
                            "num_instances": 2,
                            "n_jobs": [30, 60],
                            "n_resources": [4, 8],
                            "tightness": [0.6, 0.8],
                            "seed": 11,
                        },
                        vary=["n_jobs", "n_resources", "tightness"],
                        vary_mode="zip",
                    )
                    # Calls:
                    #   gen_rcpsp_like(..., n_jobs=30, n_resources=4, tightness=0.6, ...)
                    #   gen_rcpsp_like(..., n_jobs=60, n_resources=8, tightness=0.8, ...)
                    # Non-varied kwargs (num_instances, seed) are reused in both calls.

            Vary multiple kwargs with Cartesian product::

                .. code-block:: python

                    ds = IterableDataset.from_generator(
                        gen_rcpsp_like,
                        gen_kwargs={
                            "num_instances": 1,
                            "n_jobs": [30, 60],
                            "n_resources": [4, 8],
                            "tightness": [0.6, 0.8],
                            "seed": 11,
                        },
                        vary=["n_jobs", "n_resources", "tightness"],
                        vary_mode="product",
                    )
                    # Calls all 2 x 2 x 2 = 8 combinations
        """
        gen_kwargs = gen_kwargs or {}

        if vary is not None:
            # Variant: call generator once per expanded kwargs
            class FromGeneratorVariedDataset(IterableDataset):
                def __init__(
                    self,
                    generator: Callable,
                    gen_kwargs: dict,
                    vary: Union[str, List[str]],
                    vary_mode: str,
                ):
                    self.generator = generator
                    self.gen_kwargs = gen_kwargs
                    self.vary = vary
                    self.vary_mode = vary_mode

                def __iter__(self):
                    for kwargs in expand_varying_kwargs(
                        self.vary, self.gen_kwargs, mode=self.vary_mode
                    ):
                        for item in self.generator(**kwargs):
                            yield item

            return FromGeneratorVariedDataset(
                generator, gen_kwargs, vary, vary_mode
            )
        else:
            # Original: single call to generator
            class FromGeneratorDataset(IterableDataset):
                def __init__(self, generator: Callable, gen_kwargs: dict):
                    self.generator = generator
                    self.gen_kwargs = gen_kwargs

                def __iter__(self):
                    return self.generator(**self.gen_kwargs)

            return FromGeneratorDataset(generator, gen_kwargs)

class FileDataset(IndexedDataset):
    """
    Abstract base class for PyTorch-style datasets of file-based CO benchmarking sets.

    The `FileDataset` class provides a standardized interface for downloading and
    accessing file-backed benchmark instances. This class should not be used on its own.
    Either have a look at one of the concrete subclasses, providing access to 
    well-known datasets from the community, or use this class as the base for your own dataset.

    For a more detailed authoring guide (design patterns, metadata conventions,
    and implementation checklist), see :ref:`datasets_advanced_authoring`.
    """

    # Extension for metadata sidecar files
    METADATA_EXTENSION = ".meta.json"

    def __init__(
            self,
            dataset_dir: str = ".",
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
            download: bool = False,
            extension: str = ".txt",
            **kwargs
        ):
        """
        Constructor for the FileDataset base class.

        Arguments:
            dataset_dir (str): Path to the dataset directory.
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).
            extension (str): Extension of the instance files. Used to filter instance files from the dataset directory.
            **kwargs: Advanced options. Currently supports:
                - metadata_workers (int): Number of parallel workers for
                  metadata collection during initial download (default: 1).
                - ignore_sidecar (bool): If True, do not read/write metadata
                  sidecars and collect metadata on demand at iteration time
                  using ``collect_instance_metadata()`` (default: False).

        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
            ValueError: If the dataset directory does not contain any instance files.
        """

        self.dataset_dir = pathlib.Path(dataset_dir)
        self.extension = extension

        # Advanced options
        metadata_workers = kwargs.pop("metadata_workers", 1)
        self._ignore_sidecar = kwargs.pop("ignore_sidecar", False)

        if not self._check_exists():
            if not download:
                raise ValueError("Dataset not found. Please set download=True to download the dataset.")
            else:
                self.download()
                if not self._ignore_sidecar:
                    self._collect_all_metadata(workers=metadata_workers)
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

    @staticmethod
    @abstractmethod
    def _loader(content: str) -> cp.Model:
        """
        Loader for the dataset. Loads a CPMpy model from raw file content string.
        The content will be the raw text content of the file (already decompressed).
        
        Arguments:
            content (str): Raw file content string to load into a model.
            
        Returns:
            cp.Model: The loaded CPMpy model.
        """
        pass

    @abstractmethod
    def categories(self) -> dict:
        """
        Labels to distinguish instances into categories matching to those of the dataset.
        E.g.
            - year
            - track
        """
        pass

    @abstractmethod
    def download(self, *args, **kwargs):
        """
        Download the dataset.
        """
        pass


    # ---------------------------------------------------------------------------- #
    #                        Methods to optionally overwrite                       #
    # ---------------------------------------------------------------------------- #

    def collect_instance_metadata(self, file: pathlib.Path) -> dict:
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
        python standard library's 'open', e.g. '.xz', '.lzma'.

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


    # ---------------------------------------------------------------------------- #
    #                               Public interface                               #
    # ---------------------------------------------------------------------------- #

    def load(self, instance: Union[str, os.PathLike]) -> cp.Model:
        """
        Load a CPMpy model from an instance file.
        
        Uses `.read()` to handle reading (decompressing + reading raw contents) and then turns 
        raw contents into a CPMpy model via `.loader()`.
        
        Arguments:
            instance (str or os.PathLike): 
                - File path to the instance file
                - OR a string containing the instance content directly
            
        Returns:
            cp.Model: The loaded CPMpy model.
        """

        # If instance is a path to a file -> open file
        if isinstance(instance, (str, os.PathLike)) and os.path.exists(instance):
           # Reading - use read() to decompress and read raw file contents
            content = self.read(instance)
        # If instance is a string containing a model -> use it directly
        else:
            content = instance

        # Loading - turn raw contents into CPMpy model
        return self._loader(content)

    def instance_metadata(self, instance: os.PathLike) -> InstanceInfo:
        """
        Return the metadata for a given instance file.

        Returns an :class:`~metadata.InstanceInfo`, which is a ``dict`` subclass
        so all existing ``meta['year']``, ``meta.get('jobs')`` access is unchanged.
        Structured access via ``info.domain_metadata``, ``info.model_features``,
        ``info.id``, etc. is additive.

        Arguments:
            file (os.PathLike): Path to the instance file.

        Returns:
            InstanceInfo: The metadata for the instance.
        """
        metadata = {
            'id': str(instance),
            'dataset': self.name,
            'category': self.category(),
            'name': pathlib.Path(instance).name.replace(self.extension, ''),
            'path': instance,
        }

        # Advanced mode: bypass sidecars and collect metadata on demand.
        if self._ignore_sidecar:
            metadata.update(self.collect_instance_metadata(file=str(instance)))
            return InstanceInfo(metadata)
        else:
            # Load sidecar metadata if it exists
            meta_path = self._metadata_path(instance)
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    sidecar = json.load(f)
                # Structured: flatten instance_metadata, format_metadata, and model_features
                metadata.update(sidecar.get("instance_metadata", {}))
                metadata.update(sidecar.get("format_metadata", {}))
                metadata.update(sidecar.get("model_features", {}))
            return InstanceInfo(metadata)


    # ---------------------------------------------------------------------------- #
    #                                   Internals                                  #
    # ---------------------------------------------------------------------------- #

    # ------------------------------ Instance access ----------------------------- #

    def _list_instances(self) -> list:
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
        filename = str(file_path)

        metadata = self.instance_metadata(filename)
        if self.target_transform:
            metadata = self.target_transform(metadata)

        if self.transform:
            filename = self.transform(filename)
            # Let transforms contribute to metadata (e.g. model verification info)
            if hasattr(self.transform, 'enrich_metadata'):
                metadata = self.transform.enrich_metadata(filename, metadata)

        return filename, metadata


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

    def _collect_all_metadata(self, force: bool = False, workers: int = 1):
        """
        Collect and store structured metadata sidecar files for all instances.

        Writes a structured `.meta.json` sidecar alongside each instance with:

        - `dataset`: dataset-level metadata (name, description, url, ...)
        - `instance_name`: logical instance name (filename stem)
        - `source_file`: path to the instance file
        - `category`: dataset category labels (year, track, variant, family)
        - `instance_metadata`: portable domain-specific metadata
        - `format_metadata`: format-specific metadata from the source format

        Arguments:
            force (bool): If True, re-collect instance metadata even if sidecar
                files already exist.
            workers (int): Number of parallel workers for metadata collection.
                Default is 1 (sequential). Use >1 for parallel processing.
        """
        files = self._list_instances()

        # Filter files that need processing
        files_to_process = []
        for file_path in files:
            meta_path = self._metadata_path(file_path)
            if force or not meta_path.exists():
                files_to_process.append(file_path)

        if not files_to_process:
            return

        # Process files sequentially or in parallel
        if workers <= 1:
            # Sequential processing
            if tqdm is not None:
                file_iter = tqdm(files_to_process, desc="Collecting metadata", unit="file")
            else:
                file_iter = files_to_process
                print(f"Collecting metadata for {len(files_to_process)} instances...")

            for file_path in file_iter:
                self._collect_one_metadata(file_path)
        else:
            # Parallel processing with ProcessPoolExecutor for CPU-bound work
            print(f"Collecting metadata for {len(files_to_process)} instances using {workers} workers...")
            
            # Use ProcessPoolExecutor with fork start method (Linux) to allow bound methods
            # On Linux, fork allows sharing the dataset instance, so bound methods work
            ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {executor.submit(self._collect_one_metadata, fp): fp for fp in files_to_process}
                
                if tqdm is not None:
                    iterator = tqdm(as_completed(futures), total=len(futures), desc="Collecting metadata", unit="file")
                else:
                    iterator = as_completed(futures)
                
                for future in iterator:
                    try:
                        future.result()
                    except Exception as e:
                        fp = futures[future]
                        print(f"Error collecting metadata for {fp.name}: {e}")

    def _collect_one_metadata(self, file_path):
        """Collect metadata for a single instance file."""
        meta_path = self._metadata_path(file_path)
        metadata_error = None
        try:
            instance_meta = self.collect_instance_metadata(str(file_path))
        except Exception as e:
            instance_meta = {}
            metadata_error = str(e)

        # Separate portable from format-specific fields
        portable = portable_instance_metadata(instance_meta)
        format_specific = {
            k: v for k, v in instance_meta.items()
            if k not in portable and not k.startswith("_")
        }

        # Derive instance name (strip format-specific extensions)
        stem = file_path.stem
        for ext in (".xml", ".wcnf", ".opb"):
            if stem.endswith(ext):
                stem = stem[:len(stem) - len(ext)]
                break

        # Build structured sidecar
        sidecar = {
            "dataset": self.dataset_metadata().to_jsonable(),
            "instance_name": stem,
            "source_file": str(file_path.relative_to(self.dataset_dir)),
            "categories": self.categories(),
            "instance_metadata": portable,
            "format_metadata": format_specific,
        }

        if metadata_error is not None:
            sidecar["_metadata_error"] = metadata_error

        # Preserve previously extracted model features if present.
        # Otherwise, compute them from the parsed model when possible.
        model_features = None
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    existing = json.load(f)
                if "model_features" in existing:
                    model_features = existing["model_features"]
            except (json.JSONDecodeError, IOError):
                pass

        if model_features is None:
            if not callable(self.reader):
                raise TypeError(
                    f"Cannot extract model features for {file_path}: "
                    "no dataset reader configured. If unexpected, please open an issue on GitHub."
                )
            model = self.reader(str(file_path), open=self.open)
            model_features = extract_model_features(model)
    
        sidecar["model_features"] = model_features

        with open(meta_path, "w") as f:
            json.dump(sidecar, f, indent=2)

            
    # ----------------------------- Download methods ----------------------------- #

    @staticmethod
    def _download_file(url: str, target: str, destination: Optional[str] = None,
                        desc: str = None,
                        chunk_size: int = 1024 * 1024) -> os.PathLike:
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
            str: The destination path where the downloaded file is saved.
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
        """Download file sequentially with progress bar."""
        import sys
        
        # Convert to Path if it's a string
        if isinstance(filepath, str):
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
                downloaded = 0
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {_format_bytes(downloaded)}/{_format_bytes(total_size)} ({percent:.1f}%)")
                        else:
                            sys.stdout.write(f"\r\033[KDownloading {desc}: {_format_bytes(downloaded)}...")
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()

    @staticmethod
    def _download_parallel(urls_and_targets: List[Tuple[str, str]], base_url: str, 
                           destination_dir: str, desc_prefix: str = "Downloading",
                           chunk_size: int = 1024 * 1024,
                           max_workers: Optional[int] = None) -> List[pathlib.Path]:
        """
        Download multiple files in parallel from a base URL.
        
        Arguments:
            urls_and_targets (List[Tuple[str, str]]): List of (url_suffix, target_filename) tuples
            base_url (str): Base URL for downloads (used as fallback)
            destination_dir (str): Directory to save files
            desc_prefix (str): Prefix for progress bar descriptions
            chunk_size (int): Chunk size for downloads
            max_workers (int, optional): Maximum number of parallel workers. Defaults to min(32, num_files)
            
        Returns:
            List[pathlib.Path]: List of downloaded file paths
        """
        os.makedirs(destination_dir, exist_ok=True)
        
        if max_workers is None:
            max_workers = min(32, len(urls_and_targets))
        
        downloaded_files = []
        errors = []
        
        def download_one(url_suffix: str, target: str) -> Tuple[Optional[pathlib.Path], Optional[str]]:
            dest_path = os.path.join(destination_dir, target)
            desc = f"{desc_prefix} {target}"

            try:
                full_url = base_url.rstrip('/') + '/' + url_suffix.lstrip('/') + target
                req = Request(full_url)
                with urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                
                FileDataset._download_sequential(full_url, dest_path, total_size, desc, chunk_size)
                return pathlib.Path(dest_path), None
            except Exception as e:
                return None, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_one, url_suffix, target): (url_suffix, target)
                for url_suffix, target in urls_and_targets
            }
            
            for future in as_completed(futures):
                url_suffix, target = futures[future]
                result, error = future.result()
                if result is not None:
                    downloaded_files.append(result)
                else:
                    errors.append((target, error))
        
        if errors:
            error_msg = f"Failed to download {len(errors)}/{len(urls_and_targets)} files. "
            error_msg += f"First error: {errors[0][0]} - {errors[0][1]}"
            warnings.warn(error_msg)
        
        return downloaded_files

def from_files(dataset_dir: os.PathLike, extension: str = ".txt") -> FileDataset:
    """
    Create a FileDataset from a list of files.
    """
    class FromFilesDataset(FileDataset):
        def __init__(self, dataset_dir: os.PathLike, extension: str = ".txt"):
            super().__init__(dataset_dir=dataset_dir, extension=extension)

        @property
        def name(self) -> str:
            raise NotImplementedError("Arbitrary file dataset does not support a name. Please implement this method in a subclass, or use a more specific dataset class.")

        @property
        def description(self) -> str:
            raise NotImplementedError("Arbitrary file dataset does not support a description. Please implement this method in a subclass, or use a more specific dataset class.")

        @property
        def url(self) -> str:
            raise NotImplementedError("Arbitrary file dataset does not support a URL. Please implement this method in a subclass, or use a more specific dataset class.")

        @property
        def citation(self) -> List[str]:
            raise NotImplementedError("Arbitrary file dataset does not support a citation. Please implement this method in a subclass, or use a more specific dataset class.")

        def _loader(self, file: os.PathLike) -> cp.Model:
            raise NotImplementedError("Arbitrary file dataset does not support loading. Please implement this method in a subclass, or use a more specific dataset class.")

        def category(self) -> dict:
            raise NotImplementedError("Arbitrary file dataset does not support categories. Please implement this method in a subclass, or use a more specific dataset class.")

        def download(self) -> None:
            raise NotImplementedError("Arbitrary file dataset does not support downloading. Please implement this method in a subclass, or use a more specific dataset class.")

        def instance_metadata(self, file: os.PathLike) -> dict:
            metadata = {
                'id': str(file),
                'dataset_dir': str(self.dataset_dir),
                'name': pathlib.Path(file).name.replace(self.extension, ''),
                'path': file,
            }
            return metadata

    return FromFilesDataset(dataset_dir, extension)

# Not implemented yet
class URLDataset(IndexedDataset):
    """
    Abstract base class for URL-backed datasets.

    Each instance reference is a URL.
    """
    pass

# Not implemented yet
class StreamingDataset(IterableDataset):
    """
    Abstract base class for streaming datasets.
    """
    pass

# Not implemented yet
class GeneratedDataset(IterableDataset):
    """
    Abstract base class for generated datasets.
    """
    pass