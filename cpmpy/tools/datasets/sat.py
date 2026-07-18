"""
The SAT Competition provides benchmark instances in DIMACS CNF format.
Origin: https://benchmark-database.de/
"""

from __future__ import annotations

import io
import lzma
import os
import pathlib
import re
from typing import Any, Optional, Callable
from urllib.request import Request, urlopen

from cpmpy.tools.datasets.core import FileDataset


INSTANCE_LIST_URL = "https://benchmark-database.de/getinstances"


class SATDataset(FileDataset):
    """
    SAT competition benchmark dataset (DIMACS CNF).

    - Origin: https://benchmark-database.de/

    To load an instance into a CPMpy model, use :func:`~cpmpy.tools.io.dimacs.load_dimacs`.
    For examples of using a loader as a dataset ``transform``, see the
    :ref:`modeling guide <modeling-datasets>`.

    Arguments:
        root (str): Root directory where the dataset is stored or will be downloaded (default=".").
        year (int): Competition year (default=2025).
        track (str): Competition track (default="main").
        context (str): Context query parameter for getinstances (default="cnf").
        transform (callable, optional): Optional transform applied to the instance file path.
        target_transform (callable, optional): Optional transform applied to the metadata dict.
        download (bool): If True, download the instance list and all instances if not present (default=False).
    """

    name = "sat"
    description = "SAT competition benchmark instances (DIMACS CNF) from benchmark-database.de."
    homepage = "https://benchmark-database.de/"
    citation = []

    def __init__(
        self,
        root: str = ".",
        year: int = 2025,
        track: str = "main",
        context: str = "cnf",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        **kwargs: Any
    ):
        """
        Constructor for the SAT competition dataset.
        """
        self.root = pathlib.Path(root)
        self.year = year
        self.track = track
        self.context = context

        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. main, submissions, ...")

        dataset_dir = self.root / self.name / str(year) / track / context
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            target_transform=target_transform,
            download=download,
            extension=".cnf.xz",
            **kwargs,
        )

    def categories(self) -> dict[str, Any]:
        return {"year": self.year, "track": self.track, "context": self.context}

    @classmethod
    def open(cls, instance: os.PathLike) -> io.TextIOBase:
        path = str(instance)
        return lzma.open(instance, "rt") if path.endswith(".xz") else open(instance, "r")

    def collect_instance_metadata(self, file: pathlib.Path) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            with self.open(file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("p"):
                        match = re.search(r"p\s+cnf\s+(\d+)\s+(\d+)", line)
                        if match:
                            result["dimacs_num_variables"] = int(match.group(1))
                            result["dimacs_num_clauses"] = int(match.group(2))
                        break
        except Exception:
            pass
        return result

    def download(self):
        params = f"query=track%3D{self.track}_{self.year}&context={self.context}"
        list_url = f"{INSTANCE_LIST_URL}?{params}"
        print(f"Fetching SAT instance list from {list_url}")

        req = Request(list_url)
        with urlopen(req) as response:
            body = response.read().decode("utf-8")

        file_urls = [line.strip() for line in body.splitlines() if line.strip()]
        if not file_urls:
            raise ValueError(f"No instances returned from {list_url}. Check track/context.")

        def path_to_name(url: str) -> str:
            name = url.rstrip("/").split("/")[-1]
            if name.lower().endswith(".cnf.xz"):
                return name
            if name.lower().endswith(".cnf"):
                return f"{name}.xz"
            return f"{name}.cnf.xz"

        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        seen_targets = set()
        items = []
        for url in file_urls:
            target = path_to_name(url)
            if target not in seen_targets:
                seen_targets.add(target)
                items.append((url, target))

        print(f"Downloading {len(items)} SAT instances to {self.dataset_dir}")
        for url, target in items:
            destination = str(self.dataset_dir / target)
            self._download_file(url=url, target="", destination=destination, desc=target)


if __name__ == "__main__":
    dataset = SATDataset(year=2025, track="main", context="cnf", download=False)
    print("Dataset size:", len(dataset))
