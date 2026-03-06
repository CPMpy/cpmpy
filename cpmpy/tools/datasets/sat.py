"""
SAT Competition Dataset.

Instances are fetched from benchmark-database.de via ``getinstances``.
Each returned line is an instance URL, usually served as XZ-compressed DIMACS.
"""

import io
import lzma
import os
import pathlib
import re
import tempfile
from urllib.request import Request, urlopen

from cpmpy.tools.datasets.core import FileDataset
from cpmpy.tools.datasets.metadata import FeaturesInfo


INSTANCE_LIST_URL = "https://benchmark-database.de/getinstances"


class SATDataset(FileDataset):
    """
    SAT competition benchmark dataset (DIMACS CNF).
    """

    name = "sat"
    description = "SAT competition benchmark instances (DIMACS CNF) from benchmark-database.de."
    homepage = "https://benchmark-database.de/"
    citation = []
    version = "2025"
    license = "competition-specific"
    domain = "sat"
    tags = ["satisfaction", "sat", "cnf", "dimacs"]
    language = "DIMACS-CNF"
    features = FeaturesInfo({
        "dimacs_num_variables": ("int", "Number of propositional variables from DIMACS p-line"),
        "dimacs_num_clauses": ("int", "Number of clauses from DIMACS p-line"),
    })

    def __init__(
        self,
        root: str = ".",
        track: str = "main_2025",
        context: str = "cnf",
        transform=None,
        target_transform=None,
        download: bool = False,
        **kwargs
    ):
        """
        Constructor for the SAT competition dataset.

        Arguments:
            root (str): Root directory where the dataset is stored or will be downloaded (default=".").
            track (str): Track query parameter for getinstances (default="main_2025").
            context (str): Context query parameter for getinstances (default="cnf").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dict.
            download (bool): If True, download the instance list and all instances if not present (default=False).
            **kwargs: Passed through to download() (e.g. workers for parallel downloads).
        """
        self.root = pathlib.Path(root)
        self.track = track
        self.context = context

        dataset_dir = self.root / self.name / track / context
        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform,
            target_transform=target_transform,
            download=download,
            extension=".cnf.xz",
            **kwargs,
        )

    @staticmethod
    def _loader(content: str):
        from cpmpy.tools.io.dimacs import load_dimacs
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".cnf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            return load_dimacs(tmp_path)
        finally:
            os.unlink(tmp_path)

    def category(self) -> dict:
        return {"track": self.track, "context": self.context}

    def categories(self) -> dict:
        return self.category()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        path = str(instance)
        return lzma.open(instance, "rt") if path.endswith(".xz") else open(instance, "r")

    def collect_instance_metadata(self, file) -> dict:
        result = {}
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
        params = f"query=track%3D{self.track}&context={self.context}"
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
    dataset = SATDataset(track="main_2025", context="cnf", download=False)
    print("Dataset size:", len(dataset))
