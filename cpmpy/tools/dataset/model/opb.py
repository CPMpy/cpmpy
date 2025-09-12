"""
Pseudo Boolean Competition (PB) Dataset

https://www.cril.univ-artois.fr/PB25/
"""

import lzma
import os
import pathlib
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import tarfile

from .._base import _Dataset


class OPBDataset(_Dataset): 
    """
    Pseudo Boolean Competition (PB) benchmark dataset.

    Provides access to benchmark instances from the Pseudo Boolean 
    competitions. Instances are grouped by `year` and `track` (e.g., 
    `"OPT-LIN"`, `"DEC-LIN"`) and stored as `.opb.xz` files. 
    If the dataset is not available locally, it can be automatically 
    downloaded and extracted.

    More information on the competition can be found here: https://www.cril.univ-artois.fr/PB25/
    """

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "OPT-LIN", 
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the PB competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="OPT-LIN").
            transform (callable, optional): Optional transform applied to the instance file path.
            target_transform (callable, optional): Optional transform applied to the metadata dictionary.
            download (bool): If True, downloads the dataset if it does not exist locally (default=False).


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested year/track combination is not available.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        # Check requested dataset
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. exact-weighted, exact-unweighted, ...")

        dataset_dir = self.root / str(year) / track

        super().__init__(
            dataset_dir=dataset_dir, 
            transform=transform, target_transform=target_transform, 
            download=download, extension=".opb.xz"
        )

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def metadata(self, file) -> dict:
        # Add the author to the metadata
        return super().metadata(file) | {'author': str(file).split(os.sep)[-1].split("_")[0],}
                

    def download(self):
        # TODO: add option to filter on competition instances
        print(f"Downloading OPB {self.year} {self.track} instances...")
        
        url = f"https://www.cril.univ-artois.fr/PB24/benchs/"
        year_suffix = str(self.year)[2:]  # Drop the starting '20'
        url_path = url + f"normalized-PB{year_suffix}.tar"
        tar_path = self.root / f"normalized-extraPB{year_suffix}.tar"
        
        try:
            urlretrieve(url_path, str(tar_path))
        except (HTTPError, URLError) as e:
            raise ValueError(f"No dataset available for year {self.year}. Error: {str(e)}")
        
        # Extract only the specific track folder from the tar
        with tarfile.open(tar_path, "r:*") as tar_ref:  # r:* handles .tar, .tar.gz, .tar.bz2, etc.
            # Get the main folder name
            main_folder = None
            for name in tar_ref.getnames():
                if "/" in name:
                    main_folder = name.split("/")[0]
                    break

            if main_folder is None:
                raise ValueError(f"Could not find main folder in tar file")

            # Extract only files from the specified track
            # Get all unique track names from tar
            tracks = set()
            for member in tar_ref.getmembers():
                parts = member.name.split("/")
                if len(parts) > 2 and parts[0] == main_folder:
                    tracks.add(parts[1])

            # Check if requested track exists
            if self.track not in tracks:
                raise ValueError(f"Track '{self.track}' not found in dataset. Available tracks: {sorted(tracks)}")

            # Create track folder in root directory
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract files for the specified track
            prefix = f"{main_folder}/{self.track}/"
            for member in tar_ref.getmembers():
                if member.name.startswith(prefix) and member.isfile():
                    # Path relative to main_folder/track
                    relative_path = member.name[len(prefix):]

                    # Flatten: replace "/" with "_" to encode subfolders (some instances have clashing names)
                    flat_name = relative_path.replace("/", "_")
                    target_path = self.dataset_dir / flat_name

                    with tar_ref.extractfile(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        # Clean up the tar file
        tar_path.unlink()

    def open(self, instance: os.PathLike) -> callable:
        return lzma.open(instance, 'rt') if str(instance).endswith(".xz") else open(instance)

if __name__ == "__main__":
    dataset = OPBDataset(year=2024, track="DEC-LIN", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
