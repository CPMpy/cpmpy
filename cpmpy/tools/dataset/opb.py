"""
Pseudo Boolean Competition (PB) Dataset

https://www.cril.univ-artois.fr/PB25/
"""

import fnmatch
import lzma
import os
import pathlib
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError
import tarfile

from cpmpy.tools.dataset._base import _Dataset


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

    name = "opb"

    def __init__(
            self, 
            root: str = ".", 
            year: int = 2024, track: str = "OPT-LIN", 
            competition: bool = False,
            transform=None, target_transform=None, 
            download: bool = False
        ):
        """
        Constructor for a dataset object of the PB competition.

        Arguments:
            root (str): Root directory where datasets are stored or will be downloaded to (default="."). 
            year (int): Competition year of the dataset to use (default=2024).
            track (str): Track name specifying which subset of the competition instances to load (default="OPT-LIN").
            competition (bool): If True, the dataset will filtered on competition-used instances.
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
        self.competition = competition

        # Check requested dataset
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. exact-weighted, exact-unweighted, ...")

        dataset_dir = self.root / str(year) / track / ('selected' if self.competition else 'normalized')

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
        print(f"Downloading OPB {self.year} {self.track} {'competition' if self.competition else 'non-competition'} instances...")
        
        url = f"https://www.cril.univ-artois.fr/PB24/benchs/"
        year_suffix = str(self.year)[2:]  # Drop the starting '20'
        url_path = url + f"{'normalized' if not self.competition else 'selected'}-PB{year_suffix}.tar"
        tar_path = self.root / f"{'normalized' if not self.competition else 'selected'}-PB{year_suffix}.tar"
        
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
            if not self.competition:
                tracks = set()
                for member in tar_ref.getmembers():
                    parts = member.name.split("/")
                    if len(parts) > 2 and parts[0] == main_folder:
                        tracks.add(parts[1])
            else:
                tracks = set()
                for member in tar_ref.getmembers():
                    parts = member.name.split("/")
                    if len(parts) > 2 and parts[0] == main_folder:
                        tracks.add(parts[2])

            # Check if requested track exists
            if self.track not in tracks:
                raise ValueError(f"Track '{self.track}' not found in dataset. Available tracks: {sorted(tracks)}")

            # Create track folder in root directory
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

            # Extract files for the specified track
            if not self.competition:
                prefix = f"{main_folder}/{self.track}/"
            else:
                prefix = f"{main_folder}/*/{self.track}/"
            for member in tar_ref.getmembers():
                if fnmatch.fnmatch(member.name, prefix + "*") and member.isfile():
                    # Path relative to main_folder/track
                    # Find where the track folder ends and get everything after
                    track_marker = f"/{self.track}/"
                    marker_pos = member.name.find(track_marker)
                    relative_path = member.name[marker_pos + len(track_marker):]

                    # Flatten: replace "/" with "_" to encode subfolders (some instances have clashing names)
                    flat_name = relative_path#.replace("/", "_")
                    target_path = self.dataset_dir / flat_name

                    os.makedirs(os.path.dirname(target_path), exist_ok=True)

                    with tar_ref.extractfile(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

        # Clean up the tar file
        tar_path.unlink()

    def open(self, instance: os.PathLike) -> callable:
        return lzma.open(instance, 'rt') if str(instance).endswith(".xz") else open(instance)

if __name__ == "__main__":
    dataset = OPBDataset(year=2024, track="DEC-LIN", competition=True, download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
