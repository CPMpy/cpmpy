"""
XCS3 Dataset

https://xcsp.org/instances/
"""

import os
import lzma
import zipfile
import pathlib
import io

from cpmpy.tools.dataset._base import _Dataset


class XCSP3Dataset(_Dataset):  # torch.utils.data.Dataset compatible

    """
    XCSP3 Dataset in a PyTorch compatible format.
    
    Arguments:
        root (str): Root directory containing the XCSP3 instances (if 'download', instances will be downloaded to this location)
        year (int): Competition year (2022, 2023 or 2024)
        track (str, optional): Filter instances by track type (e.g., "COP", "CSP", "MiniCOP")
        transform (callable, optional): Optional transform to be applied on the instance data (the file path of each problem instance)
        target_transform (callable, optional): Optional transform to be applied on the metadata (the metadata dictionary of each problem instance)
        download (bool): If True, downloads the dataset from the internet and puts it in `root` directory
    """

    name = "xcsp3"
    
    def __init__(self, root: str = ".", year: int = 2024, track: str = "CSP", transform=None, target_transform=None, download: bool = False):
        """
        Initialize the XCSP3 Dataset.
        """

        self.root = pathlib.Path(root)
        self.year = year
        self.track = track

        dataset_dir = self.root / self.name / str(year) / track
        
        if not str(year).startswith('20'):
            raise ValueError("Year must start with '20'")
        if not track:
            raise ValueError("Track must be specified, e.g. COP, CSP, MiniCOP, ...")

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform, 
            download=download, extension=".xml.lzma"
        )

    def category(self) -> dict:
        return {
            "year": self.year,
            "track": self.track
        }

    def download(self):

        url = "https://www.cril.univ-artois.fr/~lecoutre/compets/"
        target = f"instancesXCSP{str(self.year)[2:]}.zip"
        target_download_path = self.root / target

        print(f"Downloading XCSP3 {self.year} instances from www.cril.univ-artois.fr")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available for year {self.year}. Error: {str(e)}")

        # Extract only the specific track folder from the zip
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:
            # Get the main folder name (e.g., "024_V3")
            main_folder = None
            for name in zip_ref.namelist():
                if '/' in name:
                    main_folder = name.split('/')[0]
                    break
            
            if main_folder is None:
                raise ValueError(f"Could not find main folder in zip file")

            # Extract only files from the specified track
            # Get all unique track names from zip
            tracks = set()
            for file_info in zip_ref.infolist():
                parts = file_info.filename.split('/')
                if len(parts) > 2 and parts[0] == main_folder:
                    tracks.add(parts[1])
            
            # Check if requested track exists
            if self.track not in tracks:
                raise ValueError(f"Track '{self.track}' not found in dataset. Available tracks: {sorted(tracks)}")
            
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files for the specified track
            prefix = f"{main_folder}/{self.track}/"
            for file_info in zip_ref.infolist():
                if file_info.filename.startswith(prefix):
                    # Extract file to track_dir, removing main_folder/track prefix
                    filename = pathlib.Path(file_info.filename).name
                    with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                        target.write(source.read())

        # Clean up the zip file
        target_download_path.unlink()

    def open(self, instance: os.PathLike) -> io.TextIOBase:
        return lzma.open(instance, mode='rt', encoding='utf-8') if str(instance).endswith(".lzma") else open(instance)


if __name__ == "__main__":
    dataset = XCSP3Dataset(year=2024, track="MiniCOP", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])
