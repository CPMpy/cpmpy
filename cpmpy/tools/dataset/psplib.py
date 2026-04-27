"""
PSPlib Dataset

https://www.om-db.wi.tum.de/psplib/getdata_sm.html
"""

import os
import pathlib
import io
import zipfile

from cpmpy.tools.dataset._base import _Dataset

class PSPLibDataset(_Dataset):  # torch.utils.data.Dataset compatible
    """
    PSPlib Dataset in a PyTorch compatible format.
    
    More information on PSPlib can be found here: https://www.om-db.wi.tum.de/psplib/main.html
    """
    
    name = "psplib"

    def __init__(self, root: str = ".", variant: str = "rcpsp", family: str = "j30", transform=None, target_transform=None, download: bool = False):
        """
        Constructor for a dataset object for PSPlib.

        Arguments:
            root (str): Root directory containing the psplib instances (if 'download', instances will be downloaded to this location)
            variant (str): scheduling variant (only 'rcpsp' is supported for now)
            family (str): family name (e.g. j30, j60, etc...)
            transform (callable, optional): Optional transform to be applied on the instance data
            target_transform (callable, optional): Optional transform to be applied on the file path
            download (bool): If True, downloads the dataset from the internet and puts it in `root` directory


        Raises:
            ValueError: If the dataset directory does not exist and `download=False`,
                or if the requested variant/family combination is not available.
        """
        
        self.root = pathlib.Path(root)
        self.variant = variant
        self.family = family
        
        self.families = dict(
            rcpsp = ["j30", "j60", "j90", "j120"]
        )
        self.family_codes = dict(rcpsp="sm", mrcpsp="mm")

        if variant != "rcpsp":
            raise ValueError("Only 'rcpsp' variant is supported for now")
        if family not in self.families[variant]:
            raise ValueError(f"Unknown problem family. Must be any of {','.join(self.families[variant])}")
        
        dataset_dir = self.root / self.name / self.variant / self.family

        super().__init__(
            dataset_dir=dataset_dir,
            transform=transform, target_transform=target_transform, 
            download=download, extension=f".{self.family_codes[self.variant]}"
        )
        
    def category(self) -> dict:
        return {
            "variant": self.variant,
            "family": self.family
        }

    def download(self):

        url = "https://www.om-db.wi.tum.de/psplib/files/"
        target = f"{self.family}.{self.family_codes[self.variant]}.zip"
        target_download_path = self.root / target
        
        print(f"Downloading PSPLib {self.variant} {self.family} instances from www.om-db.wi.tum.de")

        try:
            target_download_path = self._download_file(url, target, destination=str(target_download_path))
        except ValueError as e:
            raise ValueError(f"No dataset available for variant {self.variant} and family {self.family}. Error: {str(e)}")
        
        # make directory and extract files
        with zipfile.ZipFile(target_download_path, 'r') as zip_ref:                    
            # Create track folder in root directory, parents=True ensures recursive creation
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract files
            for file_info in zip_ref.infolist():
                # Extract file to family_dir, removing main_folder/track prefix
                filename = pathlib.Path(file_info.filename).name
                with zip_ref.open(file_info) as source, open(self.dataset_dir / filename, 'wb') as target:
                    target.write(source.read())
        # Clean up the zip file
        target_download_path.unlink()


if __name__ == "__main__":
    dataset = PSPLibDataset(variant="rcpsp", family="j30", download=True)
    print("Dataset size:", len(dataset))
    print("Instance 0:", dataset[0])