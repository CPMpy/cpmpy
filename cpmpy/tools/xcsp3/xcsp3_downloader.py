"""
Utilities for downloading instances for the XCSP3 competition.
"""

import lzma
import os
import pathlib
import zipfile
import subprocess

CHUNK_SIZE = 32768

def install_xcsp3_instances_22():
    import requests
    import tqdm
    """
        Downloads the XCSP3 2022 main and mini track problem instances.
    """

    XCSP3_INSTANCES_URL = "https://www.cril.univ-artois.fr/~lecoutre/compets/instancesXCSP22.zip"
    XCSP3_INSTANCES_DESTINATION_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "models")
    print(XCSP3_INSTANCES_DESTINATION_PATH)
    XCSP3_INSTANCES_DESTINATION_FILE = "temp.zip"
    XCSP3_INSTANCES_EXTRACTION_TIMEOUT = 100

    print("Installing XCSP3 2022 instances ...")

    session = requests.Session()
    response = session.get(XCSP3_INSTANCES_URL, stream=True)

    with open(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE), "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    with zipfile.ZipFile(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE), 'r') as zip_ref:
        zip_ref.extractall(XCSP3_INSTANCES_DESTINATION_PATH)

    pathlib.Path(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE)).unlink()

    def convert(file):
        p = subprocess.Popen(["xz", "-d", file], start_new_session=True)
        p.wait(timeout=XCSP3_INSTANCES_EXTRACTION_TIMEOUT)


    target = ".lzma"
    for root, dir, files in os.walk(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, "instancesXCSP22")):
        print(f"Extracting {root}")
        for file in tqdm.tqdm(files):
            if target in file:
                filename = os.path.join(root, file)
                convert(filename)

def install_xcsp3_instances_23():
    import requests
    import tqdm
    """
        Downloads the XCSP3 2023 main and mini track problem instances.
    """

    XCSP3_INSTANCES_URL = "https://www.cril.univ-artois.fr/~lecoutre/compets/instancesXCSP23.zip"
    XCSP3_INSTANCES_DESTINATION_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "models")
    XCSP3_INSTANCES_DESTINATION_FILE = "temp.zip"
    XCSP3_INSTANCES_EXTRACTION_TIMEOUT = 100

    print("Installing XCSP3 2023 instances ...")

    session = requests.Session()
    response = session.get(XCSP3_INSTANCES_URL, stream=True)

    with open(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE), "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

    with zipfile.ZipFile(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE), 'r') as zip_ref:
        zip_ref.extractall(XCSP3_INSTANCES_DESTINATION_PATH)

    os.rename(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, "XCSP23_V2"), os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, "instancesXCSP23"))

    pathlib.Path(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, XCSP3_INSTANCES_DESTINATION_FILE)).unlink()

    def convert(file):
        # Open the .lzma file and decompress it
        with lzma.open(file, 'rb') as compressed_file:
            # Determine the output file name by removing the .lzma extension
            output_file = file[:-5]  # Assuming the file ends with '.lzma'
            with open(output_file, 'wb') as decompressed_file:
                decompressed_file.write(compressed_file.read())

    # Example usage
    target = ".lzma"
    for root, dir, files in os.walk(os.path.join(XCSP3_INSTANCES_DESTINATION_PATH, "instancesXCSP23")):
        print(f"Extracting {root}")
        for file in tqdm.tqdm(files):
            if target in file:
                filename = os.path.join(root, file)
                convert(filename)

import argparse

def main():
    parser = argparse.ArgumentParser(description='XCSP3 Downloader')
    parser.add_argument('--install-2022', action='store_true', help='Install XCSP3 2022 instances')
    parser.add_argument('--install-2023', action='store_true', help='Install XCSP3 2023 instances')
    args = parser.parse_args()

    if args.install_2022:
        install_xcsp3_instances_22()
    if args.install_2023:
        install_xcsp3_instances_23()

if __name__ == "__main__":
    main()

