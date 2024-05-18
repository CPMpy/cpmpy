import sys, os
import requests
import pathlib
import zipfile
import subprocess
import tqdm
from git import Repo

CHUNK_SIZE = 32768

def install_solution_checker():
    """
        Downloads the SolutionChecker Jar file.
    """

    SOLUTION_CHECKER_URL = "https://drive.usercontent.google.com/u/0/uc?id=1244olX9XarR3-yzsapFOy66zPe78zc51&export=download"
    SOLUTION_CHECKER_DESTINATION_PATH = os.path.join(pathlib.Path(__file__).parent.resolve())
    SOLUTION_CHECKER_DESTINATION_FILE = "xcsp3-solutionChecker-2.5.jar"

    print("Installing solutionChecker ...")
    
    session = requests.Session()
    response = session.get(SOLUTION_CHECKER_URL, stream=True)

    with open(os.path.join(SOLUTION_CHECKER_DESTINATION_PATH, SOLUTION_CHECKER_DESTINATION_FILE), "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)

def install_xcsp3_instances():
    """
        Downloads the XCSP3 2022 main and mini track problem instances.
    """

    XCSP3_INSTANCES_URL = "https://www.cril.univ-artois.fr/~lecoutre/compets/instancesXCSP22.zip"
    XCSP3_INSTANCES_DESTINATION_PATH = os.path.join(pathlib.Path(__file__).parent.resolve())
    XCSP3_INSTANCES_DESTINATION_FILE = "temp.zip"
    XCSP3_INSTANCES_EXTRACTION_TIMEOUT = 100

    print("Installing XCSP3 instances ...")

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
    for root, dir, files in os.walk('./xcsp3/instancesXCSP22/'):
        print(f"Extracting {root}")
        for file in tqdm.tqdm(files):
            if target in file:
                filename = os.path.join(root, file)
                convert(filename)

def install_pycsp3():
    """
        Downloads the pycsp3 repository.
    """

    PYCSP3_REPO_URL = "https://github.com/xcsp3team/PyCSP3.git"
    PYCSP3_REPO_DIR = os.path.join(pathlib.Path(__file__).parent.resolve(), "pycsp3", "pycsp3")

    print("Cloning pycsp3 repository ...")

    Repo.clone_from(PYCSP3_REPO_URL, PYCSP3_REPO_DIR)

def update_pycsp3():
    """
        Removes a line of code within the pycsp3 parser that causes unwanted printing.
        Messes up the result capturing of the executable.
    """

    print("Updating pycsp3 code ...")
    
    def lines_that_contain(file_name, line_to_match):
        fp = open(file_name, 'r')
        return [i for i, line in enumerate(fp) if line_to_match in line]

    def replace_line(file_name, line_num, text):
        lines = open(file_name, 'r').readlines()
        lines[line_num] = text
        out = open(file_name, 'w')
        out.writelines(lines)
        out.close()

    code_file_to_edit = os.path.join(pathlib.Path(__file__).parent.resolve(), "pycsp3", "pycsp3", "tools", "xcsp.py")
    replace_line(code_file_to_edit, lines_that_contain(code_file_to_edit, "Warning: no variables in this model (and so, no generated file)!")[0], "")

install_solution_checker()
install_xcsp3_instances()
install_pycsp3()
update_pycsp3()