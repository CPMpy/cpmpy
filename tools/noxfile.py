from __future__ import annotations
import nox
import tempfile
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/CPMpy/cpmpy.git"
BRANCH = "release_0_9_25"
# BRANCH = "fix/tests-with-no-extras"
# BRANCH = "pypblib_test_skip"
# SOLVERS = [
#     "ortools",
#     "z3",
#     "choco",
#     "exact",
#     "minizinc",
#     "pysat",
#     "gurobi",
#     "pysdd",
#     "gcs",
#     "cpo"
# ]
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12", "3.13"]

import cpmpy as cp
SOLVERS = [solver_name for solver_name, _ in cp.SolverLookup().base_solvers()]


import subprocess

def check_minizinc_available():
    try:
        result = subprocess.run(["minizinc", "--version"], capture_output=True, text=True, check=True)
        print(f"MiniZinc is available: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("MiniZinc is not installed or not in PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"MiniZinc command failed: {e}")
        return False

MINIZINC_AVAILABLE = check_minizinc_available()


@nox.session(python=PYTHON_VERSIONS, reuse_venv=False, requires=["install_minizinc"] if "minizinc" in SOLVERS and not MINIZINC_AVAILABLE else [], venv_backend="conda")
def pytest(session):

    # Parse the Python version of current session (e.g., "3.9" or "3.10")
    major, minor = map(int, session.python.split("."))
    
    # Solver filtering
    selected_solvers = SOLVERS.copy()
    # Remove "exact" if Python < 3.10
    if ("exact" in selected_solvers) and (major, minor) < (3, 10):
        selected_solvers.remove("exact")
    # Remove "cpo" if Python >= 3.13
    if ("cpo" in selected_solvers) and (major, minor) >= (3, 13):
        selected_solvers.remove("cpo")

    # Add MiniZinc to PATH if it was installed (see "install_minizinc" session) and included in the session
    if "minizinc" in SOLVERS:
        if MINIZINC_AVAILABLE:
            pass
        else:
            if Path(".minizinc_path").exists():
                minizinc_bin = Path(".minizinc_path").read_text().strip()
                session.env["PATH"] = f"{minizinc_bin}:{session.env.get('PATH', '')}"
                session.log(f"Added MiniZinc to PATH: {minizinc_bin}")
            else:
                raise Exception("Could not find Minizinc executable")

    # Install console into environment (otherise later installs will complain)
    session.conda_install(
        "--channel", "conda-forge",
        "bash"
    )

    # Collect solver build dependencies
    solver_deps = []
    if "gcs" in selected_solvers:
        solver_deps += [
            "python-devtools",
            "cxx-compiler",
            "coreutils", "sed", "git", "boost",
            "ninja",
            "compilers", 
            "cmake", "make",
            "libgcc-ng",          # libpthread and friends
            "libstdcxx-ng",       # standard C++ symbols
            "gxx_linux-64",       # actual g++ compiler
            "gcc=13.*",
            "gxx=13.*",
        ]
    solver_deps = set(solver_deps)

    # Install solver build dependencies
    session.conda_install(
        "--channel", "conda-forge",
        *solver_deps
    )

    # Install build tools
    session.install("scikit-build-core")


    # Install optional dependencies
    optional_deps = []
    if "pysat" in selected_solvers:
        optional_deps.append("pypblib")
    if "cpo" in selected_solvers:
        optional_deps.append("cplex")
    session.install(*optional_deps, "--no-cache")


    # Install CPMpy
    cpmpy_dir = Path(__file__).parent.parent
    session.install("-e", f"{cpmpy_dir}[test,{','.join(selected_solvers)}]", "--no-cache")#, env=env)

    # Install pytest dependencies
    session.install("pytest-xdist", "--no-cache")
    
    # If manually installing CPLEX binaries
    # session.run(
    #     "bash", 
    #     "-c", 
    #     f"export PATH='{minizinc_bin}:/cw/dtaijupiter/NoCsBack/dtai/thomass/CPLEX_Studio_Community2212/cpoptimizer/bin:$PATH'"
    # )

    # Now the repo is installed, so we can import cpmpy normally
    session.log("Checking available solvers...")
    session.run("python", "-c", "import cpmpy; print(cpmpy.SolverLookup().print_status())")

    # Create results directory if needed
    results_dir = cpmpy_dir / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run pytest and save results
    import multiprocessing
    session.chdir(str(cpmpy_dir))
    result_file = results_dir / f"results_py{session.python.replace('.', '')}.txt"
    with result_file.open("w") as f:
        session.run(
            "pytest",
            "-n", str(int(multiprocessing.cpu_count()-2)),
            "tests",
            stdout=f,
        )
    session.log(f"Saved test results to {result_file}")


# CPLEX_INSTALLER_PATH = str(Path(__file__).parent / "cos_installer_preview-22.1.2.R4-M0N96ML-linux-x86-64.bin")
# CPLEX_INSTALL_DIR = Path("./CPLEX_Studio2211")

# @nox.session(default=False)
# def install_cplex(session):
#     """Run the CPLEX Optimization Studio installer in silent mode."""
#     installer = Path(CPLEX_INSTALLER_PATH)

#     if not installer.exists():
#         session.error(f"CPLEX installer not found at: {installer}")
#     if not os.access(installer, os.X_OK):
#         session.log("Making installer executable...")
#         os.chmod(installer, 0o755)

#     # Ensure target install directory exists
#     CPLEX_INSTALL_DIR.mkdir(parents=True, exist_ok=True)

#     session.log("Running CPLEX installer in silent mode...")

#     session.run(
#         str(installer),
#         "-i", "silent",
#         # f"-DUSER_INSTALL_DIR={CPLEX_INSTALL_DIR.resolve()}",
#         external=True,
#     )

#     session.log(f"CPLEX installed to: {CPLEX_INSTALL_DIR.resolve()}")



import urllib.request
import tarfile

MINIZINC_VERSION = "2.9.2"
MINIZINC_URL = f"https://github.com/MiniZinc/MiniZincIDE/releases/download/{MINIZINC_VERSION}/MiniZincIDE-{MINIZINC_VERSION}-bundle-linux-x86_64.tgz"
MINIZINC_DIR = Path(".") / f"minizinc-{MINIZINC_VERSION}"
MINIZINC_BIN = None  # will be set after installation

@nox.session(name="install_minizinc", default=False, reuse_venv=True)
def install_minizinc(session):
    """Download and install the MiniZinc binary bundle for Linux."""

    minizinc_path_file = Path(".minizinc_path")

    # Skip installation if .minizinc_path already exists
    if minizinc_path_file.exists():
        session.log("MiniZinc already installed (path file exists). Skipping installation.")
        return

    archive_path = Path("minizinc.tgz")

    if not archive_path.exists():
        session.log(f"Downloading MiniZinc {MINIZINC_VERSION} from {MINIZINC_URL}...")
        urllib.request.urlretrieve(MINIZINC_URL, archive_path)

    if not MINIZINC_DIR.exists():
        session.log("Extracting MiniZinc...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=MINIZINC_DIR)

    bin_path = next(MINIZINC_DIR.glob("MiniZincIDE*/bin"), None)
    if not bin_path:
        session.error("Failed to find MiniZinc bin/ directory.")

    # Save the bin path so other sessions can use it
    minizinc_bin = str(bin_path.resolve())
    minizinc_path_file.write_text(minizinc_bin)

    env = session.env.copy()
    env["PATH"] = f"{minizinc_bin}:{env.get('PATH', '')}"
    session.log(f"MiniZinc PATH: {minizinc_bin}")

    session.run("bash", "-c", f"export PATH='{minizinc_bin}:$PATH' && minizinc --version", env=env)

