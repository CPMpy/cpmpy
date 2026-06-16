"""
XCSP3 JAR-based solution checker.

Wraps the official XCSP3 Java checker JAR so it can be called from both the
legacy benchmark pipeline and the cpbenchy observer (XCSP3JarCheckerObserver).
"""

import subprocess
import time
from collections.abc import Callable


def _looks_like_competition_output(solution_file: str) -> bool:
    """Return True when the file contains XCSP3 competition output lines."""
    try:
        with open(solution_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.lstrip()
                if not stripped:
                    continue
                if stripped.startswith(("v ", "s ")):
                    return True
                if stripped.startswith("<instantiation"):
                    return False
    except OSError:
        return False
    return False


def run_solution_checker(
    jar: str,
    instance_path: str,
    solution_file: str,
    debug_callback: Callable[[str], None] | None = None,
) -> tuple[str, float]:
    """
    Run the official XCSP3 Java checker JAR against a solution file.

    Args:
        jar: Path to the checker JAR file.
        instance_path: Path to the XCSP3 instance (XML or XML.lzma).
        solution_file: Path to a file containing either raw ``<instantiation>``
            XML or full XCSP3 competition output (``c``/``o``/``v``/``s`` lines).
        debug_callback: Optional callback that receives the exact command and
            solution-file content that will be passed to the checker.

    Returns:
        (verdict, checker_time) where verdict is the last non-empty output line
        from the checker and checker_time is wall-clock seconds taken.
    """
    competition_mode = _looks_like_competition_output(solution_file)

    command = ["java", "-jar", jar, str(instance_path), str(solution_file)]
    if competition_mode:
        command.append("-cm")

    if debug_callback is not None:
        debug_callback("command: " + " ".join(command))
        debug_callback("solution input begin")
        with open(solution_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                debug_callback(line.rstrip("\r\n"))
        debug_callback("solution input end")

    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    checker_time = time.time() - start

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lines = [l for l in output.split("\n") if l.strip()]
    verdict = lines[-1] if lines else ""
    return verdict, checker_time
