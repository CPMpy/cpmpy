"""
XCSP3 JAR-based solution checker.

Wraps the official XCSP3 Java checker JAR so it can be called from both the
legacy benchmark pipeline and the cpbenchy observer (XCSP3JarCheckerObserver).
"""

import subprocess
import time


def run_solution_checker(jar: str, instance_path: str, solution_file: str) -> tuple[str, float]:
    """
    Run the official XCSP3 Java checker JAR against a solution file.

    Args:
        jar: Path to the checker JAR file.
        instance_path: Path to the XCSP3 instance (XML or XML.lzma).
        solution_file: Path to a file containing the ``v <xml>`` solution line.

    Returns:
        (verdict, checker_time) where verdict is the last non-empty output line
        from the checker and checker_time is wall-clock seconds taken.
    """
    command = " ".join(["java", "-jar", jar,
                        "'" + str(instance_path) + "'",
                        str(solution_file)])
    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    checker_time = time.time() - start

    lines = [l for l in result.stdout.split("\n") if l.strip()]
    verdict = lines[-1] if lines else ""
    return verdict, checker_time
