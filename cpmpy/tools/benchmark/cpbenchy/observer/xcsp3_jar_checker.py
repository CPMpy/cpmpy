"""Observer that runs the official XCSP3 Java checker JAR after a solve."""

import tempfile
import os

from cpmpy.tools.benchmark.cpbenchy.observer.base import Observer
from cpmpy.tools.benchmark.cpbenchy.runner.runner import Runner
from cpmpy.tools.xcsp3.checker import run_solution_checker


class XCSP3JarCheckerObserver(Observer):
    """
    Runs the official XCSP3 Java checker JAR on the solution after each solve.

    Prints verdict and timing as ``c ...`` comment lines, consistent with the
    XCSP3 competition output format.

    Args:
        jar_path: Path to the XCSP3 checker JAR file.
    """

    def __init__(self, jar_path: str, **kwargs):
        super().__init__(**kwargs)
        self.jar_path = jar_path
        self._output_lines = []
        self._checking = False

    def _capture_line(self, line: str):
        if self._checking:
            return
        self._output_lines.extend(str(line).rstrip("\r\n").splitlines())

    def print_raw(self, text: str):
        self._capture_line(text)

    def observe_end(self, runner: Runner):
        if not runner.is_sat:
            return  # No solution to check
        # In RunExec parent-process replay mode, the child has already run the
        # checker and only relayed raw output/metadata. The live solver object is
        # not available there, so there is no solution object to format.
        if getattr(runner, "s", None) is None:
            return

        # Write the raw competition output seen by this observer to a temp file.
        # This deliberately preserves c/o/v/s lines exactly as emitted.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write("\n".join(self._output_lines))
            tmp.write("\n")
            tmp_path = tmp.name

        self._checking = True
        try:
            verdict, checker_time = run_solution_checker(
                jar=self.jar_path,
                instance_path=runner.instance,
                solution_file=tmp_path,
                # debug_callback=lambda line: runner.print_comment(
                #     f"xcsp3-checker input: {line}"
                # ),
            )
            checker_payload = {
                "checker": "xcsp3-solutionChecker",
                "jar_path": self.jar_path,
                "verdict": verdict,
                "time": checker_time,
                "valid": verdict.startswith("OK"),
            }
            setattr(runner, "solution_checker", checker_payload)
            if hasattr(runner, "runner_metadata") and isinstance(runner.runner_metadata, dict):
                runner.runner_metadata["solution_checker"] = checker_payload
            runner.print_comment(f"xcsp3-checker result: {verdict}")
            runner.print_comment(f"xcsp3-checker time: {checker_time:.3f}s")
        except Exception as e:
            checker_payload = {
                "checker": "xcsp3-solutionChecker",
                "jar_path": self.jar_path,
                "verdict": None,
                "time": None,
                "valid": False,
                "error": str(e),
            }
            setattr(runner, "solution_checker", checker_payload)
            if hasattr(runner, "runner_metadata") and isinstance(runner.runner_metadata, dict):
                runner.runner_metadata["solution_checker"] = checker_payload
            runner.print_comment(f"xcsp3-checker failed: {e}")
        finally:
            self._checking = False
            os.unlink(tmp_path)
