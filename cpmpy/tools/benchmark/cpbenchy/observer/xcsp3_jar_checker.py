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

    def observe_end(self, runner: Runner):
        if not runner.is_sat:
            return  # No solution to check
        # In RunExec parent-process replay mode, the child has already run the
        # checker and only relayed raw output/metadata. The live solver object is
        # not available there, so there is no solution object to format.
        if getattr(runner, "s", None) is None:
            return

        from cpmpy.tools.benchmark.cpbenchy.adapter.xcsp3 import solution_xcsp3
        try:
            solution_xml = solution_xcsp3(runner.s)
        except Exception as e:
            runner.print_comment(f"xcsp3-checker: could not format solution: {e}")
            return

        # Write solution to a temp file that the JAR can read
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(solution_xml + "\n")
            tmp_path = tmp.name

        try:
            verdict, checker_time = run_solution_checker(
                jar=self.jar_path,
                instance_path=runner.instance,
                solution_file=tmp_path,
            )
            runner.print_comment(f"xcsp3-checker result: {verdict}")
            runner.print_comment(f"xcsp3-checker time: {checker_time:.3f}s")
        except Exception as e:
            runner.print_comment(f"xcsp3-checker failed: {e}")
        finally:
            os.unlink(tmp_path)
