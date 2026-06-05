from cpmpy.tools.benchmark.cpbenchy.adapter._base import InstanceAdapter
import xml.etree.ElementTree as ET

from cpmpy.tools.benchmark.cpbenchy.observer import (
    DIMACSPrintingObserver,
    HandlerObserver,
    RuntimeObserver,
    ResourceLimitObserver,
    SolverArgsObserver,
)
from cpmpy.tools.io.nurserostering import read_nurserostering


def solution_xcsp3(model, useless_style="*", boolean_style="int"):
    """
        Formats a solution according to the XCSP3 specification.

        Arguments:
            model: CPMpy model for which to format its solution (should be solved first)
            useless_style: How to process unused decision variables (with value `None`).
                           If "*", variable is included in reporting with value "*".
                           If "drop", variable is excluded from reporting.
            boolean_style: Print style for boolean constants.
                           "int" results in 0/1, "bool" results in False/True.

        Returns:
            XML-formatted model solution according to XCSP3 specification.
    """

    # CSP
    if not model.has_objective():
        root = ET.Element("instantiation", type="solution")
    # COP
    else:
        root = ET.Element("instantiation", type="optimum", cost=str(int(model.objective_value())))

    # How useless variables should be handled
    #    (variables which have value `None` in the solution)
    variables = {var.name: var for var in model.user_vars if var.name[:2] not in ["IV", "BV", "B#"]} # dirty workaround for all missed aux vars in user vars TODO fix with Ignace
    if useless_style == "*":
        variables = {k:(v.value() if v.value() is not None else "*") for k,v in variables.items()}
    elif useless_style == "drop":
        variables = {k:v.value() for k,v in variables.items() if v.value() is not None}

    # Convert booleans
    if boolean_style == "bool":
        pass
    elif boolean_style == "int":
        variables = {k:(v if (not isinstance(v, bool)) else (1 if v else 0)) for k,v in variables.items()}

    # Build XCSP3 XML tree
    ET.SubElement(root, "list").text=" " + " ".join([str(v) for v in variables.keys()]) + " "
    ET.SubElement(root, "values").text=" " + " ".join([str(v) for v in variables.values()]) + " "
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    res = ET.tostring(root).decode("utf-8")

    return str(res)


class NurseRosteringCompetitionPrintingObserver(DIMACSPrintingObserver):
    """Nurse rostering competition-style output printer."""

    def __init__(self, verbose: bool = False, **kwargs):
        super().__init__(solution_printer=solution_xcsp3, verbose=verbose, **kwargs)


class NurseRosteringAdapter(InstanceAdapter):

    default_observers = [
        NurseRosteringCompetitionPrintingObserver,
        RuntimeObserver,
        HandlerObserver,
        SolverArgsObserver,
            ResourceLimitObserver,
    ]

    # Keep parser as plain callable; avoid instance-method binding.
    reader = staticmethod(read_nurserostering)

    def cmd(self, instance: str, solver: str = "ortools", output_file: str = None, **kwargs):
        cmd = self.base_cmd(instance)
        if solver is not None:
            cmd.append("--solver")
            cmd.append(solver)
        if output_file is not None:
            cmd.append("--output_file")
            cmd.append(output_file)
        return cmd

    def print_comment(self, comment: str):
        print("c" + chr(32) + comment.rstrip("\n"), end="\r\n", flush=True)


def main():
    runner = NurseRosteringAdapter()
    parser = runner.argparser()
    args = parser.parse_args()
    runner.run(**vars(args))


if __name__ == "__main__":
    main()
