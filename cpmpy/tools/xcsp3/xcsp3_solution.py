import xml.etree.cElementTree as ET
import cpmpy as cp

def solution_xml(model, useless_style="*", boolean_style="int"):
    """
        Formats a solution according to the XCSP3 specification.
    """

    # CSP
    if not model.has_objective():
        root = ET.Element("instantiation", type="solution")
    # COP
    else:
        root = ET.Element("instantiation", type="optimum", cost=str(int(model.objective_value())))

    # How useless variables should be handled
    #    (variables which have value `None` in the solution)
    variables = {var.name: var for var in model.user_vars}
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