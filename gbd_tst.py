import cpmpy as cp
from cpmpy.tools.dataset.problem.jsplib import JSPLibDataset
from cpmpy.tools.jsplib.parser import read_jsplib, _parse_jsplib, _model_jsplib
from cpmpy.tools.dimacs import write_dimacs
import tqdm

from cpmpy.transformations.int2bool import int2bool
from cpmpy.transformations.decompose_global import decompose_in_tree
from cpmpy.transformations.flatten_model import flatten_constraint, flatten_model
from cpmpy.transformations.linearize import linearize_constraint, only_positive_bv, only_positive_coefficients
from cpmpy.transformations.normalize import simplify_boolean, toplevel_list
from cpmpy.transformations.reification import only_bv_reifies, only_implies

if __name__ == "__main__":

    dataset = JSPLibDataset(root=".", download=True)
    for instance, metadata in tqdm.tqdm(dataset):
        model = read_jsplib(instance)

        cnf_model = cp.Model()
        csemap = {}
        encoding = "auto"
        ivarmap = dict()
        for cpm_expr in model.constraints:
            cpm_cons = toplevel_list(cpm_expr)
            cpm_cons = decompose_in_tree(cpm_cons, csemap=csemap)
            cpm_cons = simplify_boolean(cpm_cons)
            cpm_cons = flatten_constraint(cpm_cons, csemap=csemap)  # flat normal form
            cpm_cons = only_bv_reifies(cpm_cons, csemap=csemap)
            cpm_cons = only_implies(cpm_cons, csemap=csemap)
            cpm_cons = linearize_constraint(cpm_cons, csemap=csemap)  # the core of the MIP-linearization
            cpm_cons = int2bool(cpm_cons, ivarmap, encoding=encoding)
            cpm_cons = only_positive_coefficients(cpm_cons)
            cpm_cons = only_positive_bv(cpm_cons, csemap=csemap)
            cnf_model += (cpm_cons)
        
        cnf_model = flatten_model(cnf_model)

        fname = instance + ".dimacs.cnf"
        write_dimacs(cnf_model, fname)
        