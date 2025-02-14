import unittest
import tempfile

import cpmpy as cp
from cpmpy.tools.dimacs import read_dimacs, write_dimacs
from cpmpy.transformations.get_variables import get_variables_model

import io

class CNFTool(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpfile = tempfile.NamedTemporaryFile()

    def dimacs_to_model(self, cnf_str):
        # return read_dimacs(io.StringIO(cnf_str))
        with open(self.tmpfile.name, "w") as f:
            f.write(cnf_str)
        return read_dimacs(self.tmpfile.name)

    def test_read_cnf(self):
        model = self.dimacs_to_model("p cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n")
        bvs = sorted(get_variables_model(model), key=str)
        self.assertEqual(str(model), str(cp.Model(
            cp.any([~bvs[1], ~bvs[2]]), cp.any([bvs[2], bvs[1],bvs[0]]), ~bvs[0])
                         ))

    def test_empty_formula(self):
        model = self.dimacs_to_model("p cnf 0 0")
        self.assertTrue(model.solve())
        self.assertEqual(model.status().exitstatus, cp.solvers.solver_interface.ExitStatus.OPTIMAL)

    def test_empty_clauses(self):
        model = self.dimacs_to_model("p cnf 0 2\n0\n0")
        self.assertFalse(model.solve())
        self.assertEqual(model.status().exitstatus, cp.solvers.solver_interface.ExitStatus.UNSATISFIABLE)

    def test_with_comments(self):
        model = self.dimacs_to_model("c this file starts with some comments\nc\np cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n")
        vars = sorted(get_variables_model(model), key=str)

        sols = set()
        addsol = lambda : sols.add(tuple([v.value() for v in vars]))

        self.assertEqual(model.solveAll(display=addsol), 2)
        self.assertSetEqual(sols, {(False, False, True), (False, True, False)})

    def test_write_cnf(self):

        a,b,c = [cp.boolvar(name=n) for n in "abc"]

        m = cp.Model()
        m += cp.any([a,b,c])
        m += b.implies(~c)
        m += a <= 0

        cnf_txt = write_dimacs(m)
        gt_cnf = "p cnf 3 3\n1 2 3 0\n-2 -3 0\n-1 0\n"

        self.assertEqual(cnf_txt, gt_cnf)


