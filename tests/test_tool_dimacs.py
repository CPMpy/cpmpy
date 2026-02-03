import os
import unittest
import tempfile

import pytest
import cpmpy as cp
from cpmpy.tools.dimacs import read_dimacs, write_dimacs
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.solvers.pindakaas import CPM_pindakaas



@pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class CNFTool(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def tearDown(self) -> None:
        self.tmpfile.close()
        os.remove(self.tmpfile.name)

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
        self.assertEqual(model.status().exitstatus, ExitStatus.FEASIBLE)

    def test_empty_clauses(self):
        model = self.dimacs_to_model("p cnf 0 2\n0\n0")
        self.assertFalse(model.solve())
        self.assertEqual(model.status().exitstatus, ExitStatus.UNSATISFIABLE)

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

        gt_cnf = "p cnf 3 3\n1 2 3 0\n-2 -3 0\n-1 0\n"
        gt_clauses = set(gt_cnf.split("\n")[1:]) # skip the p-line

        cnf_txt = write_dimacs(model=m)
        cnf_clauses = set(cnf_txt.split("\n")[1:]) # skip the p-line
       
        self.assertEqual(cnf_clauses, gt_clauses)


    def test_missing_p_line(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("1 -2 0\np cnf 2 2")

    def test_incorrect_p_line(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0")

    def test_too_many_clauses(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0\n1 0\n2 0")

    def test_too_few_clauses(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 0")

    def test_too_many_variables(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 1\n1 2 3 0")

    def test_non_int_literal(self):
        with self.assertRaises(ValueError):
            self.dimacs_to_model("p cnf 2 1\n1 b 2 0")

    def test_non_terminated_final_clause(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 2\n1 2 0\n-1 -2 0\n2")


    @pytest.mark.skip(reason="We allow fewer variables, because this is technically correct DIMACS")
    def test_too_few_variables(self):
        with self.assertRaises(AssertionError):
            self.dimacs_to_model("p cnf 2 1\n1 0")


