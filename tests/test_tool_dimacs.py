import os
import unittest
import tempfile

import pytest
import cpmpy as cp
from cpmpy.tools.dimacs import read_dimacs, write_dimacs, write_gcnf
from cpmpy.transformations.get_variables import get_variables_model
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.solvers.pindakaas import CPM_pindakaas
from test_tocnf import get_gcnf_cases



# @pytest.mark.skipif(not CPM_pindakaas.supported(), reason="Pindakaas (required for `to_cnf`) not installed")
class CnfTool(unittest.TestCase):

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
    
class TestDimacs:
    def test_gcnf(self):
        x = cp.boolvar(shape=3, name="x")
        def x_(i):
            return x[i-1]

        # example from https://satisfiability.org/competition/2011/rules.pdf
        # c
        # c Example of group oriented CNF
        # c
        # c Represents the following formula
        # c D
        # = {x1 or x2 or x3}
        # c G1 = {x1 -> x2, x2 -> x3}
        # c G2 = {x3}
        # c G3 = {x3 -> x2, -x2 or -x3}
        # c G4 = {x2 -> x3}
        # c
        hard = [cp.any(x)]
        soft = [
            x_(1).implies(x_(2)) & x_(2).implies(x_(3)),
            x_(3),
            x_(3).implies(x_(2)) & (~x_(2)) | (~x_(3)),
            x_(2).implies(x_(3)),
        ]

        # TODO current encoding is different from the example
        #         assert write_gcnf(soft, hard=hard, encoding="direct") == """p gcnf 5 7 4
        # {0} 1 2 3 0
        # {1} -1 2 0
        # {1} -2 3 0
        # {2} -3 0
        # {3} 2 -3 0
        # {3} -2 -3 0
        # {4} -2 3 0
        # """

        assert write_gcnf(soft, hard=hard, name="a", encoding="direct") == """p gcnf 5 12 4
{0} 1 2 3 0
{0} -4 5 0
{0} -2 -4 0
{0} 4 -5 2 0
{0} -5 -3 2 0
{0} 3 5 0
{0} -2 5 0
{1} -1 2 0
{1} -2 3 0
{2} 3 0
{3} 4 -3 0
{4} -2 3 0
"""

        # note: 2nd clause of group 4 is merged with 2nd clause of group 1
        assert write_gcnf(soft, hard=hard, name="a", encoding="direct", normalize=True) == """p gcnf 6 13 4
{0} 1 2 3 0
{0} -4 5 0
{0} -2 -4 0
{0} 4 -5 2 0
{0} -5 -3 2 0
{0} 3 5 0
{0} -2 5 0
{0} -6 -2 3 0
{1} -1 2 0
{1} -2 3 0
{2} 3 0
{3} 4 -3 0
{4} 6 0
"""


    @pytest.mark.parametrize(
        "case",
        get_gcnf_cases(),
    )
    def test_normalized_gcnf(self, case):
        print("case", case)
        soft, hard = case
        fname = tempfile.mktemp()
        write_gcnf(soft, hard=hard, name="a", encoding="direct", normalize=True, fname=fname)

