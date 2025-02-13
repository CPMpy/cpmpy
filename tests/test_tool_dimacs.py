import unittest
import tempfile

import cpmpy as cp
from cpmpy.tools.dimacs import read_dimacs, write_dimacs
from cpmpy.transformations.get_variables import get_variables_model
class CNFTool(unittest.TestCase):


    def setUp(self) -> None:
        self.tmpfile = tempfile.NamedTemporaryFile()

    def test_read_cnf(self):

        """
        a | b | c,
        ~b | ~c,
        ~a
        """
        cnf_txt = "p cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n"
        with open(self.tmpfile.name, "w") as f:
            f.write(cnf_txt)

        model = read_dimacs(self.tmpfile.name)
        vars = sorted(get_variables_model(model), key=str)

        sols = set()
        addsol = lambda : sols.add(tuple([v.value() for v in vars]))

        self.assertEqual(model.solveAll(display=addsol), 2)
        self.assertSetEqual(sols, {(False, False, True), (False, True, False)})


    def test_read_bigint(self):

        cnf_txt = "p cnf 3 3 \n-2   -3 0\n3 2 1 0\n-1 0\n"
        with open(self.tmpfile.name, "w") as f:
            f.write(cnf_txt)

        model = read_dimacs(self.tmpfile.name)
        vars = sorted(get_variables_model(model), key=str)

        self.assertEqual(model.solveAll(), 2)

    def test_with_comments(self):
        cnf_txt = "c this file starts with some comments\nc\np cnf 3 3\n-2 -3 0\n3 2 1 0\n-1 0\n"

        with open(self.tmpfile.name, "w") as f:
            f.write(cnf_txt)

        model = read_dimacs(self.tmpfile.name)
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


