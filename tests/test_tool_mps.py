
import pytest
import unittest
import tempfile
import os
import cpmpy as cp
from cpmpy.tools.mps import read_mps, write_mps
from cpmpy.transformations.get_variables import get_variables
from cpmpy.tools.mps.parser import MPS

class MPSTool(unittest.TestCase):

    mps = """\
NAME          CPMPYMODEL
ROWS
 N  minobj
 L  c0
 G  c1
 E  c2
COLUMNS
    MARK0000  MARKER               INTORG
    XONE      minobj               1   c0                   1
    XONE      c1                   1
    YTWO      minobj               4   c0                   1
    YTWO      c2                   -1
    ZTHREE    minobj               9   c1                   1
    ZTHREE    c2                   1
    MARK0001  MARKER               INTEND
RHS
    rhs       c0                   5   c1                   10
    rhs       c2                   7
BOUNDS
 LI bnd       XONE                 0
 UI bnd       XONE                 4
 LI bnd       YTWO                 -1
 UI bnd       YTWO                 1
 FX bnd       ZTHREE               3
ENDATA\
"""
    def setUp(self) -> None:
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def tearDown(self) -> None:
        self.tmpfile.close()
        os.remove(self.tmpfile.name)

    def test_read_mps(self):
        
        model = read_mps(self.mps, model_constants=True, filter_zeros=False)

        # 1) test variables
        variables = get_variables(model.constraints)
        for variable in variables:
            if variable.name == "XONE":
                self.assertEqual(variable.lb, 0)
                self.assertEqual(variable.ub, 4)
            elif variable.name == "YTWO":
                self.assertEqual(variable.lb, -1)
                self.assertEqual(variable.ub, 1)
            elif variable.name == "ZTHREE":
                self.assertEqual(variable.lb, 3)
                self.assertEqual(variable.ub, 3)
            else:
                self.fail(f"Unexpected variable: {variable.name}")

        # 2) test objective
        assert str(model.objective_) == str(cp.sum(cp.cpm_array([1, 4, 9])*cp.cpm_array([cp.intvar(0, 4, name="XONE"), cp.intvar(-1, 1, name="YTWO"), cp.intvar(3, 3, name="ZTHREE")])))

        # 3) test constraints
        assert str(model.constraints[0]) == str(cp.intvar(0, 4, name="XONE") + cp.intvar(-1, 1, name="YTWO") <= 5)
        assert str(model.constraints[1]) == str(cp.intvar(0, 4, name="XONE") + cp.intvar(3, 3, name="ZTHREE") >= 10)
        assert str(model.constraints[2]) == str(cp.sum(cp.cpm_array([-1, 1])*cp.cpm_array([cp.intvar(-1, 1, name="YTWO"), cp.intvar(3, 3, name="ZTHREE")])) == 7)
   

    def test_write_mps(self):

        


        model = read_mps(self.mps, model_constants=True, filter_zeros=False)
        print(model)

        # mps_obj = MPS().from_cpmpy(model)
        # print(mps_obj)


        
        mps = write_mps(model)
        assert mps == self.mps

